"""Deterministic tiny checkpoints and a separate dense, double-precision oracle.
Only Python's standard library is needed. Random weights test math, not language.
"""


import math
from pathlib import Path
import random
import struct
import sys


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def norm(x, weight):
    scale = 1 / math.sqrt(sum(v * v for v in x) / len(x) + 1e-5)
    return [a * scale * b for a, b in zip(x, weight)]


def mat(weight, x):
    return [sum(a * b for a, b in zip(weight[i:i + len(x)], x))
            for i in range(0, len(weight), len(x))]


def softmax(x):
    exps = [math.exp(v - max(x)) for v in x]
    return [v / sum(exps) for v in exps]


def make_model(root, name, kv_heads, tied):
    d, hidden, layers, heads, vocab, context = 16, 24, 2, 4, 263, 64
    hs, kv = d // heads, d // heads * kv_heads

    rng = random.Random(42 + kv_heads)

    shapes = [('embedding', vocab * d), ('att_norm', layers * d),
              ('wq', layers * d * d), ('wk', layers * kv * d), ('wv', layers * kv * d),
              ('wo', layers * d * d), ('ffn_norm', layers * d),
              ('w1', layers * hidden * d), ('w2', layers * d * hidden),
              ('w3', layers * hidden * d), ('final_norm', d), ('rope', context * hs)]
    if not tied:
        shapes.append(('classifier', vocab * d))

    w = {key: [f32(1 + rng.uniform(-0.1, 0.1) if 'norm' in key else rng.uniform(-0.15, 0.15))
                for _ in range(size)] for key, size in shapes}

    with (root / (name + '.bin')).open('wb') as file:
        file.write(struct.pack('<7i', d, hidden, layers, heads, kv_heads, vocab if tied else -vocab, context))
        for key, _ in shapes:
            file.write(struct.pack('<%df' % len(w[key]), *w[key]))
    if name == 'gqa':
        stop = {key: [1.0 if 'norm' in key or key == 'embedding' else 0.0 for _ in values]
                for key, values in w.items()}
        stop['classifier'][2 * d:3 * d] = [1.0] * d
        with (root / 'eos.bin').open('wb') as file:
            file.write(struct.pack('<7i', d, hidden, layers, heads, kv_heads, -vocab, context))
            for key, _ in shapes:
                file.write(struct.pack('<%df' % len(stop[key]), *stop[key]))
    if tied:
        w['classifier'] = w['embedding']

    keys, values = [[] for _ in range(layers)], [[] for _ in range(layers)]

    tokens = [1, 259, 260, 261, 262, 198, 172, 12, 88, 5, 259, 83, 41]
    logits = []
    for pos, token in enumerate(tokens):
        x = w['embedding'][token * d:(token + 1) * d]
        for layer in range(layers):
            xb = norm(x, w['att_norm'][layer * d:(layer + 1) * d])
            q = mat(w['wq'][layer * d * d:(layer + 1) * d * d], xb)
            k = mat(w['wk'][layer * kv * d:(layer + 1) * kv * d], xb)
            v = mat(w['wv'][layer * kv * d:(layer + 1) * kv * d], xb)

            for vector in (q, k):
                for i in range(0, len(vector), 2):
                    angle = pos / (10000 ** ((i % hs) / hs))
                    a, b = vector[i:i + 2]
                    vector[i:i + 2] = [a * math.cos(angle) - b * math.sin(angle),
                                       a * math.sin(angle) + b * math.cos(angle)]

            keys[layer].append(k)
            values[layer].append(v)
            attention = []
            for head in range(heads):
                start = (head // (heads // kv_heads)) * hs

                probs = softmax([sum(a * b for a, b in zip(q[head * hs:(head + 1) * hs],
                                                         past[start:start + hs])) / math.sqrt(hs)
                                 for past in keys[layer]])

                attention.extend(sum(prob * past[start + j] for prob, past in zip(probs, values[layer]))
                                 for j in range(hs))

            x = [a + b for a, b in zip(x, mat(w['wo'][layer * d * d:(layer + 1) * d * d], attention))]
            xb = norm(x, w['ffn_norm'][layer * d:(layer + 1) * d])

            a = mat(w['w1'][layer * hidden * d:(layer + 1) * hidden * d], xb)
            b = mat(w['w3'][layer * hidden * d:(layer + 1) * hidden * d], xb)
            hb = [u / (1 + math.exp(-u)) * v for u, v in zip(a, b)]
            x = [a + b for a, b in zip(x, mat(w['w2'][layer * hidden * d:(layer + 1) * hidden * d], hb))]

        logits.extend(mat(w['classifier'], norm(x, w['final_norm'])))

    with (root / (name + '.oracle')).open('wb') as file:
        file.write(struct.pack('<i', len(tokens)))
        file.write(struct.pack('<%di' % len(tokens), *tokens))
        file.write(struct.pack('<%df' % len(logits), *logits))


def main():
    root = Path(sys.argv[1])
    root.mkdir(parents=True, exist_ok=True)

    make_model(root, 'gqa', 2, False)
    make_model(root, 'mha', 4, True)

    vocab = ['<unk>', '<s>', '</s>'] + ['<0x%02X>' % i for i in range(256)] + [' ', 'a', 'b', 'ab']
    with (root / 'tokenizer.bin').open('wb') as file:
        file.write(struct.pack('<i', max(map(len, vocab))))
        for i, text in enumerate(vocab):
            data = text.encode()

            file.write(struct.pack('<fi', float(i), len(data)))
            file.write(data)

    (root / 'truncated.bin').write_bytes((root / 'gqa.bin').read_bytes()[:128])
    (root / 'bad-config.bin').write_bytes(struct.pack('<7i', 16, 24, 2, 3, 2, 263, 64))
    (root / 'bad-tokenizer.bin').write_bytes(struct.pack('<ifi', 6, 0.0, -1))
    (root / 'short-tokenizer.bin').write_bytes((root / 'tokenizer.bin').read_bytes()[:-1])
    (root / 'extra-tokenizer.bin').write_bytes((root / 'tokenizer.bin').read_bytes() + b'X')
    print('Created MHA/GQA fixtures and independent dense logits oracle')

if __name__ == '__main__':
    main()
