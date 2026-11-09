"""Optional real-model equivalence test against pinned karpathy/llama2.c.
Downloads upstream C source into a temporary directory and compiles it locally.
"""


import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile
import urllib.request


REVISION = '350e04fe35433e6d2941dce5a1f53308f87058eb'


WRAPPER = r'''
#define TESTING
#include <stdint.h>
#include "reference.c"
int main(int argc, char **argv) {
    if (argc != 5) return 2;
    Transformer model;
    build_transformer(&model, argv[1]);
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, argv[2], model.config.vocab_size);
    int n;
    int *tokens = malloc((strlen(argv[3]) + 3) * sizeof(int));
    encode(&tokenizer, argv[3], 1, 0, tokens, &n);
    int outputs = atoi(argv[4]);
    if (n + outputs - 1 > model.config.seq_len) return 3;
    int token = tokens[0];
    for (int pos = 0; pos < n + outputs - 1; pos++) {
        float *logits = forward(&model, token, pos);
        token = pos < n - 1 ? tokens[pos + 1] : sample_argmax(logits, model.config.vocab_size);
        if (pos >= n - 1) printf("%d ", token);
    }
    free(tokens);
    free_tokenizer(&tokenizer);
    free_transformer(&model);
    return 0;
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--model', type=Path, default=Path('models/stories15M.bin'))
    parser.add_argument('--tokenizer', type=Path, default=Path('models/tokenizer.bin'))
    parser.add_argument('--binary', type=Path, default=Path('build/nanovllm'))
    parser.add_argument('--source', type=Path, help='use already-downloaded reference run.c')
    parser.add_argument('--tokens', type=int, default=32)
    args = parser.parse_args()

    model, tokenizer, binary = (str(path.resolve()) for path in (args.model, args.tokenizer, args.binary))

    with tempfile.TemporaryDirectory(prefix='nanovllm-reference-') as directory:
        root = Path(directory)
        if args.source:
            shutil.copyfile(args.source, root / 'reference.c')
        if not args.source:
            url = f'https://raw.githubusercontent.com/karpathy/llama2.c/{REVISION}/run.c'
            with urllib.request.urlopen(url, timeout=60) as response:
                (root / 'reference.c').write_bytes(response.read())
        (root / 'wrapper.c').write_text(WRAPPER)
        reference = str(root / 'reference')

        subprocess.run(['cc', '-O3', '-std=c11', '-D_POSIX_C_SOURCE=200809L',
                        str(root / 'wrapper.c'), '-lm', '-o', reference], check=True)

        for prompt in ['Once upon a time', 'One day, Lily', '', '你好']:
            result = subprocess.run([reference, model, tokenizer, prompt, str(args.tokens)],
                                    capture_output=True, text=True, check=True, timeout=180)
            expected = [int(value) for value in result.stdout.split()]

            for extra in [[], ['--no-cache', '--block-size', '3', '--budget', '1']]:
                result = subprocess.run([binary, model, '-z', tokenizer, '-i', prompt,
                                         '-n', str(args.tokens), '-t', '0', '--ids', '--ignore-eos'] + extra,
                                        capture_output=True, text=True, check=True, timeout=180)

                actual = [int(value) for value in result.stdout.split()[1:]]
                assert actual == expected, (prompt, extra, actual, expected)
            print(f'PASS {Path(model).name}: {prompt!r}, {len(expected)} greedy tokens match upstream')

if __name__ == '__main__':
    main()
