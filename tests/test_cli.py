"""Exercise the actual process boundary, malformed files and deterministic output."""


from pathlib import Path
import subprocess
import sys


binary, root = sys.argv[1], Path(sys.argv[2])
base = [binary, str(root / 'gqa.bin'), '-z', str(root / 'tokenizer.bin')]


def run(args, ok=True):
    result = subprocess.run(args, capture_output=True, text=True, errors='replace', timeout=30)
    assert (result.returncode == 0) == ok, (args, result.returncode, result.stderr)
    return result

run([binary, '--help'])
run([binary], False)


for args in [['-n', '-1'], ['-n', '4oops'], ['-n'], ['-t', 'nan'], ['-t', 'inf'],
             ['-t', '-1'], ['-p', '0'], ['-p', '1.1'], ['--budget', '0'],
             ['--wat', '3'], ['--block-size', '65'], ['--blocks', '1'], ['-n', '9999']]:
    result = run(base + args, False)
    assert 'nanovllm:' in result.stderr

for model in ['truncated.bin', 'bad-config.bin', 'does-not-exist.bin']:
    run([binary, str(root / model)], False)

for tokenizer in ['bad-tokenizer.bin', 'short-tokenizer.bin', 'extra-tokenizer.bin']:
    run([binary, str(root / 'gqa.bin'), '-z', str(root / tokenizer)], False)

args = ['-i', 'ab', '-n', '8', '-t', '0', '--ids', '--ignore-eos']
a = run(base + args)
b = run(base + args + ['--no-cache', '--budget', '1', '--block-size', '3'])
assert a.stdout == b.stdout

assert len(a.stdout.split()) == 9

run(base + ['-i', '你好', '-n', '4', '--ids'])
run(base + ['-i', '', '-n', '4', '--ids'])


eos_base = [binary, str(root / 'eos.bin'), '-z', str(root / 'tokenizer.bin'), '-i', 'ab', '-n', '8', '-t', '0', '--ids']
assert run(eos_base).stdout.strip() == '[0] 2'

assert run(eos_base + ['--ignore-eos']).stdout.strip() == '[0] ' + ' '.join(['2'] * 8)
print('PASS EOS termination and ignore-eos')
print('PASS CLI validation, malformed checkpoints/tokenizers, deterministic output, UTF-8')
