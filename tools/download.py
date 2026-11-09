"""Download pretrained TinyStories weights + matching tokenizer (no pip packages).
Network is only used by this opt-in script, never by the C executable.
"""


import argparse
import hashlib
from pathlib import Path
import shutil
import urllib.error
import urllib.request


HF = 'https://huggingface.co/karpathy/tinyllamas/resolve/0bd21da7698eaf29a0d7de3992de8a46ef624add/'


ARTIFACTS = {
    '15m': [
        ('stories15M.bin', HF + 'stories15M.bin', 'cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a'),
        ('tokenizer.bin', 'https://raw.githubusercontent.com/karpathy/llama2.c/350e04fe35433e6d2941dce5a1f53308f87058eb/tokenizer.bin', '50a52ef822ee9e83de5ce9d0be0a025a773d019437f58b5ff9dcafb063ece361'),
    ],
    '260k': [
        ('stories260K.bin', HF + 'stories260K/stories260K.bin', 'b0a507e7ad0f626624f17112325e66691f9076d622e1d3274d103d00299f2696'),
        ('tok512.bin', HF + 'stories260K/tok512.bin', '037cb335abb25d1fa9e8ecae30ed2a3a8ace9302862ebcdc05d51a6bbb10c312'),
    ],
}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--model', choices=ARTIFACTS, default='15m')
    parser.add_argument('--directory', type=Path, default=Path(__file__).resolve().parents[1] / 'models')
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    for filename, url, expected in ARTIFACTS[args.model]:
        target = args.directory / filename

        if target.is_file() and digest(target) == expected:
            print('Verified:', target, flush=True)
            continue

        partial = target.with_suffix(target.suffix + '.part')
        print('Downloading:', url, flush=True)
        try:
            with urllib.request.urlopen(url, timeout=60) as response, partial.open('wb') as output:
                shutil.copyfileobj(response, output)
            actual = digest(partial)

            if actual != expected:
                raise ValueError(f'SHA256 mismatch for {filename}: expected {expected}, got {actual}')

            partial.replace(target)
        except (OSError, urllib.error.URLError, ValueError) as error:
            partial.unlink(missing_ok=True)
            raise SystemExit(f'Download failed: {error}') from error
        print('Saved:', target, flush=True)

if __name__ == '__main__':
    main()
