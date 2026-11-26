#!/usr/bin/env bash
set -euo pipefail
binary=${1:?Usage: profile_graph.sh BENCH_DECODE MODEL [OUTPUT_DIR] [DTYPE]}
model=${2:?model checkpoint required}
out=${3:-benchmarks/operators/round2-results/nsys}
dtype=${4:-f16}
mkdir -p "$out"
out=$(realpath "$out")
if [[ -e "$out/decode-graph.nsys-rep" ]]; then
  echo "Choose a fresh output directory" >&2; exit 2
fi
mkdir -p "$out/tmp"
export NSYS_TMPDIR="$out/tmp"
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)" ]]; then
  echo "Foreign CUDA compute processes present; refusing NSys capture" >&2; exit 2
fi
python3 "$(dirname "$0")/../audit/guard_command.py" "$out/telemetry.jsonl" -- nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop \
  --force-overwrite=true --output="$out/decode-graph" \
  "$binary" --trace-decode "$model" "$dtype" > "$out/profile.stdout" 2> "$out/profile.log"
nsys export --type=sqlite --force-overwrite=true --output="$out/decode-graph.sqlite" \
  "$out/decode-graph.nsys-rep" > "$out/export.log" 2>&1
nsys stats --force-export=true --report cuda_api_sum,cuda_gpu_kern_sum,nvtx_sum --format csv \
  --output "$out/stats" "$out/decode-graph.nsys-rep" > "$out/stats.log" 2>&1
