#!/usr/bin/env bash
# Full local fixture validation; no checkpoint download or source modification.
set -euo pipefail
nv_root=$(cd "$(dirname "$0")/.." && pwd)
nv_build=${1:-"$nv_root/build-cuda"}
nv_out=${2:-"$nv_root/benchmarks/operators/runs/$(date +%Y%m%d-%H%M%S)"}
nv_profile=${3:-}
nv_python=${NV_PYTHON:-python3}
if [[ -e "$nv_out" ]]; then printf 'Choose a fresh output directory: %s\n' "$nv_out" >&2; exit 2; fi
mkdir -p "$nv_out"
nv_build=$(realpath -m "$nv_build")
nv_out=$(realpath "$nv_out")
cmake -S "$nv_root" -B "$nv_build" -DCMAKE_BUILD_TYPE=Release \
  -DNANO_WITH_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="${NV_CUDA_ARCH:-89}" > "$nv_out/configure.log"
cmake --build "$nv_build" -j4 > "$nv_out/build.log" 2>&1
ctest --test-dir "$nv_build" --output-on-failure > "$nv_out/ctest.log" 2>&1
compute-sanitizer --tool memcheck --error-exitcode 99 --log-file "$nv_out/memcheck-final.log" \
  "$nv_build/bench_decode" --test-all "$nv_build/fixtures" > "$nv_out/memcheck.stdout" 2> "$nv_out/correctness-final.log"
compute-sanitizer --tool racecheck --error-exitcode 99 --log-file "$nv_out/racecheck-final.log" \
  "$nv_build/bench_decode" --test-all "$nv_build/fixtures" > "$nv_out/racecheck.stdout" 2> "$nv_out/racecheck-validation.log"
for nv_tool in memcheck racecheck; do
  compute-sanitizer --tool "$nv_tool" --error-exitcode 99 --log-file "$nv_out/gemv-rms-$nv_tool.log" \
    "$nv_build/bench_operators" --test > "$nv_out/gemv-rms-$nv_tool.stdout" 2> "$nv_out/gemv-rms-$nv_tool-validation.log"
done
"$nv_build/bench_decode" --attention > "$nv_out/attention.csv" 2> "$nv_out/attention.log"
"$nv_build/bench_decode" --decode "$nv_build/fixtures/perf.bin" > "$nv_out/decode.csv" 2> "$nv_out/decode.log"
nv_require=()
if [[ "$nv_profile" == "--profile" ]]; then
  sudo /usr/bin/python3 "$nv_root/benchmarks/round2/profile_all.py" "$nv_build" --out "$nv_out"
  bash "$nv_root/benchmarks/round2/profile_graph.sh" "$nv_build/bench_decode" "$nv_build/fixtures/perf.bin" "$nv_out/nsys"
  "$nv_python" "$nv_root/benchmarks/round2/analyze_nsys.py" "$nv_out/nsys/decode-graph.sqlite" > "$nv_out/nsys/analysis.log"
  nv_require=(--require-profilers)
fi
"$nv_python" "$nv_root/tools/record_cuda_environment.py" "$nv_build" "$nv_out"
"$nv_python" "$nv_root/benchmarks/round2/summarize.py" "$nv_out" "${nv_require[@]}"
printf 'Validated artifacts: %s\n' "$nv_out"
