# 冷 L2 与 Graph 评测审计

状态：冷/热Attention、低精度超L2、NCU、交错FP16 Decode、8/8 CTest及16/16项Memcheck/Racecheck矩阵已验收。16层FP16完整计算链路使用独立正确性入口完成Racecheck，保留全部动态页表与KV槽校验；旧计时循环插桩中断记录仍保留。
16层BF16动态前向超出既定CPU FP64容差，失败数据单独保留，不放宽容差、不参与正式性能矩阵。[审计报告](../../docs/PERFORMANCE_AUDIT_20261126.md)。

只读复核历史：`python3 benchmarks/audit/audit_history.py`（不启动GPU工作）。

所有 GPU 子进程串行执行。`guard.py` 在命令前、运行中每秒、完成后检查外部 CUDA
compute 进程；遇到外部工作负载只停止本脚本的子进程，整批结果拒收。图形桌面保留。
`telemetry.jsonl` 同时保存进程、显存、SM/显存频率与温度快照；未锁频时必须随数据公开。
空目录名用于新测量；旧结果不会被覆盖。每个阶段写入 `*-status.json`，只有 `complete`
代表整阶段成功；`failed/running` 的部分CSV不能冒充完整结果。当前validation为`complete`：原15项证据与新增完整模型正确性Racecheck共同闭环；补充Memcheck也通过。

```sh
cmake -S . -B build-cuda -DNANO_WITH_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-cuda -j4
python3 benchmarks/audit/make_model.py build-cuda/fixtures/audit16.bin
python3 benchmarks/audit/run_audit.py --out benchmarks/audit/new-run
python3 benchmarks/audit/profile_audit.py build-cuda --out benchmarks/audit/new-run/ncu
python3 benchmarks/audit/run_decode.py build-cuda benchmarks/audit/new-run
python3 benchmarks/audit/validate_audit.py build-cuda benchmarks/audit/new-run
python3 tools/record_cuda_environment.py build-cuda benchmarks/audit/new-run
bash benchmarks/round2/profile_graph.sh build-cuda/bench_decode build-cuda/fixtures/perf.bin benchmarks/audit/new-run/nsys
python3 benchmarks/round2/analyze_nsys.py benchmarks/audit/new-run/nsys/decode-graph.sqlite --plot
python3 benchmarks/audit/summarize_audit.py benchmarks/audit/new-run
```

- 冷缓存：分配至少 128 MiB，且不小于 `4 × cudaDeviceProp.l2CacheSize`；每次待测调用前，
  同一 stream 中完整读改写驱逐区。驱逐在 start event 之前；Attention 两段之间不驱逐。
  这是软件驱逐，不是禁用硬件缓存；允许同一次调用内的 GQA 数据复用。
- 热缓存：相同指针重复调用，warmup 后不驱逐。CSV 明示 hot/cold、驱逐字节、样本数。
- Attention：H16/KV4/D128，FP32 B1/S4096；FP16/BF16 B2/S32768（长度分别 32768/32765，
  有效唯一 KV 127.994 MiB）。三种实现为未分段标量、64-token 分段向量化、未分段向量化消融。
  未分段路径是每头一个 warp、4 warps/CTA，不是成熟 Attention 库。
- NCU Attention：`--replay-mode application --cache-control none`，应用在 profiler range 前
  显式驱逐；保留分段→merge scratch 的正常缓存依赖。热缓存对照不驱逐。整个应用每 pass
  重跑，避免 kernel replay 破坏热缓存/生产者消费者状态。分别报告两个 kernel，不能平均 DRAM%。
- NCU GEMV/RMS：单 kernel `--cache-control all`；对照 scalar/vec4，按 dtype 分别捕获。
  `metrics.csv` 来自原生报告导出，保留原始单位；`ideal_bytes/effective_GiB_s` 为逻辑有效带宽。
- GEMV：8192×4096 B1，FP32/FP16/BF16，W 分别 128/64/64 MiB；同存储精度 cuBLAS，
  pedantic FP32 累加，无 TF32。审计形状的所有实现均逐元素对照 CPU FP64，而非只对照 cuBLAS。
- RMSNorm：128×4096。baseline 与 optimized 都是 256 threads/row + warp/block reduction；
  optimized 新增向量访存与输入寄存器驻留，baseline 会在输出阶段重读输入。
- Decode：4 层 d512 与 16 层 d1024/FFN2816/H16/KV4/vocab8192，FP16、B1/seq129；均为随机权重，
  固定输入重复单步，非训练模型生成 TPS。每个样本先驱逐并同步，之后才开始计时。
  compute 只含提交到计算完成；full_cpu_greedy 包含 3 次 H2D、logits/error D2H 与 CPU argmax；
  full_gpu_greedy 包含 3 次 H2D、GPU argmax、4-byte Token/error 回传。两个 full 模式均将传输
  纳入 Graph，Host staging 拷贝也计时。采样原型仅支持有限 logits 的 B1 greedy，tie 取最低 ID。
  每个scope内双方预热20次，100组成对交错、交替先后采样，减少顺序/DVFS偏差。
  性能对照在每个 scope 内匹配 eager/Graph，不能将不同采样实现的差值全归因于 CUDA Graph。
- 原型以同一 Graph 更新长度、跨页/分段并替换页绑定；逐 logits 和全部物理 KV 槽对照独立
  CPU FP64。GPU greedy 与 GPU logits 的 CPU argmax 严格一致；FP16/BF16 不声称 CPU FP64
  bitwise 相等。零失败、Memcheck/Racecheck 零错误与非零浮点误差分开列出。

`interfered-*`、`guard-rejected-*`、`preflight-rejected-*` 为失败/受干扰记录，禁止用于性能结论。
原生 profiler 二进制遵循仓库 `.gitignore`，只保留本地；公开 CSV、命令、哈希及由真实 SQLite
绘制的 Timeline。安装 Python NumPy（生成较大模型）和 Matplotlib（绘图）后可复现全部命令。

`profile_graph.sh` 第4参数指定dtype（默认f16）；历史NSys图来自旧版默认f32，不能把其间隙比例当作FP16的同精度采样。

## 桌面资源保护

本机两次16层完整性能循环的Racecheck未写出完成摘要，期间发生桌面卡顿与系统重启。
采集到的显存仅约749 MiB，不能归因于显存耗尽；此前未记录主存峰值，根因未证明。
已将正确性和性能循环分开：`--validate-decode MODEL DT`跳过预热、计时与flush，完整
前向从原5样本入口的164次减到16次；保留原模型/精度、计算Graph、CPU/GPU greedy
传输Graph、位置127/128/129/143/144及16,777,216个物理KV槽。不使用kernel filter。
该入口普通数值、Memcheck、Racecheck全部通过，Racecheck零错误/零警告。

guard记录MemAvailable、Swap、内存PSI、进程RSS/Swap及cgroup实际限制；保留3 GiB
主存、1 GiB显存，单进程最多180秒。验证使用systemd用户scope，MemoryMax=6 GiB、
MemorySwapMax=0、nice+10；Racecheck使用1个分析线程。主机总Swap上限为1 GiB，
区别于job自身禁止Swap。默认guard的512 MiB上限不变，只有验证入口使用该覆盖值。
成功Racecheck约46.28秒，成功阶段观测到的cgroup内存峰值1562.16 MiB、job Swap为0；
精确值与限制从`racecheck-resolution/telemetry.jsonl`核对，未发生OOM kill。
先前因主机Swap超过512 MiB而主动中止的尝试也完整保留，不计为通过。

`validate_audit.py`默认完成16项矩阵，所有envelope sanitizer都使用正确性入口；
`--skip-large-racecheck`必须如实产生partial。旧完整计时循环不再作为Racecheck入口。
本次只补最后一项，没有重测合格性能，也没有放宽BF16容差。构建独立验证二进制，
避免覆盖原测量二进制身份；闭环记录在`results-20261126-final/racecheck-resolution/`。

```sh
cmake -S . -B build-cuda-validation -DNANO_WITH_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-cuda-validation --target bench_decode -j2
# 新日志路径；使用已生成的同一16层checkpoint
NV_BENCH_MAX_SWAP_MIB=1024 python3 benchmarks/audit/guard_command.py benchmarks/audit/check-telemetry.jsonl -- systemd-run --user --scope --quiet -p MemoryMax=6G -p MemorySwapMax=0 nice -n 10 compute-sanitizer --tool racecheck --racecheck-num-workers 1 --error-exitcode 99 --log-file benchmarks/audit/check-racecheck.log build-cuda-validation/bench_decode --validate-decode build-cuda/fixtures/audit16.bin f16
```

Linux复测需启用用户systemd与cgroup v2 memory控制器。`finish_validation.py`只接受
partial状态和新resolution目录；正式目录已complete，脚本会拒绝覆盖。日常复核只需
运行`summarize_audit.py`，不自动重试任何中断或重采性能。
