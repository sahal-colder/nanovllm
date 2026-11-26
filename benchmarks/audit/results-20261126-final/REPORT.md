# CUDA 评测审计与实测报告（2026-11-26，Australia/Perth）

本轮在用户释放GPU后完成正式重测。采用RTX 4060 Laptop GPU，所有GPU工作串行执行；每个阶段的进程守卫均未发现外部CUDA计算任务。此前并发VLLM干扰及失败阶段保留，但不进入正式性能结论。

正式证据目录：`benchmarks/audit/results-20261126-final/`。性能、NCU、Decode均完成；现有16/16项sanitizer矩阵通过。16层FP16完整链路的独立正确性入口已通过Racecheck；旧计时循环插桩中断记录仍保留。新结果与历史热缓存日志分开，以下除明确说明外均来自本轮真实实测。

## 1. 评测审计与实测报告

### 硬件、计时与正确性口径

RTX 4060 Laptop GPU，24 SM、CC8.9、128-bit显存总线；nvidia-smi显存8188 MiB，L2为**33,554,432 bytes = 32 MiB**。Driver615.71.09，NVCC13.4.92，cuBLAS130800，NCU2026.3.1，NSys2026.3.2。

未锁频；每秒保存进程、温度、SM/显存频率快照。实测存在2010/2580 MHz及7001/8001 MHz等动态频率状态，单次snapshot不能替代Kernel瞬时频率。表内延迟与NCU独立采样，不能直接用不同pass的时间反推同一次带宽。小幅差异不作为稳定优化承诺。

冷缓存：每次调用前在同一stream完整读改写128 MiB驱逐区（4×L2）；flush位于start event之前，不计入算子延迟。允许同一次调用内缓存复用。Attention partition与merge之间不驱逐。热缓存只用于同地址warmup重复调用的对照。每方法100个event样本，报告P50/P95；有效GiB/s按逻辑唯一张量字节计算，区别于NCU真实DRAM字节。

全量CTest **8/8**通过。算子、新增三路Attention、FP16/BF16超L2冷调用、三dtype小模型传输/greedy Graph的Memcheck/Racecheck均零错误、Racecheck零警告；16层FP16独立正确性入口也通过普通数值、Memcheck与Racecheck，未过滤任何Kernel。完整16层FP16逐16,777,216个物理KV槽对照CPU参考，并检查5个动态步的Graph/eager logits逐位一致。

CPU参考使用独立FP64算术并显式模拟存储舍入；验收是**容差内零校验失败，不是浮点零误差**。Attention atol/rtol：FP32=2e-5/3e-4，FP16=1e-3/2e-3，BF16=8e-3/1.6e-2；GEMV/RMS为1e-4/2e-4、1e-3/2e-3、8e-3/1.6e-2；完整前向为1e-4/3e-4、3e-3/8e-3、2e-2/5e-2。逐元素满足`abs_err <= atol + rtol*abs(ref)`，近零max_rel需与max_abs一起解释。

### Attention：冷/热差异及Baseline

历史73.73 µs的FP32 B1/H16/KV4/D128/S4096循环未显式驱逐，唯一KV恰为16 MiB，小于32 MiB L2；原62.60% DRAM和75.86% Occupancy来自另一轮`ncu --cache-control all`，两者不能组合为同一冷缓存测量。

| FP32 B1/S4096 | hot P50/P95 µs | cold P50/P95 µs | cold/hot P50 | cold有效GiB/s |
| --- | --- | --- | --- | --- |
| serial_online | 2009.09/2010.14 | 2917.54/3027.97 | 1.452× | 5.36 |
| unsplit_vec4 | 1809.41/1810.43 | 3126.27/3131.39 | 1.728× | 5.00 |
| split_online_vec4 | 73.73/74.75 | 120.83/125.95 | 1.639× | 129.44 |

分段路径清L2后P50由73.73增至120.83 µs，增加47.10 µs / 63.89%。NCU应用重放中，partition热缓存DRAM读取0.000 MiB，驱逐后16.040 MiB，交叉验证驱逐生效。

Baseline精确定义：`serial_online`每query head一个warp、4 warps/CTA，整条序列不分段；不是成熟Attention库，也不是“每序列恰好一个block”。新增`unsplit_vec4`保留向量布局、只去掉分段，与`split_online_vec4`比较可隔离分段并行度及合并的净收益。

FP16/BF16新形状：B2/H16/KV4/D128，长度[32768,32765]，GQA比4，唯一KV **127.994 MiB**，远超L2；Q/KV/output按对应dtype存储，均FP32累加。页表乱序、页尾投毒。

| dtype | 形状H16/KV4/D128 | KV MiB | 标量未分段P50 µs | 向量未分段P50 µs | 分段P50/P95 µs | vs标量 | vs向量未分段 | 有效GiB/s 标量/向量/分段 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f32 | B1/S4096 | 16.000 | 2917.54 | 3126.27 | 120.83/125.95 | 24.15× | 25.87× | 5.36/5.00/129.44 |
| f16 | B2/S32768 | 127.994 | 23314.43 | 24893.44 | 1168.38/1175.55 | 19.95× | 21.31× | 5.36/5.02/106.99 |
| bf16 | B2/S32768 | 127.994 | 23196.67 | 25284.61 | 1161.22/1178.62 | 19.98× | 21.77× | 5.39/4.94/107.65 |

| dtype | 每路输出元素 | 三路最大abs error | 三路最大rel error | CPU FP64判定 |
| --- | --- | --- | --- | --- |
| f32 | 2048 | 7.07805157e-08 | 3.84262625 | 全部三路通过混合容差 |
| f16 | 4096 | 3.81469727e-06 | 0.00387596899 | 全部三路通过混合容差 |
| bf16 | 4096 | 1.52587891e-05 | 0.0173410405 | 全部三路通过混合容差 |

| dtype | Kernel | DRAM % | 真实DRAM GiB/s | Achieved Occupancy % | DRAM read MiB | L2 hit % |
| --- | --- | --- | --- | --- | --- | --- |
| f32 | scalar unsplit | 2.29 | 4.77 | 8.33 | 16.242 | 5.25 |
| f32 | vec4 unsplit | 1.90 | 4.52 | 8.33 | 16.182 | 4.08 |
| f32 | partition | 61.52 | 146.52 | 72.33 | 16.040 | 75.75 |
| f32 | merge | 10.21 | 20.93 | 13.08 | 0.047 | 92.10 |
| f16 | scalar unsplit | 2.53 | 6.04 | 8.33 | 129.277 | 9.02 |
| f16 | vec4 unsplit | 2.35 | 5.59 | 8.33 | 128.736 | 8.80 |
| f16 | partition | 63.69 | 132.75 | 79.54 | 146.483 | 74.15 |
| f16 | merge | 85.13 | 177.22 | 11.71 | 7.119 | 21.73 |
| bf16 | scalar unsplit | 2.51 | 5.99 | 8.33 | 128.669 | 9.01 |
| bf16 | vec4 unsplit | 2.37 | 5.65 | 8.33 | 128.849 | 8.94 |
| bf16 | partition | 59.23 | 141.08 | 79.52 | 146.490 | 74.44 |
| bf16 | merge | 82.29 | 195.50 | 11.75 | 7.120 | 21.86 |

NCU Attention使用`--replay-mode application --cache-control none --profile-from-start off`：应用在profile range前显式驱逐，每pass重建相同输入，保留partition→merge真实缓存依赖。热缓存对照不驱逐。未分段只有4/8个CTA（B1/B2），Occupancy低；分段后展开序列并行度，但不能将partition指标称作整个Attention指标，也不能简单平均两个kernel的DRAM%/Occupancy。

FP16/BF16大形状的不驱逐结果也保存在CSV，但超L2不等于无缓存命中，且动态频率可能使不驱逐版本略慢；不把这种小幅反转解释成清缓存可加速。简历采用cold结果。

### GEMV与RMSNorm：同精度、同缓存、同Baseline

GEMV为8192×4096/B1，FP32权重128 MiB，FP16/BF16权重64 MiB，均超L2；仍对每次调用显式驱逐。cuBLAS同存储精度，FP32用Sgemv、FP16/BF16用GemmEx，PEDANTIC FP32累加，禁用低精度中间归约和TF32。审计形状的全部方法逐元素对齐CPU FP64。

| dtype | naive P50 µs | optimized P50/P95 µs | cuBLAS P50 µs | cuBLAS/optimized | 有效GiB/s | NCU DRAM GiB/s | DRAM % | Occupancy % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f32 | 708.61 | 620.54/713.73 | 558.08 | 0.899× | 201.51 | 230.37 | 96.72 | 98.64 |
| f16 | 403.46 | 409.60/415.74 | 288.77 | 0.705× | 152.64 | 215.71 | 90.57 | 97.60 |
| bf16 | 352.19 | 360.45/363.52 | 287.74 | 0.798× | 173.46 | 215.69 | 90.56 | 97.17 |

**新冷缓存结果不支持“GEMV优于cuBLAS”。** 本实现可有较高DRAM利用率，但flush后的常规event延迟仍比cuBLAS慢。NCU replay/频率与正常计时不同，不能拿profiler中的单kernel时长替代100样本P50，也不能把硬件利用率直接变成端到端加速。历史230.06 GiB/s、96.59%为旧FP32冷NCU实测，保留归档，新表不覆盖旧证据。

RMSNorm为128×4096；X+Y+gamma逻辑字节FP32为4.016 MiB、FP16/BF16为2.008 MiB，小于L2。历史8.192→6.144 µs是FP32热缓存，不能标冷。两个版本都是256 threads/row与warp/block归约；optimized的差别是向量读写、X驻留寄存器，baseline在输出阶段重读X。

| dtype | cache | 标量P50/P95 µs | 向量/寄存器P50/P95 µs | 加速比 |
| --- | --- | --- | --- | --- |
| f32 | hot | 7.30/8.19 | 6.14/7.17 | 1.19× |
| f32 | cold | 25.60/26.62 | 18.43/19.46 | 1.39× |
| f16 | hot | 6.82/7.17 | 5.12/5.31 | 1.33× |
| f16 | cold | 19.46/20.48 | 12.29/13.31 | 1.58× |
| bf16 | hot | 7.33/8.19 | 5.28/6.14 | 1.39× |
| bf16 | cold | 19.46/20.48 | 12.29/13.31 | 1.58× |

RMS有效带宽分子统一为唯一X/Y/gamma逻辑字节，baseline重读X不膨胀分子；实际DRAM计数见NCU CSV。近微秒event有量化影响，报告当前样本点估计，不宣称跨机器固定收益。

### CUDA Graph：更大模型、完整步与架构权衡

原4层d512/FFN1536/H8/KV2/vocab512/B1/seq129随机权重模型的FP16历史计算链路250.343→221.740 µs（11.43%），未显式驱逐且排除metadata与采样。它适合展示小算子/小Batch下Launch开销，不能外推成生产模型吞吐提升。

新测试包括同小模型及16层d1024/FFN2816/H16/KV4/vocab8192、容量2048、B1/seq129；均是随机权重、固定输入重复单步，非训练模型生成TPS。大模型FP32 checkpoint755,634,204 bytes，seed719；FP16权重流量远超L2。每个样本先驱逐并同步，再开始wall/event计时。**每个scope中eager/Graph按成对交错、交替先后顺序采样，预热双方各20次**，降低长批次顺序偏差。较早顺序计时数据仅归档，不作为正式收益。

`compute`=Host提交到计算完成；`full_cpu_greedy`=staging拷贝、token/length/table H2D、计算、logits/error D2H、CPU unpack+argmax；`full_gpu_greedy`=staging、3次H2D、计算、GPU argmax及4-byte Token/error回传。两个full Graph均实际捕获传输。均不含CPU页分配/请求调度、tokenizer、文本输出。

| FP16层数 | scope | eager wall P50/P95 µs | Graph wall P50/P95 µs | P50降低 | device P50 µs |
| --- | --- | --- | --- | --- | --- |
| 4 | compute | 506.02/510.41 | 473.15/480.17 | 6.50% | 500.77→467.97 |
| 4 | full_cpu_greedy | 436.29/468.16 | 410.04/436.43 | 6.02% | 428.45→401.86 |
| 4 | full_gpu_greedy | 433.79/439.96 | 406.76/410.32 | 6.23% | 428.61→401.92 |
| 16 | compute | 2719.22/3196.76 | 2609.19/3064.63 | 4.05% | 2714.34→2604.03 |
| 16 | full_cpu_greedy | 2796.06/2806.53 | 2683.44/2688.95 | 4.03% | 2727.42→2614.78 |
| 16 | full_gpu_greedy | 2735.13/2742.42 | 2622.72/2629.92 | 4.11% | 2729.57→2617.60 |

16层完整GPU greedy步的同scope Graph收益为4.11%，绝对节省112.41 µs。只能比较同scope中的eager/Graph；GPU greedy减少回读与CPU处理的收益不能全部归因于Graph。生产引擎仍使用CPU sampler，本轮传输/argmax集成为benchmark原型。

**明确数值限制：16层BF16动态前向未通过原有CPU FP64容差。** 在pos143得到-0.0849609375，对照-0.11328125，abs=0.0283203125；允许界约0.025664，故判失败。该步eager与Graph所有8192个logits逐位一致，排除了Graph更新造成差异；更符合深层BF16舍入累积的表现，但不把该推断当已证明根因。未放宽容差、未换seed、未发布失败阶段的性能。BF16 Attention超L2、原1/3层链路和新4层传输Graph仍独立通过验收。16层Graph简历收益仅引用FP16。

已知BF16数值边界复现（当前seed下应报FAIL，结果不进入正式矩阵）：`build-cuda/bench_decode --audit-decode build-cuda/fixtures/audit16.bin bf16 5`。完整失败日志与原采集命令位于`rejected-decode-bf16/`。

架构权衡：设备地址、shape与容量固定，seq_len/Block Table是设备数据，更新内容不需重建Graph。旧版把H2D放外部属于接口与pinned staging生命周期选择，并非无法捕获memcpy；固定pinned地址可把上传/回传捕获进Graph。CPU修改staging前必须确认上一轮已消费。GPU greedy可只回传Token ID，但CPU页管理、抢占与temperature/top-p语义仍需单独设计；单请求下一步依赖当前Token，不能声称所有传输可异步隐藏。跨请求双缓冲/多stream应在真正调度器中另测。

### NSys：同精度FP16新证据

NSys trace使用FP16固定输入重复计算步，未显式驱逐，仅用于Launch归因；每模式5步：eager=255次Kernel API、Graph=5次Graph API；两边各255个GPU Kernel。每步Host提交**51→1**，GPU Kernel仍51个，另有1个memset（计算Graph共52节点）。

插桩GPU活动窗口间隙比例17.12%→13.92%；包含event/同步与profiler影响，仅用于归因，不用来替换正式延迟。旧NSys默认FP32，不能将旧间隙比率当作FP16同精度数据。

![FP16 NSys真实时间线](../benchmarks/audit/results-20261126-final/nsys/decode-timeline.png)

### 复现命令与证据身份

```sh
cmake -S . -B build-cuda -DNANO_WITH_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-cuda -j4
python3 benchmarks/audit/make_model.py build-cuda/fixtures/audit16.bin
python3 benchmarks/audit/run_audit.py --out benchmarks/audit/new-run
python3 benchmarks/audit/profile_audit.py build-cuda --out benchmarks/audit/new-run/ncu
python3 benchmarks/audit/run_decode.py build-cuda benchmarks/audit/new-run
python3 benchmarks/audit/validate_audit.py build-cuda benchmarks/audit/new-run
bash benchmarks/round2/profile_graph.sh build-cuda/bench_decode build-cuda/fixtures/perf.bin benchmarks/audit/new-run/nsys f16
python3 benchmarks/round2/analyze_nsys.py benchmarks/audit/new-run/nsys/decode-graph.sqlite --plot
python3 tools/record_cuda_environment.py build-cuda benchmarks/audit/new-run
python3 benchmarks/audit/summarize_audit.py benchmarks/audit/new-run
```

`new-run`必须为新目录；各测量与已完成验证按具体状态发布，partial中的未完成项不得计为PASS。`commands.json`、`decode-commands.json`、`validation-commands.json`、`ncu/commands.json`和日志包含本次精确绝对命令；各telemetry保存时序证据。原始NCU/NSys二进制按仓库规则保留本地，公开CSV、哈希和真实数据图。

`initial-measurement-state.json`保留Attention/GEMV/RMS与NCU采集时的既有source/binary SHA256；后续只修改Decode计时/诊断后重编译，新的Decode与sanitizer使用最终environment/telemetry中的二进制身份；NSys使用`sequential-stage-environment.json`对应的二进制（计算捕获路径未改动）。不得用后续二进制哈希冒充此前采集身份。environment.json为16:53归档快照，不因后续汇总脚本修改而回写；后续CPU检查与文档身份另存publication-state.json。资源保护逻辑在卡顿后才补充，不能声称此前采集已经有主存阈值保护。

### 桌面卡顿与资源约束

两次16层整步Racecheck没有写出完成摘要，后续检测到系统重新启动。最后可用遥测显示GPU占用100%、约749MiB/8188MiB显存，不能解释为显存耗尽。16:49之后检查可用主存约10GiB、Swap为0；但这些是重启后的快照，不能反证卡顿时主存未曾紧张。之前没有主存峰值记录；现有内核日志未查到OOM或NVIDIA Xid，卡顿根因尚未证实。

完整性能循环包含大量预热、flush、eager/Graph重复，放进Racecheck给桌面GPU带来不必要的持续满载；这是验证流程安排不当。新增`--validate-decode MODEL DT`独立正确性入口：不执行预热、计时或flush，保留原模型、计算/CPU greedy/GPU greedy三种范围、两条传输Graph、5个动态位置及全部物理KV校验。原5样本入口执行164次完整前向，新入口执行16次；没有过滤Kernel、缩小模型、放宽容差或发布插桩耗时为性能。旧性能循环的Racecheck仍未完成，当前关闭的是相同完整计算链路的正确性验收。

guard记录MemAvailable/Swap/内存PSI、目标进程RSS/Swap与cgroup实际限制；默认至少预留3GiB主存、1GiB显存，单子进程超过180秒即终止自身进程组，sanitizer以nice+10运行。本次首次正确性重试因主机总Swap超过原512MiB阈值被守卫主动终止（仍有约10GiB可用内存，目标VmSwap=0），完整记录保留在`racecheck-resolution-guard-aborted/`。随后把验证放入独立systemd用户scope，实际施加MemoryMax=6GiB、MemorySwapMax=0；主机总Swap上限改为1GiB，Racecheck只用1个分析线程。成功日志、命令、资源时序和新验证二进制身份保存在`racecheck-resolution/`；既有测量binary/environment与全部性能CSV保持不变。

正确性验证实测：Racecheck总wall 46.28秒（不属于性能指标），成功阶段观测到的cgroup内存峰值1562.16 MiB、job Swap=0、最小主机MemAvailable 9719.50 MiB；cgroup未发生OOM kill。Memcheck另行完成且零错误。所有实测限制均从本次telemetry核对，不能追溯声称旧中断阶段也受这些限制。

本次闭环精确命令（其余15项不重跑）：

```sh
cmake -S . -B build-cuda-validation -DNANO_WITH_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-cuda-validation --target bench_decode -j2
python3 benchmarks/audit/finish_validation.py build-cuda-validation benchmarks/audit/results-20261126-final --model build-cuda/fixtures/audit16.bin
```

`finish_validation.py`仅接受partial状态并拒绝覆盖既有resolution目录；当前正式目录已complete，重复验证应选择新目录。单独复测Racecheck可使用下列命令；日志文件名应为新路径：

```sh
NV_BENCH_MAX_SWAP_MIB=1024 python3 benchmarks/audit/guard_command.py benchmarks/audit/check-telemetry.jsonl -- systemd-run --user --scope --quiet -p MemoryMax=6G -p MemorySwapMax=0 nice -n 10 compute-sanitizer --tool racecheck --racecheck-num-workers 1 --error-exitcode 99 --log-file benchmarks/audit/check-racecheck.log build-cuda-validation/bench_decode --validate-decode build-cuda/fixtures/audit16.bin f16
```

## 2. 简历改写版（中文，4条）

- 针对长序列Decode的串行与并行度瓶颈，实现64-token分段Online Softmax及max/sumexp归约合并；RTX4060 Laptop上FP16 B2/H16/KV4/D128、S32768（KV约128MiB），冷L2 P50 23314.43→1168.38 µs，较自有未分段基线19.95×，向量未分段消融21.31×；分段Kernel NCU DRAM 63.69%、Occupancy 79.54%。
- 针对RMSNorm重复访存，实现向量读写与输入寄存器驻留；FP32 128×4096、每调用前驱逐128MiB，对比同256-thread行归约标量基线，P50 25.60→18.43 µs（1.39×）；GEMV FP32 8192×4096/B1冷NCU达230.37 GiB/s、DRAM 96.72%，另测冷缓存P50 620.54 µs，cuBLAS 558.08 µs，明确性能边界。
- 针对B1 Decode的Host提交开销，利用固定地址与动态长度/页表复用CUDA Graph，并在benchmark内捕获元数据H2D、GPU greedy与Token ID回传；16层d1024随机模型FP16、冷L2完整步P50 2735.13→2622.72 µs（降低4.11%），小模型FP16 NSys证实每步51次Kernel提交→1次Graph提交。
- 针对分页与低精度正确性，覆盖FP32/FP16/BF16、GQA/MQA、乱序页、尾部及动态Graph更新；独立CPU FP64逐输出/逐物理KV槽验收容差内零失败，算子及小模型Memcheck/Racecheck零错误，16层FP16校验1677万KV槽并通过独立正确性入口的Memcheck/Racecheck，16/16项验收完成。另披露16层BF16深层数值容差限制。

## 3. README补强清单

- [x] 以当前cold/hot表替换混合口径绝对延迟，写明128MiB驱逐位于计时外；硬件精确L2字节、dtype与实际长度。
- [x] Attention三路Baseline、FP16/BF16超L2、P50/P95、有效GiB/s与partition/merge真实DRAM/Occupancy分列。
- [x] GEMV同精度cuBLAS对照，不声称跑赢；RMS两个版本准确命名，热/冷区分。
- [x] 小/16层FP16模型的compute与完整CPU/GPU greedy范围、交错采样、wall/device时间及随机权重限制。
- [x] 同精度FP16 NSys原始报告、SQLite、官方统计及51→1真实时间线；说明GPU Kernel数量不变。
- [x] 固定seed、完整命令、source/binary/model哈希、进程守卫、sanitizer与CPU FP64误差日志。
- [x] 保留受干扰/失败/顺序计时记录并排除，明确16层BF16失败，不更改容差或隐去反例。
- [x] 16层FP16独立正确性入口完整Racecheck零错误/零警告；16/16项矩阵闭环，保留旧性能循环中断与资源守卫终止记录。
