# CUDA 第二轮阶段验收

> 本文为历史归档验收。最新缓存口径、Baseline审计和未完成的新GPU验证见
> [2026-11-26评测审计](PERFORMANCE_AUDIT_20261126.md)。73.73µs为历史热缓存；新增正式性能已重测，15项sanitizer通过，额外16层Racecheck未完成；BF16限制见最新报告。

实现仓库为 nano-vLLM-C（`~/nanovllm`），按用户后续指示复用其 C11 模型/调度/采样框架。
MiniInfer-C 保留第一轮成果；本轮未导入其存储层。KV-Fabric 仅更新简历与面试文档。

## 工程与正确性

- C11 Host，CUDA 设备函数通过 C ABI 暴露；无新增 class/RAII/vtable/无意义 ctx。
- 完整链路：Embedding → RMSNorm → QKV SGEMV → 邻接对 RoPE / 分页 KV 写入 →
  Paged Attention → O SGEMV / 残差 → RMSNorm → Gate/Up SGEMV → SwiGLU →
  Down SGEMV / 残差 → Final RMSNorm → Classifier SGEMV / CPU 采样。
- 8/8 CTest 通过（两项为 fixture 生成）；Memcheck/Racecheck 对旧算子及新增链路均通过。
- Attention 90 项随机/边界用例：主矩阵 seq 128/1024/1037/4096、GQA 比 1/4/8、
  batch 1/3、三种 dtype；另含 D=1/31/65/127/128/256、空行、非对齐、极端 logits、
  NaN 投毒未满页尾、workspace/输出哨兵和非法设备元数据。
- CPU 参考用 FP64 两遍 Softmax，与 GPU Online Softmax 独立实现；每个输出元素检查。
- 单层 batch=1、三层 batch=2，三种 dtype；分别 21 / 83 步，逐 logits 与全部物理 KV
  槽对照独立 CPU FP64 前向，并验证同一 Graph 跨页、跨 64-token 分段读取新长度/页表。
- Graph 节点逐项检查只含 Kernel/Memset，没有 Host、内存分配或 Host 数据复制节点。
- 原有调度器在 eager/Graph、普通/紧张页池下均匹配 CPU greedy Token ID，覆盖抢占重算。
- 真实 TinyStories 260K checkpoint：4 个 prompt × 32 = 128 个生成 Token，CPU/eager/Graph
  FP32 greedy ID 完全一致；模型和 tokenizer SHA256 经固定来源校验。

## Paged Attention 代表性实测

RTX 4060 Laptop；FP32，B=1，H=16，KV=4，D=128，seq=4096。
随机输入和乱序页，warmup 10 次；60 次 Graph replay × 5 次独立 event 调用，300 样本。
对照是本项目串行 Warp 参考，不是 FlashAttention/cuDNN 等成熟库。

| 实现 | P50/P95 µs | 有效 GiB/s | max abs / max rel |
| --- | ---: | ---: | ---: |
| serial_online | 1596.42/1597.44 | 9.80 | 8.38e-08 / 0.0343 |
| split_online_vec4 | 73.73/73.73 | 212.13 | 1.02e-08 / 0.00777 |

完整 144 行覆盖 B=1/4、H=16、KV=16/4/1、D=128、seq=128/1024/1037/4096、三种 dtype。
有效字节数采用唯一 KV + Q/O：`dtype_bytes*(2*sum(lengths)*KV*D+2*B*H*D)`。
GQA 复读、分段 scratch 与元数据不膨胀分子；这是算法有效带宽，不是 DRAM 计数。
主矩阵为 warm-cache，NCU 则 cache flush；二者不能直接混用。

## 完整 Decode 与 Graph

随机权重 4 层、hidden=512、FFN=1536、H=8、KV=2、vocab=512、seq=129、B=1；
包含 embedding、全部层和最终 logits。输入/长度固定的重复单步，初始 KV 为随机数据。
warmup 20 次/mode、100 个样本；Host 单调时钟覆盖提交至完成等待，另记录 CUDA event。
以下计算区间不含 Host 元数据 H2D、logits D2H、采样、模型加载与 Graph 实例化。

| dtype | eager P50/P95 µs | Graph P50/P95 µs | P50 降低 |
| --- | ---: | ---: | ---: |
| f32 | 482.95/484.93 | 454.50/458.28 | 5.89% |
| f16 | 250.34/251.41 | 221.74/223.49 | 11.43% |
| bf16 | 249.35/249.99 | 221.49/222.39 | 11.17% |

Nsight Systems 每模式 5 步：eager 255 次 Kernel Launch，
Graph 5 次 Graph Launch；两者均执行 255 个 GPU Kernel，
即每步 Host 提交 51 → 1，GPU Kernel 数未减少。GPU 活动窗口内间隙比例
9.10% → 7.50%，并未消除所有设备调度间隙。
此统计含 profiler/event 影响，只用于归因；正式延迟使用未插桩的 100 样本 CSV。
原始 `.nsys-rep`、SQLite、官方统计及真实数据绘制的 Timeline 均归档。

## NCU 硬件对照

SGEMV：8192×4096/B1；RMSNorm：128×4096；Attention：B1/H16/KV4/D128/S4096。
每 Kernel 一次采样，cache flush、未锁时钟；利用率由硬件计数读取，非标称带宽反推。

| Kernel | dtype | DRAM % | 实测 GiB/s | Achieved Occupancy % |
| --- | --- | ---: | ---: | ---: |
| sgemv | f32 | 96.59 | 230.06 | 98.49 |
| sgemv | f16 | 90.51 | 215.59 | 97.65 |
| sgemv | bf16 | 90.50 | 215.50 | 96.70 |
| rmsnorm | f32 | 65.72 | 136.38 | 85.86 |
| rmsnorm | f16 | 51.55 | 106.97 | 86.44 |
| rmsnorm | bf16 | 51.95 | 107.67 | 87.30 |
| attention_partition | f32 | 62.60 | 130.43 | 75.86 |
| attention_merge | f32 | 38.63 | 80.24 | 11.16 |
| attention_partition | f16 | 35.32 | 73.59 | 68.72 |
| attention_merge | f16 | 39.85 | 82.47 | 11.07 |
| attention_partition | bf16 | 35.52 | 74.04 | 70.79 |
| attention_merge | bf16 | 41.20 | 85.31 | 11.08 |

Attention 的分段和合并分别列指标。合并仅 16 个 CTA，低 Occupancy 是当前小头数形状的
限制，不能将分段指标称作整条 Attention 的利用率。当前实现仍有 SFU、归约与调度开销。

## 误差与适用边界

- Attention：FP32 atol/rtol=2e-5/3e-4，FP16=1e-3/2e-3，BF16=8e-3/1.6e-2。
- 完整前向：FP32=1e-4/3e-4，FP16=3e-3/8e-3，BF16=2e-2/5e-2。每个元素满足
  `abs_err <= atol + rtol*abs(ref)`；同时保留真实 max_abs / max_rel。近零参考值的相对误差
  可能很大，不能脱离绝对误差解读；FP16/BF16 不承诺 CPU bitwise 或 Token ID 一致。
- SGEMV 适用低 batch Decode；高 batch 推荐可复用权重的 GEMM。原调度器每请求串行
  GPU forward，并非 GPU continuous-batching GEMM；独立 Kernel 测试覆盖真正 B>1。
- 只支持 legacy llama2.c FP32 checkpoint（加载后可转 FP16/BF16）和邻接对 RoPE。
  未实现 Qwen/safetensors/GGUF 加载、量化、Tensor Parallel、CUDA Prefill 优化或生产服务。
- Graph 的 batch、模型形状、页池容量与设备地址固定；token/seq_len/Block Table 内容动态。
  分配、页引用调度和 pinned 元数据上传在捕获外，单 stream 依赖及上传事件保护 staging。
- 本轮未对接 KV-Fabric 远程/分层存储；两项目分别负责计算执行与存储传输。

## 复现与证据

构建与一键验证命令见仓库 README；全部本轮结果位于 `benchmarks/operators/round2-results/`。
原始 benchmark CSV、NCU `.ncu-repz`/CSV、Nsys `.nsys-rep`/SQLite、sanitizer 日志和
checkpoint 哈希保留；environment.json 提供源文件与计时二进制 SHA256。
