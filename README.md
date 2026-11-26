# nano-vLLM-C

用 C11 与 CUDA 实现 Llama 风格推理执行。复用原有 checkpoint mmap、分词器、
采样器、请求调度、分页块引用及前缀复用，在同一执行链路接入手写 CUDA 算子与 Graph。
KV-Fabric 负责外部存储与数据传输；本仓库负责 GPU 常驻 KV 和推理计算。

本轮完整结果：[数值与性能验收](docs/CUDA_VALIDATION.md)、
[阶段交付总结（含双项目简历）](docs/ROUND2_ACCEPTANCE.md)。

## 架构与代码入口

```text
src/main.c / engine.c      C11 CLI、调度、前缀复用、抢占重算、CPU/CUDA 显式分支
src/model.c               原 CPU 前向、legacy FP32 checkpoint mmap
src/cache.c               原有块 ID、引用计数、空闲页 LRU 和前缀哈希链
src/cuda_runtime.c        C11 GPU 权重/缓冲区、页指针表、Decode Step、Graph
src/operators/            SGEMV、RMSNorm、Paged Attention、RoPE、SwiGLU 等 CUDA
include/nv_cuda.h         Host 所有权和 Device buffer 分离的扁平 C 结构
include/nv_attention.h    Paged Attention 的薄 C ABI 和设备数据布局
benchmarks/round2/        独立 FP64 参考、随机测试、计时、NCU/Nsys 分析
benchmarks/operators/     三算子原始基准和验收数据
```

调用链：`nv_engine_step → nv_cuda_forward → nv_cuda_step`。

```text
Embedding → RMSNorm → QKV SGEMV → RoPE + Paged KV Store
          → Paged Decode Attention → O SGEMV + Residual
          → RMSNorm → Gate/Up SGEMV → SwiGLU → Down SGEMV + Residual
          → 重复层 → Final RMSNorm → Classifier SGEMV → CPU Sampling
```

Host 仍为 C11；CUDA `.cu` 只暴露 C ABI。无 class、RAII、虚函数、vtable 或 ctx
分发层。第一轮两个算子从 MiniInfer-C 迁入，保留 `mi_*` C ABI；未复制其存储实现。

## 页布局与 Graph

CPU 调度器继续使用 `block_table[logical] → physical_id` 和现有页引用/LRU。
GPU 每个物理块独立分配，包含全部层；每层页载荷为 `[K/V, token, kv_head, dim]`，
设备页指针表允许物理页不连续。完整前缀只读共享，当前写入页须独占。
启用 CUDA 后释放未使用的 CPU KV 张量，保留其块管理元数据。

Decode Step 只读设备 token、seq_len 和 Block Table；shape、页池和地址固定。
所有显存分配、页表准备及 pinned 元数据上传均在捕获外；上传事件保护 staging，
同一 stream 保证 KV 写入先于 Attention。Graph 内只有 Kernel/Memset。
CPU 采样需要 logits 回读，发生在 Graph 外。改变 batch/模型形状/容量需重新实例化。

## 三个核心算子

| 算子 | 组织与复用 | 适用边界 |
| --- | --- | --- |
| SGEMV | 4 warps/CTA、1 warp/row，向量化 W/X，shuffle 归约 | 低 batch Decode；高 batch 用复用 W 的 GEMM 更合适 |
| RMSNorm | 256 threads/row，输入驻留寄存器，warp/shared 归约和向量读写 | hidden 1..16384；处理非对齐、奇数尾部及平方和溢出 |
| Paged Attention | warp/head/split，64-token 分段 Online Softmax；256-thread 合并 CTA 共享权重 | MHA/GQA/MQA、非连续页、partial tail；D≤256、容量≤65536 |

FP32/FP16/BF16 均 FP32 累加；向量地址不对齐时在同一 Kernel 内安全标量访问。
RoPE 使用原模型的邻接对形式，支持原有 legacy llama2.c FP32 checkpoint，
加载后可转为 FP16/BF16；不直接加载 safetensors/GGUF/Qwen 权重。

## 代表性实测

RTX 4060 Laptop 8188 MiB，CUDA 13.4 / cuBLAS 13.8，SM 89。
第一轮 SGEMV/RMSNorm 延迟样本保留在 `benchmarks/operators/results/`；
本轮在 nano-vLLM 中重新采集三算子的 NCU，并完成全链路回归。

| 算子与形状 | 基础 P50 | 优化 P50 | 口径 |
| --- | ---: | ---: | --- |
| FP32 SGEMV 4096×4096 B1 | 272.38 µs | 272.38 µs | 与 cuBLAS 同级；大矩阵访存受限 |
| FP32 RMSNorm 128×4096 | 8.19 µs | 6.14 µs | 约 1.33×；CPU 逐元素校验 |
| FP32 Attention B1/H16/KV4/D128/S4096 | 1596.42 µs | 73.73 µs | 对照本项目串行 Warp 参考；非成熟库比较 |

有效带宽不能冒充 DRAM 实测。上述 Attention 为 warm-cache；本轮 NCU cache flush：

| FP32 Kernel | DRAM 利用率 | 实测 GiB/s | Achieved Occupancy |
| --- | ---: | ---: | ---: |
| SGEMV | 96.59% | 230.06 | 98.49% |
| RMSNorm | 65.72% | 136.38 | 85.86% |
| Attention 分段 | 62.60% | 130.43 | 75.86% |
| Attention 合并 | 38.63% | 80.24 | 11.16% |

NCU 形状分别为 8192×4096/B1、128×4096、B1/H16/KV4/D128/S4096。
合并只有 16 个 CTA，低并行度仍是该形状的限制。

4 层随机权重 Decoder（d512、FFN1536、H8/KV2、seq129、B1），100 样本计算链路时间：

| dtype | eager P50/P95 µs | Graph P50/P95 µs |
| --- | ---: | ---: |
| FP32 | 482.95 / 484.93 | 454.50 / 458.28 |
| FP16 | 250.34 / 251.41 | 221.74 / 223.49 |
| BF16 | 249.36 / 249.99 | 221.49 / 222.39 |

计时覆盖 Host 提交至 stream 完成，不含元数据 H2D、logits D2H、采样及 Graph 实例化。
Nsight Systems 同配置各 5 步：GPU Kernel 均 255 个，Host 提交由 255 次 Kernel
Launch 变为 5 次 Graph Launch，即每步 51 → 1；未声称消除全部 GPU 调度间隙。

## 构建与一键复现

CPU 仍可 `make && make test`。CUDA 模式需要 CMake 3.18+、CUDA Toolkit 和 Python 3。
本机架构为 89，其他卡通过 `NV_CUDA_ARCH` 指定兼容架构。

```sh
# 全量构建、8 项 CTest、Memcheck/Racecheck、Attention 和 Decode 基准
bash tools/validate_cuda.sh build-cuda benchmarks/operators/runs/new-run

# 加上 NCU 和 Nsys：仅 NCU 在本机需要 sudo 计数器权限
bash tools/validate_cuda.sh build-cuda benchmarks/operators/runs/profile-run --profile
```

验收数据在 `benchmarks/operators/round2-results/`。公开副本包含实测 CSV、日志、Nsys 统计及 Timeline；
原始 NCU / Nsys 二进制报告与 SQLite 保留本地，可用 `--profile` 重新生成。
详见[公开归档范围](benchmarks/operators/PROFILING_ARTIFACTS.md)。
可选 `analyze_nsys.py ... --plot`
从真实 SQLite 生成 Timeline 图（需 Matplotlib）；官方统计不依赖它。
无 GPU 的 CUDA 测试返回 77/skip，不算 GPU 验收。

## 运行预训练模型

```sh
python3 tools/download.py --model 260k
build-cuda/nanovllm_cuda models/stories260K.bin -z models/tok512.bin \
  -i "Once upon a time" -n 32 -t 0 --cuda-graph
python3 benchmarks/round2/verify_pretrained.py build-cuda models benchmarks/operators/runs/pretrained
```

本轮真实 260K checkpoint 的 4×32 个 greedy Token 在 CPU/eager/Graph 下严格一致。
FP16/BF16 做数值误差验收，不承诺与 FP32 的生成 Token 相同。
CPU 调度每个 tick 仍串行推进各请求；未实现 GPU 批量 GEMM、并行 Prefill、量化或服务层。
TinyStories 不具备可靠中文问答能力。出处与 MIT 许可证保留于 [third_party/](third_party/README.md)。
