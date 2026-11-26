# 三个核心算子的验证入口

当前实现位于 nano-vLLM-C。第一轮 SGEMV/RMSNorm 的源码迁入 `src/operators/`，
没有引入 MiniInfer-C 的 allocator、tiered storage 或分页管理层。

- `bench_operators.c`：SGEMV / RMSNorm 基础版、优化版、cuBLAS / CPU 参考。
- `profile_ncu.py`：SGEMV/RMSNorm 指定指标采集。
- `../round2/`：Paged Attention、完整 Decoder、Graph 的 C11 测试/计时和 profiler 工具。
- `round2-results/`：本轮完整 CSV、NCU/Nsys 原始报告、sanitizer、checkpoint 与环境证据。
- `results/`、`RESULTS.md`、`AUDIT.md`：2026-11-25 在 MiniInfer-C 原型中的历史实测，
  环境文件保留当时的路径/哈希，不能当作当前 nano-vLLM 代码结构的描述。

本轮报告：[CUDA_VALIDATION.md](../../docs/CUDA_VALIDATION.md)。

## 线程组织与字节口径

| 算子 | 优化路径 | 边界 |
| --- | --- | --- |
| SGEMV | 4 warps/CTA，1 warp/row；float4 或 4×16-bit 加载、shuffle 归约，无 shared/barrier | 元素对齐即可；非向量对齐和尾部安全标量访问；面向 batch=1/低 batch |
| RMSNorm | 256 threads/row，输入驻留寄存器；warp + 跨 warp 归约，向量读写 | hidden 1..16384；奇数尾部、同 stride 原地操作；大数值平方和溢出时缩放归约 |
| Paged Attention | 4 warps/CTA，warp 处理 head 的 64-token 分段；Q/PV 寄存器；256-thread 合并 CTA 共享 split 权重 | GQA/MQA、乱序页、partial tail；head_dim 1..256，容量≤65536 |

所有 dtype 均 FP32 累加。Attention 使用稳定 Online Softmax，不保存 seq_len 大小的
score/probability 数组；跨分段通过 max/sumexp 重新归一化。`__expf` 的误差由 CPU
逐元素测试约束，没有对整个工程启用 fast-math。

有效带宽统一用逻辑唯一张量字节；它不同于 NCU 的真实 DRAM 字节计数。
GQA 复读和 batched GEMV 重读 W 不会膨胀有效带宽分子。cuBLAS 对照使用 FP32
pedantic math，并禁止低精度中间归约；FP32 不使用 TF32。

SGEMV 高 batch 会显著落后复用 W 的 cuBLAS GEMM。当前引擎每请求串行推进，
CUDA 执行主路径 batch=1；独立测试覆盖 B>1，不将它称为批量 GEMM 优化。

## 复现

```sh
cmake -S . -B build-cuda -DNANO_WITH_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-cuda -j4
build-cuda/bench_operators --test
build-cuda/bench_operators --samples 80 > gemv-rms.csv
build-cuda/bench_decode --test-attention
build-cuda/bench_decode --attention > attention.csv
bash tools/validate_cuda.sh build-cuda benchmarks/operators/runs/new-run --profile
```

NCU 输出为原生 `.ncu-rep/.ncu-repz` 及 raw CSV。NSys 同时输出 `.nsys-rep`、
SQLite 与官方统计。权限不足时采集失败，不生成占位指标。原始失败/调优记录
与最终数据名称区分，正式结论只取 `attention.csv`、`decode.csv`、`*-ncu/` 和 `nsys/`。
