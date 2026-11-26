# 编码前算子审查：2026-11-25

来源：`sahal-colder/miniinfer-c` 的唯一远端分支 `dev`，原始提交
`cb17fd4bfbca1080788ab895174489f860123307`。这是初始化提交，无父提交。
该仓库为私有仓库；HTTPS Git 未获得连接器凭据。通过已授权 GitHub
连接器读取全部 31 个文件，恢复 Git 对象并逐一验证 blob、tree、commit SHA，
经 `git fsck --full` 后从本地镜像克隆到 `/home/wty/miniinfer-c`。
未替换为公开的 MiniInfer/C++ 仓库，也未修改 `/home/wty/Infra`。

## 基线清单

| 算子 | 状态 | 实际推理路径 |
| --- | --- | --- |
| SGEMM | 未实现 | 无 |
| SGEMV | 未实现 | 无 |
| RMSNorm | 未实现 | 无 |
| Softmax | 未实现 | 无 |
| Attention | 未实现 | 无 |
| PagedAttention 计算 Kernel | 未实现 | 无 |
| FlashAttention | 未实现 | 无 |
| Gather / Scatter 计算 Kernel | 未实现 | 无 |

不存在“仅原型”的上述计算算子，也没有可据以判断“存在缺陷”的算子实现。
缺失实现不代表已实现但未优化。FP32/FP16/BF16 推理主 dtype 未定义；
存储载荷是无类型字节块。本轮三个 dtype 是新增算子覆盖范围，不能声称
“已覆盖现有推理主路径”。

## 目录、构建和调用链证据

- `CMakeLists.txt`：项目 `MiniInferCStorage`，仅启用 C；默认静态库
  `miniinfer_storage`。`MINIINFER_WITH_CUDA` 只添加两个 `.c` 文件和 cudart。
  基线无 `.cu/.cuh`、cuBLAS 依赖、算子 Benchmark 或模型执行入口。
- `src/`：allocator、graph allocator、storage backend、tiered storage、file IO、
  KV cache、paged KV；可选 CUDA allocator/pool。`include/miniinfer/` 为相应 API。
- `examples/basic.c` 是存储示例。没有权重加载、Transformer 层或 decode loop。
- `tests/` 的 pool / graph / tiered / kv_cache / paged_kv / gpu_mock 共 6 项
  验证存储功能；基线 Debug 构建 6/6 通过。
- `mi_kv_sequence_append_page -> mi_storage_create_block`；
  `mi_paged_kv_acquire -> mi_storage_acquire -> page lease`；前缀发布/匹配沿 CPU
  前缀树操作整页 token 键。没有 QK、Softmax 或 PV 计算。
- `src/cuda_adapter.c`：`cudaMalloc/cudaMallocHost`、`cudaMemcpyAsync`、
  `cudaStreamSynchronize`；`cuda_memory_pool.c` 为事件保护的显存回收。
  Graph allocator 分配捕获工作区，不是计算 CUDA Graph 或 Attention 实现。
- `test_gpu_mock.c` 使用 POSIX 主机分配、`memmove` 模拟 GPU；不能证明真实
  CUDA 计算、向量化或 GPU 内存正确性。

## 缺口、重叠与优先级

目前没有可测的计算性能瓶颈：计算层尚不存在。任何“Attention 已接入”或
“当前 GEMV 受带宽限制”的判断都缺少基线证据。

现有存储、分页、前缀共享与 KV-Fabric 的职责重叠；暂不新增第二套管理接口，
本轮也不重构原有存储库。基线存储中已有的 allocator ops/ctx 是旧接口，
新增计算库不复用或扩展它们。

优先新增无状态 C ABI 算子与独立验收程序：先建立 dtype、布局、stride、
对齐、误差和计时口径，然后优化 SGEMV、RMSNorm。下一轮需先明确模型执行
入口与页表布局，再做 Paged Decode Attention 和推理集成。

## 环境

RTX 4060 Laptop GPU，NVML 显存 8188 MiB，CUDA 可用显存容量
8272019456 字节（保留区口径不同），SM 8.9；驱动 615.71.09；
nvcc/runtime 13.4（nvcc 13.4.92）；cuBLAS runtime 130800 / 13.8.0；
NCU 2026.3.1。沙箱内部不可见设备，实际 GPU 执行使用已授权沙箱外工具。

初期 NCU 环境探针报 `ERR_NVGPUCTRPERM`，`sudo -n` 报需要密码。
后续获得用户 sudo 授权后，已成功采集两类优化 Kernel、三种 dtype 的全部
6 份原始报告及硬件指标 CSV；详见 RESULTS.md。未更改驱动配置。
