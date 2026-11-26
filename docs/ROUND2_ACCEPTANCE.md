# 第二轮阶段验收与交付

验收日期：2026-11-26（Australia/Perth）。按后续指示，计算项目采用用户的
`sahal-colder/nanovllm`，基线为 `060d2b4bb339d4feb85c4b6858e7b0eb67d1e6f9`，
分支 `codex/paged-decode-cuda`。MiniInfer-C 保留第一轮；不继续扩展其存储架构。

## 交付与验收范围

| 项目 | 已完成与证据 |
| --- | --- |
| Paged Decode Attention | FP32/FP16/BF16，非连续物理页指针表、Block Table、GQA/MQA、稳定 Online Softmax、64-token 分段及归一化合并；partial tail 和非向量对齐地址安全处理 |
| C11/CUDA 推理 | 保留原模型加载/调度/分页引用/采样；加入 Embedding、RMSNorm、QKV、RoPE/KV Store、Attention、O/残差、SwiGLU FFN、最终 logits 的真实执行链路 |
| CUDA Graph | 固定设备地址与容量，token/长度/页表内容动态；捕获内仅 Kernel/Memset；同一 GraphExec 跨页和分段边界验证 |
| 独立数值验证 | Attention 90 项随机/边界用例；独立 FP64 两遍 Softmax；单层/B1 21 步及三层/B2 83 步逐 logits 与全部物理 KV 槽对照，覆盖全部三种 dtype |
| 真实 checkpoint | 固定 SHA256 的 TinyStories 260K；4 prompt × 32 个 FP32 greedy Token，CPU/eager/Graph 完全一致 |
| 全量回归 | CMake Release 构建、8/8 CTest（含两项 fixture）；原 CPU Makefile 回归通过；旧算子及新执行链路 Memcheck/Racecheck 零错误 |
| 性能与 profiler | 144 行 Attention、6 行 Decode；9 份原始 NCU 报告含 12 条 Kernel 记录；Nsys 报告、SQLite、官方统计与数据绘制的 Timeline |
| 一键复现 | `tools/validate_cuda.sh` 在全新构建与输出目录完整执行通过；默认不要求 sudo，`--profile` 时采集 NCU/Nsys |
| 文档与简历 | README 架构/目录/复现命令、完整 CUDA 数值与性能报告；双项目一页 A4 MD/Word/PDF；面试自查另存 |

## 实测摘要

RTX 4060 Laptop（8188 MiB，SM89），CUDA 13.4，cuBLAS 13.8。
所有原始数据位于 `benchmarks/operators/round2-results/`，完整口径及三精度表见
[CUDA_VALIDATION.md](CUDA_VALIDATION.md)。

- FP32 Attention，B1/H16/KV4/D128/S4096：串行 Warp 参考 P50/P95
  **1596.42/1597.44 µs**，优化版 **73.73/73.73 µs**；优化有效带宽
  **212.13 GiB/s**。对照为本项目参考实现，不是成熟库；warm-cache 有效带宽不等于 DRAM 计数。
- 同形状 Attention 分段 Kernel：NCU DRAM **62.60% / 130.43 GiB/s**、
  Achieved Occupancy **75.86%**；合并 Kernel 仅 16 CTA，Occupancy **11.16%**。
- FP32 SGEMV，8192×4096/B1：NCU DRAM **96.59% / 230.06 GiB/s**；
  RMSNorm 128×4096：**65.72% / 136.38 GiB/s**。第一轮 RMSNorm P50
  **8.19 → 6.14 µs**，历史延迟与本轮 NCU 来源分别保留。
- 4 层随机 Decoder，d512/FFN1536/H8/KV2/B1/S129，100 样本：FP16 eager
  P50/P95 **250.34/251.41 µs**，Graph **221.74/223.49 µs**，P50 降 **11.43%**。
  计时覆盖 Host 提交至完成等待，不含元数据上传、logits 回读、采样或 Graph 实例化。
- Nsys 每模式 5 步均执行 255 个 GPU Kernel：Host 每步从 **51 次 Kernel Launch
  → 1 次 Graph Launch**；活动窗口内间隙 **9.10% → 7.50%**。GPU Kernel 数未减少，
  并未消除全部设备调度间隙。Timeline 为原始 SQLite 数据绘图，非 UI 截屏。

## 简历及隔离校验

交付路径：

- `~/Infra/docs/resume_ai_infra.md`
- `~/Infra/docs/resume_out/王廷煜_AI_Infra_研发实习生.docx`
- `~/Infra/docs/resume_out/王廷煜_AI_Infra_研发实习生.pdf`
- `~/Infra/docs/interview_prep_kv_fabric.md`
- `~/Infra/docs/resume_out/layout-validation.json` 和 `resume_preview.png`（排版校验）

Word 经文档运行时渲染并目视检查。PDF 严格 1 页、A4 210×297 mm；11 pt 正文，
纵向墨迹覆盖 **88.35%**；正文与 Markdown 一致、嵌入中文字体、5 处箭头保留。
修正“末维连续”、`ext4` 和 `5.006 s / 5.034 s` 表述；两项目各 5 条成果，
面试问答未混入简历。KV-Fabric 数字来自原有 `resume-evidence.json`，计算侧数字来自
本轮原始 CSV/NCU/Nsys；两项目不同模型和计时口径不作直接比较。

交付前对 KV-Fabric 快照的 328 个已跟踪文件逐一 SHA256 比较，发现 43 个文件在快照后
出现外部改动，因此不声称全仓哈希不变，也不回滚这些改动；对比路径清单保留于
`benchmarks/operators/round2-results/infra-source-audit.json`。本轮未编辑 KV-Fabric 源码，
在 `~/Infra` 仅更新指定简历、面试文档及其排版产物，覆盖前版本备份在
`/tmp/nano-resume/before/`。本轮未执行 Git commit 或 push。

## 复现

```sh
cd ~/nanovllm
bash tools/validate_cuda.sh build-cuda benchmarks/operators/runs/new-run
# NCU 在本机需管理员性能计数器权限：
bash tools/validate_cuda.sh build-cuda benchmarks/operators/runs/profile-run --profile
python3 tools/download.py --model 260k
build-cuda/nanovllm_cuda models/stories260K.bin -z models/tok512.bin \
  -i "Once upon a time" -n 32 -t 0 --cuda-graph
```

结果目录需不存在；脚本拒绝覆盖已有实测。`environment.json` 记录源文件、计时二进制和
fixture 哈希。失败及调优日志保留并与正式 CSV 区分；未获取 profiler 时不填充占位数字。

## 已知边界

当前调度器逐请求执行 B1 GPU 前向；独立测试支持 B>1，但高 batch 应切换可复用权重的
GEMM。只加载 legacy llama2.c FP32 checkpoint，加载后可转低精度；邻接对 RoPE。
未实现 Qwen/safetensors/GGUF、量化、TP、GPU 批量 GEMM、并行 Prefill、服务层，
也未在本轮连接 KV-Fabric 的分层/远程存储。真实生成验证为 FP32 TinyStories；
低精度以逐元素误差验收，不保证 FP32 Token ID 一致。Graph 容量、batch、模型形状
和设备地址固定，元数据内容可变。当前结果证明计算链路可运行，不代表生产模型吞吐。
