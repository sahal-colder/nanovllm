# 验证记录：2026-11-04

环境：本工作区 WSL/Linux，GCC 11.4.0，Python 3.10.12。

## 离线正确性

以下三个构建全部通过。所有测试进程均在 `tests/` 目录运行。

```bash
make test
make sanitize
OMP_NUM_THREADS=2 make BUILD=build-omp OPENMP=1 test
```

| 检查 | 结果 |
| --- | --- |
| GQA、独立输出权重，逐位置全词表 logits 对 dense Python oracle | 最大绝对误差 `2.98023224e-07`，阈值 `2e-5` |
| MHA、共享输出权重，逐位置全词表 logits 对 dense Python oracle | 最大绝对误差 `2.38418579e-07`，阈值 `2e-5` |
| 不连续物理块 `[7,2,19,4,11]` | logits 与连续布局参考一致 |
| 三请求批处理 vs 串行、开启/关闭前缀缓存 | greedy 和带 seed 的 temperature/top-p 输出逐 token 一致 |
| 4 个物理块下三请求抢占、重放 | 开启和关闭前缀缓存均触发抢占；greedy 和随机采样均与串行一致 |
| 运行中新增、取消、空闲后再次加入请求 | 状态、引用计数、保留结果、前缀命中正确 |
| 页面大小 `1,2,3,4,7,16,64`，每 tick 仅 1 token | 输出一致，所有请求可结束 |
| 完整上下文与最后一个未 forward 的输出边界 | 64 个 forward 位置，可保留第 65 个采样 token |
| BPE merge、UTF-8 byte fallback、空 prompt | 通过 |
| EOS 提前结束与 `--ignore-eos` | 通过合法确定性 EOS 模型验证 |
| 非法 CLI 数值、截断/非法模型、损坏/不配套 tokenizer | 返回非零退出状态并给出错误信息 |
| ASan / UBSan | 无报告 |
| OpenMP，2 个线程 | 同一离线测试集全部通过 |

测试权重由固定 seed 生成；参考计算使用 Python double 和连续 K/V 数组，并非重新调用 C 的数学函数。误差数值来自小模型 fixture，不能推断所有尺寸与所有平台都具有同一数值误差。

## 真实模型与上游逐 token 对照

比较对象：`karpathy/llama2.c` 固定版本 `350e04fe35433e6d2941dce5a1f53308f87058eb` 的实际 forward 和 tokenizer。测试 wrapper 预先包含 `<stdint.h>`，再包含未修改的 `run.c`，以支持该环境下的 C11 编译。

```bash
cd tests
python3 ../tools/compare_upstream.py \
  --model ../models/stories15M.bin --tokenizer ../models/tokenizer.bin \
  --binary ../build/nanovllm
python3 ../tools/compare_upstream.py \
  --model ../models/stories260K.bin --tokenizer ../models/tok512.bin \
  --binary ../build/nanovllm
```

两个模型均通过以下四个 prompt，每种生成 32 个 greedy token：

- `Once upon a time`
- `One day, Lily`
- 空字符串
- `你好`

每个 prompt 同时验证默认参数、关闭缓存并设 `block_size=3` 和 `budget=1` 两条执行路径；全部与上游 ID 完全一致。为了比较固定数量 token，双方均忽略 BOS/EOS 的停止条件。中文输入测试验证的是分词和计算一致性，不代表模型会中文问答。

预训练权重及分词器均通过下载器中的 SHA256 校验，模型下载固定在 Hugging Face revision `0bd21da7698eaf29a0d7de3992de8a46ef624add`。

## 可运行例子

```bash
make api-demo
./build/nanovllm models/stories15M.bin -i "Once upon a time" -n 64 -t 0
./build/nanovllm models/stories260K.bin -z models/tok512.bin -i "Once upon a time" -n 32 -t 0
```

均成功生成自然语言。15M 输出开头：

```text
Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine.
```

15M 双请求的前缀复用检查：

```bash
./build/nanovllm models/stories15M.bin \
  -i "Once upon a time, there was a little girl" \
  -i "Once upon a time, there was a little boy" \
  -n 32 -t 0 --block-size 4 --batch 2 --budget 32 --ids
```

结果：`generated=64 forwards=76 reused=8 preemptions=0`。

15M 双请求的紧凑 KV 池检查：

```bash
./build/nanovllm models/stories15M.bin \
  -i "Once upon a time" -i "Once upon a time" \
  -n 32 -t 0 --block-size 4 --blocks 10 --batch 2 --budget 8 --no-cache --ids
```

结果：`generated=64 forwards=128 reused=0 preemptions=4`，两个相同请求输出的全部 32 个 token ID 一致。

这些是功能正确性记录，没有进行原版 nano-vLLM、vLLM 或 llama2.c 的公平性能基准，也未验证 GPU、Windows 原生或大模型运行。


## 2026-11-06 前缀哈希链（最终采用仅哈希匹配）

按用户要求，最终移除了 parent、generation、parent_generation、本块 token 副本和沿链内容校验。缓存仅比较前缀长度和 64 位 FNV-1a 哈希，接受碰撞导致错误复用的可能性。

`make test` 与 `make sanitize` 全部通过，包括 greedy/随机采样的批处理、缓存及抢占重算一致性。直接缓存检查验证相同块内容、不同前文会产生不同前缀哈希；重新分配并重建相同前缀后，旧后续块仍可命中。

查找候选仍线性扫描物理块。每块仅保留引用数、前缀长度、哈希和空闲顺序。
