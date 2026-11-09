# nano-vLLM-C：用 C 学习 LLM 推理引擎

一个约 800 行 C11 核心代码的 **CPU 学习版 nano-vLLM**。可以加载预训练模型生成文本，同时观察请求调度、分页 KV Cache、前缀复用和缓存不足时的抢占重算。

参考 [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) 的引擎结构，改写 [llama2.c](https://github.com/karpathy/llama2.c) 的前向计算和 BPE 分词。出处、版本与 MIT 许可证见 [third_party/README.md](third_party/README.md)。这是独立的 C 教学实现，不是两个上游项目的官方移植。

## 直接运行

在 WSL/Linux 中进入本目录。编译仅依赖 C 编译器、Make 和系统数学库；下载模型与运行测试使用 Python 3 标准库，不需要 pip、PyTorch 或 GPU。

```bash
cd /mnt/c/Users/wtyjt/Desktop/nanovllm
make
python3 tools/download.py
./build/nanovllm models/stories15M.bin -i "Once upon a time" -n 64 -t 0
```

也可以执行 `make demo`，完成编译、下载及运行。模型已经在本工作区下载好；下载器会用 SHA256 校验已有文件，正确时直接复用。

15M 权重约 61 MB，默认分词器约 434 KB。实际运行示例开头：

```text
Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine.
```

这是在 TinyStories 上训练的英文续写模型；中文输入可以分词，但模型不具备可靠的中文问答能力。

更小的 260K 参数 GQA 模型只需约 1.1 MB，文本质量更低，适合快速调试：

```bash
python3 tools/download.py --model 260k
./build/nanovllm models/stories260K.bin -z models/tok512.bin -i "Once upon a time" -n 64 -t 0
```

**模型和分词器必须配套**：15M 使用 `tokenizer.bin`（32000 词），260K 使用 `tok512.bin`（512 词）。

## 能学到什么

| 能力 | 代码位置 | 本项目的实现 |
| --- | --- | --- |
| Llama Transformer | `src/model.c` | FP32、RMSNorm、RoPE、MHA/GQA、SwiGLU、残差连接 |
| PagedAttention 思路 | `src/cache.c`、`src/model.c` | 逻辑块映射到物理块；attention 直接读取分页后的 K/V |
| 连续批处理 | `src/engine.c` | 每个 tick 推进多条请求，完成即释放位置，允许执行途中加入新请求 |
| 分块 prefill | `src/engine.c` | 每个 tick 有 token 预算，长输入分多次计算 |
| 前缀缓存 | `src/cache.c` | 逐块前缀哈希链，仅比较长度和哈希，引用计数共享，空闲块 LRU 淘汰 |
| 抢占重算 | `src/engine.c` | 显存概念用 CPU 内存模拟；KV 不够时释放较新请求的页面，之后重放已有 token |
| 分词与采样 | `src/tokenizer.c`、`src/sampler.c` | BPE、UTF-8 byte fallback、greedy、temperature、top-p |

这里的“批处理”是调度层的批次：同一个 tick 内各请求依次做 CPU forward，共享模型权重。没有 GPU 的批量矩阵乘法，也没有 CUDA、FlashAttention、张量并行、量化或 HTTP 服务，不以复现原版吞吐量为目标。

权重格式是 **llama2.c legacy FP32 `.bin`**：7 个 int32 头字段加 FP32 tensor。不能直接加载 GGUF、safetensors、Qwen 权重或 llama2.c 的量化/版本化格式。当前实现固定 RoPE base=10000 和 RMSNorm epsilon=1e-5，支持共享/非共享输出权重。

## 观察调度与前缀缓存

`-i` 可以重复。单请求流式打印；多请求在完成后按请求 ID 打印，调度进度在 stderr。

```bash
./build/nanovllm models/stories15M.bin \
  -i "Once upon a time, there was a little girl" \
  -i "Once upon a time, there was a little boy" \
  -n 32 -t 0 --block-size 4 --batch 2 --budget 32 --trace
```

比较关闭缓存后的 `forwards`，检查启用时的 `reused`。前缀复用只跳过已经算好且完整的块；第二个请求太早开始、还没有完整块时，缓存命中可以为 0。

```bash
./build/nanovllm models/stories15M.bin \
  -i "Once upon a time" -i "Once upon a time" \
  -n 32 -t 0 --block-size 4 --blocks 10 --batch 2 --budget 8 --trace --no-cache
```

上面的紧凑 KV 池可用于观察 `preemptions`。`--blocks` 至少要让单条请求独立跑完：

```text
ceil((prompt_tokens + max_new_tokens - 1) / block_size) <= num_blocks
```

`-n` 是新增 token 上限；输入和实际参与 forward 的输出必须在模型上下文内。最后一个输出不必再次 forward，因此允许 `prompt_tokens + max_new_tokens = context + 1`。采样到 BOS/EOS 就提前结束；`--ignore-eos` 用于测试固定长度输出。

输出统计的 `output_tok/s` 包含 prefill 和重算耗时，不能作为独立 decode 吞吐量，也没有与上游做性能对比。所有选项见 `./build/nanovllm --help`。

## 阅读顺序

先读 [中文学习指南](docs/learning.md) 理解概念，再按 [学习路线](docs/study-guide.md) 分阶段动手——它给了每个阶段的阅读目标、动手实验和验收问题，以及三条按时间预算安排的路线。

源码内已写好详细中文注释：文件头说明这个文件在系统里的角色，结构体字段说明语义和依赖关系，函数头说明前置条件与不变量，行内注释标出容易写错的地方。然后按以下顺序对照代码：

1. `include/nv.h`：模型、序列、缓存和引擎的状态。
2. `src/main.c` → `src/engine.c`：一个请求如何从 prompt 走到输出。
3. `src/model.c`：一个 token 如何经过所有 Transformer 层。
4. `src/cache.c`：K/V 如何跨序列共享和回收。
5. `src/tokenizer.c`、`src/sampler.c`：文本和 token 的边界。

[examples/api.c](examples/api.c) 展示运行中加入第二条请求。执行：

```bash
make api-demo
```

## 验证

```bash
make test
make sanitize   # Linux + GCC/Clang 的 AddressSanitizer / UndefinedBehaviorSanitizer
```

测试无需联网；Make 会进入 `tests/` 运行测试。包含独立 dense Python 数学参考、MHA/GQA、非连续物理页面、批处理/缓存/抢占的 token 一致性、随机采样、动态加入和取消、上下文边界、EOS、分词及损坏文件检查。随机权重 fixture 用来测正确性，不用来展示语言能力。

可选的真实模型上游对照：

```bash
cd tests
python3 ../tools/compare_upstream.py \
  --model ../models/stories15M.bin --tokenizer ../models/tokenizer.bin \
  --binary ../build/nanovllm
```

该脚本下载固定版本的上游 `run.c` 到临时目录，编译并比较四种 prompt 的 32 个 greedy 输出 token。见 [验证记录](docs/validation.md)。

可选 OpenMP 并行 CPU 矩阵乘法，使用独立构建目录避免混用编译选项：

```bash
make BUILD=build-omp OPENMP=1
OMP_NUM_THREADS=4 ./build-omp/nanovllm models/stories15M.bin -n 64 -t 0
```

## 目录

```text
include/nv.h              C 数据结构与 API
src/main.c                CLI、流式输出、统计
src/engine.c              调度、请求状态、抢占重算
src/cache.c               分页 KV、引用计数、前缀缓存
src/model.c               权重 mmap 与 Llama 前向计算
src/tokenizer.c           BPE 编码与解码
src/sampler.c             greedy / temperature / top-p
examples/api.c            执行途中加入请求的示例
models/                   下载的权重和分词器（不纳入版本控制）
tests/                    离线正确性测试
tools/download.py         固定版本下载、SHA256 校验
tools/compare_upstream.py  可选真实模型上游对照
docs/                     中文学习说明、学习路线与验证记录
third_party/              原始出处和许可证
```

代码采用 MIT 许可；上游改写部分保留各自 MIT 声明。下载模型适用其原始模型许可证。
