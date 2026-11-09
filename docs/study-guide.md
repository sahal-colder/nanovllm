# 学习路线：从跑通到能改造

这份文档回答一个问题：**拿到这个仓库，按什么顺序读、每个阶段动手做什么、怎么判断自己真的看懂了。**

[learning.md](learning.md) 是按主题讲概念的（一次 forward 做了什么、分页为什么这么设计），[validation.md](validation.md) 是正确性证据。
本文是**行动路线**：把代码拆成 7 个阶段，每阶段给出阅读目标、动手实验和验收问题。

> 源码里已经写好了详细中文注释。注释的组织方式是固定的，学会用它：
> - **文件头**：这个文件在系统里扮演什么角色、解决什么问题。
> - **结构体字段**：每个字段的语义，以及它和其他字段的关系（哪一个是"真相"、哪些是派生的）。
> - **函数头**：前置条件、不变量、复杂度、为什么这样写。
> - **行内注释**：容易写错的那一行，以及"为什么不能反着写"。
>
> 读完一段代码如果答不上"如果把这一行删了会怎样"，就回去找对应的行内注释。

---

## 0. 先想清楚：你要从这个项目里学什么

这个项目刻意只做三件事，并且每件都做到"能对着代码讲明白"：

| 层次 | 具体内容 | 对应文件 |
| --- | --- | --- |
| 数学 | Llama 一次前向：RMSNorm、RoPE、GQA、causal attention、SwiGLU | `src/model.c` |
| 系统 | 推理引擎的调度与内存：连续批处理、分块 prefill、分页 KV、前缀缓存、抢占重算 | `src/engine.c`、`src/cache.c` |
| 工程 | C11 手写内存管理、mmap 权重、离线可复现测试 | 全项目、`tests/` |

它**没有**：CUDA、量化、张量并行、HTTP 服务、批矩阵乘。所以不要指望在这里学"怎么把吞吐做到生产级"；它教你的是**这些优化到底在优化什么**。

一句话定位：**用 CPU 换掉 GPU 之后，vLLM 的骨架仍然完整可见。**

---

## 1. 前置知识与自检

| 需要 | 到什么程度 | 用在哪里 | 不熟怎么办 |
| --- | --- | --- | --- |
| C 语言 | 指针、数组、`malloc/free`、`struct`、`#define`、位运算 | 全项目 | 能读懂 `memcpy`/`memmove` 和指针算术就够了，不需要会用宏写框架 |
| 线性代数 | 矩阵×向量、点积、softmax、归一化 | `model.c` | 只需理解"一行权重和输入做点积 = 一个输出神经元" |
| Transformer | embedding、attention、残差、FFN | `model.c` | 先看 [learning.md 第 2 节](learning.md)，再对照代码 |
| 操作系统 | 文件 mmap、虚拟内存（了解即可） | `nv_model_load()` | 记住"mmap = 让文件内容看起来像数组，不用先读进内存" |
| **不需要** | CUDA、PyTorch、分布式、Web 框架 | — | 这个项目一行都没有 |

**一个重要心态**：这个项目的价值不在于写得快，而在于**每个设计决定都能被一个实验验证**。所以每读一个概念，都要问"我怎么用命令行证明它对"。

---

## 2. 环境准备

项目用了 `sys/mman.h`、`unistd.h`，是 **POSIX/Linux** 代码，Windows 原生编译器编不过。

推荐 **WSL**（Windows 上最省事）：

```bash
wsl                       # 进入 WSL 后
cd /mnt/c/Users/wtyjt/Desktop/nanovllm
sudo apt install build-essential python3    # 只需一次
make && python3 tools/download.py
./build/nanovllm models/stories15M.bin -i "Once upon a time" -n 32 -t 0
```

也可以用任意 Linux 机器或 Docker 镜像。`python3` 只用于下载和测试，**编译本身只需要 cc + make + libm**。

> 如果 `make` 报错找不到 `cc`，装 `build-essential`。
> 如果模型下载失败，先检查网络；`models/` 里已经有文件时下载器会校验 SHA256 并直接复用。

---

## 3. 三条按时间预算的路线

| 你有多少时间 | 做什么 | 目标 |
| --- | --- | --- |
| **30 分钟** | 阶段 1 + 阶段 2，只读 `nv.h`、`main.c`、`engine.c` 的 `nv_engine_step()` | 能画出"文本 → token → 调度 → forward → token → 文本"的完整链路 |
| **3 小时** | 阶段 1–5，外加阶段 6 的前两个实验 | 能解释分页 KV、前缀缓存、抢占重算分别解决什么问题 |
| **3 天**（认真学） | 7 个阶段全做 + 第 8 节的 3 个改造练习 + 自测题 | 能独立修改调度策略或缓存结构，并用测试证明没算错 |

---

## 4. 七个阶段

每阶段的格式是 **目标 → 读什么 → 动手 → 验收问题**。验收问题答不上来就先别进入下一阶段——后面的内容会依赖它。

### 阶段 1：建立全局地图（约 20 分钟）

**目标**：不碰数学，先知道有哪些部件、谁调用谁。

**读什么**
1. `include/nv.h` 文件头 + 五个核心结构体的注释：`NvModel`、`NvCache`、`NvBlock`、`NvSequence`、`NvEngine`。
2. `src/main.c` 的 `main()`：注意初始化顺序（模型 → 分词器 → 引擎）和那个 `while (nv_engine_step(&engine))` 循环。

**动手**

```bash
./build/nanovllm --help
./build/nanovllm models/stories15M.bin -i "Once upon a time" -n 16 -t 0
```

观察 stderr 的两行：模型超参（`dim/layers/heads/kv_heads/vocab/context`）和 KV 池大小（MiB）。

**验收问题**
1. 一次推理需要保存哪三类状态？分别属于哪个结构体？
2. 为什么 `nv_engine_step()` 返回 0 就代表"结束了"？
3. `NvModel` 里的 `logits` 是谁分配的，为什么可以复用？

---

### 阶段 2：一条请求的一生（约 40 分钟）

**目标**：搞清 `NV_WAITING → NV_ACTIVE → NV_DONE` 的迁移，以及 `tokens` 和 `computed` 的区别。

**读什么**
1. `src/engine.c` 的 `nv_engine_add()`：看它都校验了什么（尤其是"单请求必须能装进池子"那条）。
2. `src/engine.c` 的 `nv_engine_step()`：先只读到内层循环的前半段，别管抢占。
3. `include/nv.h` 里 `NvSequence` 的注释，特别是那个 "prompt 5 个 token，prefill 后 computed=5、num_tokens=6" 的例子。

**动手**：在纸上跟着跑一遍

```bash
./build/nanovllm models/stories260K.bin -z models/tok512.bin \
  -i "Once upon a time" -n 4 -t 0 --ids --trace --batch 1 --budget 4 --ignore-eos
```

先记住编码结果：`"Once upon a time"` 在 15M 的 `tokenizer.bin` 里是 **5 个 token**
`[1, 9038, 2501, 263, 931]`，对应 `"<s>"`、`" Once"`、`" upon"`、`" a"`、`" time"`。
注意插入的那个"假空格"已经被 BPE 合并进 `" Once"` 了——所以示例不是 6 个 token。

然后看调度：`--batch 1 --budget 4` ⇒ 每条请求每 tick 的配额是
`quota = budget / batch = 4`（这条规则见 `nv_engine_step()` 里的 `quota`）。
于是 `step=1` 只做 4 次 forward（prefill 前 4 个位置），`step=2` 做第 5 次并采样出第一个新 token，
之后每个 tick 各产出 1 个新 token。`-n 4` 时总数应该是 `generated=4 forwards=8`
（5 次 prefill + 4 个输出里除最后一个外的 3 次回填 = 8）。

再试一下默认的 `--batch 4 --budget 4`（去掉 `--batch 1`）：`quota` 变成 `4/4 = 1`，
于是每 tick 每条请求只算 1 个 token，prefill 被切成 5 段。对比两次的 `step` 数量，
你就理解了 `max_sequences` 和 `token_budget` 这两个旋钮是**两个不同维度**。

**验收问题**
1. prefill 结束后 `computed` 和 `num_tokens` 差几？为什么？
2. `tokens[]` 和 `block_table[]` 各自由谁分配、容量按什么算？
3. 为什么 `nv_engine_add()` 要复制一份 tokens，而不是直接用调用方的指针？
4. 如果 `max_new_tokens` 让 `prompt + max_new - 1 > context`，`add()` 会被哪个检查拒绝？为什么允许 `prompt + max_new = context + 1`？

---

### 阶段 3：数学核心，一次 forward（约 60 分钟）

**目标**：能对着 `nv_forward()` 讲出每一行在算什么，并且知道它和 llama2.c 的差别在哪。

**读什么**
1. `src/model.c` 的 `nv_forward()`：**先假装 `cache`/`table` 不存在**，把 `nv_cache_key(...)` 理解成 `k[pos]`，就能看成标准的 llama2.c forward。
2. `rmsnorm()`、`matmul()`、`nv_softmax()` 三个小函数。
3. `learning.md` 第 2 节对照着看。

**动手**：用真实模型验证数学没算错

```bash
cd tests
python3 ../tools/compare_upstream.py \
  --model ../models/stories15M.bin --tokenizer ../models/tokenizer.bin \
  --binary ../build/nanovllm
```

这个脚本会下载固定版本的 llama2.c、编译它，然后逐 token 对比。两个完全独立的实现，32 个 greedy token 全部相同，说明你的理解（以及代码）是对的。

**验收问题**
1. `matmul(out, x, w, in, rows)` 到底算的是 `W·x` 还是 `x·W`？为什么不需要转置？
2. RoPE 里 `i % hs` 是什么意思？如果换成 `pos % hs` 会怎样？
3. GQA 中第 `head` 个 Q 头用的是第几组 K/V？`group` 怎么算？
4. attention 里为什么 Q/K 要转、V 不用转？为什么先写 K/V 再算 attention？
5. `m->xb` 在同一层里被用了两次（存 RMSNorm 结果、存 attention 输出），为什么不会冲突？
6. 为什么 `nv_forward()` 返回的是内部指针 `m->logits`，而不是拷贝？

---

### 阶段 4：分页 KV（约 50 分钟）

**目标**：理解"逻辑块 → 物理块"这层映射为什么存在、落地成了哪几行代码。

**读什么**
1. `include/nv.h` 里 `NvBlock` 和 `NvCache` 的注释：四种块状态那张表要能自己画出来。
2. `src/cache.c` 的 `nv_cache_key()`/`nv_cache_value()`：只有一处公式。
3. 回到 `src/model.c` 的 `nv_forward()`，找出**三处**分页寻址（写 K/V、读 K 打分、读 V 加权）。

**动手**：证明分页与数学无关

```bash
for bs in 1 3 4 7 16; do
  ./build/nanovllm models/stories260K.bin -z models/tok512.bin \
    -i "Once upon a time" -n 8 -t 0 --ids --ignore-eos --block-size $bs
done
```

五条输出必须**一模一样**（除了 stderr 的统计）。块大小只改变"K/V 存在哪里"，不改变"算出来是什么"。这就是最有力的分页正确性证据。

**验收问题**
1. 逻辑位置 `t` 的 K/V 在内存里的地址公式是什么？三个维度分别是什么？
2. 为什么 `block_table` 里的物理块号可以乱序、可以不连续？
3. 一个位置需要新的物理块时，`nv_engine_step()` 是靠哪个条件发现的？
4. 关闭前缀缓存（`--no-cache`）时，KV 池还会分配吗？为什么？
5. 15M 模型的默认 KV 池占多少内存？公式写出来算一遍（答案：约 27 MiB）。

---

### 阶段 5：调度与抢占（约 50 分钟）

**目标**：能解释"为什么只能抢占更年轻的请求"，并亲眼看到抢占发生。

**读什么**
1. `src/engine.c` 的 `nv_engine_step()` 全文，把 `active`/`used`/`scheduled` 三个计数器的作用标出来。
2. `allocate()`：只看"为什么 `i > s->id`"。
3. `include/nv.h` 里 `NvOptions` 的注释：`max_sequences` 和 `token_budget` 是两个不同维度。

**动手**：制造内存压力

```bash
./build/nanovllm models/stories15M.bin \
  -i "Once upon a time" -i "Once upon a time" \
  -n 32 -t 0 --block-size 4 --blocks 10 --batch 2 --budget 8 --trace --no-cache --ids
```

预期统计（来自作者的验证记录）：`generated=64 forwards=128 reused=0 preemptions=4`。
注意 `forwards=128` 正好是 `64*2`——**多出来的 64 次就是重算**。再对比一下去掉 `--no-cache` 之后 `forwards` 的变化。

然后试着把 `--blocks` 调到 5、6，看 `preemptions` 怎么变；调到 3 以下会发生什么（`add()` 直接拒绝——为什么？）。

**验收问题**
1. 同一个 tick 里，为什么抢占的受害者一定还没被处理过？
2. 如果允许 A 抢 B、B 也能抢 A，会发生什么？为什么"只能抢更年轻的"能避免？
3. `quota` 的作用是什么？`--budget 1`、`--batch 4` 时每条请求每 tick 能算几个 token？
4. 被抢占的请求恢复时，为什么不会"重新采样"一遍旧输出？
5. 抢占时 `tokens` 和 `rng` 被保留、`computed` 和 `block_table` 被清空，各自为什么？

---

### 阶段 6：前缀缓存（约 40 分钟）

**目标**：理解"为什么必须比较整段前缀"，并亲眼看到命中。

**读什么**
1. `src/cache.c` 的 `nv_cache_reuse()`：`limit` 那行和 `memcmp` 那行。
2. `src/cache.c` 的 `nv_cache_publish()`：三个 `assert` 分别在防什么。
3. `nv_cache_allocate()` 的两段式策略（干净块优先，否则 LRU 淘汰缓存块）。

**动手**：观察命中

```bash
# 命中：两个 prompt 共享较长前缀，块大小 4
./build/nanovllm models/stories15M.bin \
  -i "Once upon a time, there was a little girl" \
  -i "Once upon a time, there was a little boy" \
  -n 32 -t 0 --block-size 4 --batch 2 --budget 32 --trace
# 看 stderr 的 reused 是不是 8，forwards 是不是 76（而不是 84）
```

```bash
# 对比：完全相同的两个 prompt，但关掉缓存
./build/nanovllm models/stories15M.bin \
  -i "Once upon a time" -i "Once upon a time" \
  -n 32 -t 0 --block-size 4 --batch 2 --budget 8 --trace --no-cache
```

**再做一个反直觉实验**：让第二个请求"太早"开始（`--budget` 很大、第一条还没填满块），此时 `reused` 可能是 0。为什么？因为缓存只能发布**已填满**的块。

**验收问题**
1. 为什么命中上限是 `(num_tokens - 1) / block_size`，而不是 `num_tokens / block_size`？
2. 为什么前缀要存"整段历史"而不是"块内那几个 token"？举个会出错的例子。
3. `refs` 从 2 减到 1 时，`nv_cache_release()` 为什么什么也不做？
4. 已发布的块为什么不能再改？`publish()` 里的 `assert(b->refs == 1)` 在防什么？
5. 关闭缓存时为什么还要分配 `prefix` 数组？（提示：看 `nv_cache_init()` 的注释和"教学取舍"）

---

### 阶段 7：分词与采样（约 30 分钟）

**目标**：搞清楚文本和 token 的边界，以及"确定性"从哪来。

**读什么**
1. `src/tokenizer.c` 的 `nv_encode()`：两阶段——先按 UTF-8 字符贪心匹配，再做 BPE 合并。
2. `nv_decode()`：`previous == 1` 时吃掉空格、`<0xXX>` 还原成字节。
3. `src/sampler.c`：greedy / 温度 / top-p 三段，以及 xorshift64* 那四行。

**动手**

```bash
M=models/stories15M.bin
# 同一个 seed，输出必须逐字节可复现
./build/nanovllm $M -i "Once upon a time" -n 16 -t 0.8 -p 0.9 -s 7
./build/nanovllm $M -i "Once upon a time" -n 16 -t 0.8 -p 0.9 -s 7
# greedy 下温度/块大小/预算都不该改变 token ID
./build/nanovllm $M -i "Once upon a time" -n 16 -t 0 --ids --ignore-eos
./build/nanovllm $M -i "Once upon a time" -n 16 -t 0 --ids --ignore-eos --budget 1 --block-size 3 --no-cache
# 中文：能分词，但模型不会中文问答
./build/nanovllm $M -i "你好" -n 16 -t 0
```

**验收问题**
1. 为什么 `nv_encode()` 要在 BOS 之后插一个空格 token？解码时怎么抵消这个"假空格"？
2. 一个中文字符在词表里不存在时会变成几个 token？依据是哪条规则？
3. 为什么 `--ids` 输出是 9 段（`[0]` + 8 个数字）而不是 8 个数字？
4. 为什么随机数状态要放在 `NvSequence` 里而不是全局？如果放全局，哪个测试会失败？
5. top-p 里 `mass` 可能小于 1，为什么 `target = coin * mass` 仍然正确？

---

### 阶段 8（可选）：用测试反推设计（约 40 分钟）

**目标**：测试代码往往比实现代码更清楚地写出了"应该成立什么"。

**读什么**
1. `tests/test_engine.c` 的 `invariant()`：一个断言抓住了整个引用计数系统。
2. `oracle()`：为什么故意用乱序块表 `{7, 2, 19, 4, 11}`。
3. `tests/fixtures.py`：Python 用 double + 连续 KV 独立算一遍，这才是"独立参考实现"。
4. `tests/test_cli.py`：为什么用 `--no-cache --budget 1 --block-size 3` 对比默认参数。

**动手**

```bash
make test        # 离线正确性
make sanitize    # ASan + UBSan（需要 Linux + GCC/Clang）
```

**验收问题**
1. `invariant()` 里那个等式是什么？如果 `refs` 记少了会出什么 bug？
2. 为什么 oracle 的容差是 `2e-5` 而不是 0？为什么不设成 `1e-3`？
3. 为什么测试用随机权重而不是真实权重？
4. `test_cli.py` 里 `a.stdout == b.stdout` 这一句，实际上证明了什么性质？

---

## 5. 关键不变量清单

读代码时把这张表放在手边。任何一处实现让你觉得"它可能不成立"，就停下来验证。

| 不变量 | 在哪里维持 | 破坏了会怎样 |
| --- | --- | --- |
| `computed <= num_tokens` | `engine.c` 内层循环 | 会去读还没生成的 token |
| `num_blocks == ceil(computed / block_size)` | `engine.c` 分配块、`cache.c` 释放 | K/V 写到错误的物理位置 |
| `blocks[b].refs == 所有 block_table 里 b 出现的次数` | `release`/`reuse`/`allocate` | 块被提前复用（数据被踩）或永久泄漏 |
| `prefix_len > 0` 时是 `block_size` 的整数倍，且内容等于该位置前的完整前缀 | `publish()` | 前缀缓存假命中 → 输出错乱 |
| 采样只在 `computed == num_tokens` 时发生 | `engine.c` 的 `continue`/`break` | 重放时重复采样，输出漂移 |
| 每个 tick `used <= token_budget` | `engine.c` 外层与内层条件 | 预算失效，tick 变长 |
| 最老的未完成请求总能前进 | `add()` 的容量校验 + 只抢年轻者 | 死锁 / 活锁（`step` 返回 0 却有活请求，触发断言） |

---

## 6. 调试手册：怎么定位"算错了"

**第一原则：先建立一条黄金路径。** 用 `-t 0 --ids --ignore-eos` 让输出变成纯数字、完全确定。之后任何改动都拿它对比：

```bash
./build/nanovllm models/stories260K.bin -z models/tok512.bin \
  -i "Once upon a time" -n 16 -t 0 --ids --ignore-eos > gold.txt
# 改完代码后
./build/nanovllm ... 同样参数 > new.txt && diff gold.txt new.txt
```

**第二原则：按"层"隔离。** 从下往上二分，每一步只引入一层变化：

1. **数学层**：`tests/fixtures.py` 的 oracle 测的是"单个位置的 logits"。如果 oracle 过、端到端不过 → 问题在缓存/调度，不在数学。
2. **缓存层**：`--block-size 1` 是最"笨"的分页（每块一个 token），如果它过了而 `--block-size 4` 挂了 → 问题在块内偏移计算。
3. **调度层**：`--batch 1 --budget 8`（串行、无抢占）是基准。如果串行过、并行挂 → 问题在状态复用或引用计数。
4. **缓存命中层**：`--no-cache` 关掉前缀复用。如果关掉就正常 → 问题在 `publish`/`reuse`。

**第三原则：把隐藏状态打印出来。** 最有效的临时调试就是给 `nv_engine_step()` 加一行：

```c
fprintf(stderr, "tick used=%d active=%d seq%d computed=%d/%d blocks=%d\n",
        used, active, s->id, s->computed, s->num_tokens, s->num_blocks);
```

`--trace` 已有的统计是聚合值；这种逐请求打印才能看出"某条序列的 computed 卡住了"。

**第四原则：让编译器帮你。** 用带断言的调试构建：

```bash
make BUILD=build-debug CFLAGS='-O0 -g -std=c11 -Wall -Wextra -Wpedantic'
gdb --args ./build-debug/nanovllm models/stories260K.bin -z models/tok512.bin \
  -i "Once upon a time" -n 4 -t 0
# 在 gdb 里：
#   break nv_forward if pos == 0
#   break nv_engine_step
#   p *s          （看整条请求的状态）
#   p *cache      （看块池）
```

`nv_engine_step()` 末尾的 `assert` 和 `cache.c`/`engine.c` 里的 `assert` 都是"不变量哨兵"——让它们炸，比让结果悄悄错好查一百倍。

**常见现象 → 可能原因**

| 现象 | 先查这里 |
| --- | --- |
| 输出乱码/重复，但 greedy 每次都一样 | 前缀缓存假命中（`prefix_len`/`memcmp` 逻辑） |
| 两条请求结果互相污染 | `refs` 记账错，块被提前回收 |
| `step` 返回 0 但断言失败 | 准入校验或抢占条件被改坏 |
| 数值答案接近但超容差 | `float` 累加顺序变化 / 忘了 `-O3` 的浮点重组 |
| `forwards` 比预期多很多 | 抢占过频（池子太小）或缓存没生效 |
| 中文输出奇怪符号 | 那是字节回退 token，属正常；模型本来没学过中文 |

---

## 7. 高频困惑 FAQ

**Q1：为什么一个 tick 里每条请求最多只能生成一个新 token？**
采样出的 token 必须先写进 KV 才能算下一个位置，而它此刻还没被 forward。所以生成天然串行：`forward(pos) → logits → 采样 → tokens[n] → 下一 tick forward(n)`。

**Q2：为什么开关前缀缓存，输出必须完全一致？**
缓存只影响"哪些位置需要重新算"，不影响"每个位置算出来的 K/V 和 logits"。同样的权重、同样的输入，位置 `t` 的 logits 只由 `tokens[0..t]` 决定，与它在哪个物理块里无关。测试里的 `batching`/`pressure` 就是在验证这一点。

**Q3：为什么命中了前缀还要重算至少一个 token？**
因为缓存里只有 K/V，**没有 logits**。要采样下一个 token 就必须再做一次 forward 拿到最后一个位置的 logits。所以 `limit` 用 `(num_tokens - 1) / block_size`，故意少复用一块。

**Q4：为什么只能抢占"更年轻"的请求？**
如果互相都能抢，在内存极紧时会无限互抢重算，没有请求能完成（活锁）。限定"只能抢后来者"等价于先来先服务：最老的请求一定能把别人挤掉、跑完、释放块，然后队列继续前进。再配合 `add()` 里"单请求必须能独立装进池子"的校验，就严格保证了只要还有活请求，`step()` 就一定能推进。

**Q5：为什么 `--ids` 输出是 `[0] 1 2 3 ...` 共 9 段？**
`[0]` 是请求标签（多请求时用来区分），后面才是 `-n` 指定数量的 token ID。

**Q6：为什么多请求在 CPU 上不会提速？**
"批处理"在这个项目里是**调度层**的批次：同一个 tick 内多条请求轮流做 forward，共享的是"已加载的权重"，并没有把多个请求的矩阵乘合成一次大矩阵乘。真正的 GPU 批处理加速来自后者。所以 `output_tok/s` 不该拿去和上游比。

**Q7：为什么中文能分词，但模型答不了中文？**
词表里没有中文字符，编码会走 `<0xXX>` 字节回退（每个字节一个 token），信息是无损的；但这个模型只在 TinyStories 英文上训练过，权重里没有中文知识。**能编码 ≠ 见过**。

**Q8：为什么 `nv_engine_add()` 要拒绝"一条请求都装不下"的池子？**
否则抢占永远腾不出足够空间：它只能回收别人的块，而自己需要的新块数超过池子总量，于是永远重算。宁可提前返回 -1，也不要运行时死循环。

**Q9：`block_table` 里的块号为什么可以是乱的？**
因为所有 K/V 访问都走 `nv_cache_key/value` 的地址公式，物理位置和逻辑位置彻底解耦。`oracle()` 测试故意用 `{7,2,19,4,11}` 来证明这一点：如果哪里还残留"用 pos 直接当连续地址"的写法，结果立刻不同。

**Q10：为什么 `att` 缓冲区长度是 `n_heads * seq_len` 而不是 `n_heads * block_size`？**
attention 的分数个数取决于**上下文长度**（`0..pos`，最多 `seq_len` 个），不是块大小。

---

## 8. 动手改造清单（按难度排序，每个都有验收标准）

改代码是检验理解的唯一方式。每个练习都建议：**先写黄金输出，再改，再 diff**。

| # | 练习 | 怎么做 | 验收标准 |
| --- | --- | --- | --- |
| 1 | 打印块表 | 在 `nv_engine_step()` 里加一行打印每条请求的 `block_table` | 能亲眼看到两个请求共享同一个物理块号 |
| 2 | 分离 prefill/decode 计时 | 在 `NvEngine` 加计数，按 `s->computed < s->num_tokens` 区分 | 统计能自证：prefill 次数 + decode 次数 = `forwards` |
| 3 | 把 LRU 换成 FIFO | 改 `nv_cache_allocate()` 的 age 比较 | 换策略后 `make test` 仍全过（正确性与策略无关） |
| 4 | 加"缓存命中率"统计 | `hits / (hits + forwards)` | 与 `--no-cache` 对比看出差距 |
| 5 | 前缀缓存改哈希表 | `publish()` 存 hash+链，`reuse()` 查表 | 与线性扫描版输出逐字节相同，但 `--blocks 512` 时明显更快 |
| 6 | 支持共享"不完整的最后一块" | 引入 copy-on-write：写第一格时才复制 | `forwards` 进一步下降，输出仍与基线一致 |
| 7 | 批量矩阵乘 | 把同一 tick 多条请求的 `matmul` 合成一次 `X[rows][batch]` | `output_tok/s` 上升，`--ids` 输出不变 |
| 8 | OpenMP/SIMD 加速 | `make BUILD=build-omp OPENMP=1`，再试 `-march=native` | 输出不变，吞吐上升 |
| 9 | KV 量化到 int8 | 存 K/V 时量化，读时反量化 | oracle 误差变大但在新阈值内；端到端仍一致 |
| 10 | 加最小 HTTP 服务 | 包一层 socket，`/generate` 调引擎 | 不改变引擎内部语义，可并发（注意引擎本身单线程） |

**注意**：练习 5 之后要同步更新 `docs/learning.md` 里"为什么用线性扫描"那段，否则文档会说谎。

---

## 9. 自测题（答案要点见下）

1. 一次完整的 decode 步骤，`nv_forward()` 被调用几次？prefill 阶段呢？
2. 15M 模型的 `kv_dim` 是多少？它和 `dim` 相等说明了什么？
3. `NvCache.hits` 的单位是什么？为什么它能直接等于"省下的 forward 次数"？
4. 把 `--block-size` 设为 1 会带来什么代价？
5. `--batch 4 --budget 8` 时，每条请求每 tick 最多算几个 token？`--budget 1` 时呢？
6. 请求完成时 `nv_cache_release()` 被调用，为什么之后它的前缀还能被命中？
7. 如果 `nv_cache_publish()` 里去掉 `s->computed % c->block_size` 判断，会发生什么？
8. 为什么 `nv_sample()` 必须先减最大值再除以温度，而不是反过来？
9. `ignore_eos` 这个参数改变了引擎的哪一行判断？
10. 为什么测试里要用 `--no-cache --budget 1 --block-size 3` 这种极端参数？

<details>
<summary>答案要点（先自己想，再展开）</summary>

1. decode 每次 1 次 forward（1 个 token 1 个位置）；prefill 是 `prompt_len` 次（每个位置一次）。所以总量 = `prompt_len + max_new - 1`。
2. `kv_dim = dim / n_heads * n_kv_heads`，15M 是 `288/6*6 = 288`，等于 `dim` ⇒ 这是 MHA（`n_kv_heads == n_heads`），没有 GQA 的压缩。
3. 单位是 token 数；命中 `block_size` 个 token 就跳过 `block_size` 次 forward，所以数值上等于省下的 forward 次数。
4. 每 token 一个块：元数据开销大（块数暴涨）、`block_table` 变长、`publish/reuse` 的线性扫描变慢。但它是最容易对照调试的配置。
5. `quota = budget / max_sequences`：8/4 = 2；`budget 1` 时 1/4 = 0，被 `if (quota < 1) quota = 1` 兜到 1。
6. `release()` 只把 `refs--` 并把块标记为"空闲但仍有 `prefix_len`"；`nv_cache_allocate()` 优先用干净块，只有没干净块时才按 LRU 淘汰这些缓存块。
7. 半满块也会被发布；之后 `reuse()` 用 `prefix_len == end` 匹配时，`end` 是 `block_size` 的倍数，永远匹配不上——白占内存，并且 `assert(b->prefix_len == 0)` 在"同一块被重复发布"时可能触发。
8. 先除温度可能把 logits 放大到 `1e38` 量级溢出成 `inf`；先平移（减最大值）保证所有值 ≤ 0，除以小温度只会变成很大的负数，`exp` 后是 0，安全。
9. 结束条件那行：`!s->sampling.ignore_eos && (token == 1 || token == 2)`。
10. 它同时破坏了三个"可能让 bug 藏起来"的便利条件：缓存不能掩盖分页错误、预算迫使 prefill 分块跨 tick、块大小不整除迫使最后一块不满。如果在这种最坏调度下输出仍然一致，说明分页和状态管理真的没写错。

</details>

---

## 10. 进一步阅读

**先把这个项目读到能改，比看十个仓库更有用。** 之后按兴趣深入：

| 方向 | 材料 |
| --- | --- |
| 调度与内存管理的原始思想 | vLLM 的 PagedAttention 论文（SOSP'23）；[nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm)（本项目的调度结构来源） |
| Transformer 前向的最小实现 | [llama2.c](https://github.com/karpathy/llama2.c)（本项目的数学与权重格式来源） |
| Attention 内核优化 | FlashAttention；然后回头看本项目 `nv_forward()` 里 attention 的三重循环，思考"哪些访存可以合并" |
| 量化 | int8 KV cache、weight-only 量化；对照练习 9 |
| 生产级系统 | vLLM、SGLang、TensorRT-LLM 的调度器与 KV 管理文档 |

**建议的下一步**：把本项目的一个真实瓶颈（CPU 上单请求 decode 只有几十 tok/s）量化拆解——`matmul` 占多少、attention 占多少、内存带宽占多少。你会立刻明白为什么 GPU 上是"内存带宽受限"而不是"算力受限"。

---

## 附：一页速查

```text
text ──nv_encode──► tokens[] ──nv_engine_add──► NvSequence(WAITING)
                                                    │
                          nv_engine_step()  ┌───────┴────────┐
                                            ▼                │
                                    WAITING→ACTIVE           │ 抢不到块
                                    nv_cache_reuse()         │ （更年轻才可抢）
                                            │                │
                                    取新块 nv_cache_allocate ◄┘
                                            │
                                    nv_forward(token,pos) ──► 写 K/V
                                            │
                                    nv_cache_publish()      （块满就发布）
                                            │
                              computed < num_tokens ? ──是──► 继续（重放）
                                            │否
                                      nv_sample() ──► tokens[] ──► DONE
```

文件职责：`nv.h` 定义状态 → `main.c` 驱动 → `engine.c` 调度 → `cache.c` 管内存 → `model.c` 算数学 → `tokenizer.c`/`sampler.c` 管边界。
