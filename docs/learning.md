# 从零读懂这个 C 推理引擎

## 1. 先把请求走通

入口是 `src/main.c`。先用 `nv_model_load()` 只读 mmap 权重，再加载分词器，构造 `NvEngine`。`nv_encode()` 将 prompt 编成 token ID；`nv_engine_add()` 复制输入并返回请求 ID。之后反复调用 `nv_engine_step()`，每次执行一个有限 token 预算的调度 tick，直到它返回 0。

```text
文本 → BPE → token ID → WAITING → ACTIVE → DONE
                               ↑   ↓
                             抢占重算
```

随时可调用 `nv_engine_cancel()` 取消尚未完成的请求，释放其 KV 引用，标记为 `NV_CANCELLED`。结果及 token 数组在引擎销毁前保留；销毁引擎后再销毁模型。`NvModel` 只有一组 scratch buffer，整个 API 是单线程使用模型，不能让多个线程同时对同一模型 forward。

`NvSequence` 最关键的几个字段：

| 字段 | 含义 |
| --- | --- |
| `tokens` / `num_tokens` | 目前已知的所有 token，包括 prompt 和已经采样的输出 |
| `prompt_len` | 输入 token 数，始终不变 |
| `computed` | 已经写入 KV 的 token 数 |
| `block_table` | 当前序列的逻辑块 → 物理块 ID |
| `rng` | 当前请求的随机状态 |

例子：prompt 有 5 个 token。prefill 完成后 `computed=5`，引擎根据最后一个输入位置的 logits 采样一个输出，`num_tokens=6`。此时输出还没写进 KV。下一次 decode 才把这个输出送入 forward，再采样下一个输出。

抢占时清空 `computed` 和 `block_table`，但保留 `tokens` 和 `rng`。恢复时先重算已有 token，只有追上 `num_tokens` 才采样新的输出，因此不会把重算过程当成新生成。

## 2. 一次 forward 做什么

`nv_forward()` 的参数是“当前 token、绝对位置 pos、当前序列的块表”。输出是词表中每个候选 token 的 logit，返回缓冲区在下一次 forward 时覆盖。

```text
embedding[token]
  ↓
逐层：RMSNorm → Q/K/V → RoPE → causal attention → Wo → residual
      RMSNorm → W1/W3 → SwiGLU → W2 → residual
  ↓
final RMSNorm → classifier → logits
```

- `matmul(out, x, w, in, rows)` 算的是 `W[rows,in] × x[in]`，不依赖 BLAS。
- RMSNorm 计算 `x / sqrt(mean(x²)+1e-5) * weight`。
- RoPE 对 Q/K 每两个元素旋转，位置使用序列中的绝对 `pos`，不是物理页面的位置。
- GQA 中多组 Q 共享一个 K/V head：`head / (n_heads / n_kv_heads)`。
- attention 只读位置 `0..pos`，保证 causal；softmax 先减去最大值保持数值稳定。
- SwiGLU 是 `silu(W1*x) * (W3*x)`；经过 W2 后加回残差。

prefill 与 decode 用同一个逐 token forward。原版 GPU 引擎可以并行处理 prefill 中多个位置；本项目按顺序计算，方便调试每个位置和每一层。

## 3. 为什么需要分页 KV

没有 KV Cache 时，生成下一个 token 需要重算所有历史 token 的 K/V。有了缓存，历史 K/V 可以直接复用。这里再把历史存储拆成固定大小的物理块，让不同长度的请求按需领取内存。

假设 `block_size=4`，请求的 `block_table=[7,2,11]`：

```text
逻辑位置      0 1 2 3 | 4 5 6 7 | 8 9 10 11
物理块          7    |    2    |     11
块内偏移      0 1 2 3 | 0 1 2 3 | 0 1  2  3
```

位置 t 的访问公式：

```c
int physical = table[t / cache->block_size];
int offset = t % cache->block_size;
float *key = nv_cache_key(cache, physical, layer, offset);
```

本项目布局是 `[physical_block][layer][offset][kv_dimension]`。`nv_forward()` 的 QKV 写入和 attention 读取都走同一套映射，缓存是真正参与计算的，不是只做页面管理的示意模型。

K/V 池大小（不含权重、scratch 和前缀元数据）：

```text
2 × num_blocks × block_size × n_layers × kv_dim × sizeof(float)
kv_dim = dim / n_heads × n_kv_heads
```

每个物理块包括所有层的 K/V。最后一个未写满的页面由单个请求持有；本实现不共享部分块，因此不需要为部分块增加 copy-on-write。

## 4. 前缀复用为什么要匹配整段历史

一个 token 的深层 K/V 受到前面所有 token 的影响。即使两个块内部都是 `a b c d`，如果此前历史不同，这两个块的 K/V 也可能不同。

每个完整块保存截至本块末尾的前缀长度和 64 位 FNV-1a 哈希。首块从固定初值开始，后续块把本块 token 混入前一块哈希，得到整个前缀的哈希。

发布缓存时通过当前序列的块表读取前一块哈希；发布后只保存结果。复用时线性扫描，前缀长度与哈希相同就视为命中。新请求从自己的 token 计算哈希，因此前一物理块是否已被回收不影响已发布后继块的哈希。

该实现按用户选择仅依靠哈希识别前缀。哈希碰撞可能导致错误复用 K/V，不做额外 token 比较。

- 只有完整、已计算的页面可以发布为前缀缓存。
- 发布后的页面不可改写，可以让多个请求增加 `refs` 并共享。
- 释放请求只递减引用；`refs=0` 的缓存仍可命中。
- 分配新页面先用无效空闲块，再淘汰最久未使用的空闲缓存块；不会覆盖 `refs>0` 的页面。
- 命中最多到最后一个输入 token 之前的完整块。即使整个 prompt 都对齐，也重算至少一个输入位置，因为本缓存不保存 logits。

缓存块的识别元数据为固定大小的前缀长度和哈希值。完成请求的 token 数组也保留到 `nv_engine_free()`；这是便于查看结果的教学 API，不适合无限请求的常驻服务。

## 5. 调度与抢占

`nv_engine_step()` 按请求加入顺序遍历：

1. 有空位时将 WAITING 请求转为 ACTIVE，并尝试复用已有完整前缀。
2. 为每条被调度请求分配 prefill token 配额；每 tick 总 forward 数不超过 `token_budget`。
3. 到达新逻辑块时按需申请物理页面。
4. KV 不足时，当前请求可以抢占比自己更晚加入的 ACTIVE 请求，释放它的 KV；后者恢复为 WAITING。
5. forward 追上已知 token 后，采样一个输出；满足长度或 BOS/EOS 条件就完成并释放 KV。

decode 只需要一次 forward，因此同一 tick 内最多为每条请求生成一个新 token。prefill 可以处理多个 token；一个请求剩余的配额不一定转移给下一条，本项目优先保持调度代码清楚。

为什么只抢占较新请求？如果 A 可以抢占 B，B 也能立即抢占 A，极小内存下可能互相重算，始终没有请求完成。按先来先服务的优先级，最早的请求可以持续前进。`nv_engine_add()` 也会确保单条请求能在池中独立完成；否则直接返回 -1。

这是一种教学调度策略：极小预算下较新请求可能等到前面的完成，不提供生产级公平性、延迟保证或多租户策略。

## 6. 两个可以自己动手的实验

**实验 A：验证缓存只改变计算量。** 同一个 prompt 使用 `-t 0 --ids --ignore-eos`，分别开启和关闭前缀缓存。比较 token ID，并观察 `forwards` 与 `reused`。不要用最终自然语言是否“差不多”来判定一致。

**实验 B：验证分页与模型数学无关。** 改变 `--block-size`，用相同 seed、温度和 top-p 运行。结果应相同。再减少 `--blocks` 触发重算：每条请求的随机状态单独保存，重放旧输出不再采样，因此结果仍应一致。

`tests/test_engine.c` 已经覆盖上述场景。`tests/fixtures.py` 则用 Python double 和连续 KV 数组独立计算每个位置的 logits；C 测试故意使用乱序物理页面，对比 MHA 与 GQA 的全部词表输出。

建议先在 `nv_forward()` 的 `pos=0` 和 `layer=0` 处下断点，观察 Q、K、V、attention，再回头读 `nv_cache_allocate()` 与 `nv_engine_step()`。理解一次实际执行后，再考虑 OpenMP、向量化或者 GPU 移植。
