#ifndef NV_H
#define NV_H

/* nano-vLLM-C 的公共数据结构与接口。
 * 调用流程：加载模型和分词器 → 初始化引擎 → 编码并添加请求 → 反复 step。
 * 引擎管理请求和分页 KV，模型持有权重及 forward 临时缓冲区。
 * 同一模型的临时缓冲区会被每次 forward 覆盖，调用方应串行使用这些接口。
 */
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

/* 对应 llama2.c legacy FP32 文件头中的 7 个 int32，字段顺序必须保持一致。
 * head_size = dim / n_heads；kv_dim = head_size * n_kv_heads。
 */
typedef struct {
    int dim,        /* token 的隐藏向量宽度，也是残差流的维度。 */
    hidden_dim,     /* SwiGLU 前馈网络的中间维度。 */
    n_layers,       /* Transformer 层数。 */
    n_heads,        /* Query 头数；dim 必须能被它整除，head_size 必须为偶数。 */
    n_kv_heads,     /* Key/Value 头数；等于 n_heads 时为 MHA，更少时为 GQA。 */
    vocab_size,     /* 加载后为正的词表大小；文件头中正/负表示共享/独立输出权重。 */
    seq_len;       /* 最大 forward 上下文长度，有效位置为 [0, seq_len)。 */
} NvConfig;

/* 权重指针均借用模型的只读 mmap 区域，不应写入或逐个 free。
 * 以下形状按 C 的行优先顺序排列，L=n_layers，D=dim，H=hidden_dim，
 * K=kv_dim，V=vocab_size；矩阵乘法使用 [输出维度, 输入维度]。
 */
typedef struct {
    float *embedding,    /* [V, D]：token ID → 隐藏向量。 */
          *att_norm,     /* [L, D]：attention 之前的 RMSNorm 缩放权重。 */
          *wq,           /* [L, D, D]：Query 投影。 */
          *wk,           /* [L, K, D]：Key 投影。 */
          *wv,           /* [L, K, D]：Value 投影。 */
          *wo;           /* [L, D, D]：拼接各 attention 头后的输出投影。 */
    float *ffn_norm,     /* [L, D]：前馈网络之前的 RMSNorm 缩放权重。 */
          *w1,           /* [L, H, D]：SwiGLU 的门控分支，后接 SiLU。 */
          *w2,           /* [L, D, H]：前馈网络的降维投影。 */
          *w3,           /* [L, H, D]：与 SiLU(w1*x) 逐元素相乘的分支。 */
          *final_norm,   /* [D]：最后一层之后的 RMSNorm 缩放权重。 */
          *classifier;   /* [V, D]：输出 logits；共享权重时指向 embedding。 */
} NvWeights;

/* 一个模型实例：权重由 mmap 持有，计算缓冲区由堆分配，均由 model_free 释放。
 * 不在此处保存请求的 KV；KV 属于引擎中的 NvCache。
 */
typedef struct {
    NvConfig config;
    NvWeights weights;
    void *mapping;       /* 模型文件只读映射的起始地址。 */
    size_t mapping_size; /* 映射的字节数，用于检查 tensor 边界和解除映射。 */
    float *x,            /* [D]：当前 token 的残差流，逐层更新。 */
          *xb,           /* [D]：归一化、attention 汇总、前馈输出共用的临时空间。 */
          *xb2,          /* [D]：attention 输出投影，随后加回残差流。 */
          *hb,           /* [H]：前馈门控分支及门控乘积。 */
          *hb2,          /* [H]：前馈网络的 w3 分支。 */
          *q,            /* [D]：当前层全部 Query 头，原地应用 RoPE。 */
          *att,          /* [n_heads, seq_len]：attention 分数，softmax 后成为概率。 */
          *logits;       /* [V]：下一个 token 的未归一化分数。 */
} NvModel;

/* 一个物理 KV 块的元数据；K/V 数据本身存放在 NvCache 的连续数组中。
 * refs==0 表示可分配，但已发布的前缀仍可保留到该块被重新分配。
 */
typedef struct {
    int refs;        /* 引用此块的序列数；完整前缀块可以被多个序列共享。 */
    int prefix_len;  /* 0 表示未发布；否则为截至本块末尾的前缀 token 总数。 */
    uint64_t hash;   /* 已发布前缀的 token 哈希链，包含前驱前缀而非仅本块内容。 */
    uint64_t age;    /* 最后一次 refs 降为 0 时的逻辑时间，用于空闲缓存块的 LRU。 */
} NvBlock;

/* 全部请求共享的 CPU 分页 KV 池。
 * keys/values 均为 [num_blocks, layers, block_size, kv_dim] 的 FP32 数组。
 * 序列的逻辑块通过 block_table 映射到物理块，物理块不要求连续。
 */
typedef struct {
    int num_blocks,  /* 物理块总数。 */
        block_size,  /* 每块容纳的 token 位置数，每个位置保存所有层的 K/V。 */
        layers,      /* 模型层数。 */
        kv_dim;      /* 每层一个位置的 K 或 V 宽度：dim / n_heads * n_kv_heads。 */
    NvBlock *blocks; /* [num_blocks]：引用计数和前缀缓存元数据。 */
    float *keys, *values;
    uint64_t clock,  /* 单调递增的释放时钟，不是实际时间。 */
             hits;  /* 累计复用的 token 数，单位不是块数或请求数。 */
    int enabled;    /* 是否启用前缀复用；关闭后仍然使用分页 KV 存储。 */
} NvCache;

/* 词表的查找索引：按 text 的字典序排序，保留原始 token ID。 */
typedef struct {
    const char *text; /* 借用 vocab[id]，不单独释放。 */
    int id;
} NvTokenIndex;

/* llama2.c 风格 BPE 分词器；约定 BOS=1、EOS=2、原始字节 b 的 ID=b+3。 */
typedef struct {
    char **vocab;         /* [vocab_size]：按 ID 保存的、以 NUL 结尾的词片字符串。 */
    float *scores;       /* [vocab_size]：BPE 合并分数，优先合并分数最高的相邻词片。 */
    NvTokenIndex *sorted;/* [vocab_size]：供 bsearch 按文本查找 ID 的排序索引。 */
    int vocab_size,      /* 必须与模型词表大小一致。 */
        max_length;      /* 文件声明的最长词片字节数，不包含结尾 NUL。 */
    char bytes[256][2];  /* 单字节加 NUL 的解码缓冲，供 <0xXX> byte fallback 使用。 */
} NvTokenizer;

/* 每条请求独立的采样配置；随机数的运行时状态保存在 NvSequence.rng。 */
typedef struct {
    float temperature, /* 0 为 greedy；正值先缩放 logits 再采样。 */
          top_p;       /* (0, 1]：保留累计概率达到此阈值的候选，1 表示整个词表。 */
    uint64_t seed;     /* 初始随机种子；引擎将 0 替换为 1，避免全零随机数状态。 */
    int ignore_eos;    /* 非零时忽略采样到的 BOS/EOS，直到达到新增 token 上限。 */
} NvSampling;

/* 常规迁移为 WAITING → ACTIVE → DONE；抢占使 ACTIVE 回到 WAITING，
 * 取消可将 WAITING 或 ACTIVE 置为 CANCELLED。
 */
typedef enum {
    NV_WAITING,   /* 尚未调度，或已释放 KV、等待重算。 */
    NV_ACTIVE,    /* 已接纳执行，可继续 prefill、重算或 decode。 */
    NV_DONE,      /* 达到生成上限或遇到结束标记，KV 引用已释放。 */
    NV_CANCELLED /* 被调用方取消，KV 引用已释放。 */
} NvStatus;

/* 引擎持有的单条请求。tokens 总是保存 prompt 和已采样结果，即使 KV 被抢占。
 * computed 只计当前仍有效的 KV 前缀，包含复用位置，不等于实际 forward 次数。
 */
typedef struct {
    int id,              /* 加入引擎时的下标，等于 requests[id] 对应的请求 ID。 */
        *tokens,         /* 容量为 prompt_len+max_new_tokens；由引擎复制并持有。 */
        prompt_len,      /* 原始 prompt 的 token 数，生成过程中不变。 */
        num_tokens,      /* 已知 token 总数；生成数为 num_tokens-prompt_len。 */
        max_new_tokens;  /* 此请求最多新增的 token 数。 */
    int computed,        /* [0, computed) 的位置已有 KV；采样出的最新 token 尚未 forward。 */
        *block_table,    /* [逻辑块号] → 物理块号；位置 p 使用 table[p/block_size]。 */
        num_blocks;      /* 当前持有的块表条目数；释放 KV 时与 computed 一起清零。 */
    NvStatus status;
    NvSampling sampling;
    uint64_t rng;        /* 每请求独立的 xorshift64* 状态，避免批次交错改变随机数流。 */
} NvSequence;

/* top-p 排序所用的候选项；概率降序，相同概率按 ID 升序。 */
typedef struct {
    float prob;
    int id;
} NvCandidate;

/* 引擎初始化选项；KV 容量由 num_blocks*block_size 决定。 */
typedef struct {
    int num_blocks,      /* 共享 KV 池的物理块总数，必须为正。 */
        block_size,      /* 每块 token 数，范围为 [1, model.config.seq_len]。 */
        max_sequences,   /* 同时活跃的请求上限；一次 step 最多调度这么多条请求。 */
        token_budget,    /* 一次 step 的实际 forward token 总预算，必须为正。 */
        prefix_cache;    /* 非零时发布并复用完整前缀块。 */
} NvOptions;

/* 调度器与运行统计；模型由调用方持有，其他动态成员由 engine_free 释放。
 * 一个 step 是调度 tick，同批请求依次调用 CPU forward。
 */
typedef struct {
    NvModel *model;           /* 借用模型；其生命周期必须覆盖引擎使用期间。 */
    NvCache cache;
    NvOptions options;
    NvSequence **requests;    /* 按加入顺序保存，已完成/取消的请求也保留到引擎销毁。 */
    int num_requests;         /* 累计接纳请求数，不是当前活跃请求数。 */
    NvCandidate *candidates;  /* [vocab_size]：各次采样复用的排序缓冲区。 */
    uint64_t steps,           /* engine_step 调用次数，包括最后返回 0 的调用。 */
             forward_tokens, /* 实际 forward 次数，包含 prefill 和抢占重算。 */
             generated_tokens, /* 累计采样出的 token 数，包括作为结束标记的 token。 */
             preemptions;    /* 为释放 KV 而将活跃请求退回等待状态的次数。 */
} NvEngine;

/* 分配并清零 count*size 字节；检查乘法溢出，失败时终止进程。
 * 返回值由调用方 free；count 为 0 时按 1 个元素分配。
 */
void *nv_alloc(size_t count, size_t size);

/* 向 stderr 输出带 nanovllm 前缀的错误并以 EXIT_FAILURE 退出，不返回。 */
void nv_fatal(const char *message);

/* 对长度 n>0 的数组原地做稳定 softmax：先减最大值再取 exp，最后归一化。 */
void nv_softmax(float *x, int n);

/* 加载 legacy FP32 权重并分配 scratch；要求小端、32 位 int/float 主机。
 * model 可以尚未初始化，但不能已持有未释放资源；文件或配置无效时终止进程。
 */
void nv_model_load(NvModel *model, const char *path);

/* 解除权重映射、释放计算缓冲区并清零结构体；调用前应先销毁使用它的引擎。 */
void nv_model_free(NvModel *model);

/* 为位置 pos 的一个 token 执行全部层，写入该位置的 K/V，返回下一 token 的 logits。
 * 调用前需分配 table[pos/block_size]，且 [0, pos) 的 KV 已有效；
 * token 必须在词表内，pos 必须在上下文内，cache 的形状应与模型一致。
 * 返回值借用 model->logits，下次 forward 会覆盖；函数不推进序列的 computed。
 */
float *nv_forward(NvModel *model, NvCache *cache, const int *table, int token, int pos);

/* 加载与模型配套的分词器并建立查找索引；vocab_size 由模型提供。
 * tokenizer 不应已有未释放资源；文件无效或词条数量不匹配时终止进程。
 */
void nv_tokenizer_load(NvTokenizer *tokenizer, const char *path, int vocab_size);

/* 释放词片、分数和索引并清零；之前由 decode 返回的指针不再可用。 */
void nv_tokenizer_free(NvTokenizer *tokenizer);

/* 将 NUL 结尾的 UTF-8 文本编码为 token 数组，并通过 count 返回实际长度。
 * 先加入 BOS，非空文本再加入 dummy space，然后按最高合并分数反复执行 BPE；
 * 未命中的字符回退为字节 ID。返回数组由调用方 free，不自动追加 EOS。
 */
int *nv_encode(const NvTokenizer *tokenizer, const char *text, int *count);

/* 解码一个有效 token；previous 是前一个 token，用于去掉 BOS 后的开头空格。
 * <0xXX> 词片还原为单字节，单次结果未必是完整 UTF-8 字符。
 * 返回值借用 tokenizer 内存，不应修改或 free。
 */
const char *nv_decode(const NvTokenizer *tokenizer, int previous, int token);

/* 输出 NUL 结尾的词片：保留普通/高位字节和换行、制表符，过滤其余控制字节。 */
void nv_print_piece(FILE *file, const char *piece);

/* 按模型维度和引擎选项分配并清零 KV 池；options 应已通过引擎的合法性校验。
 * cache 不应已有未释放资源；分配失败或容量乘法溢出时终止进程。
 */
void nv_cache_init(NvCache *cache, const NvConfig *config, NvOptions options);

/* 释放整个 KV 池及元数据并清零；使用这些页面的序列不可再继续 forward。 */
void nv_cache_free(NvCache *cache);

/* 返回 refs=1、prefix_len=0 的物理块号；无空闲块时返回 -1。
 * 先找未发布的空闲块，再按 LRU 淘汰已发布的空闲缓存块；不会清零 K/V 内容。
 */
int nv_cache_allocate(NvCache *cache);

/* 减少序列所有块的引用，清零 computed 和 num_blocks；保留 tokens 和已发布前缀。
 * 此函数不改变 status，完成、取消和抢占时由引擎决定后续状态。
 */
void nv_cache_release(NvCache *cache, NvSequence *seq);

/* 为尚未持有 KV 的序列从头复用连续的完整前缀块，更新块表、引用计数和 computed。
 * 至少留下最后一个已知 token 重新 forward，以获得未缓存的 logits。
 * 仅比较前缀长度和哈希，接受哈希碰撞可能性；未命中即停止，禁用缓存时不操作。
 */
void nv_cache_reuse(NvCache *cache, NvSequence *seq);

/* 每次 forward 后先增加 computed，再调用此函数；仅在刚写满一个新块时发布。
 * 调用时 computed>0，新块须独占且尚未发布；不足整块或禁用缓存时不操作。
 */
void nv_cache_publish(NvCache *cache, const NvSequence *seq);

/* 定位某物理块、某层、块内位置的 K 或 V 向量，返回长度为 kv_dim 的借用指针。
 * block、layer、offset 分别须在 num_blocks、layers、block_size 范围内；不做边界检查。
 */
float *nv_cache_key(NvCache *cache, int block, int layer, int offset);
float *nv_cache_value(NvCache *cache, int block, int layer, int offset);

/* 验证选项并初始化引擎、KV 池和采样缓冲；借用已加载的 model。
 * engine 不应已有未释放资源；无效选项或分配失败时终止进程。
 */
void nv_engine_init(NvEngine *engine, NvModel *model, NvOptions options);

/* 释放所有请求、token 副本、块表、KV 池和采样缓冲并清零；模型仍由调用方释放。 */
void nv_engine_free(NvEngine *engine);

/* 复制 count 个输入 token，添加 WAITING 请求；成功返回 ID，非法输入返回 -1。
 * count>=1，max_new>=1，且 count+max_new-1 不超过模型上下文及单请求 KV 容量。
 * 最后一个输出无需再 forward，因此已知 token 总数可比上下文长度多 1。
 * 可在 step 调用之间加入请求；返回后调用方即可释放原始 tokens。
 */
int nv_engine_add(NvEngine *engine, const int *tokens, int count, int max_new, NvSampling sampling);

/* 取消尚未结束的请求并释放 KV 引用；成功返回 1，ID 无效或已结束返回 0。
 * 请求结构和已有 token 仍保留在 engine->requests[id] 中供调用方读取。
 */
int nv_engine_cancel(NvEngine *engine, int id);

/* 按加入顺序执行一个调度 tick，受 max_sequences 和 token_budget 限制。
 * 长 prompt 可跨 tick prefill；单请求在一次 tick 中最多采样一个新 token。
 * KV 不足时可抢占较晚加入的活跃请求；之后利用保留的 token 重算其 KV。
 * 返回本次实际 forward 的 token 数；返回 0 表示所有请求均已完成或取消。
 */
int nv_engine_step(NvEngine *engine);

/* 从 count>0 个 logits 中选出 token ID；temperature=0 时直接返回最大值位置。
 * 随机采样会原地将 logits 转为概率，使用 count 项 scratch 排序，并更新非零 *rng。
 * temperature/top_p 须有效；seed 由引擎用于初始化 rng，ignore_eos 由引擎判断结束。
 */
int nv_sample(float *logits, int count, NvSampling options, uint64_t *rng, NvCandidate *scratch);

#endif
