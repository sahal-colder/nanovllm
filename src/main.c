/* 命令行入口：解析参数 → 加载模型/分词器 → 提交请求 → 循环调度 → 输出统计。
 * 具体的分词、前向计算、KV 管理和采样交给对应模块，这里只负责组织调用。
 * stdout 输出生成结果，stderr 输出模型信息、调度轨迹和性能统计。
 */
/* 启用 POSIX.1-2008 接口，使 <time.h> 声明 clock_gettime 等计时接口。 */
#define _POSIX_C_SOURCE 200809L
#include "nv.h"
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* 帮助和参数错误共用一份说明，由调用方选择输出到 stdout 还是 stderr。 */
static void usage(FILE *file) {
    fprintf(file,
        "Usage: nanovllm MODEL [options]\n"
        "  -z PATH       tokenizer (default models/tokenizer.bin)\n"
        "  -i TEXT       prompt; repeat for continuous batching\n"
        "  -n N          max NEW tokens per request (default 64)\n"
        "  -t F          temperature; 0 = greedy (default 0.8)\n"
        "  -p F          top-p in (0,1] (default 0.9)\n"
        "  -s N          random seed, positive (default 1)\n"
        "  --blocks N    KV pool blocks (default 128)\n"
        "  --block-size N tokens per block (default 16)\n"
        "  --batch N     maximum active requests (default 4)\n"
        "  --budget N    forward tokens per scheduling tick (default 32)\n"
        "  --no-cache    disable prefix reuse\n"
        "  --trace       print scheduling progress on stderr\n"
        "  --ids         print generated token IDs instead of text\n"
        "  --ignore-eos  generate exactly N tokens\n");
}

/* 数量和种子选项只接受 int 范围内的正整数。
 * strtol 的 end 用来拒绝空串或数字后的多余字符，errno 用来识别转换溢出。
 */
static int integer(const char *s) {
    char *end;
    errno = 0;
    long n = strtol(s, &end, 10);
    if (errno || !*s || *end || n < 1 || n > INT_MAX) nv_fatal("expected positive integer option");
    return (int)n;
}

/* 先检查字符串能否完整转换为有限浮点数，再由 main 检查各选项的取值范围。
 * NaN/Inf 即使能够被 strtof 识别，也不能作为采样参数。
 */
static float number(const char *s) {
    char *end;
    errno = 0;
    float n = strtof(s, &end);
    if (errno || !*s || *end || !isfinite(n)) nv_fatal("expected finite numeric option");
    return n;
}

/* 单调时钟不受系统日期校准影响，适合用两次读数之差计算耗时。 */
static double seconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec / 1e9;
}

int main(int argc, char **argv) {
    /* argc 是命令行参数总数，argv 是字符串数组：argv[0] 通常是程序名，argv[1] 是模型路径。
     * 例如运行 nanovllm model.bin -i "hello" 时，argc 为 4，参数依次放在 argv[0..3]。
     * 单独传 --help 时无需加载任何模型文件。
     */
    if (argc == 2 && strcmp(argv[1], "--help") == 0) { usage(stdout); return 0; }

    /* C 的数组下标从 0 开始；少于两个参数意味着连 argv[1] 模型路径都没有。
     * return 1 表示异常退出；正常完成时 main 最后返回 0。
     */
    if (argc < 2) { usage(stderr); return 1; }
    const char *tokenizer_path = "models/tokenizer.bin";

    /* const char ** prompts 表示“指向字符串指针数组的指针”；prompts 不是字符缓冲区，
     * 每个元素只保存一段只读文本的地址。
     * 每个 -i 对应一条独立请求；文本来自 argv 或下方的默认字符串，因此不复制文本内容。
     * 按 argc 分配指针槽位，足够容纳所有 -i，也足够放入未指定 -i 时的默认文本。
     */
    const char **prompts = nv_alloc((size_t)argc, sizeof(char *));
    int count = 0, max_new = 64, trace = 0, ids = 0;

    /* 字段依次为：物理 KV 块数、每块 token 数、活跃请求上限、每步 forward 预算、前缀复用。 */
    NvOptions options = {128, 16, 4, 32, 1};

    /* 字段依次为：温度、top-p、随机种子、是否忽略 BOS/EOS 结束标记。
     * 配置会复制到每条请求；引擎为每条请求独立维护随机数状态。
     */
    NvSampling sampling = {0.8f, 0.9f, 1, 0};
    /* 从 argv[2] 开始，因为 argv[1] 已经固定用作模型路径。
     * 每轮处理 argv[i]；循环末尾的 i++ 再前进到下一个参数。
     */
    for (int i = 2; i < argc; i++) {
        /* flag 只是指向当前参数字符串的指针，不会创建字符串副本。 */
        const char *flag = argv[i];

        /* 无值开关处理后直接 continue，跳过下面“读取选项值”的逻辑，避免误消费下一个参数。
         * continue 不会跳出整个 for；它会进入循环更新步骤，让 i 前进一位。
         */
        if (strcmp(flag, "--no-cache") == 0) { options.prefix_cache = 0; continue; }
        if (strcmp(flag, "--trace") == 0) { trace = 1; continue; }
        if (strcmp(flag, "--ids") == 0) { ids = 1; continue; }
        if (strcmp(flag, "--ignore-eos") == 0) { sampling.ignore_eos = 1; continue; }

        /* 其余选项都需要一个值。argv[++i] 先把 i 加 1，再取该位置的字符串，
         * 所以这里会消费紧跟在选项后面的值；循环末尾的 i++ 随后移到下个选项。
         */
        if (i + 1 >= argc) nv_fatal("option needs a value; see --help");
        const char *value = argv[++i];
        if (strcmp(flag, "-z") == 0) { tokenizer_path = value; continue; }

        /* count++ 是“先使用旧值、再加一”：先写 prompts[count]，再让请求数加一。
         * 因此多次出现 -i 会把多条提示词依次放入 prompts[0]、prompts[1]……
         */
        if (strcmp(flag, "-i") == 0) { prompts[count++] = value; continue; }
        if (strcmp(flag, "-n") == 0) { max_new = integer(value); continue; }
        if (strcmp(flag, "-t") == 0) { sampling.temperature = number(value); continue; }
        if (strcmp(flag, "-p") == 0) { sampling.top_p = number(value); continue; }
        if (strcmp(flag, "-s") == 0) { sampling.seed = (uint64_t)integer(value); continue; }
        if (strcmp(flag, "--blocks") == 0) { options.num_blocks = integer(value); continue; }
        if (strcmp(flag, "--block-size") == 0) { options.block_size = integer(value); continue; }
        if (strcmp(flag, "--batch") == 0) { options.max_sequences = integer(value); continue; }
        if (strcmp(flag, "--budget") == 0) { options.token_budget = integer(value); continue; }
        nv_fatal("unknown option; see --help");
    }

    /* || 表示“或”：任一条件成立就报错。temperature 允许为 0（greedy），
     * 正温度时按 top-p 截断后的概率分布采样；top_p 必须大于 0 且不大于 1。
     */
    if (sampling.temperature < 0 || sampling.top_p <= 0 || sampling.top_p > 1)
        nv_fatal("temperature must be >=0, top-p must be in (0,1]");

    /* !count 是“count 等于 0”的简写。用户没写 -i 时，把默认提示词放到第一个槽位并令 count 为 1。 */
    if (!count) prompts[count++] = "Once upon a time";

    /* 这些结构体在栈上声明。函数参数里的 &model、&tokenizer、&engine 是取地址，
     * 让对应的 load/init 函数把结果写进这些结构体。
     * 先读模型配置，才能校验分词器的词表大小，并按模型配置创建 KV 缓存池。
     */
    NvModel model;
    nv_model_load(&model, argv[1]);
    NvTokenizer tokenizer;
    nv_tokenizer_load(&tokenizer, tokenizer_path, model.config.vocab_size);
    NvEngine engine;
    nv_engine_init(&engine, &model, options);
    /* 所有请求先进入等待队列，真正的 prefill/decode 在后面的 step 循环中执行。
     * &engine 把引擎地址传给函数；tokens 是本次临时编码结果，nv_engine_add 会复制它。
     */
    for (int i = 0; i < count; i++) {
        int n;
        /* n 是编码后的 token 数，包含 BOS；max_new 只限制新增 token，不包含 prompt。
         * &n 把 n 的地址传给 nv_encode，让它把 token 数写回来；tokens 接收返回的 token 数组指针。
         */
        int *tokens = nv_encode(&tokenizer, prompts[i], &n);

        /* 单请求所需 forward 位置数为 n + max_new - 1：最后采样的 token 无需再前向。
         * 提交时检查上下文和单请求 KV 容量；多请求争用 KV 时由调度器处理抢占重算。
         */
        if (nv_engine_add(&engine, tokens, n, max_new, sampling) < 0)
            nv_fatal("request exceeds context or KV pool capacity; reduce -n/prompt or increase --blocks");
        /* 引擎已经复制 token 数组，调用方立即释放编码结果即可。 */
        free(tokens);
    }

    fprintf(stderr, "model: dim=%d layers=%d heads=%d kv_heads=%d vocab=%d context=%d\n",
            model.config.dim, model.config.n_layers, model.config.n_heads,
            model.config.n_kv_heads, model.config.vocab_size, model.config.seq_len);

    /* 每个物理块保存所有层的 K 和 V：块数 × 块长 × 层数 × KV 宽度 × 2 × FP32 字节数。
     * 这里只计算 K/V 数组，不包含块元数据、模型权重或临时计算缓冲区。
     */
    size_t kv_bytes = (size_t)options.num_blocks * (size_t)options.block_size *
                      (size_t)model.config.n_layers * (size_t)engine.cache.kv_dim * 2 * sizeof(float);
    fprintf(stderr, "KV pool: %.2f MiB; requests=%d (CPU serial forwards within scheduling batches)\n",
            (double)kv_bytes / 1048576.0, count);

    /* engine.requests[0] 取出第一个请求指针；-> 用来访问“结构体指针”指向的成员，
     * 等价于 (*engine.requests[0]).prompt_len。printed 是下一枚待打印 token 的下标，
     * 初始设为 prompt_len 以跳过 prompt（以及其中的 BOS）；单请求文本模式边生成边打印，
     * 多请求或 --ids 模式留到全部完成后统一打印。
     */
    int printed = engine.requests[0]->prompt_len;
    if (count == 1 && !ids) { nv_print_piece(stdout, prompts[0]); fflush(stdout); }

    /* 耗时不含模型加载、分词和提交请求，包含 prefill、decode、重算及循环中的输出。 */
    double start = seconds();

    /* 一次 step 是一个调度 tick，受活跃请求数和 forward token 预算限制；同一 tick 内的请求
     * 依次执行 CPU forward，并非一次批量矩阵计算。函数返回本次实际 forward 数；C 把非零当真、
     * 把 0 当假，因此有工作时循环继续，所有请求结束后返回 0 并停止。prefill 本身也会 forward，
     * 即使还没生成新 token，也会返回非零。
     */
    while (nv_engine_step(&engine)) {
        if (trace) fprintf(stderr, "step=%" PRIu64 " forwards=%" PRIu64 " generated=%" PRIu64
                           " cache_hits=%" PRIu64 " preemptions=%" PRIu64 "\n",
                           engine.steps, engine.forward_tokens, engine.generated_tokens,
                           engine.cache.hits, engine.preemptions);

        if (count == 1 && !ids) {
            NvSequence *s = engine.requests[0];
            /* 初始化部分留空，因为 printed 已在循环外设为 prompt_len；每轮处理 printed 下标的 token，
             * 循环更新部分 printed++ 再前进一步。BOS=1、EOS=2 不显示但仍推进下标；decode 需要前一个
             * token，以处理紧跟 BOS 的词片开头空格。
             */
            for (; printed < s->num_tokens; printed++) {
                int token = s->tokens[printed];
                if (token != 1 && token != 2) nv_print_piece(stdout, nv_decode(&tokenizer, s->tokens[printed - 1], token));
            }
            /* 不等行缓冲攒到换行，及时把生成片段显示到终端。 */
            fflush(stdout);
        }
    }
    double elapsed = seconds() - start;
    if (count == 1 && !ids) fputc('\n', stdout);

    /* 多请求按提交顺序分行输出，避免文本交错；--ids 仅列新增 token ID，保留 BOS/EOS。
     * || 表示任一条件成立就进入代码块：多请求等全部完成后统一打印，--ids 模式也走这里。
     * 请求完成时只释放 KV 引用，请求对象和 token 仍保留在引擎中供这里读取。
     */
    if (count > 1 || ids) {
        for (int i = 0; i < count; i++) {
            NvSequence *s = engine.requests[i];
            printf("[%d] ", i);
            if (!ids) nv_print_piece(stdout, prompts[i]);
            for (int j = s->prompt_len; j < s->num_tokens; j++) {
                /* ?: 是条件表达式：第一个生成 ID 前不加空格，之后的 ID 前加空格。 */
                if (ids) { printf("%s%d", j == s->prompt_len ? "" : " ", s->tokens[j]); continue; }
                if (s->tokens[j] != 1 && s->tokens[j] != 2)
                    nv_print_piece(stdout, nv_decode(&tokenizer, s->tokens[j - 1], s->tokens[j]));
            }
            fputc('\n', stdout);
        }
    }

    /* generated 包括采样出的结束标记；forwards 包括 prefill/decode/重算。
     * cache.hits（轨迹中的 cache_hits、这里的 reused）按复用 token 数计，不是命中块数。
     * output_tok/s 使用整个调度循环的耗时，不能视为单独的 decode 吞吐量。
     */
    fprintf(stderr, "generated=%" PRIu64 " forwards=%" PRIu64 " reused=%" PRIu64
            " preemptions=%" PRIu64 " elapsed=%.3fs output_tok/s=%.2f\n",
            engine.generated_tokens, engine.forward_tokens, engine.cache.hits,
            engine.preemptions, elapsed, elapsed > 0 ? (double)engine.generated_tokens / elapsed : 0);

    /* &engine 等是把对象地址交给释放函数。引擎借用模型，所以先销毁引擎，
     * 再释放分词器和模型；prompts 数组只存字符串指针，最后只需释放数组本身。
     */
    nv_engine_free(&engine);
    nv_tokenizer_free(&tokenizer);
    nv_model_free(&model);
    free(prompts);
    return 0;
}
