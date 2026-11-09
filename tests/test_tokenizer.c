#include "nv.h"
#include <stdlib.h>
#include <string.h>

/* 使用 tests/fixtures.py 生成的 263 词小词表，便于手工核对 ID。
 * 0..2 是特殊 token，3..258 是字节，259..262 是空格/a/b/ab。
 */
static int check_encoding(const NvTokenizer *tokenizer, const char *text,
                           const int *expected, int expected_count) {
    int count;
    int *tokens = nv_encode(tokenizer, text, &count);
    printf("encode(\"%s\") ->", text);
    for (int i = 0; i < count; i++) printf(" %d", tokens[i]);
    putchar('\n');
    if (count != expected_count) {
        fprintf(stderr, "ERROR encode(\"%s\"): expected %d tokens, got %d\n",
                text, expected_count, count);
        free(tokens);
        return 1;
    }
    for (int i = 0; i < count; i++) {
        if (tokens[i] != expected[i]) {
            fprintf(stderr, "ERROR encode(\"%s\"): token[%d] expected %d, got %d\n",
                    text, i, expected[i], tokens[i]);
            free(tokens);
            return 1;
        }
    }
    free(tokens);
    return 0;
}

/* 按实际生成时的 previous/current 顺序逐个解码，检查完整文本往返。 */
static int check_roundtrip(const NvTokenizer *tokenizer, const char *text) {
    int count;
    int *tokens = nv_encode(tokenizer, text, &count);
    size_t text_length = strlen(text);
    char *decoded = nv_alloc(text_length + 1, 1);
    size_t length = 0;
    for (int i = 1; i < count; i++) {
        const char *piece = nv_decode(tokenizer, tokens[i - 1], tokens[i]);
        size_t piece_length = strlen(piece);
        if (length + piece_length > text_length) {
            fprintf(stderr, "ERROR roundtrip(\"%s\"): decoded text exceeds buffer\n", text);
            free(decoded);
            free(tokens);
            return 1;
        }
        memcpy(decoded + length, piece, piece_length);
        length += piece_length;
    }
    int failed = strcmp(decoded, text) != 0;
    if (failed)
        fprintf(stderr, "ERROR roundtrip: expected \"%s\", got \"%s\"\n", text, decoded);
    free(decoded);
    free(tokens);
    return failed;
}

/* 检查一个 token 是否解码成预期文本。成功返回 0，失败返回 1。 */
static int check_decoding(const NvTokenizer *tokenizer, int previous, int token,
                          const char *expected) {
    const char *actual = nv_decode(tokenizer, previous, token);
    if (strcmp(actual, expected) == 0) return 0;

    fprintf(stderr, "ERROR decode token %d: expected \"%s\", got \"%s\"\n",
            token, expected, actual);
    return 1;
}

/* 字节 token 的 ID = 字节值 + 3，逐个检查全部 256 个字节。 */
static int check_bytes(const NvTokenizer *tokenizer) {
    for (int byte = 0; byte < 256; byte++) {
        const char *actual = nv_decode(tokenizer, 259, byte + 3);
        if ((unsigned char)actual[0] != byte || actual[1] != '\0') {
            fprintf(stderr, "ERROR decode byte %d\n", byte);
            return 1;
        }
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: test_tokenizer PATH_TO_FIXTURE_TOKENIZER\n");
        return EXIT_FAILURE;
    }
    NvTokenizer tokenizer;
    nv_tokenizer_load(&tokenizer, argv[1], 263);

    int errors = 0;
    const int empty[] = {1};
    errors += check_encoding(&tokenizer, "", empty, 1);

    /* BOS + dummy space + a。 */
    const int single[] = {1, 259, 260};
    errors += check_encoding(&tokenizer, "a", single, 3);

    /* 相邻 a/b 按词表分数合并为 ab。 */
    const int merged[] = {1, 259, 262};
    errors += check_encoding(&tokenizer, "ab", merged, 3);
    const int repeated[] = {1, 259, 262, 262};
    errors += check_encoding(&tokenizer, "abab", repeated, 4);

    /* é 的 UTF-8 字节是 C3 A9，byte fallback 的 ID 为字节值 + 3。 */
    const int utf8[] = {1, 259, 198, 172};
    errors += check_encoding(&tokenizer, "é", utf8, 4);

    /* BOS 后去掉自动添加的空格，其他位置保留空格。 */
    errors += check_decoding(&tokenizer, 1, 259, "");
    errors += check_decoding(&tokenizer, 260, 259, " ");
    errors += check_decoding(&tokenizer, 259, 262, "ab");
    errors += check_bytes(&tokenizer);

    /* 编码后再解码，应该还原原文。 */
    errors += check_roundtrip(&tokenizer, "");
    errors += check_roundtrip(&tokenizer, "ab");
    errors += check_roundtrip(&tokenizer, "abab");
    errors += check_roundtrip(&tokenizer, "a b");
    errors += check_roundtrip(&tokenizer, "  ab");
    errors += check_roundtrip(&tokenizer, "é");
    errors += check_roundtrip(&tokenizer, "你好");
    errors += check_roundtrip(&tokenizer, "😀");
    errors += check_roundtrip(&tokenizer, "ab你好😀");
    errors += check_roundtrip(&tokenizer, "a\nb\t");

    nv_tokenizer_free(&tokenizer);
    if (errors) {
        fprintf(stderr, "FAIL tokenizer: %d checks failed\n", errors);
        return EXIT_FAILURE;
    }
    puts("PASS tokenizer: BOS, dummy prefix, BPE, byte decoding, UTF-8 roundtrip");
    return EXIT_SUCCESS;
}
