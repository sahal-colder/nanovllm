#include "nv.h"
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static int compare(const void *a, const void *b) {
    return strcmp(((const NvTokenIndex *)a)->text, ((const NvTokenIndex *)b)->text);
}

static int lookup(const NvTokenizer *t, const char *text) {
    NvTokenIndex key = {.text = text};
    const NvTokenIndex *result = bsearch(&key, t->sorted, (size_t)t->vocab_size,
                                        sizeof(*t->sorted), compare);
    return result ? result->id : -1;
}

void nv_tokenizer_load(NvTokenizer *t, const char *path, int vocab_size) {
    *t = (NvTokenizer){.vocab_size = vocab_size};
    FILE *file = fopen(path, "rb");
    if (!file) nv_fatal("cannot open tokenizer; run python3 tools/download.py");

    if (fread(&t->max_length, 4, 1, file) != 1 || t->max_length < 1 || t->max_length > 1048576)
        nv_fatal("invalid tokenizer header");
    t->vocab = nv_alloc((size_t)vocab_size, sizeof(char *));
    t->scores = nv_alloc((size_t)vocab_size, sizeof(float));
    t->sorted = nv_alloc((size_t)vocab_size, sizeof(NvTokenIndex));
    for (int i = 0; i < vocab_size; i++) {
        int len;
        if (fread(t->scores + i, 4, 1, file) != 1 || !isfinite(t->scores[i]) ||
            fread(&len, 4, 1, file) != 1 || len < 1 || len > t->max_length)
            nv_fatal("invalid tokenizer entry");
        t->vocab[i] = nv_alloc((size_t)len + 1, 1);
        if (fread(t->vocab[i], 1, (size_t)len, file) != (size_t)len ||
            memchr(t->vocab[i], 0, (size_t)len)) nv_fatal("invalid tokenizer text");
        t->sorted[i] = (NvTokenIndex){t->vocab[i], i};
    }

    if (fgetc(file) != EOF) nv_fatal("tokenizer/model vocabulary size mismatch");
    fclose(file);
    qsort(t->sorted, (size_t)vocab_size, sizeof(*t->sorted), compare);
    if (lookup(t, " ") < 0) nv_fatal("tokenizer lacks dummy space token");

    for (int i = 0; i < 256; i++) t->bytes[i][0] = (char)i;
}

void nv_tokenizer_free(NvTokenizer *t) {
    for (int i = 0; i < t->vocab_size; i++) free(t->vocab[i]);
    free(t->vocab); free(t->scores); free(t->sorted);
    *t = (NvTokenizer){0};
}

int *nv_encode(const NvTokenizer *t, const char *text, int *count) {
    size_t bytes = strlen(text);
    if (bytes > INT_MAX - 3) nv_fatal("prompt too long");
    int *tokens = nv_alloc(bytes + 3, sizeof(int));
    char *buffer = nv_alloc((size_t)t->max_length * 2 + 5, 1);
    *count = 0;
    tokens[(*count)++] = 1;
    if (bytes) tokens[(*count)++] = lookup(t, " ");
    int length = 0;
    for (const unsigned char *c = (const unsigned char *)text; *c; c++) {
        if ((*c & 0xc0) != 0x80) length = 0;
        buffer[length++] = (char)*c;
        buffer[length] = 0;

        if ((c[1] & 0xc0) == 0x80 && length < 4) continue;
        int id = lookup(t, buffer);
        if (id >= 0) tokens[(*count)++] = id;

        if (id < 0) {
            for (int i = 0; i < length; i++) tokens[(*count)++] = (unsigned char)buffer[i] + 3;
        }
        length = 0;
    }

    for (;;) {
        float score = -INFINITY;
        int best = -1, position = -1;
        for (int i = 0; i + 1 < *count; i++) {
            snprintf(buffer, (size_t)t->max_length * 2 + 5, "%s%s",
                     t->vocab[tokens[i]], t->vocab[tokens[i + 1]]);
            int id = lookup(t, buffer);

            if (id >= 0 && t->scores[id] > score) {
                score = t->scores[id]; best = id; position = i;
            }
        }
        if (position < 0) break;
        tokens[position] = best;

        memmove(tokens + position + 1, tokens + position + 2,
                (size_t)(*count - position - 2) * sizeof(int));
        (*count)--;
    }
    free(buffer);
    return tokens;
}

const char *nv_decode(const NvTokenizer *t, int previous, int token) {
    const char *piece = t->vocab[token];
    if (previous == 1 && piece[0] == ' ') piece++;
    unsigned int byte;
    if (strlen(piece) == 6 && piece[0] == '<' && piece[1] == '0' && piece[2] == 'x' &&
        piece[5] == '>' && sscanf(piece, "<0x%2x>", &byte) == 1 && byte < 256)
        return t->bytes[byte];
    return piece;
}

void nv_print_piece(FILE *file, const char *piece) {
    for (const unsigned char *c = (const unsigned char *)piece; *c; c++)
        if ((*c >= 32 && *c != 127) || *c == '\n' || *c == '\t') fputc(*c, file);
}
