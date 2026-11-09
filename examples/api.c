#include "nv.h"
#include <stdlib.h>

int main(void) {
    NvModel model;
    nv_model_load(&model, "models/stories15M.bin");
    NvTokenizer tokenizer;
    nv_tokenizer_load(&tokenizer, "models/tokenizer.bin", model.config.vocab_size);

    NvEngine engine;
    nv_engine_init(&engine, &model, (NvOptions){64, 4, 2, 16, 1});

    NvSampling sampling = {0, 1, 123, 0};
    int count;

    int *tokens = nv_encode(&tokenizer, "Once upon a time", &count);
    if (nv_engine_add(&engine, tokens, count, 32, sampling) < 0) nv_fatal("first request rejected");
    free(tokens);

    nv_engine_step(&engine);
    tokens = nv_encode(&tokenizer, "Once upon a time, there was a little girl", &count);
    if (nv_engine_add(&engine, tokens, count, 32, sampling) < 0) nv_fatal("second request rejected");
    free(tokens);

    while (nv_engine_step(&engine)) {}

    for (int i = 0; i < engine.num_requests; i++) {
        NvSequence *s = engine.requests[i];
        printf("[%d] ", s->id);
        for (int j = 1; j < s->num_tokens; j++) {
            if (s->tokens[j] == 1 || s->tokens[j] == 2) continue;
            nv_print_piece(stdout, nv_decode(&tokenizer, s->tokens[j - 1], s->tokens[j]));
        }
        putchar('\n');
    }

    nv_engine_free(&engine);
    nv_tokenizer_free(&tokenizer);
    nv_model_free(&model);
    return 0;
}
