# 编译器、编译选项、头文件目录、链接库。
CC = cc
CFLAGS = -O3 -std=c11 -Wall -Wextra -Wpedantic
CPPFLAGS = -Iinclude
LDLIBS = -lm
BUILD = build

# 主程序和测试共用的实现文件。
CORE = src/model.c src/tokenizer.c src/cache.c src/sampler.c src/engine.c

# make OPENMP=1 时启用 CPU 多线程。
ifeq ($(OPENMP),1)
CFLAGS += -fopenmp
LDLIBS += -fopenmp
endif

# 默认执行 make，生成主程序。
all: $(BUILD)/nanovllm

# 每个程序直接从 C 文件编译。修改源码或头文件后会重新编译。
$(BUILD)/nanovllm: src/main.c $(CORE) include/nv.h
	mkdir -p $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) src/main.c $(CORE) $(LDLIBS) -o $(BUILD)/nanovllm

$(BUILD)/test_tokenizer: tests/test_tokenizer.c $(CORE) include/nv.h
	mkdir -p $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_tokenizer.c $(CORE) $(LDLIBS) -o $(BUILD)/test_tokenizer

$(BUILD)/test_engine: tests/test_engine.c $(CORE) include/nv.h
	mkdir -p $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_engine.c $(CORE) $(LDLIBS) -o $(BUILD)/test_engine

$(BUILD)/api: examples/api.c $(CORE) include/nv.h
	mkdir -p $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/api.c $(CORE) $(LDLIBS) -o $(BUILD)/api

# 单独测试 tokenizer。先生成小词表，再运行测试程序。
test-tokenizer: $(BUILD)/test_tokenizer
	cd tests && python3 fixtures.py $(abspath $(BUILD))/fixtures
	cd tests && $(abspath $(BUILD))/test_tokenizer $(abspath $(BUILD))/fixtures/tokenizer.bin

# 运行全部测试。
test: $(BUILD)/nanovllm $(BUILD)/test_engine test-tokenizer
	cd tests && $(abspath $(BUILD))/test_engine $(abspath $(BUILD))/fixtures
	cd tests && python3 test_cli.py $(abspath $(BUILD))/nanovllm $(abspath $(BUILD))/fixtures

# 下载模型并运行文本生成。
demo: all
	python3 tools/download.py
	$(BUILD)/nanovllm models/stories15M.bin -i "Once upon a time" -n 64 -t 0

# 运行动态加入请求的 API 示例。
api-demo: $(BUILD)/api
	python3 tools/download.py
	$(BUILD)/api

# 用独立目录构建，检查内存访问和未定义行为。
sanitize:
	$(MAKE) BUILD=build-sanitize CFLAGS='-O1 -g -std=c11 -Wall -Wextra -Wpedantic -fsanitize=address,undefined -fno-omit-frame-pointer -fno-pie' LDLIBS='-lm -fsanitize=address,undefined -no-pie' test

# 删除当前构建目录。
clean:
	rm -rf $(BUILD)

# 这些名字表示命令，不表示同名文件。
.PHONY: all test test-tokenizer demo api-demo sanitize clean
