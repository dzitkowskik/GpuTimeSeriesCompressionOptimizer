uname_S := $(shell sh -c 'uname -s 2>/dev/null || echo not')
uname_M := $(shell sh -c 'uname -m 2>/dev/null || echo not')

PROGRAM_NAME := gpuStore

RM := rm -rf
OS := $(shell uname)

SRC_PATH := src
BUILD_PATH := build
BIN_PATH := bin

SRC_CUDA_EXT := cu
SRC_EXT := cpp

COMPILER := nvcc
STANDART := --std=c++11 
NVCC_FLAGS := --cudart static --relocatable-device-code=false --default-stream per-thread

LIBS := -lcudart -lboost_system -lboost_thread -lpthread \
	-lboost_program_options -llog4cplus -lgtest -lbenchmark -lcurand -lcelero

GENCODE_SM30    := -gencode arch=compute_30,code=sm_30
GENCODE_SM35	:= -gencode arch=compute_35,code=sm_35

ifeq ($(OS),Darwin)
	LIB_DIRS := -L"/usr/local/cuda/lib" -L"/usr/local/lib"
	DATE := gdate
	GENCODE_FLAGS   := $(GENCODE_SM30)
else
	LIB_DIRS := -L"/usr/local/cuda/lib64" -L"/usr/local/lib"
	DATE := date
	GENCODE_FLAGS   := $(GENCODE_SM35)
endif

INCLUDES := -I"src"
DEFINES := #-D __GXX_EXPERIMENTAL_CXX0X__ -DBOOST_HAS_INT128=1 -D_GLIBCXX_USE_CLOCK_REALTIME -DHAVE_WTHREAD_SAFETY
WARNINGS_ERRORS := -pedantic -Wall -Wextra -Wno-deprecated -Wno-unused-parameter  -Wno-enum-compare -Weffc++

MODULES := data
MODULE_LIBS := $(MODULES:%=-l%)
MODULE_LIB_DIRS := $(MODULES:%=-L"modules/%/lib")
MODULE_INCLUDES := $(MODULES:%=-I"modules/%/include")

LIBS := $(LIBS) $(MODULE_LIBS)
LIB_DIRS := $(LIB_DIRS) $(MODULE_LIB_DIRS)
INCLUDES := $(INCLUDES) $(MODULE_INCLUDES)

# Macros for timing compilation
TIME_FILE = $(dir $@).$(notdir $@)_time
START_TIME = $(DATE) '+%s' > $(TIME_FILE)
END_TIME = read st < $(TIME_FILE) ; \
	$(RM) $(TIME_FILE) ; \
	st=$$((`$(DATE) '+%s'` - $$st - 86400)) ; \
	echo `$(DATE) -u -d @$$st '+%H:%M:%S'`

# Verbose option, to output compile and link commands
export V := false
export CMD_PREFIX := @
ifeq ($(V),true)
	CMD_PREFIX :=
endif

debug: export CODE_FLAGS := -G -g -O0 --debug --device-debug
debug: export EXCLUDED_FILES := \
	-not -name '*_unittest*' \
	-not -name '*_benchmark*'
debug: export BUILD_PATH := build/debug
debug: export BIN_PATH := bin/debug

release: export CODE_FLAGS := -O3
release: export EXCLUDED_FILES := \
	-not -name '*_unittest*' \
	-not -name '*_benchmark*'
release: export BUILD_PATH := build/release
release: export BIN_PATH := bin/release

test: export CODE_FLAGS := -G -g -O0 --debug --device-debug
test: export EXCLUDED_FILES := \
	-not -iname 'main.cpp' \
	-not -name '*_benchmark*'
test: export BUILD_PATH := build/test
test: export BIN_PATH := bin/test

benchmark: export CODE_FLAGS := -O3
benchmark: export EXCLUDED_FILES := \
	-not -iname 'main.cpp' \
	-not -name '*_unittest*'
benchmark: export BUILD_PATH := build/benchmark
benchmark: export BIN_PATH := bin/benchmark

SRC_FILES := $(shell find $(SRC_PATH)/ -name '*.$(SRC_EXT)' \
	$(EXCLUDED_FILES) -o -name '*.$(SRC_CUDA_EXT)' $(EXCLUDED_FILES) \
	| sort -k 1nr | cut -f2-)

OBJS := $(SRC_FILES:$(SRC_PATH)/%.$(SRC_EXT)=$(BUILD_PATH)/%.o)
OBJS := $(OBJS:$(SRC_PATH)/%.$(SRC_CUDA_EXT)=$(BUILD_PATH)/%.o)
DEP := $(OBJS:.o=.d)

################################################################################

.PHONY: modules
modules: data_module
	@echo "All modules built!"

.PHONY: data_module
data_module:
	@echo "Beginning module DATA build"
	@$(START_TIME)
	@cmake -Bmodules/data -Hmodules/data -DCMAKE_BUILD_TYPE=Release
	@$(MAKE) -C "modules/data" --no-print-directory
	@echo "Total build time: "
	@$(END_TIME)

.PHONY: release
release: dirs modules
	@echo "Beginning release build"
	@$(START_TIME)
	@$(MAKE) all --no-print-directory
	@echo "Total build time: "
	@$(END_TIME)

# Debug build for gdb debugging
.PHONY: debug
debug: dirs modules
	@echo "Beginning debug build"
	@$(START_TIME)
	@$(MAKE) all --no-print-directory
	@echo "Total build time: "
	@$(END_TIME)

# Test build for gtests
.PHONY: test
test: dirs modules
	@echo "Beginning test build"
	@$(START_TIME)
	@$(MAKE) all --no-print-directory
	@echo "Total build time: "
	@$(END_TIME)

# Benchmark build for google benchmark
.PHONY: benchmark
benchmark: dirs modules
	@echo "Beginning benchmark build"
	@$(START_TIME)
	@$(MAKE) all --no-print-directory
	@echo "Total build time: "
	@$(END_TIME)

.PHONY: dirs
dirs:
	@echo "Creating directories"
	@mkdir -p $(dir $(OBJS))
	@mkdir -p $(BIN_PATH)
	@echo "Directories created"

# Removes all build files
.PHONY: clean
clean:
	@echo "Deleting $(PROGRAM_NAME) symlink"
	@$(RM) $(PROGRAM_NAME)
	@echo "Deleting directories"
	@$(RM) -r build
	@$(RM) -r bin

.PHONY: all
all: $(BIN_PATH)/$(PROGRAM_NAME)
	@echo "Making symlink: $(PROGRAM_NAME) -> $<"
	@$(RM) $(PROGRAM_NAME)
	@ln -s $(BIN_PATH)/$(PROGRAM_NAME) $(PROGRAM_NAME)

.PHONY: run
run:
	./$(PROGRAM_NAME)

################################################################################

$(BIN_PATH)/$(PROGRAM_NAME): $(OBJS)
	@echo 'Linking target: $@'
	@echo 'Invoking: $(NVCC) Linker'
	$(COMPILER) $(LIB_DIRS) $(CODE_FLAGS) $(STANDART) $(NVCC_FLAGS) $(LIBS) $(GENCODE_FLAGS) -link -o $(BIN_PATH)/$(PROGRAM_NAME) $(OBJS)
	chmod +x $(BIN_PATH)/$(PROGRAM_NAME)
	@echo 'Finished building target: $@'
	@echo ' '

$(BUILD_PATH)/%.o: $(SRC_PATH)/%.$(SRC_EXT)
	@$(START_TIME)
	@echo 'Building file: $< -> $@'
	@echo 'Invoking: $(COMPILER) Compiler'
	$(COMPILER) $(INCLUDES) $(CODE_FLAGS) $(STANDART) --compile -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo '\t Compile time: '
	@$(END_TIME)
	@echo ' '

$(BUILD_PATH)/%.o: $(SRC_PATH)/%.$(SRC_CUDA_EXT)
	@$(START_TIME)
	@echo 'Building file: $< -> $@'
	@echo 'Invoking: $(NVCC) Compiler'
	$(COMPILER) $(INCLUDES) $(CODE_FLAGS) $(STANDART) --compile $(CUDA_FLAGS) $(GENCODE_FLAGS) -x $(SRC_CUDA_EXT) -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo '\t Compile time:'
	@$(END_TIME)
	@echo ' '
