uname_S := $(shell sh -c 'uname -s 2>/dev/null || echo not')
uname_M := $(shell sh -c 'uname -m 2>/dev/null || echo not')

PROGRAM_NAME := gpuStore

RM := rm -rf
OS := $(shell uname)
DATE := gdate

SRC_PATH := src
BUILD_PATH := build
BIN_PATH := bin

SRC_CUDA_EXT := cu
SRC_EXT := cpp

COMPILER := nvcc
STANDART := --std=c++11
NVCC_FLAGS := --cudart static --relocatable-device-code=false

ifeq ($(OS),Darwin)
	LIB_DIRS := #-L"/usr/local/cuda/lib" -L"/usr/local/lib"
	LIBS := -lcudart -lboost_system -lboost_thread -lpthread -lboost_thread \
		-lboost_program_options -llog4cplus -lgtest -lbenchmark
	STANDART := -std=c++11
else
	LIB_DIRS := -L"/usr/local/cuda/lib64"
	LIBS := -lcudart -lboost_system -lboost_thread -lpthread -lboost_thread \
		-lboost_program_options -llog4cplus -lgtest -lbenchmark
	STANDART := -std=c++0x
endif

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

CUDA_INCLUDES := -I"/usr/local/cuda/include"
DEFINES := #-D __GXX_EXPERIMENTAL_CXX0X__ -DBOOST_HAS_INT128=1 -D_GLIBCXX_USE_CLOCK_REALTIME -DHAVE_WTHREAD_SAFETY
WARNINGS_ERRORS := -pedantic -Wall -Wextra -Wno-deprecated -Wno-unused-parameter  -Wno-enum-compare -Weffc++

GENCODE_SM20    := -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_20
GENCODE_SM21    := -gencode arch=compute_20,code=compute_21 -gencode arch=compute_20,code=sm_21
GENCODE_SM30    := -gencode arch=compute_30,code=compute_30 -gencode arch=compute_30,code=sm_30
GENCODE_SM35		:= -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35
GENCODE_FLAGS   := $(GENCODE_SM20)

debug: COMPILER += -DDEBUG -g
debug: NVCC += --debug --device-debug
debug: export EXCLUDED_MAIN_FILE := main_tests.cpp
debug: export EXCLUDED_DIRECTORY := */tests/*
debug: export BUILD_PATH := build/debug
debug: export BIN_PATH := bin/debug

release: COMPILER += -O3
release: NVCC += -O3
release: export EXCLUDED_MAIN_FILE := main_tests.cpp
release: export EXCLUDED_DIRECTORY := */tests/*
release: export BUILD_PATH := build/release
release: export BIN_PATH := bin/release

test: export EXCLUDED_MAIN_FILE := main.cpp
test: export EXCLUDED_DIRECTORY :=
test: export BUILD_PATH := build/test
test: export BIN_PATH := bin/test

SRC_FILES := $(shell find $(SRC_PATH)/ -name '*.$(SRC_EXT)' \
	-not -iname '$(EXCLUDED_MAIN_FILE)' -not -path '$(EXCLUDED_DIRECTORY)' \
	-o -name '*.$(SRC_CUDA_EXT)' | sort -k 1nr | cut -f2-)

OBJS := $(SRC_FILES:$(SRC_PATH)/%.$(SRC_EXT)=$(BUILD_PATH)/%.o)
OBJS := $(OBJS:$(SRC_PATH)/%.$(SRC_CUDA_EXT)=$(BUILD_PATH)/%.o)
DEP := $(OBJS:.o=.d)

################################################################################

.PHONY: release
release: dirs
	@echo "Beginning release build"
	@$(START_TIME)
	@$(MAKE) all --no-print-directory
	@echo "Total build time: "
	@$(END_TIME)

# Debug build for gdb debugging
.PHONY: debug
debug: dirs
	@echo "Beginning debug build"
	@$(START_TIME)
	@$(MAKE) all --no-print-directory
	@echo "Total build time: "
	@$(END_TIME)

# Test build for gtests
.PHONY: test
test: dirs
	@echo "Beginning test build"
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
	$(COMPILER) $(STANDART) $(NVCC_FLAGS) $(LIBS) $(GENCODE_FLAGS) -link -o $(BIN_PATH)/$(PROGRAM_NAME) $(OBJS)
	chmod +x $(BIN_PATH)/$(PROGRAM_NAME)
	@echo 'Finished building target: $@'
	@echo ' '

$(BUILD_PATH)/%.o: $(SRC_PATH)/%.$(SRC_EXT)
	@$(START_TIME)
	@echo 'Building file: $< -> $@'
	@echo 'Invoking: $(COMPILER) Compiler'
	$(COMPILER) $(STANDART) --compile -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo '\t Compile time: '
	@$(END_TIME)
	@echo ' '

$(BUILD_PATH)/%.o: $(SRC_PATH)/%.$(SRC_CUDA_EXT)
	@$(START_TIME)
	@echo 'Building file: $< -> $@'
	@echo 'Invoking: $(NVCC) Compiler'
	$(COMPILER) $(STANDART) --compile $(CUDA_FLAGS) $(GENCODE_FLAGS) -x $(SRC_CUDA_EXT) -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo '\t Compile time:'
	@$(END_TIME)
	@echo ' '
