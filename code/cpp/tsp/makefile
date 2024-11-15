# --- SYSTEM ---

SYSTEM     = x86-64_linux
LIBFORMAT  = static_pic
BASISDIR   = /opt
PLATFORM   = linux64


# --- DIRECTORIES ---

CCC = g++

BOOSTDIR   = $(BASISDIR)/boost


# --- FLAGS ---

CCOPT = -std=c++11 -m64 -fPIC -fno-strict-aliasing -fexceptions -DIL_STD -Wno-deprecated-declarations -Wno-ignored-attributes

# --- OPTIMIZATION FLAGS ---

DEBUG_OPT = -DNDEBUG -O3
#DEBUG_OPT = -g3 -O0

#PROF = -pg
PROF =


# Flags
CFLAGS  = -I$(BOOSTDIR)/include -c $(PROF) $(DEBUG_OPT)
LDFLAGS = -lm -pthread


# --- APPLICATION-SPECIFIC OPTIONS ---

# number of objective functions
NUM_OBJS=3
CFLAGS += -DNOBJS=$(NUM_OBJS)


# ---- COMPILE  ----
SRC_DIR   := src
OBJ_DIR   := obj

SRC_DIRS  := $(shell find $(SRC_DIR) -type d)
OBJ_DIRS  := $(addprefix $(OBJ_DIR)/,$(SRC_DIRS))

SOURCES   := $(shell find $(SRC_DIR) -name '*.cpp')
OBJ_FILES := $(addprefix $(OBJ_DIR)/, $(SOURCES:.cpp=.o))

vpath %.cpp $(SRC_DIRS)

# ---- TARGETS ----

EXECUTABLE=multiobj

all: $(EXECUTABLE)

$(EXECUTABLE): makedir $(SOURCES) $(OBJ_FILES) 
	$(CCC) $(OBJ_FILES) $(LDFLAGS) $(PROF) -o $@

$(OBJ_DIR)/%.o: %.cpp
	$(CCC) $(CFLAGS) $< -o $@

makedir: $(OBJ_DIRS)

$(OBJ_DIRS):
	@mkdir -p $@

clean:
	@rm -rf obj 
	@rm -rf $(EXECUTABLE)
