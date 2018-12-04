CXX := nvcc
CC := nvcc 
LINK := nvcc
NVCC := nvcc

#INCLUDES    = -I. $$CUDA_INC
#LIBS        = $$CUDA_LIB
CFLAGS      += -Wno-deprecated-gpu-targets

OBJS = mmio.o main.o kernels.o pagerank.o poisson.o
TARGET = sparse
LINKLINE = $(LINK) $(CFLAGS) -o $(TARGET) $(OBJS) $(LIBS)

.SUFFIXES: .c .cpp .cu .o

$(TARGET): $(OBJS)
	$(LINKLINE)
%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@
%.o: %.cu
	$(NVCC) $(CFLAGS) $(INCLUDES) -c $< -o $@
clean:
	rm -rf *.o $(TARGET)

all: $(TARGET)

