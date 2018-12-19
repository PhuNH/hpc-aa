CXX := nvcc 
CC := nvcc
LINK := nvcc
NVCC := nvcc

INCLUDES    = -I. 
LIBS        = -lcublas -lcusparse

OBJS = mmio.c.o gnuplot_i.c.o main.c.o kernels.cu.o pagerank.c.o poisson.c.o
TARGET = sparse
LINKLINE = $(LINK) -Wno-deprecated-gpu-targets -o $(TARGET) $(OBJS) $(LIBS)
CFLAGS += -Wno-deprecated-gpu-targets

.SUFFIXES: .c .cpp .cu .o

$(TARGET): $(OBJS)
	$(LINKLINE)
%.c.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@
%.cu.o: %.cu
	$(NVCC) $(CFLAGS) $(INCLUDES) -c $< -o $@
clean:
	rm -rf *.o $(TARGET) gnuplot_tmp*
all: $(TARGET)
