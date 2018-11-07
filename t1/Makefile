CC		= nvcc 
EXEC	= out

SOURCES = \
cuda_mmult_kernels.cu \
cuda_mmult.cu

OBJS = $(SOURCES:.cu=.o)

CUFLAGS += -O3 -Wno-deprecated-gpu-targets

%.o: %.cu
	$(CC) -c $(CUFLAGS) -o $@ $<

all: $(OBJS)
	$(CC) -link -L/usr/local/cuda/lib64/ $(CUFLAGS) $(OBJS) -o $(EXEC)

clean:
	@rm -f $(OBJS) $(EXEC)
