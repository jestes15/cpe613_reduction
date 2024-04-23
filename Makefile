ADA:
	nvcc main.cu -o main -arch=sm_89 -Xcompiler '-fopenmp'

A100:
	nvcc -O3 main.cu -o main_a100 -arch=sm_80 -Xcompiler '-fopenmp'

H100:
	nvcc -O3 main.cu -o main_h100 -arch=sm_90 -Xcompiler '-fopenmp'