ada:
	nvcc -O3 main.cu -o main -arch=sm_89 -Xcompiler '-fopenmp'

xavier:
	/usr/local/cuda-11/bin/nvcc -g  main.cu -o main -arch=sm_72 -Xcompiler '-fopenmp'

a100:
	nvcc -g  main.cu -o main -arch=sm_80 -Xcompiler '-fopenmp'

H100:
	nvcc -g  main.cu -o main -arch=sm_90 -Xcompiler '-fopenmp'