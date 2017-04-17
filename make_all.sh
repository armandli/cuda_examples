#!/bin/sh

nvcc -std=c++11 -O3 -m64 -Wno-deprecated-gpu-targets -o add1d add1d.cu
nvcc -std=c++11 -O3 -m64 -Wno-deprecated-gpu-targets -o o2 o2.cu
nvcc -std=c++11 -O3 -m64 -Wno-deprecated-gpu-targets -o matrix_multiply_v1 matrix_multiply_v1.cu
nvcc -std=c++11 -O3 -m64 -Wno-deprecated-gpu-targets -o matrix_multiply_v2 matrix_multiply_v2.cu
