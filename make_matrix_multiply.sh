#!/bin/sh

nvcc -std=c++11 -O3 -m64 -Wno-deprecated-gpu-targets -o matrix_multiply_v1 matrix_multiply_v1.cu
