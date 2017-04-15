#!/bin/sh

nvcc -std=c++11 -O3 -m64 -Wno-deprecated-gpu-targets -o add1d add1d.cu
