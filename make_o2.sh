#!/bin/sh

nvcc -std=c++11 -O3 -m64 -Wno-deprecated-gpu-targets -o o2 o2.cu
