//matrix multiply

#include <cstdlib>
#include <ctime>
#include <iostream>

#define BSZ 256
#define TSZ 16
#define SZ (BSZ * TSZ)
#define TT float

using namespace std;

template <typename T>
void random_matrix(T* m, size_t sz){
  srand(time(0));

  for (size_t i = 0; i < sz; ++i)
    m[i] = (TT)rand() / 100.F;
}

template <typename T>
struct Mtx {
  T* data;
  size_t rows;
  size_t cols;
};

template <typename T>
__global__ matrix_multiply_cuda_v1(Mtx<T>* c, Mtx<T>* a, Mtx<T>* b){
  size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  size_t y = blockIdx.y * blockDim.y + threadIdx.y;

  T cval = 0.F;
  for (size_t i = 0; i < a->cols; ++i)
    cval += a->data[x * a.cols + i] * b->data[i * b.cols + y];
  c->data[x * c.cols + y] = cval;
}

template <typename T>
clock_t matrix_multiply_v1(T* c, const T* a, const T* b, size_t r, size_t c, size_t m){
  for (size_t i = 0; i < r; ++i)
    for (size_t j = 0; j < c; ++j)
      for (size_t k = 0; k < m; ++k)
        c[i * c + j] += a[i * m + k] * b[k * c + j];

  return clock();
}

int main(){
  TT* c = new TT[SZ * SZ], *a = new TT[SZ * SZ], *b = new TT[SZ * SZ], *d = new TT[SZ * SZ];
  TT* dc, *da, *db;

  random_matrix(a, SZ * SZ);
  random_matrix(b, SZ * SZ);

  cudaMalloc(&da, sizeof(TT) * SZ * SZ);
  cudaMalloc(&db, sizeof(TT) * SZ * SZ);
  cudaMalloc(&dc, sizeof(TT) * SZ * SZ);

  clock_t timing_start = clock();

  cudaMemcpy(da, a, sizeof(TT) * SZ * SZ, cudaMemcpyHostToDevice);
  cudaMemcpy(db, b, sizeof(TT) * SZ * SZ, cudaMemcpyHostToDevice);

  dim3 dblock(BSZ, BSZ);
  dim3 dthread(BSZ, BSZ);
  matrix_multiply_cuda_v1<<<dblock, dthread>>>(dc, da, db);

  cudaMemcpy(c, dc, sizeof(TT) * SZ * SZ, cudaMemcpyDeviceToHost);

  cout << "CUDA time: " << (clock() - timing_start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << endl;

  cudaFree(da);
  cudaFree(db);
  cudaFree(dc);

  timing_start = clock();

  clock_t timing_end = matrix_multiply_v1(d, a, b);

  cout << "CPU time: " << (timing_end - timing_start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << endl;

  bool match = true;
  for (size_t i = 0; i < SZ * SZ; ++i)
    if (c[i] - d[i] > 1e-5F){
      cout << "Values does not match" << endl;
      match = false;
    }

  if (match) cout << "All values match" << endl;
}
