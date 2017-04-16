//example where there is heavy computation done
//using very little data, this example GPU outperforms CPU
//by 100 of times at least

#include <cstdlib>
#include <ctime>
#include <iostream>

#define TSZ 1024
#define BSZ 1024

#define N (BSZ * TSZ)
#define M 100000
#define TT float

using namespace std;

template <typename T>
__global__ void o2_cuda(T* a){
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  a[i] = (T)i / (T)N;
  for (size_t j = 0; j < M; ++j)
    a[i] = a[i] * a[i] - 0.25;
}

template <typename T>
clock_t o2(T* a){
  for (size_t i = 0; i < N; ++i){
    a[i] = (T)i / (T)N;
    for (int j = 0; j < M; ++j)
      a[i] = a[i] * a[i] - 0.25F;
  }

  return clock();
}

int main(){
  TT* a = new TT[N], *b = new TT[N];
  TT* db;

  cudaMalloc(&db, N * sizeof(TT));

  clock_t timing_start = clock();

  o2_cuda<<<BSZ, TSZ>>>(db);

  cudaMemcpy(b, db, sizeof(TT) * N, cudaMemcpyDeviceToHost);

  cout << "CUDA time: " << (clock() - timing_start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << endl;

  cudaFree(db);

  timing_start = clock();

  clock_t timing_end = o2(a);

  cout << "CPU time: " << (timing_end - timing_start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << endl;

  bool is_same = true;
  for (size_t i = 0; i < N; ++i)
    if (a[i] != b[i]){
      cout << "Index " << i << " is different" << endl;
      is_same = false;
      break;
    }
  
  if (is_same) cout << "Answer match" << endl;
}
