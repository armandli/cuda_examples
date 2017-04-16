//add 2 arrays in parallel, often this is faster on CPU than GPU
//reason is computation is not intense, require more data than
//computation

#include <cstdlib>
#include <ctime>
#include <iostream>

#define BSZ 2048
#define TSZ 1024
#define TEST_SIZE BSZ * TSZ
#define TT float
#define EPS 10e-6

using namespace std;

template <typename T>
__global__ void add1d_cuda(T* c, T* a, T* b){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  c[idx] = a[idx] + b[idx];
}

template <typename T>
clock_t add1d(T* c, T* a, T* b, size_t sz){
  for (size_t i = 0; i < sz; ++i)
    c[i] = a[i] + b[i];

  return clock();
}

template <typename T>
void random_array(T* array, size_t sz){
  srand(time(0));

  for (size_t i = 0; i < sz; ++i)
    array[i] = (TT)rand() / 100.F;
}

int main(){
  TT *a = new TT[TEST_SIZE], *b = new TT[TEST_SIZE], *c = new TT[TEST_SIZE], *d = new TT[TEST_SIZE];
  TT* da, *db, *dc;

  random_array(a, TEST_SIZE);
  random_array(b, TEST_SIZE);


  cudaMalloc((void**)&da, sizeof(TT) * TEST_SIZE);
  cudaMalloc((void**)&db, sizeof(TT) * TEST_SIZE);
  cudaMalloc((void**)&dc, sizeof(TT) * TEST_SIZE);

  clock_t timing_start = clock();

  cudaMemcpy(da, a, sizeof(TT) * TEST_SIZE, cudaMemcpyHostToDevice);
  cudaMemcpy(db, b, sizeof(TT) * TEST_SIZE, cudaMemcpyHostToDevice);


  add1d_cuda<<<BSZ, TSZ>>>(dc, da, db);

  cudaMemcpy(c, dc, sizeof(TT) * TEST_SIZE, cudaMemcpyDeviceToHost);

  cout << "CUDA time: " << (clock() - timing_start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << endl;

  cudaFree(da);
  cudaFree(db);
  cudaFree(dc);

  timing_start = clock();

  clock_t timing_end = add1d(d, a, b, TEST_SIZE);

  cout << "CPU time: " << (timing_end - timing_start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << endl;

  bool match = true;
  for (size_t i = 0; i < TEST_SIZE; ++i)
    if (c[i] - d[i] > EPS){
      cout << "value does not match" << endl;
      match = false;
    }

  if (match) cout << "All values match" << endl;
}
