//matrix multiply, with 2048 by 2048 square matrix, gpu is faster even with the least efficient implementation

#include <cassert>
#include <cstdlib>
#include <ctime>
#include <random>
#include <iostream>

#define BSZ 64
#define TSZ 16
#define SZ (BSZ * TSZ)
#define TT float

using namespace std;

template <typename T>
void random_matrix(T* m, size_t sz){
  uniform_real_distribution<T> dist(-100.F, 100.F);
  default_random_engine eng(time(0));

  for (size_t i = 0; i < sz; ++i)
    m[i] = dist(eng);
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

template <typename T>
struct CudaMtx {
  T* data;
  size_t rows;
  size_t cols;
};

template <typename T>
struct Mtx {
  T* data;
  size_t rows;
  size_t cols;
  bool is_cuda;

  Mtx(bool is_cuda, size_t rows, size_t cols):
    data(nullptr), is_cuda(is_cuda), rows(rows), cols(cols) {
    if (is_cuda) { gpuErrchk(cudaMalloc(&data, sizeof(T) * rows * cols)); }
    else         data = new T[rows * cols];
  }

  ~Mtx(){
    if (is_cuda) cudaFree(data);
    else         delete[] data;
  }

  CudaMtx<T> cuda_mtx(){
    assert(is_cuda);
    CudaMtx<T> ret;
    ret.data = data;
    ret.rows = rows;
    ret.cols = cols;
    return ret;
  }
};

template <typename T>
__global__ void matrix_multiply_cuda_v1(CudaMtx<T> m, CudaMtx<T> a, CudaMtx<T> b){
  size_t r = blockIdx.x * blockDim.x + threadIdx.x;
  size_t c = blockIdx.y * blockDim.y + threadIdx.y;

  T mval = 0.F;
  for (size_t i = 0; i < a.cols; ++i)
    mval += a.data[r * a.cols + i] * b.data[i * b.cols + c];
  m.data[r * m.cols + c] = mval;
}

template <typename T>
clock_t matrix_multiply_v1(Mtx<T>& c, Mtx<T>& a, Mtx<T>& b){
  for (size_t i = 0; i < c.rows; ++i)
    for (size_t j = 0; j < c.cols; ++j){
      c.data[i * c.cols + j] = 0.;
      for (size_t k = 0; k < a.cols; ++k)
        c.data[i * c.cols + j] += a.data[i * a.cols + k] * b.data[k * b.cols + j];
    }

  return clock();
}

int main(){
  Mtx<TT> c(false, SZ, SZ), a(false, SZ, SZ), b(false, SZ, SZ), d(false, SZ, SZ);
  Mtx<TT> dc(true, SZ, SZ), da(true, SZ, SZ), db(true, SZ, SZ);

  random_matrix(a.data, SZ * SZ);
  random_matrix(b.data, SZ * SZ);

  clock_t timing_start = clock();

  gpuErrchk(cudaMemcpy(da.data, a.data, sizeof(TT) * SZ * SZ, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(db.data, b.data, sizeof(TT) * SZ * SZ, cudaMemcpyHostToDevice));

  dim3 dblock(BSZ, BSZ);
  dim3 dthread(TSZ, TSZ);
  matrix_multiply_cuda_v1<<<dblock, dthread>>>(dc.cuda_mtx(), da.cuda_mtx(), db.cuda_mtx());
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  gpuErrchk(cudaMemcpy(c.data, dc.data, sizeof(TT) * SZ * SZ, cudaMemcpyDeviceToHost));

  cout << "CUDA time: " << (clock() - timing_start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << endl;

  timing_start = clock();

  clock_t timing_end = matrix_multiply_v1(d, a, b);

  cout << "CPU time: " << (timing_end - timing_start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << endl;

  size_t mismatch = 0;
  for (size_t i = 0; i < SZ * SZ; ++i)
    if ((fabs(c.data[i] - d.data[i]) / d.data[i]) > 5e-3F){
      cout << "difference " << (fabs(c.data[i] - d.data[i]) / d.data[i]) << endl;
      mismatch++;
    }

  if (mismatch == 0) cout << "All values match" << endl;
  else               cout << mismatch << " differences" << endl;
}
