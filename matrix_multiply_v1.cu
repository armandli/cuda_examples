//matrix multiply, with 2048 by 2048 square matrix, gpu is faster even with the least efficient implementation

#include <cassert>
#include <cstdlib>
#include <ctime>
#include <iostream>

#define BSZ 128
#define TSZ 16
#define SZ (BSZ * TSZ)
#define TT double

using namespace std;

template <typename T>
void random_matrix(T* m, size_t sz){
  srand(time(0));

  for (size_t i = 0; i < sz; ++i)
    m[i] = (TT)rand() / 100.F;
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
    assert(is_cuda, "matrix is not allocated on CUDA");
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
  for (size_t i = 0; i < k; ++i)
    mval += a.data[r * a.cols + i] * b[i * b.cols + c];
  m[r * m.cols + c] = mval;
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
  Mtx<TT> da(true, SZ, SZ), db(true, SZ, SZ), dc(true, SZ, SZ);

  memset(a.data, 0, sizeof(TT) * SZ * SZ);
  memset(b.data, 0, sizeof(TT) * SZ * SZ);
  a.data[0] = 1.; a.data[1] = 2.; a.data[SZ] = 3.; a.data[SZ+1] = 4.;
  b.data[0] = 2.; b.data[1] = 3.; b.data[SZ] = 4.; b.data[SZ+1] = 5.;

  matrix_multiply_v1(c, a, b);

  cout << c.data[0] << " " << c.data[1] << " " << c.data[SZ] << " " << c.data[SZ+1] << endl;

  gpuErrchk(cudaMemcpy(da.data, a.data, sizeof(TT) * SZ * SZ, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(db.data, b.data, sizeof(TT) * SZ * SZ, cudaMemcpyHostToDevice));

  dim3 dblock(BSZ, BSZ);
  dim3 dthread(TSZ, TSZ);
  matrix_multiply_cuda_v1<<<1, dthread>>>(dc.cuda_mtx(), da.cuda_mtx(), db.cuda_mtx());
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  gpuErrchk(cudaMemcpy(d.data, dc.data, sizeof(TT) * SZ * SZ, cudaMemcpyDeviceToHost));

  cout << d.data[0] << " " << d.data[1] << " " << d.data[SZ] << " " << d.data[SZ+1] << endl;

//  Mtx<TT> c(false, SZ, SZ), a(false, SZ, SZ), b(false, SZ, SZ), d(false, SZ, SZ);
//  Mtx<TT> dc(true, SZ, SZ), da(true, SZ, SZ), db(true, SZ, SZ);
//
//  random_matrix(a.data, SZ * SZ);
//  random_matrix(b.data, SZ * SZ);
//
//  clock_t timing_start = clock();
//
//  gpuErrchk(cudaMemcpy(da.data, a.data, sizeof(TT) * SZ * SZ, cudaMemcpyHostToDevice));
//  gpuErrchk(cudaMemcpy(db.data, b.data, sizeof(TT) * SZ * SZ, cudaMemcpyHostToDevice));
//
//  dim3 dblock(BSZ, BSZ);
//  dim3 dthread(TSZ, TSZ);
//  matrix_multiply_cuda_v1<<<dblock, dthread>>>(dc.cuda_mtx(), da.cuda_mtx(), db.cuda_mtx());
//  gpuErrchk(cudaPeekAtLastError());
//  gpuErrchk(cudaDeviceSynchronize());
//
//  gpuErrchk(cudaMemcpy(c.data, dc.data, sizeof(TT) * SZ * SZ, cudaMemcpyDeviceToHost));
//
//  cout << "CUDA time: " << (clock() - timing_start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << endl;
//
//  timing_start = clock();
//
//  clock_t timing_end = matrix_multiply_v1(d, a, b);
//
//  cout << "CPU time: " << (timing_end - timing_start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << endl;
//
//  bool match = true;
//  for (size_t i = 0; i < SZ * SZ; ++i)
//    if (abs(c.data[i] - d.data[i]) > 1e-5F){
//      cout << "Values does not match. difference " << abs(c.data[i] - d.data[i]) << endl;
//      match = false;
//      break;
//    }
//
//  if (match) cout << "All values match" << endl;
}
