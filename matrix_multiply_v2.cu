//matrix multiply using shared memory for optimization

#include <cassert>
#include <cstdlib>
#include <ctime>
#include <random>
#include <iostream>

#define BSZ 128
#define TSZ 16
#define SZ (BSZ * TSZ)
#define TT double

using namespace std;

default_random_engine& get_default_random_engine(){
  static default_random_engine eng(time(0));
  return eng;
}

template <typename T>
void random_matrix(T* m, size_t sz){
  uniform_real_distribution<T> dist(-100.F, 100.F);
  default_random_engine& eng = get_default_random_engine();

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
  size_t stride;
};

template <typename T>
struct Mtx {
public:
  T* data;
  size_t rows;
  size_t cols;
  bool is_cuda;

  Mtx(bool is_cuda, size_t rows, size_t cols):
    data(nullptr), rows(rows), cols(cols), is_cuda(is_cuda) {
    if (is_cuda) { gpuErrchk(cudaMalloc(&data, sizeof(T) * rows * cols)); }
    else         data = new T[rows * cols];
  }

  ~Mtx(){
    if (is_cuda) { gpuErrchk(cudaFree(data)); }
    else         delete[] data;
  }

  CudaMtx<T> cuda_mtx(){
    assert(is_cuda);
    CudaMtx<T> ret;
    ret.data = data;
    ret.rows = rows;
    ret.cols = ret.stride = cols;
    return ret;
  }
};

template <typename T>
__device__ T get_elem(CudaMtx<T>& a, size_t i, size_t j){
  return a.data[i * a.stride + j];
}

template <typename T>
__device__ void set_elem(CudaMtx<T>& a, size_t i, size_t j, T val){
  a.data[i * a.stride + j] = val;
}

template <typename T>
__device__ CudaMtx<T> sub_matrix_stride(CudaMtx<T>& m, size_t row_stride, size_t col_stride){
  CudaMtx<T> ret;
  ret.data = &m.data[m.cols * TSZ * row_stride + TSZ * col_stride];
  ret.rows = ret.cols = TSZ;
  ret.stride = m.stride;
  return ret;
}

template <typename T>
__global__ void matrix_multiply_cuda_v2(CudaMtx<T> c, CudaMtx<T> a, CudaMtx<T> b){
  size_t bx = blockIdx.x, by = blockIdx.y;
  CudaMtx<T> csub = sub_matrix_stride(c, bx, by);

  T cval = 0.;

  size_t row = threadIdx.x, col = threadIdx.y;

  for (size_t i = 0; i < BSZ; ++i){
    CudaMtx<T> asub = sub_matrix_stride(a, bx, i);
    CudaMtx<T> bsub = sub_matrix_stride(b, i, by);

    __shared__ T amem[TSZ][TSZ];
    __shared__ T bmem[TSZ][TSZ];

    amem[row][col] = get_elem(asub, row, col);
    bmem[row][col] = get_elem(bsub, row, col);

    __syncthreads();

    for (size_t j = 0; j < TSZ; ++j)
      cval += amem[row][j] * bmem[j][col];

    __syncthreads();
  }

  set_elem(csub, row, col, cval);
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
  matrix_multiply_cuda_v2<<<dblock, dthread>>>(dc.cuda_mtx(), da.cuda_mtx(), db.cuda_mtx());
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  gpuErrchk(cudaMemcpy(c.data, dc.data, sizeof(TT) * SZ * SZ, cudaMemcpyDeviceToHost));

  cout << "CUDA time: " << (clock() - timing_start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << endl;

  timing_start = clock();

  clock_t timing_end = matrix_multiply_v1(d, a, b);

  cout << "CPU time: " << (timing_end - timing_start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << endl;

  size_t mismatch = 0;
  for (size_t i = 0; i < SZ * SZ; ++i)
    if (fabs(c.data[i] - d.data[i]) / d.data[i] > 5e-3F){
      cout << "difference: " << (fabs(c.data[i] - d.data[i]) / d.data[i]) << endl;
      mismatch++;
      break;
    }

  if (mismatch == 0) cout << "All values match" << endl;
  else               cout << mismatch << " differences" << endl;
}
