//matrix multiply using shared memory for optimization

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

template <typename T>
struct Mtx {
public:
  T* data;
  size_t rows;
  size_t cols;
  bool is_cuda;

  Mtx(bool is_cuda, size_t rows, size_t cols):
    data(nullptr), rows(rows), cols(cols), is_cuda(is_cuda) {
    if (is_cuda) cudaMalloc(&data, sizeof(T) * rows * cols);
    else         data = new T[rows * cols];
  }

  ~Mtx(){
    if (is_cuda) cudaFree(data);
    else         delete[] data;
  }
};

template <typename T>
struct SubMtx {
  T* data;
  size_t rows;
  size_t cols;
  size_t stride;
};

template <typename T>
__device__ T get_elem(SubMtx<T>& a, size_t i, size_t j){
  return a.data[i * a.stride + j];
}

template <typename T>
__device__ void set_elem(SubMtx<T>& a, size_t i, size_t j, T val){
  a.data[i * a.stride + j] = val;
}

template <typename T>
__device__ SubMtx<T> sub_matrix_stride(Mtx<T>& m, size_t row_stride, size_t col_stride){
  SubMtx<T> ret;
  ret.data = &m.data[m.cols * TSZ * row_stride + TSZ * col_stride];
  ret.rows = ret.cols = TSZ;
  ret.stride = m.cols;
  return ret;
}

template <typename T>
__global__ void matrix_multiply_cuda_v2(Mtx<T>& c, Mtx<T>& a, Mtx<T>& b){
  size_t bx = blockIdx.x, by = blockIdx.y;
  SubMtx<T> csub = sub_matrix_stride(c, bx, by);

  T cval = 0.;

  size_t row = threadIdx.x, col = threadIdx.y;

  for (size_t i = 0; i < BSZ; ++i){
    SubMtx<T> asub = sub_matrix_stride(a, bx, i);
    SubMtx<T> bsub = sub_matrix_stride(b, i, by);

    __shared__ T amem[TSZ][TSZ];
    __shared__ T bmem[TSZ][TSZ];

    amem[row][col] = get_elem(asub, row, col);
    bmem[row][col] = get_elem(bsub, row, col);

    __syncthreads();

    for (size_t j = 0; j < TSZ; ++j)
      cval += amem[row][j] * bmem[j][col];

    __syncthreads();
  }

//  set_elem(csub, bx * blockDim.x + row, by * blockDim.y + col, cval);
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

  cudaMemcpy(da.data, a.data, sizeof(TT) * SZ * SZ, cudaMemcpyHostToDevice);
  cudaMemcpy(db.data, b.data, sizeof(TT) * SZ * SZ, cudaMemcpyHostToDevice);

  dim3 dblock(BSZ, BSZ);
  dim3 dthread(BSZ, BSZ);
  matrix_multiply_cuda_v2<<<dblock, dthread>>>(dc, da, db);

  cudaMemcpy(c.data, dc.data, sizeof(TT) * SZ * SZ, cudaMemcpyDeviceToHost);

  cout << "CUDA time: " << (clock() - timing_start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << endl;

  timing_start = clock();

  clock_t timing_end = matrix_multiply_v1(d, a, b);

  cout << "CPU time: " << (timing_end - timing_start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << endl;

  bool match = true;
  for (size_t i = 0; i < SZ * SZ; ++i)
    if (c.data[i] - d.data[i] > 1e-5F){
      cout << "Values does not match" << endl;
      match = false;
      break;
    }

  if (match) cout << "All values match" << endl;
}
