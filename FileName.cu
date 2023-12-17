#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>

#define BLOCK_SIZE 16
#define SHARED_BLOCK_SIZE 16
#define SUMMATION_BLOCK_SIZE BLOCK_SIZE * BLOCK_SIZE

#define CEILDIV(x, y) (x/y + (x % y != 0))

template <typename T>
class CPUArray {
private:
	size_t dim1;
	size_t dim2;
	T* data;

public:

	T& operator[] (size_t n) {	//for dim2 = 0, i.e. vectors
		return data[n];
	};
	T operator[] (size_t n) const {	//for dim2 = 0, i.e. vectors
		return data[n];
	};
	T& operator() (size_t n, size_t m) {
		return data[n * dim2 + m];
	}
	T operator() (size_t n, size_t m) const {
		return data[n * dim2 + m];
	};
	CPUArray(size_t d1, size_t d2 = 0U) {
		dim1 = d1;
		dim2 = d2;
		if (dim2 == 0U)
			data = new T[dim1];
		else
			data = new T[dim1 * dim2];
	}
	CPUArray(size_t d1, size_t d2, T* dta) {
		dim1 = d1;
		dim2 = d2;
		data = dta;
	}
	size_t getDim1() const {
		return dim1;
	}
	size_t getDim2() const {
		return dim2;
	}

	T const* getData() const {
		return data;
	}
	void add(const CPUArray& arr) {
		if (dim2 == 0) {
			for (int i = 0; i < dim1; i++) {
				data[i] += arr[i];
			}
		}
		else {
			for (int i = 0; i < dim1; i++) {
				for (int j = 0; j < dim2; j++)
				{
					data[i * dim2 + j] += arr(i, j);
				}
			}
		}
	};
	void subtract(const CPUArray& arr) {
		if (dim2 == 0) {
			for (int i = 0; i < dim1; i++) {
				data[i] -= arr[i];
			}
		}
		else {
			for (int i = 0; i < dim1; i++) {
				for (int j = 0; j < dim2; j++)
				{
					data[i * dim2 + j] -= arr(i, j);
				}
			}
		}
	};
	void print() const {
		if (dim2 == 0) {
			std::cout << "{ ";
			for (int i = 0; i < dim1; i++) {
				std::cout << data[i] << " ";
			}
			std::cout << "}";
		}
		else {
			std::cout << "[";
			for (int i = 0; i < dim1; i++) {
				std::cout << "[ ";
				for (int j = 0; j < dim2; j++)
				{
					std::cout << data[i * dim2 + j] << " ";
				}
				if (i == dim1 - 1)
					std::cout << "]";
				else
					std::cout << "]\n";
			}
			std::cout << "]";
		}
	};
	void fill(T** arr) {
		for (int i = 0; i < dim1; i++) {
			for (int j = 0; j < dim2; j++)
			{
				data[i * dim2 + j] = arr[i][j];
			}
		}
	};
	void fill(T* arr) {
		if (dim2 == 0) {
			for (int i = 0; i < dim1; i++) {
				data[i] = arr[i];
			}
		}
		else {
			for (int i = 0; i < dim1; i++) {
				for (int j = 0; j < dim2; j++)
				{
					data[i * dim2 + j] = arr[i * dim2 + j];
				}
			}
		}
	};
	void setValue(T value) {
		if (dim2 == 0) {
			for (int i = 0; i < dim1; i++) {
				data[i] = value;
			}
		}
		else {
			for (int i = 0; i < dim1; i++) {
				for (int j = 0; j < dim2; j++)
				{
					data[i * dim2 + j] = value;
				}
			}
		}
	}

};

template<typename T>
CPUArray<T> operator+(CPUArray<T>& op1, CPUArray<T>& op2) {
	auto dim1 = op1.getDim1(), dim2 = op1.getDim2();
	//auto sum = CPUArray<T>(dim1, dim2);
	T* data;
	if (dim2 == 0)
		data = new T[dim1];
	else
		data = new T[dim1 * dim2];
	if (dim2 == 0) {
		for (int i = 0; i < dim1; i++) {
			data[i] = op1[i] + op2[i];
		}
	}
	else {
		for (int i = 0; i < dim1; i++) {
			for (int j = 0; j < dim2; j++)
			{
				data[i * dim2 + j] = op1(i, j) + op2(i, j);
			}
		}
	}
	//sum.setValue(0.);
	return CPUArray<T>(dim1, dim2, data);
};

template<typename T>
CPUArray<T> operator*(const CPUArray<T>& op1, const CPUArray<T>& op2) {
	auto dim1 = op1.getDim1(), dim2 = op2.getDim2(), dim_interim = op1.getDim2();
	//auto sum = CPUArray<T>(dim1, dim2);
	//sum.setValue(0);
	T* data;
	if (dim2 == 0)
		data = new T[dim1];
	else
		data = new T[dim1 * dim2];
	if (dim2 == 0) {
		for (int i = 0; i < dim1; i++) {
			data[i] = op1[i] + op2[i];
		}
	}
	else {
		for (int i = 0; i < dim1; i++) {
			for (int j = 0; j < dim2; j++)
			{
				T temp = 0;
				for (int k = 0; k < dim_interim; k++) {
					temp += op1(i, k) * op2(k, j);
				}
				data[i * dim2 + j] = temp;
			}
		}
	}
	return CPUArray<T>(dim1, dim2, data);
};
//Mat_mul << <blocks, threadsPerBlock >> > (op1.getData(), op2.getData(), data, dim1, dim2, dim_interim);
template <typename T>
__global__ void Mat_mul(T const* op1, T const* op2, T* res, size_t dim1, size_t dim2, size_t dim_interim) {
	size_t i = blockIdx.y * blockDim.y + threadIdx.y;
	size_t j = blockIdx.x * blockDim.x + threadIdx.x;

	if ((i >= dim1) || (j >= dim2))
		return;

	T sum = 0;
	for (int k = 0; k < dim_interim; k++) {
		sum += op1[i * dim_interim + k] * op2[k * dim2 + j];
	}
	res[i * dim2 + j] = sum;
	//printf("%lf", res[i * dim2 + j]);
}

template <typename T>
__global__ void Sum(T const* op1, T const* op2, T * sum, int dim) {
	auto tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < dim) {
		sum[tid] = op1[tid] + op2[tid];
	}
}

template <typename T>
__global__ void Mat_sum(T const* op1, T const* op2, T* res, size_t dim1, size_t dim2) {
	size_t i = blockIdx.y * blockDim.y + threadIdx.y;
	size_t j = blockIdx.x * blockDim.x + threadIdx.x;

	if ((i >= dim1) || (j >= dim2))
		return;

	res[i * dim2 + j] = op1[i * dim2 + j] + op2[i * dim2 + j];
	//printf("%lf", res[i * dim2 + j]);
}

template <typename T>
__global__ void Mat_mul_shared(T const* op1, T const* op2, T* res, size_t dim1, size_t dim2, size_t dim_interim) {
	__shared__ T op1_tile[SHARED_BLOCK_SIZE][SHARED_BLOCK_SIZE];
	__shared__ T op2_tile[SHARED_BLOCK_SIZE][SHARED_BLOCK_SIZE];

	size_t _i = blockIdx.y * blockDim.y + threadIdx.y;
	size_t _j = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("I was here 0");
	if ((_i >= dim1) || (_j >= dim2))
		return;

	T sum = 0;

	for (int ind_tile = 0; ind_tile < CEILDIV(dim_interim, BLOCK_SIZE); ind_tile++) {
		size_t i = blockIdx.y * blockDim.y + threadIdx.y;
		size_t j = ind_tile * blockDim.x + threadIdx.x;
		if ((i < dim1) && (j < dim_interim))
		{
			op1_tile[threadIdx.y][threadIdx.x] = op1[i * dim_interim + j];
		}
		else
		{
			op1_tile[threadIdx.y][threadIdx.x] = 0;
		}
		i = ind_tile * blockDim.y + threadIdx.y;
		j = blockIdx.x * blockDim.x + threadIdx.x;
		if ((i < dim_interim) && (j < dim2))
		{
			op2_tile[threadIdx.y][threadIdx.x] = op2[i * dim2 + j];
		}
		else
		{
			op2_tile[threadIdx.y][threadIdx.x] = 0;
		}
		__syncthreads();
		for (int k = 0; k < SHARED_BLOCK_SIZE; k++) {
			sum += op1_tile[threadIdx.y][k] * op2_tile[k][threadIdx.x];
		}
		__syncthreads();
	}
	res[_i * dim2 + _j] = sum;
	//printf("I was here");
	//printf("%lf", res[_i * dim2 + _j]);
}


template<typename T>
class GPUArray {
private:
	size_t dim1;
	size_t dim2;
	size_t size = 0;
	T* data;

public:
	GPUArray(const CPUArray<T>& arr) {
		dim1 = arr.getDim1();
		dim2 = arr.getDim2();
		auto fillerData = arr.getData();
		if (dim2 == 0U)
			size = sizeof(T) * dim1;
		else
			size = sizeof(T) * dim1 * dim2;
		auto rc = cudaMalloc((void**)&data, size);
		rc = cudaMemcpy(data, fillerData, size, cudaMemcpyHostToDevice);
	}

	GPUArray(size_t d1, size_t d2, T* arr) {
		dim1 = d1;
		dim2 = d2;
		data = arr;
		if (dim2 == 0)
			size = dim1 * sizeof(T);
		else
			size = dim1 * dim2 * sizeof(T);
	}

	void fill(const CPUArray<T>& arr) {
		auto fillerData = arr.getData();
		auto rc = cudaMemcpy(data, fillerData, size, cudaMemcpyHostToDevice);
	}

	void fill(T* arr) {
		auto rc = cudaMemcpy(data, arr, size, cudaMemcpyHostToDevice);
	};

	void print() const {
		if (dim2 == 0) {
			T* buf = new T[dim1];
			auto rc = cudaMemcpy(buf, data, size, cudaMemcpyDeviceToHost);
			std::cout << "{ ";
			for (int i = 0; i < dim1; i++) {
				std::cout << buf[i] << " ";
			}
			std::cout << "}";
			delete[] buf;
		}
		else {
			T* buf = new T[dim1 * dim2];
			auto rc = cudaMemcpy(buf, data, size, cudaMemcpyDeviceToHost);
			std::cout << "[";
			for (int i = 0; i < dim1; i++) {
				std::cout << "[ ";
				for (int j = 0; j < dim2; j++)
				{
					std::cout << buf[i * dim2 + j] << " ";
				}
				if (i == dim1 - 1)
					std::cout << "]";
				else
					std::cout << "]\n";
			}
			std::cout << "]";
			delete[] buf;
		}
	};


	size_t getDim1() const {
		return dim1;
	}
	size_t getDim2() const {
		return dim2;
	}

	T const* getData() const {
		return data;
	}

	~GPUArray() {
		cudaFree(data);
	};

};

template<typename T>
GPUArray<T> operator*(const GPUArray<T>& op1, const GPUArray<T>& op2) {
	size_t dim1 = op1.getDim1(), dim2 = op2.getDim2(), dim_interim = op1.getDim2();
	T* data;
	auto rc = cudaMalloc((void**)&data, dim1 * dim2 * sizeof(T));
	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocks(CEILDIV(dim1, BLOCK_SIZE), CEILDIV(dim2, BLOCK_SIZE));
	T* buffer = new T[dim1 * dim2];
	cudaMemcpy(buffer, data, dim1 * dim2 * sizeof(T), cudaMemcpyDeviceToHost);
	Mat_mul << <blocks, threadsPerBlock >> > (op1.getData(), op2.getData(), data, dim1, dim2, dim_interim);
	rc = cudaDeviceSynchronize();
	cudaMemcpy(buffer, data, dim1 * dim2 * sizeof(T), cudaMemcpyDeviceToHost);

	return GPUArray<T>(dim1, dim2, data);
};

template<typename T>
GPUArray<T> operator+(const GPUArray<T>& op1, const GPUArray<T>& op2) {
	size_t dim1 = op1.getDim1(), dim2 = op2.getDim2();
	T* data;
	if (dim2 == 0) {
		auto rc = cudaMalloc((void**)&data, dim1 * sizeof(T));
		dim3 blocks(CEILDIV(dim1, SUMMATION_BLOCK_SIZE));
		int num_of_threads = 0;
		if (dim1 <= SUMMATION_BLOCK_SIZE)
			num_of_threads = dim1;
		else
			num_of_threads = SUMMATION_BLOCK_SIZE;
		dim3 threadsPerBlock(num_of_threads);
		Sum << <blocks, threadsPerBlock >> > (op1.getData(), op2.getData(), data, dim1);
		rc = cudaDeviceSynchronize();
	}
	else {
		auto rc = cudaMalloc((void**)&data, dim1 * dim2 * sizeof(T));
		dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
		dim3 blocks(CEILDIV(dim1, BLOCK_SIZE), CEILDIV(dim2, BLOCK_SIZE));
		Mat_sum << <blocks, threadsPerBlock >> > (op1.getData(), op2.getData(), data, dim1, dim2);
		rc = cudaDeviceSynchronize();
	}
	//auto rc = cudaMalloc((void**)&data, dim1 * dim2 * sizeof(T));
	//Mat_mul << <blocks, threadsPerBlock >> > (op1.getData(), op2.getData(), data, dim1, dim2, dim_interim);
	return GPUArray<T>(dim1, dim2, data);
};

template<typename T>
GPUArray<T> sharedMul(const GPUArray<T>& op1, const GPUArray<T>& op2) {
	size_t dim1 = op1.getDim1(), dim2 = op2.getDim2(), dim_interim = op1.getDim2();
	T* data;
	auto rc = cudaMalloc((void**)&data, dim1 * dim2 * sizeof(T));
	dim3 threadsPerBlock(SHARED_BLOCK_SIZE, SHARED_BLOCK_SIZE);
	dim3 blocks(CEILDIV(dim1, SHARED_BLOCK_SIZE), CEILDIV(dim2, SHARED_BLOCK_SIZE));
	T* buffer = new T[dim1 * dim2];
	cudaMemcpy(buffer, data, dim1 * dim2 * sizeof(T), cudaMemcpyDeviceToHost);
	Mat_mul_shared << <blocks, threadsPerBlock >> > (op1.getData(), op2.getData(), data, dim1, dim2, dim_interim);
	rc = cudaDeviceSynchronize();
	cudaMemcpy(buffer, data, dim1 * dim2 * sizeof(T), cudaMemcpyDeviceToHost);
	return GPUArray<T>(dim1, dim2, data);
};