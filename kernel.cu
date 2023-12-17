
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
//#include "FileName.cu"
#include "Array.h"
#include <random>
#include <chrono>
#include <vector>
#include <numeric>
void VectorTest() {
	int dim = 3;
	auto cpuVec1 = CPUArray<double>(dim);
	auto cpuVec2 = CPUArray<double>(dim);
    cpuVec1[0] = -1.135;
    cpuVec1[1] = 12.67;
    cpuVec1[2] = 41.135;
    cpuVec2[0] = 9.7;
    cpuVec2[1] = 3.;
    cpuVec2[2] = 5.81;
    std::cout << "CPU Vectors sum:" << std::endl;
    cpuVec1.print();
    std::cout << " + ";
    cpuVec2.print();
    std::cout << " = ";
	(cpuVec1 + cpuVec2).print();
    std::cout << std::endl << "GPU Vectors sum:" << std::endl;
    auto gpuVec1 = GPUArray<double>(cpuVec1);
    auto gpuVec2 = GPUArray<double>(cpuVec2);
    gpuVec1.print();
    std::cout << " + ";
    gpuVec2.print();
    std::cout << " = ";
    (gpuVec1 + gpuVec2).print();
    std::cout << std::endl;

}

template<typename T, int TYPE_NUM>
void FillMatrix(T* mat, int dim1, int dim2) {
    std::random_device rd;  // a seed source for the random number engine
    std::mt19937 gen(rd()); // mersenne_twister_engine seeded with rd()
    if (TYPE_NUM == 0) {
        std::uniform_int_distribution<> distrib(-10, 10);
        for (int i = 0; i < dim1; i++) {
            for (int j = 0; j < dim2; j++)
            {
                mat[i * dim2 + j] = distrib(gen);
            }
        }
    }
    else {
        std::uniform_real_distribution<> distrib(-10, 10);
        for (int i = 0; i < dim1; i++) {
            for (int j = 0; j < dim2; j++)
            {
                mat[i * dim2 + j] = distrib(gen);
            }
        }

    }

}

template<typename T, int TYPE_NUM>
void CompareMethods(int dim = 100, int reps = 1) {
    T* A_raw = new T[dim * dim];
    T* B_raw = new T[dim * dim];
    auto A = CPUArray<T>(dim, dim);
    auto B = CPUArray<T>(dim, dim);
    A.setValue((T)0);
    B.setValue((T)0);
    auto A_gpu = GPUArray<T>(A);
    auto B_gpu = GPUArray<T>(B);
    long long naiveDur = 0;
    long long GPUDur = 0;
    long long GPUSharedDur = 0;
    auto begin = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();

    std::vector<long long> naiveDurs;
    std::vector<long long> GPUDurs;
    std::vector<long long> GPUSharedDurs;
    for (int i = 0; i < reps; i++) {
        FillMatrix<T, TYPE_NUM>(A_raw, dim, dim);
        FillMatrix<T, TYPE_NUM>(B_raw, dim, dim);
        A.fill(A_raw);
        B.fill(B_raw);
        A_gpu.fill(A_raw);
        B_gpu.fill(B_raw);

        if (i == 0) {
            begin = std::chrono::high_resolution_clock::now();
            auto C = A * B;
            end = std::chrono::high_resolution_clock::now();
            naiveDur = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
            naiveDurs.push_back(naiveDur);
        }
        begin = std::chrono::high_resolution_clock::now();
        auto C_gpu = A_gpu * B_gpu;
        end = std::chrono::high_resolution_clock::now();
        GPUDur = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
        GPUDurs.push_back(GPUDur);

        begin = std::chrono::high_resolution_clock::now();
        auto C_gpu_shared = sharedMul(A_gpu, B_gpu);
        end = std::chrono::high_resolution_clock::now();
        GPUSharedDur = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
        GPUSharedDurs.push_back(GPUSharedDur);
        //std::cout << C_gpu(0, 0) << C_gpu(3, 5) << C_gpu_shared(0, 0) << C_gpu_shared(3, 5);
    }
    long long sum = std::accumulate(naiveDurs.begin(), naiveDurs.end(), 0.0);
    double mean = sum / naiveDurs.size();
    std::cout << sum << "ns total, average : " << mean / reps << "ns." << std::endl;
    for (int i = 0; i < naiveDurs.size(); i++)
        std::cout << naiveDurs[i] << std::endl;
    std::cout << std::endl;
    sum = std::accumulate(GPUDurs.begin(), GPUDurs.end(), 0.0);
    mean = sum / GPUDurs.size();
    std::cout << sum << "ns total, average : " << mean / reps << "ns." << std::endl;
    for (int i = 0; i < GPUDurs.size(); i++)
        std::cout << GPUDurs[i] << std::endl;
    std::cout << std::endl;
    sum = std::accumulate(GPUSharedDurs.begin(), GPUSharedDurs.end(), 0.0);
    mean = sum / GPUSharedDurs.size();
    std::cout << sum << "ns total, average : " << mean / reps << "ns." << std::endl;
    for (int i = 0; i < GPUSharedDurs.size(); i++)
        std::cout << GPUSharedDurs[i] << std::endl;
    std::cout << std::endl;
}



int main() {
    VectorTest();
    CompareMethods<int, 0>(500, 100);
	return 0;
}
