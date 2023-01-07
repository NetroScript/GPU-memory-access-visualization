#include <vector>
#include <numeric>
#include <iostream>
#include "cuda_mav.cuh"

inline void checkCudaError(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "\rCuda Error " << err << ": " << cudaGetErrorString(err) << std::endl;
        std::cerr << "Aborting..." << std::endl;
        exit(1);
    }
}


__global__ void kernel(int prob_size, CudaMav<int> * input, CudaMav<int> * output){
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < prob_size) {
        (*output)[id] = (*input)[id];
    }

}

int main(){
    constexpr int prob_size = 100;
    
    std::vector<int> h_input(prob_size);
    std::iota(h_input.begin(), h_input.end(), 0);
    int * d_input = nullptr;
    checkCudaError(cudaMalloc((void**) &d_input, sizeof(int)*prob_size));

    std::vector<int> h_output(prob_size, 0);
    int * d_output = nullptr;
    checkCudaError(cudaMalloc((void**) &d_output, sizeof(int)*prob_size));

    checkCudaError(cudaMemcpy(d_input, h_input.data(), sizeof(int)* prob_size, cudaMemcpyHostToDevice));

    CudaMav<int> input(d_input);
    CudaMav<int> output(d_output);

    constexpr int threads = 32;
    constexpr int blocks = (prob_size/threads)+1;

    kernel<<<blocks, threads>>>(prob_size, input.getDevicePointer(), output.getDevicePointer());
    checkCudaError(cudaGetLastError());
    cudaDeviceSynchronize();

    auto data = input.getGlobalSettings();

    input.analyze("../../../html/basic_template.html", "../../../out/basic_input.html");
    output.analyze("../../../html/basic_template.html", "../../../out/basic_output.html");

    checkCudaError(cudaMemcpy(h_output.data(), d_output, sizeof(int)*prob_size, cudaMemcpyDeviceToHost));

    for(auto i = 0; i < h_input.size(); ++i){
        if(h_input[i] != h_output[i]){
            std::cerr << "Element at position " << i << "is not equal (input - output): " << h_input[i] << " != " << h_output[i] << std::endl;
            std::exit(1); 
        }
    }

    checkCudaError(cudaFree(d_input));
    checkCudaError(cudaFree(d_output));

    std::cout << "kernel finished successful" << std::endl;
    return 0;
}
