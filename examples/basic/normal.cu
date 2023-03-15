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


__global__ void kernel(int prob_size, CudaMemAccessLogger<int>* input, CudaMemAccessLogger<int>* output){
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < prob_size) {
        (*output)[id] = (*input)[id];
    }

}

int main(){
    constexpr int prob_size = 1000;
    
    std::vector<int> h_input(prob_size);
    std::iota(h_input.begin(), h_input.end(), 0);
    int * d_input = nullptr;
    checkCudaError(cudaMalloc((void**) &d_input, sizeof(int)*prob_size));

    std::vector<int> h_output(prob_size, 0);
    int * d_output = nullptr;
    checkCudaError(cudaMalloc((void**) &d_output, sizeof(int)*prob_size));

    checkCudaError(cudaMemcpy(d_input, h_input.data(), sizeof(int)* prob_size, cudaMemcpyHostToDevice));

    // The overloaded new operator generates a managed memory object
    CudaMemAccessStorage<int>* memAccessStorage = new CudaMemAccessStorage<int>(10000);

    // The overloaded new operator generates a managed memory object
    CudaMemAccessLogger<int>* input = new CudaMemAccessLogger<int>(d_input, prob_size, "Input Datastructure", *memAccessStorage);
    CudaMemAccessLogger<int>* output = new CudaMemAccessLogger<int>(d_output, prob_size, "Output Datastructure", *memAccessStorage);

    constexpr int threads = 32;
    constexpr int blocks = (prob_size/threads)+1;

    //kernel<<<blocks, threads>>>(prob_size, input.getDevicePointer(), output.getDevicePointer());
    kernel <<<blocks, threads >>> (prob_size, input, output);
    checkCudaError(cudaGetLastError());
    cudaDeviceSynchronize();

    memAccessStorage->generateTemplatedOutput("../../../templates/basic_template.html", "../../../out/basic_html.html",
                                              CudaMemAccessStorage<int>::parseDataForStaticHTML);

    checkCudaError(cudaMemcpy(h_output.data(), d_output, sizeof(int)*prob_size, cudaMemcpyDeviceToHost));

    for(auto i = 0; i < h_input.size(); ++i){
        if(h_input[i] != h_output[i]){
            std::cerr << "Element at position " << i << "is not equal (input - output): " << h_input[i] << " != " << h_output[i] << std::endl;
            std::exit(1); 
        }
    }

    checkCudaError(cudaFree(d_input));
    checkCudaError(cudaFree(d_output));

    // Free up the managed memory objects
    delete memAccessStorage;
    delete input;
    delete output;

    std::cout << "kernel finished successful" << std::endl;
    return 0;
}
