#include <vector>
#include <numeric>
#include <iostream>
#include <fstream>

inline void checkCudaError(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "\rCuda Error " << err << ": " << cudaGetErrorString(err) << std::endl;
        std::cerr << "Aborting..." << std::endl;
        exit(1);
    }
}

struct MemAccessData {
    int id = 0;
};

__device__ int profile_access(int id, MemAccessData * mem_access){
    mem_access[id].id = id;
    return id;
}

// int * const
// mem_access<int * const>
__global__ void kernel(int prob_size, int * const input, int * output, MemAccessData * mem_access){
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if(id < prob_size){
        // output[id] = input[id];
        output[id] = input[profile_access(id, mem_access)];
    }
}

// for 1D and 2D: common image format (in best case without extra library)
// or HTML
void visualize(std::vector<MemAccessData> const & mem_accs){
    std::ofstream fs("visu.txt");
    fs << "data\n";
    fs.close();
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

    std::vector<MemAccessData> h_mem_access(prob_size);
    MemAccessData * d_mem_access = nullptr;
    checkCudaError(cudaMalloc((void**) &d_mem_access, sizeof(MemAccessData)*prob_size));

    constexpr int threads = 32;
    constexpr int blocks = (prob_size/threads)+1;

    kernel<<<blocks, threads>>>(prob_size, d_input, d_output, d_mem_access);
    checkCudaError(cudaGetLastError());

    checkCudaError(cudaMemcpy(h_output.data(), d_output, sizeof(int)*prob_size, cudaMemcpyDeviceToHost));
    checkCudaError(cudaMemcpy(h_mem_access.data(), d_mem_access, sizeof(MemAccessData)*prob_size, cudaMemcpyDeviceToHost));


    for(auto i = 0; i < h_input.size(); ++i){
        if(h_input[i] != h_output[i]){
            std::cerr << "Element at position " << i << "is not equal (input - output): " << h_input[i] << " != " << h_output[i] << std::endl;
            std::exit(1); 
        }
    }

    visualize(h_mem_access);

    checkCudaError(cudaFree(d_input));
    checkCudaError(cudaFree(d_output));
    checkCudaError(cudaFree(d_mem_access));

    std::cout << "kernel finished successful" << std::endl;
    return 0;
}
