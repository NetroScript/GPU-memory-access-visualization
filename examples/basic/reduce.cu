#include <iostream>
#include <random>
#include <vector>
#include <algorithm>

// The wrapper macro is required, that __LINE__ is correct pointing to the line, where the check fails
#define checkCudaError(ans)                            \
    {                                                  \
        checkCudaErrorFunc((ans), __FILE__, __LINE__); \
    }

inline void checkCudaErrorFunc(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess)
    {
        std::cout << "\r" << file << ":" << line << " -> Cuda Error " << err << ": " << cudaGetErrorString(err) << std::endl;
        std::cout << "Aborting..." << std::endl;
        exit(0);
    }
}

// The reduction algorithms divide all elements in logical blocks with the size of threads.
// Each local block is reduced to a single element.
// A grid stride loop maps the logical blocks to cuda blocks (both has the same size).
// The output array has the size of the number of logical blocks.
__global__ void reduce_gm(unsigned int const size, unsigned int *const input, unsigned int *const output)
{
    int const id = threadIdx.x + blockIdx.x * blockDim.x;
    int const stride = blockDim.x * gridDim.x;
    // use grid stride loop to distribute the logical blocks to cuda blocks.
    for (int block_offset_id = id, virtual_block_id = blockIdx.x; block_offset_id < size; block_offset_id += stride, virtual_block_id += gridDim.x)
    {
        // reduce all elements of logical block to a single element.
        for (int max_threads_blocks = blockDim.x / 2; max_threads_blocks > 0; max_threads_blocks /= 2)
        {
            if (threadIdx.x < max_threads_blocks)
            {
                input[block_offset_id] += input[block_offset_id + max_threads_blocks];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0)
        {
            // write single element to output
            output[virtual_block_id] = input[block_offset_id];
        }
        __syncthreads();
    }
}

// Helper function -> should be replaced by html visualization ;-) 
template <typename T>
void print_vec(std::vector<T> vec)
{
    for (auto const v : vec)
    {
        std::cout << v << " ";
    }
    std::cout << std::endl;
}

int main(int argc, char **argv)
{
    int const blocks = 10;
    int const threads = 32;

    // number of input elements
    unsigned int const size = 1000;
    size_t const data_size_byte = sizeof(unsigned int) * size;

    // number of logical blocks
    size_t output_elements = size / threads;
    // add an extra element, if logical blocks does not fit in cuda blocks 
    output_elements += (size % threads == 0) ? 0 : 1;
    size_t const output_size_byte = sizeof(unsigned int) * output_elements;

    std::vector<unsigned int> h_data(size);
    std::vector<unsigned int> h_output(output_elements, 0);

    // initialize data matrix with random numbers betweem 0 and 10
    std::uniform_int_distribution<unsigned int> distribution(
        0,
        10);
    std::default_random_engine generator;
    std::generate(
        h_data.begin(),
        h_data.end(),
        [&distribution, &generator]()
        { return distribution(generator); });

    // calculate result for verification
    unsigned int const expected_result = std::reduce(h_data.begin(), h_data.end());

    unsigned int *d_data = nullptr;
    unsigned int *d_output = nullptr;

    checkCudaError(cudaMalloc((void **)&d_data, data_size_byte));
    checkCudaError(cudaMalloc((void **)&d_output, output_size_byte));
    checkCudaError(cudaMemcpy(d_data, h_data.data(), data_size_byte, cudaMemcpyHostToDevice));

    reduce_gm<<<blocks, threads>>>(size, d_data, d_output);
    checkCudaError(cudaGetLastError());

    checkCudaError(cudaMemcpy(h_output.data(), d_output, output_size_byte, cudaMemcpyDeviceToHost));

    unsigned int sum = 0;

    // Reduce all sums of the logical blocks on CPU.
    // Otherwise a second kernel or cuda cooperative groups are required to performe block synchronization.   
    for (unsigned int const v : h_output)
    {
        sum += v;
    }

    if (sum == expected_result)
    {
        std::cout << "reduction kernel works correctly" << std::endl;
    }
    else
    {
        std::cout << "sum: " << sum << std::endl;
        std::cout << "expected result: " << expected_result << std::endl;
    }

    return 0;
}
