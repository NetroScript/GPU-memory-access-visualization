// The applicaiton creates a 2D matrix and initialize each element randomly with a value between 0 and 10.
// The kernel is simply decrementing each element until 0 in a very ineffective way.

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>

// The wrapper macro is required, that __LINE__ is correct pointing to the line, where the check fails
#define checkCudaError(ans)                          \
   {                                                 \
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

__global__ void decrement(unsigned int const size, unsigned int *data, unsigned int *control)
{
   int index = threadIdx.x + blockIdx.x * blockDim.x;
   int stride = blockDim.x * gridDim.x;

   for (int i = index; i < size; i += stride)
   {
      while (data[i] > 0)
      {
         data[i] = data[i] - 1;
         control[i] = control[i] + 1;
      }
   }
}

/// @brief Increment all values in a specific area by the value of increment. The maximum value of an entry is clamp to 10.
/// @param data Data to increment.
/// @param dim Dimensions of the 2D matrix.
/// @param y_start Y start coordinate of the area to increment.
/// @param x_start X start coordinate of the area to increment.
/// @param size Size of the Y and X direction of the area to increment. 
/// @param increment Value to increment.
void hot_spot(std::vector<unsigned int> &data, unsigned int const dim, unsigned int const y_start, unsigned int const x_start, unsigned int const size, unsigned int const increment)
{
   for (auto y = y_start; y < y_start + size; ++y)
   {
      for (auto x = x_start; x < x_start + size; ++x)
      {
         if (data[y * dim + x] + increment > 10)
         {
            data[y * dim + x] = 10;
         }
         else
         {
            data[y * dim + x] += increment;
         }
      }
   }
}

int main(int argc, char **argv)
{
   unsigned int dim = 100;

   std::vector<unsigned int> h_data(dim * dim);
   // create a 2D matrix where all elements are 0
   std::vector<unsigned int> h_control(dim * dim, 0);

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

   // enable me, to create a hot spot area
   // the hot spot area should looks interessting in the memory access visualization
   if (false)
   {
      hot_spot(h_data, dim, 8, 10, 10, 3);
   }

   // enable me, to print the matrix
   if (false)
   {
      for (auto y = 0; y < dim; ++y)
      {
         for (auto x = 0; x < dim; ++x)
         {
            if (h_data[y * dim + x] < 10)
            {
               std::cout << " " << h_data[y * dim + x] << " ";
            }
            else
            {
               std::cout << h_data[y * dim + x] << " ";
            }
         }
         std::cout << std::endl;
      }
   }

   unsigned int *d_data = nullptr;
   unsigned int *d_control = nullptr;

   size_t const buffer_size_byte = sizeof(unsigned int) * dim * dim;

   checkCudaError(cudaMalloc((void **)&d_data, buffer_size_byte));
   checkCudaError(cudaMalloc((void **)&d_control, buffer_size_byte));

   checkCudaError(cudaMemcpy(d_data, h_data.data(), buffer_size_byte, cudaMemcpyHostToDevice));
   // copy h_controll to initialize all values with 0 on the GPU
   checkCudaError(cudaMemcpy(d_control, h_control.data(), buffer_size_byte, cudaMemcpyHostToDevice));

   // change me and look, how the visulization looks like
   int const blockSize = 32;
   int const numBlocks = ((dim * dim) + blockSize - 1) / blockSize;

   decrement<<<numBlocks, blockSize>>>(dim * dim, d_data, d_control);
   checkCudaError(cudaGetLastError());

   checkCudaError(cudaMemcpy(h_control.data(), d_control, buffer_size_byte, cudaMemcpyDeviceToHost));

   bool success = true;

   for (auto y = 0; y < dim; ++y)
   {
      for (auto x = 0; x < dim; ++x)
      {
         if (h_control[y * dim + x] != h_data[y * dim + x])
         {
            std::cout << "h_control[" << y << ", " << x << "] != h_data[" << y << ", " << x << "]" << std::endl;
            std::cout << h_control[y * dim + x] << " != " << h_data[y * dim + x] << std::endl;
            success = false;
         }
      }
   }

   checkCudaError(cudaMemcpy(h_data.data(), d_data, buffer_size_byte, cudaMemcpyDeviceToHost));

   for (auto y = 0; y < dim; ++y)
   {
      for (auto x = 0; x < dim; ++x)
      {
         if (h_data[y * dim + x] != 0)
         {
            std::cout << "h_data[" << y << ", " << x << "] != 0" << std::endl;
            std::cout << "value is: " << h_data[y * dim + x] << std::endl;
            success = false;
         }
      }
   }

   if (success)
   {
      std::cout << "The kernel worked correctly" << std::endl;
   }

   checkCudaError(cudaFree(d_data));
   checkCudaError(cudaFree(d_control));

   return 0;
}
