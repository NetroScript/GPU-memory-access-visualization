<a name="readme-top"></a>

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MPL-2.0 License][license-shield]][license-url]

<br />
<div align="center">


<h3 align="center">GPU-memory-access-visualization</h3>

  <p align="center">
    A single header CUDA library that allows logging individual memory accesses of a GPU kernel with as little changes to the code as possible. An export to JSON together with a web-based visualization is included and allows for easy analysis of the memory access patterns.
    <br />
    <a href="https://github.com/NetroScript/GPU-memory-access-visualization/tree/master/html"><b>View Documentation for the visualization</b></a>
    ·
    <a href="https://github.com/NetroScript/GPU-memory-access-visualization/issues">Report Bug</a>
    ·
    <a href="https://github.com/NetroScript/GPU-memory-access-visualization/issues">Request Feature</a>
  </p>
</div>

<!-- TOC -->
  * [About The Project](#about-the-project)
  * [Usage](#usage)
    * [Simple Example](#simple-example)
      * [Example Kernel](#example-kernel)
      * [Creating an instance to store the memory](#creating-an-instance-to-store-the-memory)
      * [Wrapping the data arrays](#wrapping-the-data-arrays)
      * [Changing the kernel](#changing-the-kernel)
      * [Getting the data](#getting-the-data)
      * [Full example](#full-example)
    * [Gotchas](#gotchas)
      * [Passing a `CudaMemAccessLogger` pointer to a kernel](#passing-a-cudamemaccesslogger-pointer-to-a-kernel)
      * [Doing operations besides just assignment](#doing-operations-besides-just-assignment)
      * [Synchronizing the device](#synchronizing-the-device)
  * [Contributing](#contributing)
  * [License](#license)
<!-- TOC -->


## About The Project

![Application Preview](https://user-images.githubusercontent.com/18115780/218279005-7b91f1ed-f029-4e75-90d8-c6d1c5dcc3fc.png)

This repository contains a single header CUDA library that allows logging individual memory accesses of a GPU kernel with as little changes to the code as possible. **Internally the library uses CUDA Unified Memory to store the memory access information. Because of this, please make sure your targeted architecture supports this feature.** The library was tested on a GeForce RTX 2070 Ti and on a GTX 1060.

The overall design was to require as little changes to the code as possible. The concrete usage is shown in the [Usage section](#usage). 

The library takes care of storing all the memory accesses using the provided data structure (which is almost equivalent to a normal array). Besides that, the library provides functionality to store this data to the filesystem. By default, an extremely basic HTML output is provided together with a JSON output. The JSON output can be used to create a custom visualization, this repository already includes one application which can visualize this data in a browser. You can find that application in the `html` [html](https://github.com/NetroScript/GPU-memory-access-visualization/tree/master/html) together with a documentation on how to use it. For easier usage, the releases section already contains on default pre-built version of the application.

Should these data formats not be sufficient for your needs, you can easily pass in a custom callback function to the library. This callback function will be called for every memory access and can be used to store the data in any format you like. To get an idea how to use it (as it is not documented), you can take a look inside the `generateTemplatedOutput` function.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

This section shows the most basic usage of the library for use with CUDA code. If you want to know instead how to use the visualization, please take a look at the [documentation here](https://github.com/NetroScript/GPU-memory-access-visualization/tree/master/html).

For complete working examples you can also take a look at the `examples` folder. This folder contains multiple simple CUDA applications that already use the library (files ending with `_original` are the cuda files before adapting them to use the kernel). 

To use the library, just include the header file `cuda_mav.cuh` in your project. The library is a single header file and does not require any additional files to be included.

### Simple Example

#### Example Kernel

Let's assume we have the following code right now:

```cpp

__global__ void kernel(int* data, int* result) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    result[idx] = data[idx] * 2;
}

int main() {
    // Data on the host
    int* h_data = new int[100];
    int* h_result = new int[100];
    
    // Data on the device
    int* d_data;
    int* d_result;
    
    // Allocate memory on the device
    cudaMalloc(&d_data, 100 * sizeof(int));
    cudaMalloc(&d_result, 100 * sizeof(int));
    
    // Copy data to the device
    cudaMemcpy(d_data, h_data, 100 * sizeof(int), cudaMemcpyHostToDevice);
    
    // Execute the kernel
    kernel<<<10, 10>>>(d_data, d_result);
    
    // Copy data back to the host
    cudaMemcpy(h_result, d_result, 100 * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Free the device memory
    cudaFree(d_data);
    cudaFree(d_result);
    
    // Free the host memory
    delete[] h_data;
    delete[] h_result;
}
```

We can leave the old code almost entirely intact and only slightly need to change the kernel and kernel call.

#### Creating an instance to store the memory

The first thing you need to do is create a new instance of a `CudaMemAccessStorage`. This class stores all the memory access information, and you will need to pass in an expected size for all the memory accesses. This is mainly limited by your available host memory, as internally the library uses CUDA Unified Memory to store the data.

```cpp
// We are using auto with make_unique here to automatically free the memory when the scope ends
// You can also use a normal pointer and free it manually
auto memAccessStorage = std::make_unique<CudaMemAccessStorage<int>>(100*2);
```

#### Wrapping the data arrays

After that we need to wrap our data arrays in our custom class `CudaMemAccessLogger`, provide the length of the array, and a custom description / name for the visualization and provide the Logger with a reference to a `CudaMemAccessStorage` instance. This class will then intercept all accesses to the original data using a proxy. The proxy class will then forward the memory access information to the `CudaMemAccessStorage` instance.

```cpp
// Wrap the (device) data arrays in the CudaMemAccessLogger class
// Get the object itself from the smart pointer first
auto data = CudaMemAccessLogger<int>(d_data, 100, "Input data", *memAccessStorage);
auto result = CudaMemAccessLogger<int>(d_result, 100, "Result data", *memAccessStorage);
// Once again, you can also use a normal pointer here, but then you need to make sure to free the memory manually
// As the CudaMemAccessLogger class does not allocate any memory, you do not need to use a smart pointer here
```

#### Changing the kernel

Next we need to change the kernel slightly to take in the `CudaMemAccessLogger` instances instead of the original data arrays. 

```cpp
// This is all you need to change
__global__ void kernel(CudaMemAccessLogger<int> data, CudaMemAccessLogger<int> result) {
```

Additionally, we now need to change the call to the kernel to pass in the wrapped data arrays instead of the original ones.

```cpp
kernel<<<10, 10>>>(data, result);
```

#### Getting the data

Now the code continues working as expected, but  you also want to get the stored data of the accesses.

For this you can just use the `generateJSONOutput` function which you would want to use in the most cases as this produces just one JSON file which you can then drag and drop into web based visualization.

```cpp
// Get the data from the storage
// Make sure the kernel has finished executing before calling this function
memAccessStorage->generateJSONOutput("./my_data_output.json");
```

If you instead for example want to use the HTML template to directly embed the data in the HTML file already _(warning: loading then is much slower)_ the code would look like this:

```cpp
memAccessStorage->generateTemplatedOutput("./path_to_template_file.html", "./path_to_output_file.html", CudaMemAccessStorage<int>::parseDataForJSPage)
```

#### Full example

Click below to open the spoiler and see the full example code.

<details>
  <summary>Full example code</summary>
  
```cpp
#include <memory>
#include "cuda_mav.cuh"

__global__ void kernel(CudaMemAccessLogger<int> data, CudaMemAccessLogger<int> result) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    result[idx] = data[idx] * 2;
}

int main() {
    // Data on the host
    int* h_data = new int[100];
    int* h_result = new int[100];
    
    // Data on the device
    int* d_data;
    int* d_result;
    
    // Allocate memory on the device
    cudaMalloc(&d_data, 100 * sizeof(int));
    cudaMalloc(&d_result, 100 * sizeof(int));
    
    // Copy data to the device
    cudaMemcpy(d_data, h_data, 100 * sizeof(int), cudaMemcpyHostToDevice);
    
    auto memAccessStorage = std::make_unique<CudaMemAccessStorage<int>>(100*2);
    auto data = CudaMemAccessLogger<int>(d_data, 100, "Input data", *memAccessStorage);
    auto result = CudaMemAccessLogger<int>(d_result, 100, "Result data", *memAccessStorage);
    
    // Execute the kernel
    kernel<<<10, 10>>>(data, result);
    
    // Copy data back to the host
    cudaMemcpy(h_result, d_result, 100 * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Store the memory access data
    memAccessStorage->generateJSONOutput("./my_data_output.json");
    
    // Free the device memory
    cudaFree(d_data);
    cudaFree(d_result);
    
    // Free the host memory
    delete[] h_data;
    delete[] h_result;
}
```

As you can see only 4 lines of code were added, and 2 lines of code were changed. The remaining code is the same as before.

</details>


<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Gotchas

There are some things you need to watch out for when using this library.
The two main things have to do with how the submitted array is wrapped.

#### Passing a `CudaMemAccessLogger` pointer to a kernel

Assuming you pass in a pointer of a `CudaMemAccessLogger` instance, instead of the instance itself, you will need to dereference the pointer before using it, as otherwise the array operator is not called and then wrong memory is accessed.

The previously shown example code would then look like this:

```cpp
__global__ void kernel(CudaMemAccessLogger<int>* data, CudaMemAccessLogger<int>* result) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    (*result)[idx] = (*data)[idx] * 2;
}
```

#### Doing operations besides just assignment

The wrapper class only implements assigning to the templated type, or assigning to another instance of the wrapper class. 
This means you can only use the `=` operator. If you for example want to use the `++` operation, you will have to change your kernel from:

```cpp
__global__ void kernel(CudaMemAccessLogger<int> data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    data[idx]++;
}
```

to this:

```cpp
__global__ void kernel(CudaMemAccessLogger<int> data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    data[idx] = data[idx] + 1;
}
```

#### Synchronizing the device

The library does not synchronize the device after each kernel call. This means that if you want to get the data from the device, you need to synchronize the device manually. You do this either by explicitly calling `cudaDeviceSynchronize()` before using any of the `CudaMemAccessStorage` functions to output the data, or you can just place the call to for example `generateJSONOutput` below a synchronous memory operation, like `cudaMemcpy`.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## License

Distributed under the MPL-2.0 License. See `LICENSE.md` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

[contributors-shield]: https://img.shields.io/github/contributors/NetroScript/GPU-memory-access-visualization.svg?style=for-the-badge
[contributors-url]: https://github.com/NetroScript/GPU-memory-access-visualization/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/NetroScript/GPU-memory-access-visualization.svg?style=for-the-badge
[forks-url]: https://github.com/NetroScript/GPU-memory-access-visualization/network/members
[stars-shield]: https://img.shields.io/github/stars/NetroScript/GPU-memory-access-visualization.svg?style=for-the-badge
[stars-url]: https://github.com/NetroScript/GPU-memory-access-visualization/stargazers
[issues-shield]: https://img.shields.io/github/issues/NetroScript/GPU-memory-access-visualization.svg?style=for-the-badge
[issues-url]: https://github.com/NetroScript/GPU-memory-access-visualization/issues
[license-shield]: https://img.shields.io/github/license/NetroScript/GPU-memory-access-visualization.svg?style=for-the-badge
[license-url]: https://github.com/NetroScript/GPU-memory-access-visualization/blob/master/LICENSE.md