#pragma once

#include <stdexcept>
#include <cstdio>
#include <fstream>
#include <functional>
#include <sstream>
#include <vector>
#include <iostream>

// Define a custom template class which holds the data for the CUDA kernel
template <typename T>
class CudaMav
{
    struct GlobalSettings {
        int gridDimX;
        int gridDimY;
        int gridDimZ;
        int blockDimX;
        int blockDimY;
        int blockDimZ;
        int warpSize;
        unsigned int originalSize_read;
        unsigned int currentSize_read;
        unsigned int originalSize_write;
        unsigned int currentSize_write;

    };

    // Have a struct to store logging data
    struct MemoryAccessLog {
        // Store the address which was addressed
        T* address;

        // Store the thread id which accessed the address
        unsigned int threadId;
        // Store the block id which accessed the address
        unsigned int blockId;

        // Constructor which decomposes the block and thread id into the packed long
        __host__ __device__ MemoryAccessLog(T* address, int blockId, int threadId) : address(address), threadId(threadId), blockId(blockId) {}

        // Empty constructor
        __host__ __device__ MemoryAccessLog() : address(nullptr), threadId(0), blockId(0) {}
    };

private:
    // Have a array of ints on the device used to store grid dimensions, and block dimensions, and the warp size
    // This array will always have 10 elements (x,y,z) for grid dimensions, (x,y,z) for block dimensions, and 1 for warp size, 1 for status messages, 1 for the original log size, 1 for the current log size
    GlobalSettings* d_constantData;
    // Have the same array on the host
    GlobalSettings* h_constantData;

    // Have an internal pointer to the data, this pointer is a device pointer
    T* d_data;

    // Also have an instance of this class allocated on the device
    CudaMav<T>* d_this;

    // Store if memory was fetched from the device
    bool fetchedFromDevice = false;

    // Have a pointer to a list of memory access logs for reading
    MemoryAccessLog* d_memoryAccessLog_reading = nullptr;
    // Have a pointer to a list of memory access logs on the host
    MemoryAccessLog* h_memoryAccessLog_reading = nullptr;

    // Have a pointer to a list of memory access logs for writing
    MemoryAccessLog* d_memoryAccessLog_writing = nullptr;
    // Have a pointer to a list of memory access logs on the host
    MemoryAccessLog* h_memoryAccessLog_writing = nullptr;


    // Implement a proxy class so we can both read and write from the array when accessing the array operator
    class AccessProxy {
        // Have a reference to the CudaMav class
        CudaMav<T>* cudaMav;
        // Have a reference to the index
        int index;

    public:
        // Constructor which takes a reference to the CudaMav class and the index
        __device__ AccessProxy(CudaMav<T>* cudaMav, int index) : cudaMav(cudaMav), index(index) {}
        AccessProxy() = delete;

        // Overload the assignment operator so we can write to the array
        __device__ AccessProxy &operator = (const T &value) {
            cudaMav->set(index, value);
            return *this;
        }

        // When accessing the array, and also assign a value to the access, we assign AccessProxy to AccessProxy
        // For this reason we need to define the assignment operator for AccessProxy, so that the actual values get changed
        __device__ AccessProxy &operator = (const AccessProxy &other) {
            if (this != &other) {
                cudaMav->set(index, other.cudaMav->get(other.index));
            }
            return *this;
        }

        // Overload the cast operator, so we can read from the array
        // Leaving the explicit out, won't throw an error, but might result in unexpected behaviour
        __device__ /*explicit*/ operator T() const {
            return cudaMav->get(index);
        }
    };

    // Helper function to format and throw CUDA errors
    void checkCudaError(cudaError_t err, std::string const& message = "Cuda Error.") {
        if (err != cudaSuccess) {
            const char* errorString = cudaGetErrorString(err);
            throw std::runtime_error(message + "\n Error: \n" + std::string(errorString));
        }
    }

    // Define function to load back data
    void fetchData()
    {
        // Check if data was fetched from the device
        if (fetchedFromDevice) {
            // If so, return
            return;
        }

        // First fetch the h_constantData from the device
        checkCudaError(cudaMemcpy(h_constantData, d_constantData, sizeof(GlobalSettings), cudaMemcpyDeviceToHost), "Could not copy constant data from device.");

        // Copy the data back from the device for reading
        checkCudaError(cudaMemcpy(h_memoryAccessLog_reading, d_memoryAccessLog_reading, sizeof(MemoryAccessLog) * h_constantData->originalSize_read, cudaMemcpyDeviceToHost), "Could not copy memory access logs from device.");
        // Copy the data back from the device for writing
        checkCudaError(cudaMemcpy(h_memoryAccessLog_writing, d_memoryAccessLog_writing, sizeof(MemoryAccessLog) * h_constantData->originalSize_write, cudaMemcpyDeviceToHost), "Could not copy memory access logs from device.");


        // Free up the memory on the device
        checkCudaError(cudaFree(d_constantData), "Could not free constant data on device.");
        checkCudaError(cudaFree(d_memoryAccessLog_reading), "Could not free memory access logs (reading) on device.");
        checkCudaError(cudaFree(d_memoryAccessLog_writing), "Could not free memory access logs (writing) on device.");
        checkCudaError(cudaFree(d_this), "Could not free class instance pointer on device.");

        // Set the fetched from device flag to true
        fetchedFromDevice = true;
    }

    // Define the default function for basic HTML processing
    // The first element in the tuple is the HTML code, the second the JS code
    // For this function the second will be empty
    std::tuple<std::string, std::string> parseDataForStaticHTML(GlobalSettings settingsStruct, std::vector<MemoryAccessLog> readLogs, std::vector<MemoryAccessLog> writeLogs) {

        std::stringstream htmlStream;

        // Add section for global settings
        htmlStream << "<h1>Kernel Settings</h2>" << std::endl;

        // Add section for grid dimensions
        htmlStream << "<h3>Grid Dimensions</h3>" << std::endl;
        htmlStream << "<p>Grid Dimensions X: " << settingsStruct.gridDimX << "</p>" << std::endl;
        htmlStream << "<p>Grid Dimensions Y: " << settingsStruct.gridDimY << "</p>" << std::endl;
        htmlStream << "<p>Grid Dimensions Z: " << settingsStruct.gridDimZ << "</p>" << std::endl;

        // Add section for block dimensions
        htmlStream << "<h3>Block Dimensions</h3>" << std::endl;
        htmlStream << "<p>Block Dimensions X: " << settingsStruct.blockDimX << "</p>" << std::endl;
        htmlStream << "<p>Block Dimensions Y: " << settingsStruct.blockDimY << "</p>" << std::endl;
        htmlStream << "<p>Block Dimensions Z: " << settingsStruct.blockDimZ << "</p>" << std::endl;

        // Add section for warp size
        htmlStream << "<h3>Warp Size</h3>" << std::endl;
        htmlStream << "<p>Warp Size: " << settingsStruct.warpSize << "</p>" << std::endl;



        // Add the section for the read logs
        htmlStream << "<h2>Memory Accesses (Reading)</h2>" << std::endl;

        // Add the table for the read logs
        htmlStream << "<table>" << std::endl;
        htmlStream << "<tr>" << std::endl;
        htmlStream << "<th>Address</th>" << std::endl;
        htmlStream << "<th>Block Id</th>" << std::endl;
        htmlStream << "<th>Thread Id</th>" << std::endl;
        htmlStream << "</tr>" << std::endl;

        // Loop through the read logs
        for (auto& log : readLogs) {
            htmlStream << "<tr>" << std::endl;
            htmlStream << "<td>" << log.address << "</td>" << std::endl;
            htmlStream << "<td>" << log.blockId << "</td>" << std::endl;
            htmlStream << "<td>" << log.threadId << "</td>" << std::endl;
            htmlStream << "</tr>" << std::endl;
        }

        htmlStream << "</table>" << std::endl;

        // Add the section for the write logs

        htmlStream << "<h2>Memory Accesses (Writing)</h2>" << std::endl;

        // Add the table for the write logs
        htmlStream << "<table>" << std::endl;
        htmlStream << "<tr>" << std::endl;
        htmlStream << "<th>Address</th>" << std::endl;
        htmlStream << "<th>Block Id</th>" << std::endl;
        htmlStream << "<th>Thread Id</th>" << std::endl;
        htmlStream << "</tr>" << std::endl;

        // Loop through the write logs
        for (auto& log : writeLogs) {
            htmlStream << "<tr>" << std::endl;
            htmlStream << "<td>" << log.address << "</td>" << std::endl;
            htmlStream << "<td>" << log.blockId << "</td>" << std::endl;
            htmlStream << "<td>" << log.threadId << "</td>" << std::endl;
            htmlStream << "</tr>" << std::endl;
        }

        htmlStream << "</table>" << std::endl;

        // Return the HTML code
        return std::make_tuple(htmlStream.str(), "");
    }

    // Define the default function for basic HTML processing
    // The first element in the tuple is the HTML code, the second the JS code
    // For this function the first will be empty
    std::tuple<std::string, std::string> parseDataForJSPage(GlobalSettings settingsStruct, std::vector<MemoryAccessLog> readLogs, std::vector<MemoryAccessLog> writeLogs) {

        std::stringstream jsStream;

        return std::make_tuple("", jsStream.str());
    }

public:

    // Constructor to create an empty class
    __device__ __host__ CudaMav() {
        // Set the data pointer to null
        d_data = nullptr;
        // Set the memory access log pointer to null for reading
        d_memoryAccessLog_reading = nullptr;
        // Set the memory access log pointer to null for writing
        d_memoryAccessLog_writing = nullptr;
    }

    // Constructor which allocates the memory on the device
    __host__ CudaMav(T* array_data, unsigned int size = 100000)
    {

        h_constantData = new GlobalSettings{ -1, -1, -1, -1, -1, -1, -1, size, 0, size, 0};

        // Allocate the memory on the device for the d_constantData and check if it was successful
        checkCudaError(cudaMalloc(&d_constantData, sizeof(GlobalSettings)), "Could not allocate array to store kernel data on device.");
        // Copy over the host data to the device
        checkCudaError(cudaMemcpy(d_constantData, h_constantData, sizeof(GlobalSettings), cudaMemcpyHostToDevice), "Could not copy constant data to device.");

        // Store the passed data pointer within the class
        d_data = array_data;

        // Allocate the memory on the device for the d_memoryAccessLog_reading and check if it was successful
        checkCudaError(cudaMalloc(&d_memoryAccessLog_reading, sizeof(MemoryAccessLog) * size), "Could not allocate array to store memory access logs on device. (reading)");
        // Also allocate the memory on the host for the h_memoryAccessLog_reading and check if it was successful
        h_memoryAccessLog_reading = new MemoryAccessLog[size];

        // Copy the empty data to the device
        checkCudaError(cudaMemcpy(d_memoryAccessLog_reading, h_memoryAccessLog_reading, sizeof(MemoryAccessLog) * size, cudaMemcpyHostToDevice), "Could not copy memory access logs to device. (reading)");

        // To the same for writing
        checkCudaError(cudaMalloc(&d_memoryAccessLog_writing, sizeof(MemoryAccessLog) * size), "Could not allocate array to store memory access logs on device. (writing)");
        h_memoryAccessLog_writing = new MemoryAccessLog[size];
        checkCudaError(cudaMemcpy(d_memoryAccessLog_writing, h_memoryAccessLog_writing, sizeof(MemoryAccessLog) * size, cudaMemcpyHostToDevice), "Could not copy memory access logs to device. (writing)");

        // Now we finished initializing the class, so we need to create the copy of this class on the device
        // Allocate the memory on the device for the d_this and check if it was successful
        checkCudaError(cudaMalloc(&d_this, sizeof(CudaMav<T>)), "Could not allocate array to store this class on device.");
        // Copy the empty data to the device
        checkCudaError(cudaMemcpy(d_this, this, sizeof(CudaMav<T>), cudaMemcpyHostToDevice), "Could not copy this class to device.");

    }

    __device__ T get(unsigned int index) {
        // Atomically increase the currentSize by 1
        int current_index = atomicAdd(&d_constantData->currentSize_read, 1);

        // First check if the currentSize is zero, if so we need to initialize the additional data variables, needed later to restore the data
        if (current_index == 0) {
            // Store the grid dimensions
            d_constantData->gridDimX = gridDim.x;
            d_constantData->gridDimY = gridDim.y;
            d_constantData->gridDimZ = gridDim.z;
            // Store the block dimensions
            d_constantData->blockDimX = blockDim.x;
            d_constantData->blockDimY = blockDim.y;
            d_constantData->blockDimZ = blockDim.z;
            // Store the warp size
            d_constantData->warpSize = warpSize;
        }

        // Get the block and thread id
        unsigned int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
        unsigned int threadId = threadIdx.x + threadIdx.y * blockDim.x + blockDim.x * blockDim.y * threadIdx.z;

        // Get the address of the data
        T* address = &d_data[index];

        // Check that our current index is less than the original size
        if (current_index < d_constantData->originalSize_read) {
            // Store the data in the memory access log
            d_memoryAccessLog_reading[current_index] = MemoryAccessLog(address, blockId, threadId);
        }


        // Print the accessed data
        //printf("Accessed data at address %p by thread %d in block %d\n", &d_data[index], threadId, blockId);

        return static_cast<T>(d_data[index]);
    }

    __device__ void set(unsigned int index, T value) {

        //printf("Writing to index %d \n", index);

        // Atomically increase the currentSize by 1
        int current_index = atomicAdd(&d_constantData->currentSize_write, 1);

        // Write the value to the data
        d_data[index] = value;


        // First check if the currentSize is zero, if so we need to initialize the additional data variables, needed later to restore the data
        if (current_index == 0) {
            // Store the grid dimensions
            d_constantData->gridDimX = gridDim.x;
            d_constantData->gridDimY = gridDim.y;
            d_constantData->gridDimZ = gridDim.z;
            // Store the block dimensions
            d_constantData->blockDimX = blockDim.x;
            d_constantData->blockDimY = blockDim.y;
            d_constantData->blockDimZ = blockDim.z;
            // Store the warp size
            d_constantData->warpSize = warpSize;
        }

        // Get the block and thread id
        unsigned int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
        unsigned int threadId = threadIdx.x + threadIdx.y * blockDim.x + blockDim.x * blockDim.y * threadIdx.z;

        // Get the address of the data
        T* address = &d_data[index];

        // Check that our current index is less than the original size
        if (current_index < d_constantData->originalSize_write) {
            // Store the data in the memory access log
            d_memoryAccessLog_writing[current_index] = MemoryAccessLog(address, blockId, threadId);
        }

        // Print the accessed data
        //printf("Wrote data at address %p by thread %d in block %d\n", &d_data[index], threadId, blockId);
    }


    // Destructor to free the memory on the device
    __device__ __host__ ~CudaMav()
    {
        // Free the memory on the host
        delete[] h_memoryAccessLog_reading;
        delete[] h_memoryAccessLog_writing;
        delete h_constantData;
    }

    // Array operator overload on the device
    __device__ AccessProxy operator[](size_t i) {
        return AccessProxy(this, i);
    }

    __device__ AccessProxy operator[](size_t i) const {
        return AccessProxy(this, i);
    }



    /*
    // Also define the const version of the array operator overload on the device
    __device__ const T& operator[](size_t i) const {
        return AccessProxy(*this, i);
    }
     */

    __device__ __host__ void PrintPointer() {
#ifdef  __CUDA_ARCH__
        printf("Pointer within Class (CUDA): %p\n", d_data);
#else
        printf("Pointer within Class (CPU): %p\n", d_data);
#endif
    }

    __host__ CudaMav<T>* getDevicePointer() {
        return d_this;
    }

    // Function to analyze the data which also frees up all used memory
    // As third parameter a function lambda can be passed which is called for each memory access
    void analyze(const std::string template_file, const std::string& output_path, std::function<std::tuple<std::string, std::string>(GlobalSettings settingsStruct, std::vector<MemoryAccessLog> readLogs, std::vector<MemoryAccessLog> writeLogs)> customGenerationFunction = nullptr)
    {
        // Fetch the data back from the device
        fetchData();

        // Data processing code here

        // Load the template file and check for errors
        std::ifstream file(template_file);
        if (!file) {
            std::cout << "Could not open template file at " << template_file << std::endl;
            return;
        }

        // Load the template file into a string
        std::string template_file_string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

        // Replace the placeholders in the template file with the actual data
        std::tuple<std::string, std::string> placeholderReplacement;

        // Generate a vector for the read and write logs
        std::vector<MemoryAccessLog> readLogs;
        std::vector<MemoryAccessLog> writeLogs;

        // Loop over the read logs
        for (int i = 0; i < h_constantData->currentSize_read; i++) {
            // Add the log to the vector
            readLogs.push_back(h_memoryAccessLog_reading[i]);
        }

        // Loop over the write logs
        for (int i = 0; i < h_constantData->currentSize_write; i++) {
            // Add the log to the vector
            writeLogs.push_back(h_memoryAccessLog_writing[i]);
        }

        // Run the custom generation function, if one was passed
        if (customGenerationFunction != nullptr) {
            placeholderReplacement = customGenerationFunction(*h_constantData, readLogs, writeLogs);
        }
        // If none was passed, use a default function
        else {
            placeholderReplacement = parseDataForStaticHTML(*h_constantData, readLogs, writeLogs);
        }

        // Replace "<!-- HTML_TEMPLATE -->" with the HTML template
        std::string::size_type pos = template_file_string.find("<!-- HTML_TEMPLATE -->");
        template_file_string.replace(pos, 22, std::get<0>(placeholderReplacement));

        // Replace "<!-- JS_TEMPLATE -->" with the JS template
        pos = template_file_string.find("// JS_TEMPLATE");
        template_file_string.replace(pos, 14, std::get<1>(placeholderReplacement));


        // Write the data to the output file
        std::ofstream output_file(output_path);
        if (!output_file) {
            std::cout << "Could not open output file at " << output_path << std::endl;
            return;
        }
        output_file << template_file_string;

        // Close the output file
        output_file.close();

    }

    // Get the GlobalSettings data
    GlobalSettings getGlobalSettings()
    {
        // Fetch the data back from the device
        fetchData();

        return *h_constantData;
    }

};
