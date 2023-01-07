#pragma once

#include <stdexcept>
#include <cstdio>
#include <fstream>
#include <functional>
#include <sstream>
#include <vector>
#include <iostream>

// Define a custom template class, which takes care of storing the memory accesses of a given type.
template<typename T>
class CudaMemAccessStorage {

    struct GlobalSettings {
        int gridDimX;
        int gridDimY;
        int gridDimZ;
        int blockDimX;
        int blockDimY;
        int blockDimZ;
        int warpSize;
        unsigned int originalSize;
        unsigned int currentSize;
    };

    // Have a struct to store logging data
    struct MemoryAccessLog {

    private:

        // Store the address which was addressed
        T *address;
        // Store the thread id which accessed the address, additionally the uppermost bit is used to store if the access was a read or write, 0 for read, 1 for write
        unsigned int threadId_accessType;
        // Store the block id which accessed the address
        unsigned int blockId;

    public:

        // Constructor which decomposes the block and thread id into the packed long
        __host__ __device__ MemoryAccessLog(T *address, unsigned int blockId, unsigned int threadId, bool read = true)
                : address(address), threadId_accessType(threadId), blockId(blockId) {
            // Set the uppermost bit to 1 if the access was a write, 0 if it was a read
            if (!read) {
                threadId_accessType |= static_cast<unsigned int>(1 << 31);
            } else {
                threadId_accessType &= static_cast<unsigned int>(~(1 << 31));
            }
        }

        // Empty constructor
        __host__ __device__ MemoryAccessLog() : address(nullptr), threadId_accessType(0), blockId(0) {}

        // Getter for the address
        __host__ T *Address() const {
            return address;
        }

        // Getter for the thread id
        __host__ int ThreadId() const {
            return threadId_accessType & ~(1 << 31);
        }

        // Getter for the block id
        __host__ int BlockId() const {
            return blockId;
        }

        // Getter for the access type
        __host__ bool IsRead() const {
            return (threadId_accessType & static_cast<unsigned int>(1 << 31)) == 0;
        }
    };

private:

    // Have a array of ints on the device used to store grid dimensions, and block dimensions, and the warp size
    // This array will always have 10 elements (x,y,z) for grid dimensions, (x,y,z) for block dimensions, and 1 for warp size, 1 for status messages, 1 for the original log size, 1 for the current log size
    GlobalSettings *d_constantData = nullptr;
    // Have the same array on the host
    GlobalSettings *h_constantData = nullptr;

    // Have a pointer to a list of memory access logs
    MemoryAccessLog *d_memoryAccessLog = nullptr;
    // Have a pointer to a list of memory access logs on the host
    MemoryAccessLog *h_memoryAccessLog = nullptr;


    // Store if memory was fetched from the device
    bool fetchedFromDevice = false;

    // Helper function to format and throw CUDA errors
    void checkCudaError(cudaError_t err, std::string const &message = "Cuda Error.") {
        if (err != cudaSuccess) {
            const char *errorString = cudaGetErrorString(err);
            throw std::runtime_error(message + "\n Error: \n" + std::string(errorString));
        }
    }

    // Define the default function for basic HTML processing
    // The first element in the tuple is the HTML code, the second the JS code
    // For this function the second will be empty
    std::tuple<std::string, std::string>
    parseDataForStaticHTML(GlobalSettings settingsStruct, std::vector<MemoryAccessLog> accessLogs) {

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
        htmlStream << "<h2>Memory Accesses</h2>" << std::endl;

        // Add the table for the read logs
        htmlStream << "<table>" << std::endl;
        htmlStream << "<tr>" << std::endl;
        htmlStream << "<th>Address</th>" << std::endl;
        htmlStream << "<th>Block Id</th>" << std::endl;
        htmlStream << "<th>Thread Id</th>" << std::endl;
        htmlStream << "<th>Access Type</th>" << std::endl;
        htmlStream << "</tr>" << std::endl;

        // Loop through the read logs
        for (auto &log: accessLogs) {
            htmlStream << "<tr>" << std::endl;
            htmlStream << "<td>" << log.Address() << "</td>" << std::endl;
            htmlStream << "<td>" << log.BlockId() << "</td>" << std::endl;
            htmlStream << "<td>" << log.ThreadId() << "</td>" << std::endl;
            htmlStream << "<td>" << (log.IsRead() ? "Read" : "Write") << "</td>" << std::endl;
            htmlStream << "</tr>" << std::endl;
        }

        htmlStream << "</table>" << std::endl;

        // Return the HTML code
        return std::make_tuple(htmlStream.str(), "");
    }

    // Define the default function for basic HTML processing
    // The first element in the tuple is the HTML code, the second the JS code
    // For this function the first will be empty
    std::tuple<std::string, std::string>
    parseDataForJSPage(GlobalSettings settingsStruct, std::vector<MemoryAccessLog> accessLogs) {

        std::stringstream jsStream;

        return std::make_tuple("", jsStream.str());
    }


    // Define function to load back data
    void fetchData() {
        // Check if data was fetched from the device
        if (fetchedFromDevice) {
            // If so, return
            return;
        }

        // First fetch the h_constantData from the device
        checkCudaError(cudaMemcpy(h_constantData, d_constantData, sizeof(GlobalSettings), cudaMemcpyDeviceToHost),
                       "Could not copy constant data from device.");

        // Copy the data back from the device
        checkCudaError(
                cudaMemcpy(h_memoryAccessLog, d_memoryAccessLog, sizeof(MemoryAccessLog) * h_constantData->originalSize,
                           cudaMemcpyDeviceToHost), "Could not copy memory access logs from device.");

        // Set the fetched from device flag to true
        fetchedFromDevice = true;
    }

    CudaMemAccessStorage<T> *d_this;

public:

    __device__ CudaMemAccessStorage() {
        // Set the default values
        d_memoryAccessLog = nullptr;
    }

    // Constructor which allocates the memory on the device
    __host__ CudaMemAccessStorage(unsigned int size) {

        h_constantData = new GlobalSettings{-1, -1, -1, -1, -1, -1, -1, size, 0};

        // Allocate the memory on the device for the d_constantData and check if it was successful
        checkCudaError(cudaMalloc(&d_constantData, sizeof(GlobalSettings)),
                       "Could not allocate array to store kernel data on device.");
        // Copy over the host data to the device
        checkCudaError(cudaMemcpy(d_constantData, h_constantData, sizeof(GlobalSettings), cudaMemcpyHostToDevice),
                       "Could not copy constant data to device.");

        // Allocate the memory on the device for the d_memoryAccessLog and check if it was successful
        checkCudaError(cudaMalloc(&d_memoryAccessLog, sizeof(MemoryAccessLog) * size),
                       "Could not allocate array to store memory access logs on device. (reading)");
        // Also allocate the memory on the host for the h_memoryAccessLog
        h_memoryAccessLog = new MemoryAccessLog[size];

        // Copy the empty data to the device
        checkCudaError(cudaMemcpy(d_memoryAccessLog, h_memoryAccessLog, sizeof(MemoryAccessLog) * size,
                                  cudaMemcpyHostToDevice), "Could not copy memory access logs to device.");

        // Now we finished initializing the class, so we need to create the copy of this class on the device
        // Allocate the memory on the device for the d_this and check if it was successful
        checkCudaError(cudaMalloc(&d_this, sizeof(CudaMemAccessStorage<T>)),
                       "Could not allocate array to store this class on device.");
        // Copy the empty data to the device
        checkCudaError(cudaMemcpy(d_this, this, sizeof(CudaMemAccessStorage<T>), cudaMemcpyHostToDevice),
                       "Could not copy this class to device.");

    }

    __host__ void free() {
        // Free the memory on the device
        checkCudaError(cudaFree(d_memoryAccessLog), "Could not free memory access logs on device.");
        checkCudaError(cudaFree(d_constantData), "Could not free constant data on device.");
        checkCudaError(cudaFree(d_this), "Could not free this class on device.");

        // Free the memory on the host
        delete[] h_memoryAccessLog;
        delete h_constantData;
    }

    __device__ int getStorageIndex() {// Atomically increase the currentSize by 1
        int current_index = atomicAdd(&d_constantData->currentSize, 1);

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
        return current_index;
    }

    __host__ __device__ CudaMemAccessStorage<T> *getDevicePointer() const {
        return d_this;
    }

    __device__ void pushReadLog(T *address) {
        int current_index = getStorageIndex();

        // Get the block and thread id
        unsigned int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
        unsigned int threadId = threadIdx.x + threadIdx.y * blockDim.x + blockDim.x * blockDim.y * threadIdx.z;

        // Check that our current index is less than the original size
        if (current_index < d_constantData->originalSize) {
            // Store the data in the memory access log
            d_memoryAccessLog[current_index] = MemoryAccessLog(address, blockId, threadId, true);
        }
    }

    __device__ void pushWriteLog(T *address) {
        int current_index = getStorageIndex();

        // Get the block and thread id
        unsigned int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
        unsigned int threadId = threadIdx.x + threadIdx.y * blockDim.x + blockDim.x * blockDim.y * threadIdx.z;

        // Check that our current index is less than the original size
        if (current_index < d_constantData->originalSize) {
            // Store the data in the memory access log
            d_memoryAccessLog[current_index] = MemoryAccessLog(address, blockId, threadId, false);
        }
    }

    // Function to analyze the data which also frees up all used memory
    // As third parameter a function lambda can be passed which is called for each memory access
    void generateOutput(const std::string template_file, const std::string &output_path,
                        std::function<std::tuple<std::string, std::string>(GlobalSettings settingsStruct,
                                                                           std::vector<MemoryAccessLog> accessLogs)> customGenerationFunction = nullptr) {
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

        // Generate a vector for the accessLogs
        std::vector<MemoryAccessLog> accessLogs;

        // Loop over the read logs
        for (int i = 0; i < h_constantData->currentSize; i++) {
            // Add the log to the vector
            accessLogs.push_back(h_memoryAccessLog[i]);
        }

        // Run the custom generation function, if one was passed
        if (customGenerationFunction != nullptr) {
            placeholderReplacement = customGenerationFunction(*h_constantData, accessLogs);
        }
            // If none was passed, use a default function
        else {
            placeholderReplacement = parseDataForStaticHTML(*h_constantData, accessLogs);
        }

        // Replace "<!-- HTML_TEMPLATE -->" with the HTML template
        std::string::size_type pos = template_file_string.find("<!-- HTML_TEMPLATE -->");
        // Check if the placeholder was found
        if (pos != std::string::npos) {
            // Replace the placeholder with the actual data
            template_file_string.replace(pos, 22, std::get<0>(placeholderReplacement));
        }

        // Replace "<!-- JS_TEMPLATE -->" with the JS template
        pos = template_file_string.find("// JS_TEMPLATE");
        // Check if the placeholder was found
        if (pos != std::string::npos) {
            // Replace the placeholder with the actual data
            template_file_string.replace(pos, 14, std::get<1>(placeholderReplacement));
        }

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

    __host__ const GlobalSettings getGlobalSettings() {
        // Fetch the data back from the device
        fetchData();

        return *h_constantData;
    }

};


// Define a custom template class which holds the data for the CUDA kernel
template<typename T>
class CudaMemAccessLogger {

private:

    // Have an internal pointer to the data, this pointer is a device pointer
    T *d_data;

    // Have a pointer to the storage class we are using
    CudaMemAccessStorage<T> *h_storage;
    // We also need to store its device pointer
    CudaMemAccessStorage<T> *d_storage;

    // Also have an instance of this class allocated on the device
    CudaMemAccessLogger<T> *d_this;

    // Implement a proxy class so we can both read and write from the array when accessing the array operator
    class AccessProxy {
        // Have a reference to the CudaMemAccessLogger class
        CudaMemAccessLogger<T> *cudaMav;
        // Have a reference to the index
        int index;

    public:

        // Constructor which takes a reference to the CudaMemAccessLogger class and the index
        __device__ AccessProxy(CudaMemAccessLogger<T> *cudaMav, int index) : cudaMav(cudaMav), index(index) {}

        AccessProxy() = delete;

        // Overload the assignment operator so we can write to the array
        __device__ AccessProxy &operator=(const T &value) {
            cudaMav->set(index, value);
            return *this;
        }

        // When accessing the array, and also assign a value to the access, we assign AccessProxy to AccessProxy
        // For this reason we need to define the assignment operator for AccessProxy, so that the actual values get changed
        __device__ AccessProxy &operator=(const AccessProxy &other) {
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
    void checkCudaError(cudaError_t err, std::string const &message = "Cuda Error.") {
        if (err != cudaSuccess) {
            const char *errorString = cudaGetErrorString(err);
            throw std::runtime_error(message + "\n Error: \n" + std::string(errorString));
        }
    }

public:

    // Constructor to create an empty class
    __device__ __host__ CudaMemAccessLogger() {
        // Set the data pointer to null
        d_data = nullptr;
    }

    // Constructor which allocates the memory on the device
    __host__ CudaMemAccessLogger(T *array_data, CudaMemAccessStorage<T> *storage = nullptr) {

        // Store the passed data pointer within the class
        d_data = array_data;

        // If the storage class is not null, we need to store it
        if (storage != nullptr) {
            // Store the storage class
            h_storage = storage;
        }
            // If it is null, we need to create a new storage class
        else {
            // Create a new storage class
            h_storage = new CudaMemAccessStorage<T>(10000);
        }

        // Also store the device pointer
        d_storage = h_storage->getDevicePointer();



        // We need to create the copy of this class on the device
        // Allocate the memory on the device for the d_this and check if it was successful
        checkCudaError(cudaMalloc(&d_this, sizeof(CudaMemAccessLogger<T>)),
                       "Could not allocate array to store this class on device.");
        // Copy the empty data to the device
        checkCudaError(cudaMemcpy(d_this, this, sizeof(CudaMemAccessLogger<T>), cudaMemcpyHostToDevice),
                       "Could not copy this class to device.");

    }

    __device__ T get(unsigned int index) {

        // Get the address of the data
        T *address = &d_data[index];

        // Push the read log
        d_storage->pushReadLog(address);

        // Print the accessed data
        //printf("Accessed data at address %p by thread %d in block %d\n", &d_data[index], threadId, blockId);

        return static_cast<T>(d_data[index]);
    }

    __device__ void set(unsigned int index, T value) {

        // Get the address of the data
        T *address = &d_data[index];

        // Push the write log
        d_storage->pushWriteLog(address);

        // Write the value to the data
        d_data[index] = value;

        // Print the accessed data
        //printf("Wrote data at address %p by thread %d in block %d\n", &d_data[index], threadId, blockId);
    }


    // Destructor to free the memory on the device
    __host__ ~CudaMemAccessLogger() {
        // Free up the memory on the device
        checkCudaError(cudaFree(d_this), "Could not free class instance pointer on device.");
    }

    // Array operator overload on the device
    __device__ AccessProxy operator[](size_t i) {
        return AccessProxy(this, i);
    }

    __device__ AccessProxy operator[](size_t i) const {
        return AccessProxy(this, i);
    }

    __device__ __host__ void PrintPointer() {
#ifdef  __CUDA_ARCH__
        printf("Pointer within Class (CUDA): %p\n", d_data);
#else
        printf("Pointer within Class (CPU): %p\n", d_data);
#endif
    }

    __host__ CudaMemAccessLogger<T> *getDevicePointer() {
        return d_this;
    }

    CudaMemAccessStorage<T> getStorage() {
        return *h_storage;
    }
};
