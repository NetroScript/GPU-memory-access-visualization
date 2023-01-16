#pragma once

#include <stdexcept>
#include <cstdio>
#include <fstream>
#include <functional>
#include <sstream>
#include <vector>
#include <iostream>

// Base class for creating managed memory objects
// source: https://developer.nvidia.com/blog/unified-memory-in-cuda-6/
class Managed
{
public:
    void* operator new(size_t len) {
        void* ptr;
        cudaMallocManaged(&ptr, len);
        cudaDeviceSynchronize();
        return ptr;
    }

    void operator delete(void* ptr) {
        cudaDeviceSynchronize();
        cudaFree(ptr);
    }
};

// Define a custom template class, which takes care of storing the memory accesses of a given type.
template<typename T>
class CudaMemAccessStorage : public Managed{

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
    GlobalSettings constantData;

    // Have a pointer to a list of memory access logs
    MemoryAccessLog* memoryAccessLog = nullptr;

    // Store the memory regions by storing the starting address, the amount of elements, the size of a single element and a name
    std::vector<std::tuple<T*, size_t, size_t, std::string>> memoryRegions;

    // Helper function to format and throw CUDA errors
    void checkCudaError(cudaError_t err, std::string const &message = "Cuda Error.") {
        if (err != cudaSuccess) {
            const char *errorString = cudaGetErrorString(err);
            throw std::runtime_error(message + "\n Error: \n" + std::string(errorString));
        }
    }

public:
    // Define the default function for basic HTML processing
    // The first element in the tuple is the HTML code, the second the JS code
    // For this function the second will be empty
    static std::tuple<std::string, std::string>
    parseDataForStaticHTML(GlobalSettings settingsStruct,std::vector<std::tuple<T*, size_t, size_t, std::string>> memoryRegions, std::vector<MemoryAccessLog> accessLogs) {

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

        // Add a section for each memory region and its name
        htmlStream << "<h1>Memory Regions</h1>" << std::endl;

        for (auto region : memoryRegions) {
                htmlStream << "<h2>Region: " << std::get<3>(region) << "</h2>" << std::endl;
                htmlStream << "<p>Start Address: " << std::get<0>(region) << " (End Address:) " << std::get<0>(region)+ std::get<1>(region) * std::get<2>(region) << "</p>" << std::endl;
                htmlStream << "<p>Number of Elements: " << std::get<1>(region) << "</p>" << std::endl;
                htmlStream << "<p>Size of Single Element: " << std::get<2>(region) << "</p>" << std::endl;
        }

        // Add the section for the memory access logs
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
    static std::tuple<std::string, std::string>
    parseDataForJSPage(GlobalSettings settingsStruct,std::vector<std::tuple<T*, size_t, size_t, std::string>> memoryRegions, std::vector<MemoryAccessLog> accessLogs) {

        std::stringstream jsStream;

        // Create a JSON object which describes all the available data
        // We will have one key GlobalSettings, one key MemoryRegions, and one key MemoryAccessLogs

        // Start the JSON object
        jsStream << "{";

        // Add the GlobalSettings key
        jsStream << "\"GlobalSettings\": {";

        // Store all settings

        // First store the grid dimensions
        jsStream << "\"GridDimensions\": {";
        jsStream << "\"x\": " << settingsStruct.gridDimX << ",";
        jsStream << "\"y\": " << settingsStruct.gridDimY << ",";
        jsStream << "\"z\": " << settingsStruct.gridDimZ;
        jsStream << "},";

        // Then store the block dimensions
        jsStream << "\"BlockDimensions\": {";
        jsStream << "\"x\": " << settingsStruct.blockDimX << ",";
        jsStream << "\"y\": " << settingsStruct.blockDimY << ",";
        jsStream << "\"z\": " << settingsStruct.blockDimZ;
        jsStream << "},";

        // Then store the warp size
        jsStream << "\"WarpSize\": " << settingsStruct.warpSize;

        // Close the GlobalSettings object
        jsStream << "},";

        // Add the MemoryRegions key
        jsStream << "\"MemoryRegions\": [";

        // Loop through all memory regions and store them
        for (auto region : memoryRegions) {
            // Store the start address, number of elements and the size of a single element
            jsStream << "{";
            jsStream << "\"StartAddress\": \"" << std::hex << std::get<0>(region) << "\",";
            jsStream << "\"EndAddress\": \"" << std::get<0>(region) + std::get<1>(region) * std::get<2>(region) << "\",";
            jsStream << "\"NumberOfElements\": " << std::dec << std::get<1>(region) << ",";
            jsStream << "\"SizeOfSingleElement\": " << std::get<2>(region) << ",";
            jsStream << "\"Name\": \"" << std::get<3>(region) << "\"";
            jsStream << "}";

            // If this is not the last element, add a comma
            if (region != memoryRegions.back()) {
                jsStream << ",";
            }
        }

        // Close the MemoryRegions array
        jsStream << "],";

        // Add the MemoryAccessLogs key
        jsStream << "\"MemoryAccessLogs\": [";

        // Loop through all memory access logs and store them
        // To save on storage space and parsing time (as this is the biggest part of the data) we store an array with the data
        // entries directly instead of using an object with names for each entry
        for (auto &log: accessLogs) {
            jsStream << "[";
            jsStream << "\"" <<  std::hex << log.Address() << "\",";
            jsStream << std::dec << log.BlockId() << ",";
            jsStream << log.ThreadId() << ",";
            jsStream << (log.IsRead() ? "true" : "false");
            jsStream << "]";

            // If this is not the last element, add a comma
            if (&log != &accessLogs.back()) {
                jsStream << ",";
            }
        }

        // Close the MemoryAccessLogs array
        jsStream << "]";

        // Close the JSON object
        jsStream << "}";

        // Return the JS code
        return std::make_tuple("", jsStream.str());
    }

private:

    __device__ int getStorageIndex() {// Atomically increase the currentSize by 1
        int current_index = atomicAdd(&constantData.currentSize, 1);

        // First check if the currentSize is zero, if so we need to initialize the additional data variables, needed later to restore the data
        if (current_index == 0) {
            // Store the grid dimensions
            constantData.gridDimY = gridDim.y;
            constantData.gridDimZ = gridDim.z;
            constantData.gridDimX = gridDim.x;
            // Store the block dimensions
            constantData.blockDimX = blockDim.x;
            constantData.blockDimY = blockDim.y;
            constantData.blockDimZ = blockDim.z;
            // Store the warp size
            constantData.warpSize = warpSize;
        }
        return current_index;
    }

public:

    // Constructor which allocates the memory on the device
    __host__ CudaMemAccessStorage(unsigned int size) {

        constantData = {-1, -1, -1, -1, -1, -1, -1, size, 0};

        // Allocate the memory for the memeoryAccessLog and check if it was successful
        checkCudaError(cudaMallocManaged(&memoryAccessLog, sizeof(MemoryAccessLog) * size),
                       "Could not copy constant data to device.");
    }

    __host__ void registerArray(T *array, size_t size, std::string name = "") {
        memoryRegions.push_back(std::make_tuple(array, size, sizeof(array[0]), name));
    }

    __host__ ~CudaMemAccessStorage() {
        checkCudaError(cudaFree(memoryAccessLog), "Could not free memory access logs.");
    }

    __device__ void pushReadLog(T *address) {
        int current_index = getStorageIndex();

        // Get the block and thread id
        unsigned int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
        unsigned int threadId = threadIdx.x + threadIdx.y * blockDim.x + blockDim.x * blockDim.y * threadIdx.z;

        // Check that our current index is less than the original size
        if (current_index < constantData.originalSize) {
            // Store the data in the memory access log
            memoryAccessLog[current_index] = MemoryAccessLog(address, blockId, threadId, true);
        }
    }

    __device__ void pushWriteLog(T *address) {
        int current_index = getStorageIndex();

        // Get the block and thread id
        unsigned int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
        unsigned int threadId = threadIdx.x + threadIdx.y * blockDim.x + blockDim.x * blockDim.y * threadIdx.z;

        // Check that our current index is less than the original size
        if (current_index < constantData.originalSize) {
            // Store the data in the memory access log
            memoryAccessLog[current_index] = MemoryAccessLog(address, blockId, threadId, false);
        }
    }

        // Function to analyze the data which also frees up all used memory
    // As third parameter a function lambda can be passed which is called for each memory access
    void generateOutput(const std::string template_file, const std::string &output_path,
                        std::function<std::tuple<std::string, std::string>(GlobalSettings settingsStruct, std::vector<std::tuple<T*, size_t, size_t, std::string>> memoryRegions,
                                                                           std::vector<MemoryAccessLog> accessLogs)> customGenerationFunction = nullptr) {

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
        for (int i = 0; i < constantData.currentSize; i++) {
            // Add the log to the vector
            accessLogs.push_back(memoryAccessLog[i]);
        }

        // Run the custom generation function, if one was passed
        if (customGenerationFunction != nullptr) {
            placeholderReplacement = customGenerationFunction(constantData, memoryRegions, accessLogs);
        }
            // If none was passed, use a default function
        else {
            placeholderReplacement = parseDataForStaticHTML(constantData, memoryRegions, accessLogs);
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
        return constantData;
    }

};


// Define a custom template class which holds the data for the CUDA kernel
template<typename T>
class CudaMemAccessLogger : public Managed {

private:

    // Have an internal pointer to the data, this pointer is a device pointer
    T *d_data;

    // Have a pointer to the storage class we are using
    CudaMemAccessStorage<T> *storage;

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
    __host__ CudaMemAccessLogger(T *array_data, size_t array_length, std::string description_name, CudaMemAccessStorage<T> *cma_storage = nullptr) {

        // Store the passed data pointer within the class
        d_data = array_data;

        // If the storage class is not null, we need to store it
        if (cma_storage != nullptr) {
            // Store the storage class
            storage = cma_storage;
        }
            // If it is null, we need to create a new storage class
        else {
            // Create a new storage class
            storage = new CudaMemAccessStorage<T>(10000);
        }

        storage->registerArray(array_data, array_length, description_name);
    }

    __device__ T get(unsigned int index) {

        // Get the address of the data
        T *address = &d_data[index];

        // Push the read log
        storage->pushReadLog(address);

        // Print the accessed data
        //printf("Accessed data at address %p by thread %d in block %d\n", &d_data[index], threadId, blockId);

        return static_cast<T>(d_data[index]);
    }

    __device__ void set(unsigned int index, T value) {

        // Get the address of the data
        T *address = &d_data[index];

        // Push the write log
        storage->pushWriteLog(address);

        // Write the value to the data
        d_data[index] = value;

        // Print the accessed data
        //printf("Wrote data at address %p by thread %d in block %d\n", &d_data[index], threadId, blockId);
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

    CudaMemAccessStorage<T>* getStorage() {
        return storage;
    }
};
