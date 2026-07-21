#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <memory>
#include <numeric>
#include <algorithm>
#include <sstream>
#include <cstdint>
#include <cstdlib>
#include <sys/stat.h>
#include <chrono>
#include <iomanip>
#include "acl/acl.h"

#define ACL_CHECK(ret, msg) \
    do { \
        if ((ret) != ACL_SUCCESS) { \
            std::cerr << "[ERROR] " << msg << ", errorCode=" << ret << std::endl; \
            return ret; \
        } \
    } while (0)

struct InputConfig {
    std::string name;
    std::vector<int64_t> shape;
    aclDataType dataType;
    aclFormat format;
    std::string dataFile;
};

static bool ReadBinFile(const std::string &filePath, void *&data, uint32_t &dataSize)
{
    std::ifstream file(filePath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Failed to open file: " << filePath << std::endl;
        return false;
    }
    dataSize = static_cast<uint32_t>(file.tellg());
    file.seekg(0, std::ios::beg);
    data = malloc(dataSize);
    if (data == nullptr) {
        std::cerr << "[ERROR] Failed to allocate host memory for file: " << filePath << std::endl;
        return false;
    }
    file.read(static_cast<char *>(data), dataSize);
    file.close();
    return true;
}

static std::string ShapeToString(const std::vector<int64_t> &shape);

static bool IsNpyFile(const std::string &filePath)
{
    return filePath.size() >= 4 && filePath.substr(filePath.size() - 4) == ".npy";
}

static std::string ExtractNpyHeaderField(const std::string &header, const std::string &field)
{
    std::string key = "'" + field + "'";
    size_t pos = header.find(key);
    if (pos == std::string::npos) {
        key = "\"" + field + "\"";
        pos = header.find(key);
    }
    if (pos == std::string::npos) return "";

    pos = header.find(':', pos);
    if (pos == std::string::npos) return "";
    pos++;
    while (pos < header.size() && header[pos] == ' ') pos++;

    if (pos >= header.size()) return "";

    if (header[pos] == '\'') {
        size_t end = header.find('\'', pos + 1);
        if (end == std::string::npos) return "";
        return header.substr(pos + 1, end - pos - 1);
    }

    size_t end = pos;
    while (end < header.size() && header[end] != ',' && header[end] != '}' && header[end] != '\n') end++;
    std::string val = header.substr(pos, end - pos);
    while (!val.empty() && val.back() == ' ') val.pop_back();
    return val;
}

static std::vector<int64_t> ParseNpyShape(const std::string &header)
{
    std::vector<int64_t> shape;
    size_t pos = header.find("'shape'");
    if (pos == std::string::npos) pos = header.find("\"shape\"");
    if (pos == std::string::npos) return shape;

    size_t start = header.find('(', pos);
    size_t end = header.find(')', start);
    if (start == std::string::npos || end == std::string::npos) return shape;

    std::string shapeStr = header.substr(start + 1, end - start - 1);
    std::istringstream iss(shapeStr);
    std::string token;
    while (std::getline(iss, token, ',')) {
        while (!token.empty() && (token.front() == ' ' || token.front() == '\t')) token.erase(0, 1);
        while (!token.empty() && (token.back() == ' ' || token.back() == '\t' || token.back() == ',')) token.pop_back();
        if (!token.empty()) {
            try { shape.push_back(std::stoll(token)); } catch (...) {}
        }
    }
    return shape;
}

static size_t NpyDescToSize(const std::string &descr)
{
    if (descr.find("f8") != std::string::npos || descr.find("float64") != std::string::npos) return 8;
    if (descr.find("f4") != std::string::npos || descr.find("float32") != std::string::npos) return 4;
    if (descr.find("f2") != std::string::npos || descr.find("float16") != std::string::npos) return 2;
    if (descr.find("i8") != std::string::npos || descr.find("int64") != std::string::npos) return 8;
    if (descr.find("i4") != std::string::npos || descr.find("int32") != std::string::npos) return 4;
    if (descr.find("i2") != std::string::npos || descr.find("int16") != std::string::npos) return 2;
    if (descr.find("i1") != std::string::npos || descr.find("int8") != std::string::npos) return 1;
    if (descr.find("u8") != std::string::npos || descr.find("uint64") != std::string::npos) return 8;
    if (descr.find("u4") != std::string::npos || descr.find("uint32") != std::string::npos) return 4;
    if (descr.find("u2") != std::string::npos || descr.find("uint16") != std::string::npos) return 2;
    if (descr.find("u1") != std::string::npos || descr.find("uint8") != std::string::npos) return 1;
    if (descr.find("b1") != std::string::npos || descr.find("bool") != std::string::npos) return 1;
    return 0;
}

static bool ReadNpyFile(const std::string &filePath, void *&data, uint32_t &dataSize,
                        std::vector<int64_t> &outShape, size_t &elemSize)
{
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Failed to open npy file: " << filePath << std::endl;
        return false;
    }

    char magic[6];
    file.read(magic, 6);
    if (magic[0] != '\x93' || magic[1] != 'N' || magic[2] != 'U' ||
        magic[3] != 'M' || magic[4] != 'P' || magic[5] != 'Y') {
        std::cerr << "[ERROR] Invalid .npy magic bytes: " << filePath << std::endl;
        return false;
    }

    uint8_t major, minor;
    file.read(reinterpret_cast<char*>(&major), 1);
    file.read(reinterpret_cast<char*>(&minor), 1);

    uint32_t headerLen = 0;
    if (major == 1) {
        uint16_t hlen;
        file.read(reinterpret_cast<char*>(&hlen), 2);
        headerLen = hlen;
    } else if (major == 2 || major == 3) {
        file.read(reinterpret_cast<char*>(&headerLen), 4);
    } else {
        std::cerr << "[ERROR] Unsupported .npy version: " << (int)major << std::endl;
        return false;
    }

    std::string header(headerLen, '\0');
    file.read(&header[0], headerLen);

    std::string descr = ExtractNpyHeaderField(header, "descr");
    outShape = ParseNpyShape(header);

    elemSize = NpyDescToSize(descr);
    if (elemSize == 0) {
        std::cerr << "[ERROR] Unsupported .npy dtype: " << descr << std::endl;
        return false;
    }

    size_t elemCount = 1;
    for (auto d : outShape) elemCount *= static_cast<size_t>(d);
    dataSize = static_cast<uint32_t>(elemCount * elemSize);

    data = malloc(dataSize);
    if (data == nullptr) {
        std::cerr << "[ERROR] Failed to allocate memory for npy data: " << filePath << std::endl;
        return false;
    }

    file.read(static_cast<char*>(data), dataSize);
    file.close();

    std::cout << "[INFO] Loaded .npy: shape=" << ShapeToString(outShape)
              << ", dtype=" << descr << ", elemSize=" << elemSize
              << ", dataSize=" << dataSize << std::endl;
    return true;
}

static size_t GetDataTypeSize(aclDataType dataType)
{
    switch (dataType) {
        case ACL_FLOAT:   return 4;
        case ACL_FLOAT16: return 2;
        case ACL_INT8:    return 1;
        case ACL_INT16:   return 2;
        case ACL_INT32:   return 4;
        case ACL_INT64:   return 8;
        case ACL_UINT8:   return 1;
        case ACL_UINT16:  return 2;
        case ACL_UINT32:  return 4;
        case ACL_UINT64:  return 8;
        case ACL_DOUBLE:  return 8;
        case ACL_BOOL:    return 1;
        default:          return 0;
    }
}

static size_t CalcTensorSize(const std::vector<int64_t> &shape, aclDataType dataType)
{
    size_t elemCount = 1;
    for (auto dim : shape) {
        elemCount *= static_cast<size_t>(dim);
    }
    return elemCount * GetDataTypeSize(dataType);
}

static std::string ShapeToString(const std::vector<int64_t> &shape)
{
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << shape[i];
    }
    oss << "]";
    return oss.str();
}

struct DumpConfig {
    bool enabled;
    std::string configPath;
    std::string dumpPath;
    std::string dumpMode;
    std::string dumpLevel;
    std::string dumpData;
    std::string modelName;
    std::vector<std::string> layers;
};

struct ProfilingConfig {
    bool enabled;
    std::string outputPath;
    std::string aicMetrics;
    bool taskTime;
    bool runtimeApi;
    bool ascendcl;
};

static std::string EscapeJsonString(const std::string &s)
{
    std::string result;
    for (char c : s) {
        if (c == '"') result += "\\\"";
        else if (c == '\\') result += "\\\\";
        else result += c;
    }
    return result;
}

static bool GenerateAclJson(const DumpConfig &dumpCfg, const ProfilingConfig &profCfg, const std::string &outputPath)
{
    std::ofstream file(outputPath);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Failed to create acl.json at: " << outputPath << std::endl;
        return false;
    }

    file << "{\n";
    
    // Dump section
    if (dumpCfg.enabled) {
        file << "    \"dump\": {\n";

        file << "        \"dump_list\": [\n";
        if (dumpCfg.layers.empty() && dumpCfg.modelName.empty()) {
            file << "            {}\n";
        } else {
            file << "            {\n";
            if (!dumpCfg.modelName.empty()) {
                file << "                \"model_name\": \"" << EscapeJsonString(dumpCfg.modelName) << "\"";
            }
            if (!dumpCfg.layers.empty()) {
                if (!dumpCfg.modelName.empty()) file << ",\n";
                file << "                \"layer\": [\n";
                for (size_t i = 0; i < dumpCfg.layers.size(); ++i) {
                    file << "                    \"" << EscapeJsonString(dumpCfg.layers[i]) << "\"";
                    if (i + 1 < dumpCfg.layers.size()) file << ",";
                    file << "\n";
                }
                file << "                ]\n";
            } else {
                file << "\n";
            }
            file << "            }\n";
        }
        file << "        ],\n";

        file << "        \"dump_path\": \"" << EscapeJsonString(dumpCfg.dumpPath) << "\",\n";
        file << "        \"dump_mode\": \"" << EscapeJsonString(dumpCfg.dumpMode) << "\",\n";
        file << "        \"dump_level\": \"" << EscapeJsonString(dumpCfg.dumpLevel) << "\",\n";
        file << "        \"dump_data\": \"" << EscapeJsonString(dumpCfg.dumpData) << "\",\n";
        file << "        \"dump_op_switch\": \"off\"\n";

        file << "    }";
        if (profCfg.enabled) {
            file << ",\n";
        } else {
            file << "\n";
        }
    }
    
    // Profiling section
    if (profCfg.enabled) {
        file << "    \"profiler\": {\n";
        file << "        \"switch\": \"on\",\n";
        file << "        \"output\": \"" << EscapeJsonString(profCfg.outputPath) << "\"";
        
        if (!profCfg.aicMetrics.empty()) {
            file << ",\n        \"aic_metrics\": \"" << EscapeJsonString(profCfg.aicMetrics) << "\"";
        }
        
        file << ",\n        \"task_time\": \"" << (profCfg.taskTime ? "on" : "off") << "\"";
        file << ",\n        \"runtime_api\": \"" << (profCfg.runtimeApi ? "on" : "off") << "\"";
        file << ",\n        \"ascendcl\": \"" << (profCfg.ascendcl ? "on" : "off") << "\"";
        
        file << "\n    }\n";
    }
    
    file << "}\n";

    file.close();
    return true;
}

static DumpConfig ParseDumpConfig(int argc, char *argv[])
{
    DumpConfig cfg;
    cfg.enabled = false;
    cfg.dumpPath = "./dump_data";
    cfg.dumpMode = "output";
    cfg.dumpLevel = "op";
    cfg.dumpData = "tensor";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--dump") {
            cfg.enabled = true;
        } else if (arg == "--dump_config" && i + 1 < argc) {
            cfg.configPath = argv[++i];
            cfg.enabled = true;
        } else if (arg == "--dump_path" && i + 1 < argc) {
            cfg.dumpPath = argv[++i];
        } else if (arg == "--dump_mode" && i + 1 < argc) {
            cfg.dumpMode = argv[++i];
        } else if (arg == "--dump_level" && i + 1 < argc) {
            cfg.dumpLevel = argv[++i];
        } else if (arg == "--dump_data" && i + 1 < argc) {
            cfg.dumpData = argv[++i];
        } else if (arg == "--dump_model_name" && i + 1 < argc) {
            cfg.modelName = argv[++i];
        } else if (arg == "--dump_layer" && i + 1 < argc) {
            std::string layerStr = argv[++i];
            std::istringstream iss(layerStr);
            std::string layer;
            while (std::getline(iss, layer, ',')) {
                if (!layer.empty()) {
                    cfg.layers.push_back(layer);
                }
            }
        }
    }

    const char *envEnabled = getenv("DUMP_ENABLED");
    if (envEnabled != nullptr && std::string(envEnabled) == "1") {
        cfg.enabled = true;
    }
    const char *envConfigPath = getenv("DUMP_CONFIG");
    if (envConfigPath != nullptr && cfg.configPath.empty()) {
        cfg.configPath = envConfigPath;
    }
    const char *envDumpPath = getenv("DUMP_PATH");
    if (envDumpPath != nullptr) {
        cfg.dumpPath = envDumpPath;
    }
    const char *envDumpMode = getenv("DUMP_MODE");
    if (envDumpMode != nullptr) {
        cfg.dumpMode = envDumpMode;
    }

    return cfg;
}

static ProfilingConfig ParseProfilingConfig(int argc, char *argv[])
{
    ProfilingConfig cfg;
    cfg.enabled = false;
    cfg.outputPath = "./profiling_data";
    cfg.aicMetrics = "";
    cfg.taskTime = true;
    cfg.runtimeApi = true;
    cfg.ascendcl = true;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--profiling") {
            cfg.enabled = true;
        } else if (arg == "--profiling_output" && i + 1 < argc) {
            cfg.outputPath = argv[++i];
            cfg.enabled = true;
        } else if (arg == "--profiling_aic_metrics" && i + 1 < argc) {
            cfg.aicMetrics = argv[++i];
        } else if (arg == "--profiling_no_task_time") {
            cfg.taskTime = false;
        } else if (arg == "--profiling_no_runtime_api") {
            cfg.runtimeApi = false;
        } else if (arg == "--profiling_no_ascendcl") {
            cfg.ascendcl = false;
        }
    }

    const char *envEnabled = getenv("PROFILING_ENABLED");
    if (envEnabled != nullptr && std::string(envEnabled) == "1") {
        cfg.enabled = true;
    }
    const char *envOutputPath = getenv("PROFILING_OUTPUT");
    if (envOutputPath != nullptr) {
        cfg.outputPath = envOutputPath;
    }

    return cfg;
}

class ModelInfer {
public:
    ModelInfer() : modelId_(0), modelDesc_(nullptr), input_(nullptr), output_(nullptr),
                   isDevice_(false), stream_(nullptr), deviceId_(0), dumpActive_(false) {}
    ~ModelInfer() { Destroy(); }

    aclError Init(int32_t deviceId, const char *aclConfigPath = nullptr)
    {
        deviceId_ = deviceId;
        aclError ret = aclInit(aclConfigPath);
        ACL_CHECK(ret, "aclInit failed");

        ret = aclrtSetDevice(deviceId_);
        ACL_CHECK(ret, "aclrtSetDevice failed");

        aclrtRunMode runMode;
        ret = aclrtGetRunMode(&runMode);
        ACL_CHECK(ret, "aclrtGetRunMode failed");
        isDevice_ = (runMode == ACL_DEVICE);

        ret = aclrtCreateStream(&stream_);
        ACL_CHECK(ret, "aclrtCreateStream failed");

        std::cout << "[INFO] ACL initialized, deviceId=" << deviceId
                  << ", runMode=" << (isDevice_ ? "DEVICE" : "HOST") << std::endl;
        return ACL_SUCCESS;
    }

    aclError InitDump()
    {
        aclError ret = aclmdlInitDump();
        ACL_CHECK(ret, "aclmdlInitDump failed");
        dumpActive_ = true;
        std::cout << "[INFO] Dump initialized" << std::endl;
        return ACL_SUCCESS;
    }

    aclError SetDump(const std::string &configPath)
    {
        if (!dumpActive_) {
            std::cerr << "[ERROR] Dump not initialized, call InitDump first" << std::endl;
            return ACL_ERROR_INTERNAL_ERROR;
        }
        aclError ret = aclmdlSetDump(configPath.c_str());
        ACL_CHECK(ret, "aclmdlSetDump failed, configPath=" << configPath);
        std::cout << "[INFO] Dump config set: " << configPath << std::endl;
        return ACL_SUCCESS;
    }

    aclError FinalizeDump()
    {
        if (!dumpActive_) return ACL_SUCCESS;
        aclError ret = aclmdlFinalizeDump();
        ACL_CHECK(ret, "aclmdlFinalizeDump failed");
        dumpActive_ = false;
        std::cout << "[INFO] Dump finalized" << std::endl;
        return ACL_SUCCESS;
    }

    aclError LoadModel(const std::string &omPath)
    {
        std::cout << "[INFO] Loading model from: " << omPath << std::endl;
        aclError ret = aclmdlLoadFromFile(omPath.c_str(), &modelId_);
        ACL_CHECK(ret, "aclmdlLoadFromFile failed");

        modelDesc_ = aclmdlCreateDesc();
        if (modelDesc_ == nullptr) {
            std::cerr << "[ERROR] aclmdlCreateDesc returned nullptr" << std::endl;
            return ACL_ERROR_INTERNAL_ERROR;
        }
        ret = aclmdlGetDesc(modelDesc_, modelId_);
        ACL_CHECK(ret, "aclmdlGetDesc failed");

        inputCount_ = aclmdlGetNumInputs(modelDesc_);
        outputCount_ = aclmdlGetNumOutputs(modelDesc_);

        std::cout << "[INFO] Model loaded, modelId=" << modelId_
                  << ", inputs=" << inputCount_
                  << ", outputs=" << outputCount_ << std::endl;

        for (size_t i = 0; i < inputCount_; ++i) {
            const char *name = aclmdlGetInputNameByIndex(modelDesc_, i);
            size_t size = aclmdlGetInputSizeByIndex(modelDesc_, i);
            std::cout << "[INFO]   Input[" << i << "]: name="
                      << (name ? name : "unknown")
                      << ", staticSize=" << size
                      << (size == 0 ? " (DYNAMIC)" : "") << std::endl;
        }
        for (size_t i = 0; i < outputCount_; ++i) {
            const char *name = aclmdlGetOutputNameByIndex(modelDesc_, i);
            size_t size = aclmdlGetOutputSizeByIndex(modelDesc_, i);
            std::cout << "[INFO]   Output[" << i << "]: name="
                      << (name ? name : "unknown")
                      << ", staticSize=" << size
                      << (size == 0 ? " (DYNAMIC)" : "") << std::endl;
        }

        return ACL_SUCCESS;
    }

    aclError PrepareInputOutput(const std::vector<InputConfig> &inputConfigs, bool useStatic = false)
    {
        if (inputConfigs.size() != inputCount_) {
            std::cerr << "[ERROR] Input config count (" << inputConfigs.size()
                      << ") != model input count (" << inputCount_ << ")" << std::endl;
            return ACL_ERROR_INVALID_PARAM;
        }

        input_ = aclmdlCreateDataset();
        output_ = aclmdlCreateDataset();
        if (input_ == nullptr || output_ == nullptr) {
            std::cerr << "[ERROR] Failed to create dataset" << std::endl;
            return ACL_ERROR_INTERNAL_ERROR;
        }

        for (size_t i = 0; i < inputCount_; ++i) {
            const auto &cfg = inputConfigs[i];
            size_t tensorSize;
            if (useStatic) {
                tensorSize = aclmdlGetInputSizeByIndex(modelDesc_, i);
                if (tensorSize == 0) {
                    std::cerr << "[ERROR] Static model input[" << i << "] has size 0, "
                              << "this input may be dynamic. Use dynamic mode instead." << std::endl;
                    return ACL_ERROR_INVALID_PARAM;
                }
                std::cout << "[INFO] Input[" << i << "] using model-defined size: " << tensorSize << " bytes" << std::endl;
            } else {
                tensorSize = CalcTensorSize(cfg.shape, cfg.dataType);
            }
            void *deviceBuf = nullptr;
            aclError ret = aclrtMalloc(&deviceBuf, tensorSize, ACL_MEM_MALLOC_HUGE_FIRST);
            ACL_CHECK(ret, "aclrtMalloc for input[" << i << "] failed, size=" << tensorSize);

            void *hostData = nullptr;
            uint32_t hostDataSize = 0;
            if (!cfg.dataFile.empty()) {
                if (IsNpyFile(cfg.dataFile)) {
                    std::vector<int64_t> npyShape;
                    size_t npyElemSize = 0;
                    if (!ReadNpyFile(cfg.dataFile, hostData, hostDataSize, npyShape, npyElemSize)) {
                        aclrtFree(deviceBuf);
                        return ACL_ERROR_READ_MODEL_FAILURE;
                    }
                    if (hostDataSize > tensorSize) {
                        std::cerr << "[WARN] .npy data size (" << hostDataSize
                                  << ") > tensor size (" << tensorSize
                                  << "), will truncate" << std::endl;
                        hostDataSize = static_cast<uint32_t>(tensorSize);
                    }
                } else {
                    if (!ReadBinFile(cfg.dataFile, hostData, hostDataSize)) {
                        aclrtFree(deviceBuf);
                        return ACL_ERROR_READ_MODEL_FAILURE;
                    }
                    if (hostDataSize > tensorSize) {
                        std::cerr << "[WARN] Input file size (" << hostDataSize
                                  << ") > tensor size (" << tensorSize
                                  << "), will truncate" << std::endl;
                        hostDataSize = static_cast<uint32_t>(tensorSize);
                    }
                }
            }

            if (!isDevice_ && hostData != nullptr) {
                void *hostMem = nullptr;
                ret = aclrtMallocHost(&hostMem, hostDataSize);
                if (ret == ACL_SUCCESS && hostMem != nullptr) {
                    memcpy(hostMem, hostData, hostDataSize);
                    free(hostData);
                    hostData = hostMem;
                    ret = aclrtMemcpy(deviceBuf, tensorSize, hostData, hostDataSize,
                                      ACL_MEMCPY_HOST_TO_DEVICE);
                    aclrtFreeHost(hostData);
                    hostData = nullptr;
                } else {
                    std::cerr << "[ERROR] aclrtMallocHost for input[" << i << "] failed" << std::endl;
                    free(hostData);
                    aclrtFree(deviceBuf);
                    return ACL_ERROR_INTERNAL_ERROR;
                }
            } else if (hostData != nullptr) {
                ret = aclrtMemcpy(deviceBuf, tensorSize, hostData, hostDataSize,
                                  ACL_MEMCPY_DEVICE_TO_DEVICE);
                free(hostData);
                hostData = nullptr;
            }
            if (ret != ACL_SUCCESS) {
                aclrtFree(deviceBuf);
                std::cerr << "[ERROR] aclrtMemcpy for input[" << i << "] failed" << std::endl;
                return ret;
            }

            aclDataBuffer *dataBuf = aclCreateDataBuffer(deviceBuf, tensorSize);
            if (dataBuf == nullptr) {
                aclrtFree(deviceBuf);
                std::cerr << "[ERROR] aclCreateDataBuffer for input[" << i << "] failed" << std::endl;
                return ACL_ERROR_INTERNAL_ERROR;
            }
            ret = aclmdlAddDatasetBuffer(input_, dataBuf);
            ACL_CHECK(ret, "aclmdlAddDatasetBuffer for input[" << i << "] failed");

            inputBuffers_.push_back(deviceBuf);
            std::cout << "[INFO] Input[" << i << "] prepared, shape="
                      << ShapeToString(cfg.shape)
                      << ", size=" << tensorSize << " bytes" << std::endl;
        }

        for (size_t i = 0; i < outputCount_; ++i) {
            size_t bufSize = aclmdlGetOutputSizeByIndex(modelDesc_, i);
            if (bufSize == 0) {
                bufSize = 256 * 1024 * 1024;
                std::cout << "[INFO] Output[" << i << "] is dynamic, pre-allocating "
                          << bufSize << " bytes" << std::endl;
            }
            void *outputBuf = nullptr;
            aclError ret = aclrtMalloc(&outputBuf, bufSize, ACL_MEM_MALLOC_HUGE_FIRST);
            ACL_CHECK(ret, "aclrtMalloc for output[" << i << "] failed");

            aclDataBuffer *dataBuf = aclCreateDataBuffer(outputBuf, bufSize);
            if (dataBuf == nullptr) {
                aclrtFree(outputBuf);
                std::cerr << "[ERROR] aclCreateDataBuffer for output[" << i << "] failed" << std::endl;
                return ACL_ERROR_INTERNAL_ERROR;
            }
            ret = aclmdlAddDatasetBuffer(output_, dataBuf);
            ACL_CHECK(ret, "aclmdlAddDatasetBuffer for output[" << i << "] failed");

            outputBuffers_.push_back({outputBuf, bufSize});
        }

        return ACL_SUCCESS;
    }

    aclError SetDynamicInputTensorDesc(const std::vector<InputConfig> &inputConfigs)
    {
        for (size_t i = 0; i < inputConfigs.size(); ++i) {
            const auto &cfg = inputConfigs[i];
            aclTensorDesc *tensorDesc = aclCreateTensorDesc(
                cfg.dataType,
                static_cast<int32_t>(cfg.shape.size()),
                cfg.shape.data(),
                cfg.format);
            if (tensorDesc == nullptr) {
                std::cerr << "[ERROR] aclCreateTensorDesc for input[" << i << "] failed" << std::endl;
                return ACL_ERROR_INTERNAL_ERROR;
            }

            aclError ret = aclmdlSetDatasetTensorDesc(input_, tensorDesc, i);
            if (ret != ACL_SUCCESS) {
                aclDestroyTensorDesc(tensorDesc);
                std::cerr << "[ERROR] aclmdlSetDatasetTensorDesc for input[" << i
                          << "] failed, ret=" << ret << std::endl;
                return ret;
            }
            aclDestroyTensorDesc(tensorDesc);

            std::cout << "[INFO] Set input[" << i << "] tensorDesc, name="
                      << cfg.name << ", shape=" << ShapeToString(cfg.shape) << std::endl;
        }
        return ACL_SUCCESS;
    }

    aclError Execute(bool quiet = false)
    {
        if (!quiet) std::cout << "[INFO] Executing model..." << std::endl;
        aclError ret = aclmdlExecute(modelId_, input_, output_);
        ACL_CHECK(ret, "aclmdlExecute failed");
        if (!quiet) std::cout << "[INFO] Model execution completed" << std::endl;
        return ACL_SUCCESS;
    }

    aclError ExecuteAsync(bool quiet = false)
    {
        if (!quiet) std::cout << "[INFO] Executing model (async)..." << std::endl;
        aclError ret = aclmdlExecuteAsync(modelId_, input_, output_, stream_);
        ACL_CHECK(ret, "aclmdlExecuteAsync failed");

        ret = aclrtSynchronizeStream(stream_);
        ACL_CHECK(ret, "aclrtSynchronizeStream failed");
        if (!quiet) std::cout << "[INFO] Async model execution completed" << std::endl;
        return ACL_SUCCESS;
    }

    aclError GetOutputResults(const std::string &outputDir, bool useStatic = false)
    {
        for (size_t i = 0; i < outputCount_; ++i) {
            size_t actualSize;
            std::vector<int64_t> outShape;
            aclDataType outDtype;
            aclFormat outFormat;

            if (useStatic) {
                actualSize = aclmdlGetOutputSizeByIndex(modelDesc_, i);
                const char *outName = aclmdlGetOutputNameByIndex(modelDesc_, i);

                aclmdlIODims dims;
                aclmdlGetOutputDims(modelDesc_, i, &dims);
                outShape.resize(dims.dimCount);
                for (int32_t d = 0; d < dims.dimCount; ++d) {
                    outShape[d] = dims.dims[d];
                }
                outDtype = aclmdlGetOutputDataType(modelDesc_, i);
                outFormat = aclmdlGetOutputFormat(modelDesc_, i);

                std::cout << "[INFO] Output[" << i << "]: name="
                          << (outName ? outName : "unknown")
                          << ", shape=" << ShapeToString(outShape)
                          << ", actualSize=" << actualSize
                          << ", dtype=" << outDtype
                          << ", format=" << outFormat
                          << " (static)" << std::endl;
            } else {
                aclTensorDesc *outputDesc = aclmdlGetDatasetTensorDesc(output_, i);
                if (outputDesc == nullptr) {
                    std::cerr << "[ERROR] aclmdlGetDatasetTensorDesc for output[" << i << "] failed" << std::endl;
                    return ACL_ERROR_INTERNAL_ERROR;
                }

                actualSize = aclGetTensorDescSize(outputDesc);
                int32_t ndim = aclGetTensorDescNumDims(outputDesc);
                outShape.resize(ndim);
                for (int32_t d = 0; d < ndim; ++d) {
                    aclGetTensorDescDimV2(outputDesc, d, &outShape[d]);
                }
                outDtype = aclGetTensorDescType(outputDesc);
                outFormat = aclGetTensorDescFormat(outputDesc);

                std::cout << "[INFO] Output[" << i << "]: shape=" << ShapeToString(outShape)
                          << ", actualSize=" << actualSize
                          << ", dtype=" << outDtype
                          << ", format=" << outFormat
                          << " (dynamic)" << std::endl;
            }

            aclDataBuffer *dataBuffer = aclmdlGetDatasetBuffer(output_, i);
            void *deviceData = aclGetDataBufferAddr(dataBuffer);

            std::string outFileName = outputDir + "/output_" + std::to_string(i) + ".bin";
            FILE *outFile = fopen(outFileName.c_str(), "wb");
            if (outFile == nullptr) {
                std::cerr << "[ERROR] Failed to open output file: " << outFileName << std::endl;
                continue;
            }

            if (!isDevice_) {
                void *hostData = nullptr;
                aclError ret = aclrtMallocHost(&hostData, actualSize);
                if (ret == ACL_SUCCESS && hostData != nullptr) {
                    ret = aclrtMemcpy(hostData, actualSize, deviceData, actualSize,
                                      ACL_MEMCPY_DEVICE_TO_HOST);
                    if (ret == ACL_SUCCESS) {
                        fwrite(hostData, actualSize, 1, outFile);
                    } else {
                        std::cerr << "[ERROR] aclrtMemcpy D2H for output[" << i << "] failed" << std::endl;
                    }
                    aclrtFreeHost(hostData);
                } else {
                    std::cerr << "[ERROR] aclrtMallocHost for output[" << i << "] failed" << std::endl;
                }
            } else {
                fwrite(deviceData, actualSize, 1, outFile);
            }
            fclose(outFile);
            std::cout << "[INFO] Output[" << i << "] saved to: " << outFileName << std::endl;
        }
        return ACL_SUCCESS;
    }

    void Destroy()
    {
        if (stream_ != nullptr) {
            aclrtSynchronizeStream(stream_);
        }

        if (dumpActive_) {
            aclmdlFinalizeDump();
            dumpActive_ = false;
        }

        if (input_ != nullptr) {
            for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(input_); ++i) {
                aclDataBuffer *buf = aclmdlGetDatasetBuffer(input_, i);
                aclDestroyDataBuffer(buf);
            }
            aclmdlDestroyDataset(input_);
            input_ = nullptr;
        }
        for (void *buf : inputBuffers_) {
            if (buf != nullptr) aclrtFree(buf);
        }
        inputBuffers_.clear();

        if (output_ != nullptr) {
            for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(output_); ++i) {
                aclDataBuffer *buf = aclmdlGetDatasetBuffer(output_, i);
                aclDestroyDataBuffer(buf);
            }
            aclmdlDestroyDataset(output_);
            output_ = nullptr;
        }
        for (auto &pair : outputBuffers_) {
            if (pair.first != nullptr) aclrtFree(pair.first);
        }
        outputBuffers_.clear();

        if (modelDesc_ != nullptr) {
            aclmdlDestroyDesc(modelDesc_);
            modelDesc_ = nullptr;
        }

        if (modelId_ != 0) {
            aclmdlUnload(modelId_);
            modelId_ = 0;
        }

        if (stream_ != nullptr) {
            aclrtDestroyStream(stream_);
            stream_ = nullptr;
        }

        aclrtResetDevice(deviceId_);
        aclFinalize();
    }

    void ResetInputOutput()
    {
        if (input_ != nullptr) {
            for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(input_); ++i) {
                aclDataBuffer *buf = aclmdlGetDatasetBuffer(input_, i);
                aclDestroyDataBuffer(buf);
            }
            aclmdlDestroyDataset(input_);
            input_ = nullptr;
        }
        for (void *buf : inputBuffers_) {
            if (buf != nullptr) aclrtFree(buf);
        }
        inputBuffers_.clear();

        if (output_ != nullptr) {
            for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(output_); ++i) {
                aclDataBuffer *buf = aclmdlGetDatasetBuffer(output_, i);
                aclDestroyDataBuffer(buf);
            }
            aclmdlDestroyDataset(output_);
            output_ = nullptr;
        }
        for (auto &pair : outputBuffers_) {
            if (pair.first != nullptr) aclrtFree(pair.first);
        }
        outputBuffers_.clear();
    }

private:
    uint32_t modelId_;
    aclmdlDesc *modelDesc_;
    aclmdlDataset *input_;
    aclmdlDataset *output_;
    bool isDevice_;
    aclrtStream stream_;
    int32_t deviceId_;
    size_t inputCount_ = 0;
    size_t outputCount_ = 0;
    std::vector<void *> inputBuffers_;
    std::vector<std::pair<void *, size_t>> outputBuffers_;
    bool dumpActive_;
};

static void PrintUsage(const char *prog)
{
    std::cout << "Usage: " << prog << " [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --model <path>           Path to .om model file (required)" << std::endl;
    std::cout << "  --output_dir <dir>       Output directory (default: ./output)" << std::endl;
    std::cout << "  --device_id <id>         Device ID (default: 0)" << std::endl;
    std::cout << "  --async                  Use async execution" << std::endl;
    std::cout << "  --static                 Static model (skip SetDynamicInputTensorDesc)" << std::endl;
    std::cout << "  --input <spec>           Input specification (repeatable):" << std::endl;
    std::cout << "                             name:shape:dtype:format:datafile" << std::endl;
    std::cout << "                           shape: comma-separated, e.g. 1,3,224,224" << std::endl;
    std::cout << "                           dtype: float|float16|int8|int32|int64|uint8" << std::endl;
    std::cout << "                           format: NCHW|NHWC|ND|NC1HWC0|..." << std::endl;
    std::cout << "                           datafile: path to input file (.bin or .npy)" << std::endl;
    std::cout << std::endl;
    std::cout << "Dump Options (using aclmdlInitDump/aclmdlSetDump API):" << std::endl;
    std::cout << "  --dump                   Enable data dump with auto-generated acl.json" << std::endl;
    std::cout << "  --dump_config <path>     Use existing acl.json config file for dump" << std::endl;
    std::cout << "  --dump_path <dir>        Dump output directory (default: ./dump_data)" << std::endl;
    std::cout << "  --dump_mode <mode>       Dump mode: input|output|all (default: output)" << std::endl;
    std::cout << "  --dump_level <level>     Dump level: op|kernel|all (default: op)" << std::endl;
    std::cout << "  --dump_data <type>       Dump data type: tensor|stats (default: tensor)" << std::endl;
    std::cout << "  --dump_model_name <name> Model name for dump config (optional)" << std::endl;
    std::cout << "  --dump_layer <layers>    Comma-separated layer names to dump (optional)" << std::endl;
    std::cout << std::endl;
    std::cout << "Dump Environment Variables (alternative to CLI options):" << std::endl;
    std::cout << "  DUMP_ENABLED=1           Enable data dump" << std::endl;
    std::cout << "  DUMP_CONFIG=<path>       Path to acl.json config file" << std::endl;
    std::cout << "  DUMP_PATH=<dir>          Dump output directory" << std::endl;
    std::cout << "  DUMP_MODE=<mode>         Dump mode: input|output|all" << std::endl;
    std::cout << std::endl;
    std::cout << "Profiling Options (performance data collection):" << std::endl;
    std::cout << "  --profiling              Enable profiling with auto-generated acl.json" << std::endl;
    std::cout << "  --profiling_output <dir> Profiling output directory (default: ./profiling_data)" << std::endl;
    std::cout << "  --profiling_aic_metrics <metrics> AI Core metrics: PipeUtilization|ArithmeticUtilization|Memory|etc" << std::endl;
    std::cout << "  --profiling_no_task_time Disable task_time collection" << std::endl;
    std::cout << "  --profiling_no_runtime_api Disable runtime_api collection" << std::endl;
    std::cout << "  --profiling_no_ascendcl  Disable ascendcl collection" << std::endl;
    std::cout << std::endl;
    std::cout << "Benchmark Options:" << std::endl;
    std::cout << "  --warmup <N>             Number of warmup runs before timing (default: 0)" << std::endl;
    std::cout << "  --bench <N>              Number of benchmark runs to average (default: 1)" << std::endl;
    std::cout << std::endl;
    std::cout << "Profiling Environment Variables (alternative to CLI options):" << std::endl;
    std::cout << "  PROFILING_ENABLED=1      Enable profiling" << std::endl;
    std::cout << "  PROFILING_OUTPUT=<dir>   Profiling output directory" << std::endl;
    std::cout << std::endl;
    std::cout << "Example:" << std::endl;
    std::cout << "  " << prog << " --model model.om --output_dir ./output \\" << std::endl;
    std::cout << "    --input \"input_0:1,3,224,224:float:NCHW:data/input0.bin\" \\" << std::endl;
    std::cout << "    --input \"input_1:1,64,56,56:float:NCHW:data/input1.bin\"" << std::endl;
    std::cout << std::endl;
    std::cout << "Example with dump (auto-generate acl.json):" << std::endl;
    std::cout << "  " << prog << " --model model.om --dump --dump_mode all \\" << std::endl;
    std::cout << "    --dump_path ./dump_data --dump_level op \\" << std::endl;
    std::cout << "    --input \"input_0:1,3,224,224:float:NCHW:data/input0.bin\"" << std::endl;
    std::cout << std::endl;
    std::cout << "Example with dump (specific layers):" << std::endl;
    std::cout << "  " << prog << " --model model.om --dump \\" << std::endl;
    std::cout << "    --dump_layer \"Softmax,MatMul_1\" --dump_mode output \\" << std::endl;
    std::cout << "    --input \"input_0:1,3,224,224:float:NCHW:data/input0.bin\"" << std::endl;
    std::cout << std::endl;
    std::cout << "Example with custom acl.json:" << std::endl;
    std::cout << "  " << prog << " --model model.om --dump_config ./acl.json \\" << std::endl;
    std::cout << "    --input \"input_0:1,3,224,224:float:NCHW:data/input0.bin\"" << std::endl;
    std::cout << std::endl;
    std::cout << "Example with static model:" << std::endl;
    std::cout << "  " << prog << " --model model.om --static \\" << std::endl;
    std::cout << "    --input \"input_0:1,3,224,224:float:NCHW:data/input0.bin\"" << std::endl;
    std::cout << std::endl;
    std::cout << "Example with benchmark:" << std::endl;
    std::cout << "  " << prog << " --model model.om --warmup 10 --bench 50 \\" << std::endl;
    std::cout << "    --input \"input_0:1,3,224,224:float:NCHW:data/input0.bin\"" << std::endl;
}

static aclDataType ParseDataType(const std::string &s)
{
    if (s == "float")    return ACL_FLOAT;
    if (s == "float16")  return ACL_FLOAT16;
    if (s == "int8")     return ACL_INT8;
    if (s == "int16")    return ACL_INT16;
    if (s == "int32")    return ACL_INT32;
    if (s == "int64")    return ACL_INT64;
    if (s == "uint8")    return ACL_UINT8;
    if (s == "uint16")   return ACL_UINT16;
    if (s == "uint32")   return ACL_UINT32;
    if (s == "uint64")   return ACL_UINT64;
    if (s == "double")   return ACL_DOUBLE;
    if (s == "bool")     return ACL_BOOL;
    std::cerr << "[WARN] Unknown dtype '" << s << "', defaulting to ACL_FLOAT" << std::endl;
    return ACL_FLOAT;
}

static aclFormat ParseFormat(const std::string &s)
{
    if (s == "NCHW")     return ACL_FORMAT_NCHW;
    if (s == "NHWC")     return ACL_FORMAT_NHWC;
    if (s == "ND")       return ACL_FORMAT_ND;
    if (s == "NC1HWC0")  return ACL_FORMAT_NC1HWC0;
    if (s == "NCDHW")    return ACL_FORMAT_NCDHW;
    if (s == "NDC1HWC0") return ACL_FORMAT_NDC1HWC0;
    if (s == "FRACTAL_Z") return ACL_FORMAT_FRACTAL_Z;
    std::cerr << "[WARN] Unknown format '" << s << "', defaulting to ACL_FORMAT_ND" << std::endl;
    return ACL_FORMAT_ND;
}

static bool ParseInputSpec(const std::string &spec, InputConfig &cfg)
{
    std::vector<std::string> parts;
    std::istringstream iss(spec);
    std::string part;
    while (std::getline(iss, part, ':')) {
        parts.push_back(part);
    }
    if (parts.size() < 5) {
        std::cerr << "[ERROR] Invalid input spec (need 5 colon-separated fields): " << spec << std::endl;
        return false;
    }
    cfg.name = parts[0];

    std::istringstream shapeStream(parts[1]);
    std::string dimStr;
    while (std::getline(shapeStream, dimStr, ',')) {
        cfg.shape.push_back(std::stoll(dimStr));
    }

    cfg.dataType = ParseDataType(parts[2]);
    cfg.format = ParseFormat(parts[3]);
    cfg.dataFile = parts[4];
    return true;
}

int main(int argc, char *argv[])
{
    auto total_start = std::chrono::high_resolution_clock::now();
    
    std::string modelPath;
    std::string outputDir = "./output";
    int32_t deviceId = 0;
    bool useAsync = false;
    bool useStatic = false;
    int32_t warmupRuns = 0;
    int32_t benchRuns = 1;
    std::vector<std::string> inputSpecs;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            modelPath = argv[++i];
        } else if (arg == "--output_dir" && i + 1 < argc) {
            outputDir = argv[++i];
        } else if (arg == "--device_id" && i + 1 < argc) {
            deviceId = std::stoi(argv[++i]);
        } else if (arg == "--async") {
            useAsync = true;
        } else if (arg == "--static") {
            useStatic = true;
        } else if (arg == "--warmup" && i + 1 < argc) {
            warmupRuns = std::stoi(argv[++i]);
        } else if (arg == "--bench" && i + 1 < argc) {
            benchRuns = std::stoi(argv[++i]);
        } else if (arg == "--input" && i + 1 < argc) {
            inputSpecs.push_back(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            PrintUsage(argv[0]);
            return 0;
        }
    }

    if (modelPath.empty() || inputSpecs.empty()) {
        PrintUsage(argv[0]);
        return 1;
    }

    std::vector<InputConfig> inputConfigs;
    for (const auto &spec : inputSpecs) {
        InputConfig cfg;
        if (!ParseInputSpec(spec, cfg)) {
            return 1;
        }
        inputConfigs.push_back(cfg);
        std::cout << "[INFO] Parsed input: name=" << cfg.name
                  << ", shape=" << ShapeToString(cfg.shape)
                  << ", file=" << cfg.dataFile << std::endl;
    }

    mkdir(outputDir.c_str(), 0755);

    DumpConfig dumpCfg = ParseDumpConfig(argc, argv);
    ProfilingConfig profCfg = ParseProfilingConfig(argc, argv);
    std::string aclConfigPath;

    // Generate acl.json if either dump or profiling is enabled
    if (dumpCfg.enabled || profCfg.enabled) {
        if (!dumpCfg.configPath.empty()) {
            aclConfigPath = dumpCfg.configPath;
            std::cout << "[INFO] Using provided acl.json config: " << aclConfigPath << std::endl;
        } else {
            aclConfigPath = outputDir + "/acl.json";
            mkdir(outputDir.c_str(), 0755);
            if (!GenerateAclJson(dumpCfg, profCfg, aclConfigPath)) {
                std::cerr << "[ERROR] Failed to generate acl.json" << std::endl;
                return 1;
            }
            std::cout << "[INFO] Auto-generated acl.json config: " << aclConfigPath << std::endl;
            
            if (dumpCfg.enabled) {
                std::cout << "[INFO] Dump settings:" << std::endl;
                std::cout << "  dump_path=" << dumpCfg.dumpPath << std::endl;
                std::cout << "  dump_mode=" << dumpCfg.dumpMode << std::endl;
                std::cout << "  dump_level=" << dumpCfg.dumpLevel << std::endl;
                std::cout << "  dump_data=" << dumpCfg.dumpData << std::endl;
                if (!dumpCfg.modelName.empty()) {
                    std::cout << "  model_name=" << dumpCfg.modelName << std::endl;
                }
                if (!dumpCfg.layers.empty()) {
                    std::cout << "  layers=";
                    for (size_t i = 0; i < dumpCfg.layers.size(); ++i) {
                        if (i > 0) std::cout << ",";
                        std::cout << dumpCfg.layers[i];
                    }
                    std::cout << std::endl;
                }
                mkdir(dumpCfg.dumpPath.c_str(), 0755);
                
                // Convert to absolute path for clarity
                char *abs_dump_path = realpath(dumpCfg.dumpPath.c_str(), nullptr);
                if (abs_dump_path) {
                    std::cout << "  dump_path (absolute)=" << abs_dump_path << std::endl;
                    free(abs_dump_path);
                }
            }
            
            if (profCfg.enabled) {
                std::cout << "[INFO] Profiling settings:" << std::endl;
                std::cout << "  output=" << profCfg.outputPath << std::endl;
                if (!profCfg.aicMetrics.empty()) {
                    std::cout << "  aic_metrics=" << profCfg.aicMetrics << std::endl;
                }
                std::cout << "  task_time=" << (profCfg.taskTime ? "on" : "off") << std::endl;
                std::cout << "  runtime_api=" << (profCfg.runtimeApi ? "on" : "off") << std::endl;
                std::cout << "  ascendcl=" << (profCfg.ascendcl ? "on" : "off") << std::endl;
                mkdir(profCfg.outputPath.c_str(), 0755);
            }
        }
    }

    ModelInfer infer;

    // CANN provides two separate approaches for dump configuration:
    //   Approach 1: aclInit(configPath) - reads dump section from acl.json
    //   Approach 2: aclmdlInitDump() + aclmdlSetDump(configPath) - explicit dump APIs
    //
    // Based on testing, Approach 1 (aclInit with config) is more reliable for all dump levels.
    // Strategy: Always use Approach 1 when dump is enabled.
    bool useAclInitForDump = dumpCfg.enabled;

    const char *initConfigPath = nullptr;
    if (useAclInitForDump) {
        initConfigPath = aclConfigPath.c_str();
        std::cout << "[INFO] Using aclInit(configPath) for dump, dump_level=" << dumpCfg.dumpLevel << std::endl;
    } else if (profCfg.enabled && !dumpCfg.enabled) {
        initConfigPath = aclConfigPath.c_str();
    }
    if (dumpCfg.enabled && profCfg.enabled) {
        std::cout << "[WARN] Both dump and profiling enabled. Profiling config may not be applied." << std::endl;
        std::cout << "[WARN] To collect profiling data, run again with --profiling only (without --dump)." << std::endl;
    }
    
    auto model_load_start = std::chrono::high_resolution_clock::now();
    aclError ret = infer.Init(deviceId, initConfigPath);
    if (ret != ACL_SUCCESS) return 1;

    // When using aclInit(configPath) for dump, skip aclmdlInitDump/SetDump
    // The dump configuration is already loaded by aclInit

    ret = infer.LoadModel(modelPath);
    if (ret != ACL_SUCCESS) return 1;
    auto model_load_end = std::chrono::high_resolution_clock::now();

    auto data_prep_start = std::chrono::high_resolution_clock::now();
    ret = infer.PrepareInputOutput(inputConfigs, useStatic);
    if (ret != ACL_SUCCESS) return 1;

    if (!useStatic) {
        ret = infer.SetDynamicInputTensorDesc(inputConfigs);
        if (ret != ACL_SUCCESS) return 1;
    } else {
        std::cout << "[INFO] Static model mode: skipping SetDynamicInputTensorDesc" << std::endl;
    }
    auto data_prep_end = std::chrono::high_resolution_clock::now();

    // Warmup runs (not timed)
    if (warmupRuns > 0) {
        std::cout << "[INFO] Running " << warmupRuns << " warmup iteration(s)..." << std::endl;
        for (int32_t w = 0; w < warmupRuns; ++w) {
            if (useAsync) {
                ret = infer.ExecuteAsync(true);
            } else {
                ret = infer.Execute(true);
            }
            if (ret != ACL_SUCCESS) {
                std::cerr << "[ERROR] Warmup run " << w << " failed" << std::endl;
                return 1;
            }
        }
        std::cout << "[INFO] Warmup completed" << std::endl;
    }

    // Benchmark runs (timed)
    std::vector<double> infer_times;
    infer_times.reserve(benchRuns);

    for (int32_t b = 0; b < benchRuns; ++b) {
        auto infer_start = std::chrono::high_resolution_clock::now();
        if (useAsync) {
            ret = infer.ExecuteAsync(true);
        } else {
            ret = infer.Execute(true);
        }
        if (ret != ACL_SUCCESS) {
            std::cerr << "[ERROR] Benchmark run " << b << " failed" << std::endl;
            return 1;
        }
        auto infer_end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration_cast<std::chrono::microseconds>(infer_end - infer_start).count() / 1000.0;
        infer_times.push_back(ms);
    }

    // Sort for percentile calculation
    std::sort(infer_times.begin(), infer_times.end());

    double infer_avg = 0.0;
    for (double t : infer_times) infer_avg += t;
    infer_avg /= infer_times.size();

    double infer_min = infer_times.front();
    double infer_max = infer_times.back();
    double infer_p50 = infer_times[infer_times.size() / 2];
    double infer_p99 = infer_times[static_cast<size_t>(infer_times.size() * 0.99)];

    auto output_start = std::chrono::high_resolution_clock::now();
    ret = infer.GetOutputResults(outputDir, useStatic);
    if (ret != ACL_SUCCESS) return 1;
    auto output_end = std::chrono::high_resolution_clock::now();

    // When using aclInit(configPath) for dump, no need to call FinalizeDump
    // The dump data is automatically flushed during aclFinalize()
    if (dumpCfg.enabled && !useAclInitForDump) {
        ret = infer.FinalizeDump();
        if (ret != ACL_SUCCESS) {
            std::cerr << "[WARN] FinalizeDump returned error, but continuing cleanup" << std::endl;
        } else {
            std::cout << "[INFO] Dump finalized. Check for data in: " << dumpCfg.dumpPath << std::endl;
        }
    } else if (dumpCfg.enabled && useAclInitForDump) {
        std::cout << "[INFO] Dump enabled via aclInit. Data will be flushed during cleanup." << std::endl;
        std::cout << "[INFO] Check for dump data in: " << dumpCfg.dumpPath << std::endl;
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    
    // Calculate timing
    auto model_load_ms = std::chrono::duration_cast<std::chrono::microseconds>(model_load_end - model_load_start).count() / 1000.0;
    auto data_prep_ms = std::chrono::duration_cast<std::chrono::microseconds>(data_prep_end - data_prep_start).count() / 1000.0;
    auto output_ms = std::chrono::duration_cast<std::chrono::microseconds>(output_end - output_start).count() / 1000.0;
    auto total_ms = std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start).count() / 1000.0;
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "End-to-End Performance Summary" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "  Model Loading:      " << std::fixed << std::setprecision(2) << model_load_ms << " ms" << std::endl;
    std::cout << "  Data Preparation:   " << std::fixed << std::setprecision(2) << data_prep_ms << " ms" << std::endl;
    if (benchRuns == 1) {
        std::cout << "  Inference:          " << std::fixed << std::setprecision(2) << infer_avg << " ms" << std::endl;
    } else {
        std::cout << "  Inference (avg):    " << std::fixed << std::setprecision(2) << infer_avg << " ms"
                  << "  (min=" << infer_min << ", p50=" << infer_p50 << ", p99=" << infer_p99 << ", max=" << infer_max << ")" << std::endl;
    }
    std::cout << "  Output Saving:      " << std::fixed << std::setprecision(2) << output_ms << " ms" << std::endl;
    std::cout << "  " << std::string(56, '-') << std::endl;
    std::cout << "  Total E2E:          " << std::fixed << std::setprecision(2) << total_ms << " ms" << std::endl;
    if (warmupRuns > 0 || benchRuns > 1) {
        std::cout << "  (warmup=" << warmupRuns << ", bench_runs=" << benchRuns << ")" << std::endl;
    }
    std::cout << std::string(60, '=') << std::endl;

    std::cout << "[INFO] Inference completed successfully. Results in: " << outputDir << std::endl;
    if (dumpCfg.enabled) {
        std::cout << "[INFO] Dump data saved to: " << dumpCfg.dumpPath << std::endl;
    }
    return 0;
}
