#pragma once
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <limits>
#include <algorithm>
#include <numeric>
#define ASSERT(x) do { if(!(x)) throw std::runtime_error(#x); } while(0)
#define CHECK(x) do { auto err=(x); if(err!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(err)); } while(0)
namespace tensorrt_log {
class Logger: public nvinfer1::ILogger {
 public:
 Severity threshold=Severity::kWARNING;
 void log(Severity s,const char* msg) noexcept override { if(s<=std::max(threshold,Severity::kWARNING)) std::cerr<<msg<<std::endl; }
 Severity getReportableSeverity() const {return threshold;}
 Logger& getTRTLogger(){return *this;}
};
inline Logger gLogger;
inline void setReportableSeverity(Logger::Severity s){gLogger.threshold=s;}
}
namespace tensorrt_buffer {
template<typename T> using TensorRTUniquePtr=std::unique_ptr<T>;
struct StreamDeleter {void operator()(cudaStream_t* p)const {if(p){cudaStreamDestroy(*p);delete p;}}};
inline auto makeCudaStream(){std::unique_ptr<cudaStream_t,StreamDeleter> p(new cudaStream_t);CHECK(cudaStreamCreate(p.get()));return p;}
inline void enableDLA(nvinfer1::IBuilder*,nvinfer1::IBuilderConfig* c,int core){if(core>=0){c->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);c->setDLACore(core);c->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);}}
inline int indexOf(const nvinfer1::ICudaEngine& e,const char* name){for(int i=0;i<e.getNbIOTensors();++i)if(std::strcmp(e.getIOTensorName(i),name)==0)return i;return -1;}
inline size_t getElementSize(nvinfer1::DataType t){using T=nvinfer1::DataType;switch(t){case T::kFLOAT:case T::kINT32:return 4;case T::kHALF:return 2;case T::kINT64:return 8;case T::kINT8:case T::kBOOL:case T::kUINT8:return 1;default:throw std::runtime_error("unsupported I/O dtype");}}
inline size_t volume(nvinfer1::Dims d){if(d.nbDims<0)throw std::runtime_error("invalid rank");size_t n=1;for(int i=0;i<d.nbDims;++i){if(d.d[i]<0)throw std::runtime_error("unresolved dynamic dimension");if(d.d[i] && n>std::numeric_limits<size_t>::max()/d.d[i])throw std::overflow_error("tensor dimensions");n*=d.d[i];}return n;}
}
