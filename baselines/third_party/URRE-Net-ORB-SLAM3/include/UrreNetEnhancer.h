#ifndef URRENETENHANCER_H
#define URRENETENHANCER_H

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>
#include <memory>

namespace ORB_SLAM3
{

class UrreNetEnhancer
{
public:
    // 构造函数：传入模型路径和 SLAM 配置的宽高
    UrreNetEnhancer(const std::string& strOnnxPath, int nWidth, int nHeight);
    ~UrreNetEnhancer();

    // 核心处理函数
    cv::Mat Process(const cv::Mat& imLowLight);

private:
    int mWidth;
    int mHeight;
    
    // ONNX Runtime 核心组件
    Ort::Env mEnv;
    Ort::SessionOptions mSessionOptions;
    std::unique_ptr<Ort::Session> mpSession;
    Ort::MemoryInfo mMemoryInfo;

    // IO 节点名称管理
    std::vector<const char*> mInputNodeNames;
    std::vector<const char*> mOutputNodeNames;

    // 张量维度
    std::vector<int64_t> mInputDims;
    std::vector<int64_t> mOutputDims;

    // 预分配内存，避免每帧重新申请 (零拷贝与极速显存交互的关键)
    std::vector<float> mInputTensorValues;
    std::vector<float> mOutputTensorValues;
};

} // namespace ORB_SLAM3

#endif // URRENETENHANCER_H