#include "UrreNetEnhancer.h"
#include <iostream>
#include <cstring>

namespace ORB_SLAM3
{

UrreNetEnhancer::UrreNetEnhancer(const std::string& strOnnxPath, int nWidth, int nHeight)
    : mWidth(nWidth), 
      mHeight(nHeight),
      mEnv(ORT_LOGGING_LEVEL_WARNING, "UrreNet"),
      mMemoryInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
{
    // 1. 配置 Session
    mSessionOptions.SetIntraOpNumThreads(1);
    mSessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    
    // 2. 唤醒 RTX 5090 CUDA 加速
    OrtCUDAProviderOptions cuda_options;
    cuda_options.device_id = 0;
    cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
    cuda_options.arena_extend_strategy = 0;
    
    try {
        mSessionOptions.AppendExecutionProvider_CUDA(cuda_options);
        std::cout << "[UrreNetEnhancer] CUDA Execution Provider appended successfully." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[UrreNetEnhancer] WARNING: Failed to append CUDA EP: " << e.what() << std::endl;
        std::cerr << "Falling back to CPU. Inference will be slow." << std::endl;
    }

    // 3. 加载模型
    try {
        mpSession = std::make_unique<Ort::Session>(mEnv, strOnnxPath.c_str(), mSessionOptions);
        std::cout << "[UrreNetEnhancer] ONNX Model loaded successfully!" << std::endl;
    } catch (const Ort::Exception& e) {
        std::cerr << "[UrreNetEnhancer] ERROR loading ONNX model: " << e.what() << std::endl;
        exit(-1);
    }

    // 4. 精准打击：配置输入输出名称
    mInputNodeNames = {"input"};
    // 【核心差异】：我们网络有 4 个输出，但这里只向引擎请求 "enhance_image"
    // ONNX Runtime 会智能截断不需要的计算分支，最大化节省 5090 的算力
    mOutputNodeNames = {"enhance_image"}; 

    mInputDims = {1, 3, mHeight, mWidth};
    mOutputDims = {1, 3, mHeight, mWidth};

    // 5. 预分配固定大小的内存池
    size_t tensorSize = 3 * mHeight * mWidth;
    mInputTensorValues.resize(tensorSize);
    mOutputTensorValues.resize(tensorSize);
}

UrreNetEnhancer::~UrreNetEnhancer()
{
}

cv::Mat UrreNetEnhancer::Process(const cv::Mat& imLowLight)
{
    if (imLowLight.empty()) return imLowLight;

    // ============================================
    // 1. 预处理 (OpenCV CPU -> 内存池)
    // ============================================
    cv::Mat imRgb, imResized;
    // SLAM 传入的可能是灰度图也可能是 BGR
    if (imLowLight.channels() == 1) {
        cv::cvtColor(imLowLight, imRgb, cv::COLOR_GRAY2RGB);
    } else {
        cv::cvtColor(imLowLight, imRgb, cv::COLOR_BGR2RGB);
    }
    
    // 调整为网络所需的输入尺寸 (如 752x480)
    cv::resize(imRgb, imResized, cv::Size(mWidth, mHeight));

    // 归一化 (1.0/255.0) 并完成 HWC 到 NCHW 的格式转换
    cv::Mat blob = cv::dnn::blobFromImage(imResized, 1.0 / 255.0, cv::Size(mWidth, mHeight), cv::Scalar(0, 0, 0), false, false);

    // 将数据塞入我们预分配的输入内存池
    std::memcpy(mInputTensorValues.data(), blob.ptr<float>(), mInputTensorValues.size() * sizeof(float));

    // 创建零拷贝 Tensor
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        mMemoryInfo, mInputTensorValues.data(), mInputTensorValues.size(), mInputDims.data(), mInputDims.size());
    Ort::Value outputTensor = Ort::Value::CreateTensor<float>(
        mMemoryInfo, mOutputTensorValues.data(), mOutputTensorValues.size(), mOutputDims.data(), mOutputDims.size());

    // ============================================
    // 2. RTX 5090 极速推理
    // ============================================
    try {
        mpSession->Run(Ort::RunOptions{nullptr}, 
                       mInputNodeNames.data(), &inputTensor, 1, 
                       mOutputNodeNames.data(), &outputTensor, 1);
    } catch (const Ort::Exception& e) {
        std::cerr << "[UrreNetEnhancer] Inference error: " << e.what() << std::endl;
        // 若推理崩溃，直接返回原图灰度，强行保住 SLAM 不掉线
        cv::Mat fallbackGray;
        if(imLowLight.channels() == 3) cv::cvtColor(imLowLight, fallbackGray, cv::COLOR_BGR2GRAY);
        else fallbackGray = imLowLight.clone();
        return fallbackGray;
    }

    // ============================================
    // 3. 后处理 (内存池 -> OpenCV 灰度图)
    // ============================================
    float* outPtr = mOutputTensorValues.data();
    
    // 剥离 NCHW 的 R, G, B 三个内存面
    cv::Mat matR(mHeight, mWidth, CV_32FC1, outPtr);
    cv::Mat matG(mHeight, mWidth, CV_32FC1, outPtr + mHeight * mWidth);
    cv::Mat matB(mHeight, mWidth, CV_32FC1, outPtr + 2 * mHeight * mWidth);

    // 重新组合为 OpenCV 习惯的 BGR 格式
    std::vector<cv::Mat> channels = {matB, matG, matR};
    cv::Mat imEnhancedF;
    cv::merge(channels, imEnhancedF);

    // 反归一化 (乘以 255) 并压缩回 8U 格式
    cv::Mat imEnhanced8U;
    imEnhancedF.convertTo(imEnhanced8U, CV_8UC3, 255.0);

    // 转回灰度图，适配 ORB-SLAM3 底层特征提取器
    //cv::Mat imGrayOut;
    //cv::cvtColor(imEnhanced8U, imGrayOut, cv::COLOR_BGR2GRAY);

    // 如果发生了缩放，必须将增强后的图像还原回相机的原始分辨率
    if (imEnhanced8U.size() != imLowLight.size()) {
        cv::resize(imEnhanced8U, imEnhanced8U, imLowLight.size());
    }

    return imEnhanced8U;
}

} // namespace ORB_SLAM3