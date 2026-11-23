#include "EnlightenGAN.h"
#include <iostream>
#include <vector>
#include <algorithm>

namespace ORB_SLAM2
{

EnlightenGAN::EnlightenGAN(const std::string& modelPath) 
    : env(ORT_LOGGING_LEVEL_WARNING, "EnlightenGAN_Inference") {
    
    Ort::SessionOptions session_options;
    
    // 1. 针对 RTX 5090 的 CUDA 配置
    try {
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = 0;
        // 开启常用的针对 Blackwell 架构的优化
        cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
        cuda_options.gpu_mem_limit = 2 * 1024 * 1024 * 1024; // 限制 2GB 显存，防止抢占 SLAM 资源
        session_options.AppendExecutionProvider_CUDA(cuda_options);
        std::cout << "[EnlightenGAN] CUDA Execution Provider 成功加载 (RTX 5090)" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[EnlightenGAN] CUDA 启动失败，回退到 CPU: " << e.what() << std::endl;
    }

    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // 2. 创建 Session
    // 注意：在 Windows 下路径需转为宽字符，Linux 下直接使用 string
    session = new Ort::Session(env, modelPath.c_str(), session_options);
    
    memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
}

EnlightenGAN::~EnlightenGAN() {
    if (session) delete session;
}

cv::Mat EnlightenGAN::Enhance(cv::Mat& inputImg) {
    if (inputImg.empty()) return inputImg;

    // --- 1. 统一通道处理 ---
    cv::Mat colorImg;
    if (inputImg.channels() == 1) {
        cv::cvtColor(inputImg, colorImg, cv::COLOR_GRAY2BGR);
    } else {
        colorImg = inputImg;
    }

    // --- 2. 颜色空间转换 (核心修改：必须转为 RGB) ---
    // OpenCV 默认是 BGR，而开源模型是在 RGB 空间训练的
    cv::Mat rgbImg;
    cv::cvtColor(colorImg, rgbImg, cv::COLOR_BGR2RGB);

    int orig_h = rgbImg.rows;
    int orig_w = rgbImg.cols;

    // --- 3. 尺寸对齐 (32的倍数) ---
    int new_h = (orig_h % 32 == 0) ? orig_h : ((orig_h / 32) + 1) * 32;
    int new_w = (orig_w % 32 == 0) ? orig_w : ((orig_w / 32) + 1) * 32;

    cv::Mat resizedImg;
    cv::resize(rgbImg, resizedImg, cv::Size(new_w, new_h));

    // --- 4. 归一化 (核心修改：对齐 Python 库的 / 255.0) ---
    // 不要使用 127.5 和 -1.0，直接映射到 [0, 1]
    cv::Mat floatImg;
    resizedImg.convertTo(floatImg, CV_32FC3, 1.0 / 255.0);

    // --- 5. HWC 转 NCHW ---
    std::vector<float> inputTensorValues(1 * 3 * new_h * new_w);
    float* inputPtr = inputTensorValues.data();
    
    std::vector<cv::Mat> channels(3);
    cv::split(floatImg, channels); 
    for (int i = 0; i < 3; ++i) {
        // 由于之前转了 RGB，这里 channels[0]=R, [1]=G, [2]=B
        std::memcpy(inputPtr + i * new_h * new_w, channels[i].data, new_h * new_w * sizeof(float));
    }

    // --- 6. 推理准备 ---
    std::vector<int64_t> inputShape = {1, 3, new_h, new_w};
    auto inputTensor = Ort::Value::CreateTensor<float>(
        memory_info, inputTensorValues.data(), inputTensorValues.size(), 
        inputShape.data(), inputShape.size()
    );

    const char* inputNames[] = {"input"};
    const char* outputNames[] = {"output"}; // 如果还是黑的，请尝试改为 "255"

    try {
        // --- 7. GPU 推理 ---
        auto outputTensors = session->Run(
            Ort::RunOptions{nullptr}, 
            inputNames, &inputTensor, 1, 
            outputNames, 1
        );

        float* outputData = outputTensors[0].GetTensorMutableData<float>();
        
        // --- 8. 后处理 (NCHW -> HWC) ---
        cv::Mat outChannels[3];
        for (int i = 0; i < 3; ++i) {
            outChannels[i] = cv::Mat(new_h, new_w, CV_32FC1, outputData + i * new_h * new_w);
        }
        
        cv::Mat mergedImg;
        cv::merge(outChannels, 3, mergedImg);

        // --- 9. 数值还原与 Clip ---
        // 1. 将 0-1 恢复到 0-255
        cv::Mat finalImg;
        mergedImg.convertTo(finalImg, CV_32FC3, 255.0); 

        // 2. 限制边界 (对齐 np.clip)
        cv::threshold(finalImg, finalImg, 255.0, 255.0, cv::THRESH_TRUNC);
        cv::threshold(finalImg, finalImg, 0.0, 0.0, cv::THRESH_TOZERO);

        // 3. 转换为 8 位 RGB
        cv::Mat final8U;
        finalImg.convertTo(final8U, CV_8UC3);

        // --- 10. 恢复尺寸与颜色空间 ---
        cv::Mat result;
        cv::resize(final8U, result, cv::Size(orig_w, orig_h));

        // 转回 BGR 供 SLAM 和 OpenCV 显示
        cv::cvtColor(result, result, cv::COLOR_RGB2BGR);

        if (inputImg.channels() == 1) {
            cv::cvtColor(result, result, cv::COLOR_BGR2GRAY);
        }

        return result;

    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime 异常: " << e.what() << std::endl;
        return inputImg;
    }
}
}