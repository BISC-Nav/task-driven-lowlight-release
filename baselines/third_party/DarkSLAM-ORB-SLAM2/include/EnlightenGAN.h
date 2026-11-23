#ifndef ENLIGHTENGAN_H
#define ENLIGHTENGAN_H

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <string>
namespace ORB_SLAM2
{
class EnlightenGAN {
public:
    EnlightenGAN(const std::string& modelPath);
    ~EnlightenGAN();

    // 输入弱光图像，返回增强后的图像
    cv::Mat Enhance(cv::Mat& inputImg);

private:
    Ort::Env env;
    Ort::Session* session;
    Ort::MemoryInfo memory_info{nullptr};
};
}
#endif