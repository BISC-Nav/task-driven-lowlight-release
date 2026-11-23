#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<iomanip> // 用于处理文件名填充

#include<opencv2/core/core.hpp>
#include<System.h>

using namespace std;

cv::Mat SimulateLowLight(const cv::Mat& inputImg, float lowLightFactor = 0.1f, float noiseVariance = 0.01f)
{
    if (inputImg.empty()) return inputImg;

    cv::Mat imgFloat;
    inputImg.convertTo(imgFloat, CV_32F, 1.0 / 255.0);

    // 1. 逆 Gamma 校正回线性空间 (x^2.2)
    cv::Mat imgLinear;
    cv::pow(imgFloat, 2.2, imgLinear);

    // 2. 降低物理照度
    cv::Mat imgDark = imgLinear * lowLightFactor;

    // 3. 注入传感器噪声 (高斯噪声)
    cv::Mat noise = cv::Mat::zeros(imgDark.size(), imgDark.type());
    cv::randn(noise, 0, noiseVariance); 
    cv::Mat imgNoisy = imgDark + noise;
    
    // 限制范围在 [0, 1]
    cv::Mat imgClipped;
    cv::threshold(imgNoisy, imgClipped, 1.0, 1.0, cv::THRESH_TRUNC); // 大于 1.0 的变 1.0
    cv::threshold(imgClipped, imgClipped, 0.0, 0.0, cv::THRESH_TOZERO); // 小于 0.0 的变 0.0

    // 4. Gamma 校正回 sRGB 空间 (x^(1/2.2))
    cv::Mat imgFinal;
    cv::pow(imgClipped, 1.0 / 2.2, imgFinal);

    cv::Mat outImg;
    imgFinal.convertTo(outImg, CV_8U, 255.0);
    return outImg;
}

int main(int argc, char **argv) {
    if(argc != 4) {
        cerr << endl << "Usage: ./mono_tartanair path_to_vocabulary path_to_settings path_to_image_folder" << endl;
        return 1;
    }

    // 初始化 SLAM 系统
    ORB_SLAM2::System SLAM(argv[1], argv[2], ORB_SLAM2::System::MONOCULAR, true);

    string imgDir = string(argv[3]);
    if(imgDir.back() != '/') imgDir += "/";

    cout << "Start processing sequence..." << endl;

    // 循环读取图片：000000.png, 000001.png ...
    for(int ni=0; ; ni++) {
        // 构建文件名 000000.png
        stringstream ss;
        ss << setw(6) << setfill('0') << ni << ".png";
        string imgPath = imgDir + ss.str();

        // 尝试读取
        cv::Mat im = cv::imread(imgPath, cv::IMREAD_COLOR);
        
        if(im.empty()) {
            if (ni == 0) {
                cerr << "Error: Could not find first image at " << imgPath << endl;
                return 1;
            }
            cout << "End of sequence reached at frame " << ni << endl;
            break; 
        }

        cv::Mat imLowLight = SimulateLowLight(im, 0.1f, 0.01f);

        double tframe = ni * 0.1; // 假设 10fps

        // 追踪
        SLAM.TrackMonocular(imLowLight, tframe);
    }

    // 停止并保存轨迹
    SLAM.Shutdown();
    SLAM.SaveTrajectoryTUM("Trajectory_TartanAir.txt");

    return 0;
}