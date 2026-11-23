#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<iomanip>
#include<dirent.h>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
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

// 自动读取文件夹内所有 png 图片并排序
void LoadImages(const string &strPath, vector<string> &vstrImageFilenames, vector<double> &vTimestamps) {
    vstrImageFilenames.clear();
    vTimestamps.clear();

    DIR *dir;
    struct dirent *ptr;
    if ((dir = opendir(strPath.c_str())) == NULL) {
        cerr << "Error: Cannot open directory " << strPath << endl;
        exit(-1);
    }

    vector<string> vFiles;
    while ((ptr = readdir(dir)) != NULL) {
        string name = string(ptr->d_name);
        if (name.size() > 4 && name.substr(name.size() - 4) == ".png") {
            vFiles.push_back(name);
        }
    }
    closedir(dir);

    if (vFiles.empty()) {
        cerr << "Error: No png files found in " << strPath << endl;
        exit(-1);
    }

    // 必须排序，确保 000000.png 在 000001.png 之前
    sort(vFiles.begin(), vFiles.end());

    for (size_t i = 0; i < vFiles.size(); i++) {
        vstrImageFilenames.push_back(strPath + "/" + vFiles[i]);
        // 假设采样频率为 10Hz (0.1s 间隔)
        vTimestamps.push_back(i * 0.1);
    }
    
    cout << "Successfully loaded " << vstrImageFilenames.size() << " images." << endl;
}

int main(int argc, char **argv) {
    if(argc != 4) {
        cerr << "Usage: ./mono_tartanair path_to_vocabulary path_to_settings path_to_sequence" << endl;
        return 1;
    }

    // 1. 加载图像列表
    vector<string> vstrImageFilenames;
    vector<double> vTimestamps;
    LoadImages(string(argv[3]), vstrImageFilenames, vTimestamps);
    int nImages = vstrImageFilenames.size();

    // 2. 初始化 SLAM 系统
    // 参数：词库路径, 配置文件路径, 传感器类型, 是否开启可视化界面
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::MONOCULAR, true);

    cout << endl << "------- Start processing sequence -------" << endl;

    // 3. 主循环：逐帧送入 SLAM
    for(int ni=0; ni<nImages; ni++) {
        cv::Mat im = cv::imread(vstrImageFilenames[ni], cv::IMREAD_UNCHANGED);
        double tframe = vTimestamps[ni];

        if(im.empty()) {
            cerr << "Failed to load image at: " << vstrImageFilenames[ni] << endl;
            continue;
        }

        cv::Mat imLowLight = SimulateLowLight(im, 0.1f, 0.01f);

        // 核心：传递给 ORB-SLAM3
        SLAM.TrackMonocular(imLowLight, tframe);

        // 如果处理太快，可以根据时间戳稍微 sleep 一下，但这在离线跑数据时通常不需要
    }

    // 4. 停止并保存位姿（如果有设置输出文件）
    SLAM.Shutdown();

    SLAM.SaveTrajectoryEuRoC("Trajectory_TartanAir.txt");
    
    cout << "Finished. Press enter to exit." << endl;
    return 0;
}
