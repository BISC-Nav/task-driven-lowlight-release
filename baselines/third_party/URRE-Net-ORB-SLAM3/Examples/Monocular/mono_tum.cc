/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>

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

void LoadImages(const string &strFile, vector<string> &vstrImageFilenames,
                vector<double> &vTimestamps);

int main(int argc, char **argv)
{
    if(argc != 4)
    {
        cerr << endl << "Usage: ./mono_tum path_to_vocabulary path_to_settings path_to_sequence" << endl;
        return 1;
    }

    // Retrieve paths to images
    vector<string> vstrImageFilenames;
    vector<double> vTimestamps;
    string strFile = string(argv[3])+"/rgb.txt";
    LoadImages(strFile, vstrImageFilenames, vTimestamps);

    int nImages = vstrImageFilenames.size();

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(argv[1],argv[2],ORB_SLAM3::System::MONOCULAR,true);
    float imageScale = SLAM.GetImageScale();

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    double t_resize = 0.f;
    double t_track = 0.f;

    std::ofstream f_clear("./eval_metrics_tmp.txt", std::ios_base::trunc);
    f_clear.close();
    
    // Main loop
    cv::Mat im;
    for(int ni=0; ni<nImages; ni++)
    {
        // Read image from file
        im = cv::imread(string(argv[3])+"/"+vstrImageFilenames[ni],cv::IMREAD_UNCHANGED); //,cv::IMREAD_UNCHANGED);
        double tframe = vTimestamps[ni];

        if(im.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(argv[3]) << "/" << vstrImageFilenames[ni] << endl;
            return 1;
        }

        if(imageScale != 1.f)
        {
#ifdef REGISTER_TIMES
    #ifdef COMPILEDWITHC11
            std::chrono::steady_clock::time_point t_Start_Resize = std::chrono::steady_clock::now();
    #else
            std::chrono::monotonic_clock::time_point t_Start_Resize = std::chrono::monotonic_clock::now();
    #endif
#endif
            int width = im.cols * imageScale;
            int height = im.rows * imageScale;
            cv::resize(im, im, cv::Size(width, height));
#ifdef REGISTER_TIMES
    #ifdef COMPILEDWITHC11
            std::chrono::steady_clock::time_point t_End_Resize = std::chrono::steady_clock::now();
    #else
            std::chrono::monotonic_clock::time_point t_End_Resize = std::chrono::monotonic_clock::now();
    #endif
            t_resize = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t_End_Resize - t_Start_Resize).count();
            SLAM.InsertResizeTime(t_resize);
#endif
        }

        cv::Mat imLowLight = SimulateLowLight(im, 0.1f, 0.01f);

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        // Pass the image to the SLAM system
        SLAM.TrackMonocular(imLowLight,tframe);

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

#ifdef REGISTER_TIMES
            t_track = t_resize + std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t2 - t1).count();
            SLAM.InsertTrackTime(t_track);
#endif

        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesTrack[ni]=ttrack;

        // Wait to load the next frame
        double T=0;
        if(ni<nImages-1)
            T = vTimestamps[ni+1]-tframe;
        else if(ni>0)
            T = tframe-vTimestamps[ni-1];

        if(ttrack<T)
            usleep((T-ttrack)*1e6);
    }

    // Stop all threads
    SLAM.Shutdown();

    SLAM.PrintEfficiencyAnalysis();

    // Tracking time statistics
    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    cout << "mean tracking time: " << totaltime/nImages << endl;
    
    std::ifstream f_in("./eval_metrics_tmp.txt");
    float e = 0.0f, r = 0.0f;
    float sum_entropy = 0.0f, sum_inlier_ratio = 0.0f;
    int valid_frames = 0;

    if(f_in.is_open()) {
        // 逐行读取文件中保存的 (entropy, inlierRatio)
        while(f_in >> e >> r) {
            sum_entropy += e;
            sum_inlier_ratio += r;
            valid_frames++;
        }
        f_in.close();

        if(valid_frames > 0) {
            cout << "------- Image Enhancement Evaluation -------" << endl;
            cout << "Processed Frames : " << valid_frames << endl;
            cout << "Average Entropy  : " << sum_entropy / valid_frames << endl;
            cout << "Avg Inlier Ratio : " << (sum_inlier_ratio / valid_frames) * 100.0f << " %" << endl;
            cout << "--------------------------------------------" << endl;
        }
    } else {
        cerr << "Warning: Could not read evaluation metrics file." << endl;
    }
    // Save camera trajectory
    SLAM.SaveTrajectoryEuRoC("Trajectory.txt");

    return 0;
}

void LoadImages(const string &strFile, vector<string> &vstrImageFilenames, vector<double> &vTimestamps)
{
    ifstream f;
    f.open(strFile.c_str());

    // skip first three lines
    string s0;
    getline(f,s0);
    getline(f,s0);
    getline(f,s0);

    while(!f.eof())
    {
        string s;
        getline(f,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            string sRGB;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenames.push_back(sRGB);
        }
    }
}
