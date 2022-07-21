#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <thread>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <unistd.h> 
#include "FaceRecognize.h"

#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <unistd.h>
#include <algorithm>
#include <tools.h>



float l2_norm(vector<float>& u) {
    float accum = 0.;
    for (int i =0; i<u.size(); i++) {
        accum += u[i] * u[i];
    }
    return sqrt(accum);
}

void L2_normalization(vector<float>& u){
    float l2Norm = l2_norm(u);
    for(int i = 0; i < u.size(); i++)
    {
        u[i] /= l2Norm;
    }
}




vector<float> FaceRecognize::getFeature(const cv::Mat& image)
{

    vector<float> feature_result;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(image.data, ncnn::Mat::PIXEL_BGR, image.cols, image.rows, 112, 112);


    ncnn::Extractor ex = fr_model.create_extractor();
    ex.set_num_threads(1);
    ex.input(0, in);

    for (int k = 0; k < in.total(); k++) {
        in[k] = (in[k] - 127.5) / 128.0;}

    ncnn::Mat out;
    #if fr_use_ncnn_optimize
    ex.extract(120, out);  //opt
    #else
    ex.extract(455, out);  //origin
    #endif

    for (int j=0; j<out.w; j++){
       // feature_result.data[j] = out[j];
       feature_result.push_back(out[j]);
    }


    return feature_result;
}




vector<float> FaceRecognize::getFeature(std::string imagepath)
{

    cv::Mat image = cv::imread(imagepath);
    return getFeature(image);
}


int main(int argc, char** argv){    
    cv::Mat image = cv::imread("/media/leyan/E/Git/ViaFaceRec/tests/data/00000001.jpg");
    ncnn::Net fr_model;

    // const std::string modeldir = "/media/leyan/D/D_Disk/11/src/nullImpl/old_fd";
    // fr_model.load_param_bin((modeldir +"/frparam.so").c_str());
    // fr_model.load_model((modeldir+ "/frbin.so").c_str());

    // const std::string modeldir = "/media/leyan/D/D_Disk/11/src/nullImpl/modeldir/ResCustom_3_49_70_3";
    const std::string modeldir = "/media/leyan/D/D_Disk/11/src/nullImpl/modeldir/resnext/resnext200_32x4d";
    // const std::string modeldir = "/media/leyan/D/D_Disk/11/src/nullImpl/modeldir/resnest/resnest152_8x14d";
    // const std::string modeldir = "/media/leyan/D/D_Disk/11/src/nullImpl/modeldir/resnest/resnest152_32x4d";
    // const std::string modeldir = "/media/leyan/D/D_Disk/11/src/nullImpl/modeldir/resnest/resnest200_1x64d";
    // const std::string modeldir = "/media/leyan/D/D_Disk/11/src/nullImpl/modeldir/resnest/resnest200_1x64d_r4";
    // const std::string modeldir = "/media/leyan/D/D_Disk/11/src/nullImpl/modeldir/Res269";
    // fr_model.load_param((modeldir +"/fr.param").c_str());
    // fr_model.load_model((modeldir+ "/fr.bin").c_str());

    fr_model.load_param((modeldir +"/fr-opt.param").c_str());
    fr_model.load_model((modeldir+ "/fr-opt.bin").c_str());
    
    std::string rep_str = "/home/leyan/DataSet/ms1m-retinaface-t1/ms1m_retinaface/";

    // auto imgPaths = exec(("find " + rep_str + " -name *.jpg").c_str());
    // cout << imgPaths.size() << endl;


    double if_time;
    double if_time_all;


    std::vector<double> if_time_v;
    std::vector<double> if_time_all_v;

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    for(int i = 0 ; i < 1000 ; i++){   
        cout << "---------------roumd: " << i << "---------------" << endl;
        vector<float> feature_result;
        // cv::Mat image = cv::imread(imgPaths[i]);

        auto st_all = std::chrono::high_resolution_clock::now();

        ncnn::Mat in = ncnn::Mat::from_pixels_resize(image.data, ncnn::Mat::PIXEL_BGR, image.cols, image.rows, 112, 112);

        auto st = std::chrono::high_resolution_clock::now();


        ncnn::Extractor ex = fr_model.create_extractor();
        ex.set_num_threads(1);
        ex.input(0, in);

        for (int k = 0; k < in.total(); k++) {
            in[k] = (in[k] - 127.5) / 128.0;}

        ncnn::Mat out;     
        // ex.extract(455, out);  //origin   
        ex.extract("output0", out);  //origin  

        // ex.extract("1518", out);  //origin   
        // ex.extract("1479_splitncnn_0", out);  //origin 
        // ex.extract("1520", out);  //origin    

        // ex.extract("1522", out);  //origin  

        for (int j=0; j<out.w; j++){
        // feature_result.data[j] = out[j];
        feature_result.push_back(out[j]);
        }

        auto en = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(en - st);
        if_time = time_span.count()*1000;

        std::chrono::duration<double> time_span_all = std::chrono::duration_cast<std::chrono::duration<double>>(en - st_all);
        if_time_all = time_span_all.count()*1000;

        if_time_v.push_back(if_time);
        if_time_all_v.push_back(if_time_all);
    }

    cout << "---------------------------------------" << endl;
    cout << "finished" << endl;

    double sum_of_elems = std::accumulate(if_time_v.begin(), if_time_v.end(), 0);
    cout << "Average Inference duration time: " << sum_of_elems/if_time_v.size() << " ms" << endl;

    double all_sum_of_elems = std::accumulate(if_time_all_v.begin(), if_time_all_v.end(), 0);
    cout << "Average All Inference duration time: " << all_sum_of_elems/if_time_all_v.size() << " ms" << endl; 

}