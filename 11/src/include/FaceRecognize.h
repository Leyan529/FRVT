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
#include "facestructs.h"
#include "platform.h"
#include "net.h"
#include <numeric>
#include <vector>
using namespace std;

#if fr_use_ncnn_optimize
#include "fr_opt_id.h" //opimize
#else
#include "fr_id.h"   //origin
#endif

class FaceRecognize{

public:
    ncnn::Net fr_model;

    void initialize(std::string modeldir){
        #if fr_use_ncnn_optimize
        fr_model.load_param_bin( (modeldir+"/fr_optparam.so").c_str());
        fr_model.load_model((modeldir+"/fr_optbin.so").c_str());
        #else
        fr_model.load_param_bin( (modeldir+"/frparam.so").c_str());
        fr_model.load_model((modeldir+"/frbin.so").c_str());
        #endif
    }

    vector<float> getFeature(std::string imagepath);
    vector<float> getFeature(const cv::Mat& image);


};

void L2_normalization(vector<float>& u);