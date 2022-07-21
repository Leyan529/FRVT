#include <iostream>
#include "opencv2/opencv.hpp"
#include "anchor_generator.h"
#include "fd_config.h"
#include "tools.h"
#include <vector>
#include "facestructs.h"
#include "stdlib.h"
#include <chrono>
#include <numeric>

#if fd_use_ncnn_optimize
#include "fd_opt_id.h"  //optimize
#else
#include "fd_id.h"    //origin
#endif

using namespace cv;
using namespace std;


class FaceDetection{

public:
    const static int input_size = 256;
    static const  std::string  mt;

    ncnn::Net _net;
    // ncnn::Net fd_model;
    void initialize(std::string modeldir){
        // #if fd_use_ncnn_optimize
        // _net.load_param_bin((modeldir+"/fd_optparam.so").c_str());
        // _net.load_model((modeldir+"/fd_optbin.so").c_str());
        // #else
        // _net.load_param_bin((modeldir+"/fdparam.so").c_str());
        // _net.load_model((modeldir+"/fdbin.so").c_str());
        // #endif

        #if fd_use_ncnn_optimize
        _net.load_param_bin((modeldir+"/scrfd_2.5g.param").c_str());
        _net.load_model((modeldir+"/scrfd_2.5g.bin").c_str());
        #else
        _net.load_param_bin((modeldir+"/scrfd_2.5g-opt.param").c_str());
        _net.load_model((modeldir+"/scrfd_2.5g-opt.bin").c_str());
        #endif
    }  

    
void findFace_old(cv::Mat &image, std::vector<cv::Mat>&faces, std::vector<cv::Rect >&faceboxs, std::vector<std::vector<cv::Point> >&landmarks );
void findFace(cv::Mat &image, std::vector<cv::Mat>&faces, std::vector<cv::Rect >&faceboxs, std::vector<std::vector<cv::Point> >&landmarks );
void findFace(string imagepath, std::vector<cv::Mat>&faces, std::vector<cv::Rect >&faceboxs, std::vector<std::vector<cv::Point> >&landmarks );
void findbiggestface( string imagepath, cv::Mat &face, std::vector<cv::Point>&landmark);
void findbiggestface( cv::Mat &image, cv::Mat &face, std::vector<cv::Point>&landmark) ;
}; 


