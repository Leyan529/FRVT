#ifndef __VIA_FACESTRUCT_H 
#define __VIA_FACESTRUCT_H

#include <opencv2/opencv.hpp>
#include <istream>
#include <ostream>
#include <iostream>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace cv;

#define fr_use_ncnn_optimize 0
#define fd_use_ncnn_optimize 0
#define showdetail 0


struct face_landmarks
{	
	std::vector< Point > points_data;

	float get_angle()
	{
		float angle = (float)atan(((points_data[1].y - points_data[0].y)*1.0 / (points_data[1].x - points_data[0].x)))* 180.0 / 3.1415;
		std::cout<<"angle :"<< angle <<std::endl;
		return angle;
	}

	float get_radian()
	{
		float radian = (float)atan(((points_data[1].y - points_data[0].y)*1.0 / (points_data[1].x - points_data[0].x)));
	   std::cout<<"radian :"<< radian <<std::endl;
		return radian;
	}

};



#endif
