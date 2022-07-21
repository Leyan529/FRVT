/*
 * This software was developed at the National Institute of Standards and
 * Technology (NIST) by employees of the Federal Government in the course
 * of their official duties. Pursuant to title 17 Section 105 of the
 * United States Code, this software is not subject to copyright protection
 * and is in the public domain. NIST assumes no responsibility  whatsoever for
 * its use by other parties, and makes no guarantees, expressed or implied,
 * about its quality, reliability, or any other characteristic.
 */

#include <cstring>
#include <cstdlib>
#include<math.h>
#include <sys/time.h>
#include "nullimplfrvt11.h"


using namespace std;
using namespace FRVT;
using namespace FRVT_11;


cv::Mat makeImg(FRVT::Image img){

    cv::Mat SynthesisImg = cv::Mat::zeros( img.height,img.width, CV_8UC3 );
    unsigned char* pxvec = SynthesisImg.ptr<uchar>(0);

    int i, j;
    for (i = 0; i < SynthesisImg.rows; i++)
    {
        pxvec = SynthesisImg.ptr<uchar>(i);
        for (j = 0; j < SynthesisImg.cols*SynthesisImg.channels(); j++)
        {
            pxvec[j] = int(img.data.get()[(i*img.width*3)+j ])  ;
        }
    }

    return SynthesisImg;
}


double calEuldistance(float *arrayA, float *arrayB)
{
    double similar=0;
    for (int j = 0; j < 512; j++) {

        similar+=(arrayA[j]-arrayB[j])*(arrayA[j]-arrayB[j]);
    }
    return (4.0-similar);
}



NullImplFRVT11::NullImplFRVT11() {}

NullImplFRVT11::~NullImplFRVT11() {}

ReturnStatus
NullImplFRVT11::initialize(const std::string &configDir)
{
    fd_detector.initialize(configDir);
    fr_detector.initialize(configDir);  
    return ReturnStatus(ReturnCode::Success);
}

ReturnStatus
NullImplFRVT11::createTemplate(
        const Multiface &faces,
        TemplateRole role,
        std::vector<uint8_t> &templ,
        std::vector<EyePair> &eyeCoordinates)
{

    
    for (unsigned int i=0; i<faces.size(); i++){

        #if showdetail
        struct timeval timeStart, timeEnd, timeSystemStart; 
        gettimeofday(&timeStart, NULL );
        #endif

        cv::Mat rgb_img = makeImg(faces[i]);
        cv::Mat bgr_img;
        cv::cvtColor(rgb_img, bgr_img, cv::COLOR_RGB2BGR);

        Mat rgb_aligned_face;
        Mat bgr_aligned_face;

        vector<cv::Point> landmark;
        fd_detector.findbiggestface(bgr_img, bgr_aligned_face, landmark);

        cv::cvtColor(bgr_aligned_face, rgb_aligned_face, cv::COLOR_BGR2RGB);


        //cv::Mat flip_rgb_aligned_face;
        //cv::flip(rgb_aligned_face, flip_rgb_aligned_face,1);

            
        vector<float> fresult = fr_detector.getFeature(rgb_aligned_face);
        //vector<float> flip_fresult = fr_detector.getFeature(flip_rgb_aligned_face);

        //fresult.insert(fresult.end(),flip_fresult.begin(),flip_fresult.end());
        L2_normalization(fresult);

        eyeCoordinates.push_back(EyePair(true, true, landmark[0].x, landmark[0].y, landmark[1].x, landmark[1].y ));


        uint8_t *p = reinterpret_cast<uint8_t*>(&fresult[0]);
        for(int j=0;j<sizeof(fresult[0])*fresult.size();j++){
            templ.push_back(*p);
            p++;
        }
        #if showdetail
        gettimeofday( &timeEnd, NULL ); 
        double runTime = (timeEnd.tv_sec - timeStart.tv_sec ) + (double)(timeEnd.tv_usec -timeStart.tv_usec)/1000000;  
        cout<<"use time "<<  runTime  <<endl;
        #endif
    }

    return ReturnStatus(ReturnCode::Success);
}

ReturnStatus
NullImplFRVT11::matchTemplates(
        const std::vector<uint8_t> &verifTemplate,
        const std::vector<uint8_t> &enrollTemplate,
        double &similarity)
{

    float *verifTemplate_array = (float *)verifTemplate.data();
    float *enrollTemplate_array = (float *)enrollTemplate.data();
    similarity = calEuldistance(verifTemplate_array, enrollTemplate_array);
    #if showdetail
    cout<<"smi"<<similarity<<endl;
    #endif
    return ReturnStatus(ReturnCode::Success);
}

std::shared_ptr<Interface>
Interface::getImplementation()
{
    return std::make_shared<NullImplFRVT11>();
}





