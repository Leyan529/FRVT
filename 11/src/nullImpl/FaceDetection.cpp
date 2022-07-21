#include "FaceDetection.h"
#include<cmath>
#include"cpu.h"

#include <string>
#include <dirent.h>
#include <vector>
#include <map>

#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <unistd.h>
#include <algorithm>
#include <tools.h>


#define min(x,y) ((x)<(y)?(x):(y))
#define max(x,y) ((x)>(y)?(x):(y))

// const std::string  FaceDetection::mt = "scrfd_500m";
const std::string  FaceDetection::mt = "scrfd_2.5g";
// const std::string  FaceDetection::mt = "scrfd_10g";

#if fd_use_ncnn_optimize
int target_blob_cls[3] = {191,207,218};
int target_blob_reg[3] = {182,202,214};
int target_blob_pts[3] = {183,203,215};
#else
int target_blob_cls[3] = {345,373,386};
int target_blob_reg[3] = {328,363,382};
int target_blob_pts[3] = {329,364,383};
#endif

struct FaceObject
{
    cv::Rect_<float> rect;
    float prob;    
    vector<cv::Point> landmark_vec_tmp;
};

cv::Mat meanAxis0(const cv::Mat &src){
    int num = src.rows;
    int dim = src.cols;

    cv::Mat output(1,dim,CV_32F);
    for(int i = 0 ; i <  dim; i ++)
    {
        float sum = 0 ;
        for(int j = 0 ; j < num ; j++)
        {
            sum+=src.at<float>(j,i);
        }
        output.at<float>(0,i) = sum/num;
    }

    return output;
}

cv::Mat elementwiseMinus(const cv::Mat &A,const cv::Mat &B){
    cv::Mat output(A.rows,A.cols,A.type());

    assert(B.cols == A.cols);
    if(B.cols == A.cols)
    {
        for(int i = 0 ; i <  A.rows; i ++)
        {
            for(int j = 0 ; j < B.cols; j++)
            {
                output.at<float>(i,j) = A.at<float>(i,j) - B.at<float>(0,j);
            }
        }
    }
    return output;
}

cv::Mat varAxis0(const cv::Mat &src){
    cv::Mat temp_ = elementwiseMinus(src,meanAxis0(src));
    cv::multiply(temp_ ,temp_ ,temp_ );
    return meanAxis0(temp_);
}

int MatrixRank(cv::Mat M){
    cv::Mat w, u, vt;
    cv::SVD::compute(M, w, u, vt);
    cv::Mat1b nonZeroSingularValues = w > 0.0001;
    int rank = cv::countNonZero(nonZeroSingularValues);
    return rank;
}

cv::Mat similarTransform(cv::Mat src,cv::Mat dst) {
    int num = src.rows;
    int dim = src.cols;
    cv::Mat src_mean = meanAxis0(src);
    cv::Mat dst_mean = meanAxis0(dst);
    cv::Mat src_demean = elementwiseMinus(src, src_mean);
    cv::Mat dst_demean = elementwiseMinus(dst, dst_mean);
    cv::Mat A = (dst_demean.t() * src_demean) / static_cast<float>(num);
    cv::Mat d(dim, 1, CV_32F);
    d.setTo(1.0f);
    if (cv::determinant(A) < 0) {
        d.at<float>(dim - 1, 0) = -1;

    }
    cv::Mat T = cv::Mat::eye(dim + 1, dim + 1, CV_32F);
    cv::Mat U, S, V;
    cv::SVD::compute(A, S,U, V);

    int rank = MatrixRank(A);
    if (rank == 0) {
        assert(rank == 0);

    } else if (rank == dim - 1) {
        if (cv::determinant(U) * cv::determinant(V) > 0) {
            T.rowRange(0, dim).colRange(0, dim) = U * V;
        } else {

            int s = d.at<float>(dim - 1, 0) = -1;
            d.at<float>(dim - 1, 0) = -1;

            T.rowRange(0, dim).colRange(0, dim) = U * V;
            cv::Mat diag_ = cv::Mat::diag(d);
            cv::Mat twp = diag_*V; //np.dot(np.diag(d), V.T)
            cv::Mat B = cv::Mat::zeros(3, 3, CV_8UC1);
            cv::Mat C = B.diag(0);
            T.rowRange(0, dim).colRange(0, dim) = U* twp;
            d.at<float>(dim - 1, 0) = s;
        }
    }
    else{
        cv::Mat diag_ = cv::Mat::diag(d);
        cv::Mat twp = diag_*V.t(); //np.dot(np.diag(d), V.T)
        cv::Mat res = U* twp; // U
        T.rowRange(0, dim).colRange(0, dim) = -U.t()* twp;
    }
    cv::Mat var_ = varAxis0(src_demean);
    float val = cv::sum(var_).val[0];
    cv::Mat res;
    cv::multiply(d,S,res);
    float scale =  1.0/val*cv::sum(res).val[0];
    T.rowRange(0, dim).colRange(0, dim) = - T.rowRange(0, dim).colRange(0, dim).t();
    cv::Mat  temp1 = T.rowRange(0, dim).colRange(0, dim); // T[:dim, :dim]
    cv::Mat  temp2 = src_mean.t(); //src_mean.T
    cv::Mat  temp3 = temp1*temp2; // np.dot(T[:dim, :dim], src_mean.T)
    cv::Mat temp4 = scale*temp3;
    T.rowRange(0, dim).colRange(dim, dim+1)=  -(temp4 - dst_mean.t()) ;
    T.rowRange(0, dim).colRange(0, dim) *= scale;
    return T;
}

cv::Mat MappingRotate(cv::Mat &img, float* landmark){

  cv::Mat target_landmark_mat = cv::Mat(5, 2, CV_32F, landmark);

  float base_landmark[10] = { 38.29459953,  51.69630051,  73.53179932,  51.50139999,
    56.02519989,  71.73660278,  41.54930115,  92.3655014 ,
    70.72990036,  92.20410156 };
  cv::Mat base_landmark_mat = cv::Mat(5, 2, CV_32F, base_landmark);

  cv::Mat transMattmp = similarTransform(target_landmark_mat, base_landmark_mat);
  float outMat[6]={0};

  int i =0;
  int j =0;
  for( i =0; i<2; i++){
    const float*indata = transMattmp.ptr<float>(i);
    for ( j =0; j<3; j++)
    outMat[(3*i+j)] = indata[j];
  }

    cv::Mat transMat = cv::Mat(2, 3, CV_32F, outMat);

    Mat warped;
    cv::warpAffine(img, warped, transMat, Size(112,112));

    return warped;
}


void FaceDetection::findFace_old(cv::Mat &image, std::vector<cv::Mat>&faces, std::vector<cv::Rect >&faceboxs, std::vector<std::vector<cv::Point> >&landmarks){

    cv::Mat new_image;
    float pixel_mean[3] = {0, 0, 0};
    float pixel_std[3] = {1, 1, 1};
    float pixel_scale = 1.0;


    int target_size = 160;
    int max_size = 240;
    int img_width = image.cols;
    int img_height = image.rows;
    int im_size_min = min(img_width, img_height);
    int im_size_max = max(img_width, img_height);

    float im_scale = float(target_size) / float(im_size_min);

    if (round(im_scale * im_size_max) > max_size){
        im_scale = float(max_size) / float(im_size_max);
    }

    int new_height = int(img_height*im_scale);
    int new_width = int(img_width*im_scale);

    ncnn::Mat input = ncnn::Mat::from_pixels_resize(image.data, ncnn::Mat::PIXEL_BGR2RGB, image.cols, image.rows, new_width, new_height);
    cv::resize(image, new_image, cv::Size(new_width, new_height));
    input.substract_mean_normalize(pixel_mean, pixel_std);
    ncnn::Extractor _extractor = _net.create_extractor();
    _extractor.set_num_threads(1);
    _extractor.input(0, input);


    std::vector<AnchorGenerator> ac(_feat_stride_fpn.size());
    for (int i = 0; i < _feat_stride_fpn.size(); ++i) {
        int stride = _feat_stride_fpn[i];
        ac[i].Init(stride, anchor_cfg[stride], false);
    }

    std::vector<Anchor> proposals;
    proposals.clear();

    for (int i = 0; i < _feat_stride_fpn.size(); ++i) { 
        ncnn::Mat cls;
        ncnn::Mat reg;
        ncnn::Mat pts;

        _extractor.extract(target_blob_cls[i], cls);
        _extractor.extract(target_blob_reg[i], reg);
        _extractor.extract(target_blob_pts[i], pts);

        ac[i].FilterAnchor(cls, reg, pts, proposals);

    }

    std::vector<Anchor> result;

    nms_cpu(proposals, nms_threshold, result);

    for(int i = 0; i < result.size(); i ++)
    {
        faceboxs.push_back( cv::Rect(cv::Point((int)result[i].finalbox.x, (int)result[i].finalbox.y), cv::Point((int)result[i].finalbox.width, (int)result[i].finalbox.height)) );
        vector<cv::Point> landmark_vec_tmp;
        float landmark_tmp[10] = {0};
        for (int j = 0; j < result[i].pts.size(); ++j) {
            landmark_vec_tmp.push_back( cv::Point((int)result[i].pts[j].x, (int)result[i].pts[j].y) );
            landmark_tmp[2*j] = result[i].pts[j].x;
            landmark_tmp[2*j+1] = result[i].pts[j].y;
        }
        landmarks.push_back(landmark_vec_tmp);

        // cv::rectangle(new_image, cv::Rect(cv::Point((int)result[i].finalbox.x, (int)result[i].finalbox.y), cv::Point((int)result[i].finalbox.width, (int)result[i].finalbox.height)), cv::Scalar(255,0,0),4);
        // for (int j = 0; j < result[i].pts.size(); ++j) {
        //     cv::circle(new_image, cv::Point((int)result[i].pts[j].x, (int)result[i].pts[j].y), 3, cv::Scalar(0,255,0),-1);
        // }

        // cv::imshow("test", new_image);
        // cv::waitKey(0);

        cv::Mat aligned_img = MappingRotate(new_image, landmark_tmp);
        faces.push_back(aligned_img);
    }

}


static ncnn::Mat generate_anchor_centers(int base_size, const ncnn::Mat& ratios, const ncnn::Mat& scales, const ncnn::Mat& score, int stride, int square)
{   
    ncnn::Mat anchors;
    // anchors.create(num_ratio * num_scale, score.h);


    // *    #solution-1, c style:
    // *    anchor_centers = np.zeros( (height, width, 2), dtype=np.float32 )
    // *    for i in range(height):
    // *        anchor_centers[i, :, 1] = i
    // *    for i in range(width):
    // *        anchor_centers[:, i, 0] = i
    // *
    // *    anchor_centers = (anchor_centers * stride).reshape( (-1, 2) )
    // *    if self._num_anchors>1:
    // *        anchor_centers = np.stack([anchor_centers]*self._num_anchors, axis=1).reshape( (-1,2) )

    int height = square / stride;
    int width = square / stride;
    anchors.create(height, width, 2);

    // cout << "height: " <<  height << endl;
    // cout << "width: " <<  width << endl;

    // cout << "anchors.w: " <<  anchors.w << endl;
    // cout << "anchors.h: " <<  anchors.h << endl;
    // cout << "anchors.c: " <<  anchors.c << endl;

    for (int y=0; y < anchors.h; y++)
    {
        for (int x=0; x < anchors.w ; x++)
        {   
            anchors[x * anchors.h * 2 + y * 2 + 1] = x * stride;  // anchor_centers[i, :, 1] = i
        }
    }


    for (int y=0; y < anchors.h; y++)
    {
        for (int x=0; x < anchors.w ; x++)
        {   
            anchors[y * anchors.h * 2 + x * 2 + 0] = x * stride;  // anchor_centers[:, i, 0] = i
        }
    }
 
    ncnn::Mat anchors_reshape = anchors.reshape(anchors.w * anchors.h, 2);


    // pretty_print(anchors_reshape, "anchors_" + stride); 

    // cout << "anchors_reshape.w: " <<  anchors_reshape.w << endl;
    // cout << "anchors_reshape.h: " <<  anchors_reshape.h << endl;
    // cout << "anchors_reshape.c: " <<  anchors_reshape.c << endl;


    // * anchor_centers = np.stack([anchor_centers]*self._num_anchors, axis=1).reshape( (-1,2) )

    ncnn::Mat anchors_stack;
    anchors_stack.create(anchors_reshape.w * 2, 2);

    for (int y=0; y < anchors_stack.h; y++)
    {
        for (int x=0; x < anchors_stack.w ; x+=2)
        {   
            anchors_stack[x * anchors_reshape.h + y ] = anchors_reshape[x/2 * anchors_reshape.h + y ];  // 1 clone copy
            anchors_stack[x * anchors_reshape.h + y + 2] = anchors_reshape[x/2 * anchors_reshape.h + y ];  // 2 clone copy
        }
    }

    // pretty_print(anchors_stack, ("anchors_stack_" + std::to_string(stride)).c_str()); 

    // cout << "anchors_stack.w: " <<  anchors_stack.w << endl;
    // cout << "anchors_stack.h: " <<  anchors_stack.h << endl;
    // cout << "anchors_stack.c: " <<  anchors_stack.c << endl;
    // cout << "anchor finshed" << endl;
    return anchors_stack;
}

static void generate_new_proposals(const ncnn::Mat& anchors, int feat_stride, const ncnn::Mat& score_blob, const ncnn::Mat& bbox_blob, const ncnn::Mat& kps_blob, float prob_threshold, std::vector<FaceObject>& faceobjects)
{
    int w = bbox_blob.w;
    int h = bbox_blob.h;   

    // generate face proposal from bbox deltas and shifted anchors
    const int num_anchors = anchors.h;       
    const int landmark_num = kps_blob.w;

    for(int rows = 0 ; rows < h ; rows++){
        float prob = score_blob[rows];  

        if (prob >= prob_threshold)
        {
            // insightface/detection/scrfd/mmdet/core/bbox/transforms.py distance2bbox()
            // Transform outputs for a single batch item into labeled boxes
            float cx = anchors[rows * 2];
            float cy = anchors[rows * 2 + 1];

            float dx = bbox_blob[rows*4 + 0] * feat_stride;
            float dy = bbox_blob[rows*4 + 1] * feat_stride;
            float dw = bbox_blob[rows*4 + 2] * feat_stride;
            float dh = bbox_blob[rows*4 + 3] * feat_stride; 

            float x0 = cx - dx;
            float y0 = cy - dy;
            float x1 = cx + dw;
            float y1 = cy + dh;

            // Insightface distance2kps(points, distance, max_shape=None):  
            std::vector<cv::Point> keep_kps;
            for(int l = 0 ; l < landmark_num; l+=2){
                float px = anchors[rows * 2 + l%2 ] + kps_blob[rows*10 + l] * feat_stride;
                float py = anchors[rows * 2 + l%2 + 1 ] + kps_blob[rows*10 + l + 1] * feat_stride;
                cv:Point2f p;
                p.x = px;
                p.y = py;
                keep_kps.push_back(p);
            }

            FaceObject obj;
            obj.rect.x = x0;
            obj.rect.y = y0;
            obj.rect.width = x1 - x0 + 1;
            obj.rect.height = y1 - y0 + 1;
            obj.prob = prob;
            obj.landmark_vec_tmp = keep_kps;
            faceobjects.push_back(obj);          
        }        
    }
}


static inline float intersection_area(const FaceObject& a, const FaceObject& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<FaceObject>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    //     #pragma omp parallel sections
    {
        //         #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        //         #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<FaceObject>& faceobjects)
{
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<FaceObject>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const FaceObject& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const FaceObject& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            //             float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

std::pair<std::vector<double>, cv::Mat> findFace(cv::Mat &image, std::vector<cv::Mat>&faces, std::vector<cv::Rect >&faceboxs, std::vector<std::vector<cv::Point> >&landmarks){

    cv::Mat new_image;
    float pixel_mean[3] = {127.5f, 127.5f, 127.5f};
    float pixel_std[3] = {1/128.f, 1/128.f, 1/128.f};
    float pixel_scale = 1.0;

    // commit
    ncnn::Net _net;
    const std::string modeldir = "/media/leyan/D/D_Disk/11_new/src/nullImpl/modeldir/";
    _net.load_param((modeldir + std::to_string(FaceDetection::input_size) + "/" + FaceDetection::mt + ".param").c_str());
    _net.load_model((modeldir+ std::to_string(FaceDetection::input_size) + "/" + FaceDetection::mt + ".bin").c_str());
    ////

    auto st_all = std::chrono::high_resolution_clock::now();


    // pad to multiple of 32
    int target_size = FaceDetection::input_size;
    int width = image.cols;
    int height = image.rows;

    int new_width = width;
    int new_height = height;
    float im_scale = 1.f;

    if (new_width > new_height)
    {
        im_scale = (float)target_size / new_width;
        new_width = target_size;
        new_height = new_height * im_scale;
    }
    else
    {
        im_scale = (float)target_size / new_height;
        new_height = target_size;
        new_width = new_width * im_scale;
    }
    
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(image.data, ncnn::Mat::PIXEL_BGR2RGB, width, height, new_width, new_height);
    cv::resize(image, new_image, cv::Size(new_width, new_height));

    // pad to target_size rectangle
    int square = max(new_width, new_height);
    int wpad = (square + 31) / 32 * 32 - new_width;
    int hpad = (square + 31) / 32 * 32 - new_height;
    ncnn::Mat in_pad;

    // copy_make_border(const Mat& src, Mat& dst, 
    // int top, int bottom, int left, int right, 
    // int type, float v, const Option& opt = Option());

    // ncnn::copy_make_border(in, in_pad, 
    //     hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, 
    //     ncnn::BORDER_CONSTANT, 0.f);
    
    ncnn::copy_make_border(in, in_pad, 
        0, hpad, 0, wpad, 
        ncnn::BORDER_CONSTANT, 0.f);

    in_pad.substract_mean_normalize(pixel_mean, pixel_std);

    auto st = std::chrono::high_resolution_clock::now();

    ncnn::Extractor _extractor = _net.create_extractor();
    _extractor.set_num_threads(1);
    // _extractor.input(0, input);
    _extractor.input(0, in_pad);

    // cout << "width: " << width << endl;
    // cout << "height: " << height << endl;

    // cout << "in w: " << in.w << endl;
    // cout << "in h: " << in.h << endl;

    // cout << "im_scale: " << im_scale << endl; 

    // cout << "wpad: " << wpad << endl;
    // cout << "hpad: " << hpad << endl; 

    // cout << "in_pad w: " << in_pad.w << endl;
    // cout << "in_pad h: " << in_pad.h << endl;     


    // cout << "new_width: " << new_width << endl;
    // cout << "new_height: " << new_height << endl;

    // cout << "in_pad.w: " << in_pad.w << endl;
    // cout << "in_pad.h: " << in_pad.h << endl;

    std::vector<FaceObject> faceproposals;
    faceproposals.clear();

    ncnn::Mat ratios(1);
    ratios[0] = 1.f;
    ncnn::Mat scales(2);
    scales[0] = 1.f;
    scales[1] = 2.f;

    double if_time;
    double if_time_all;

    if(true){    
        for (int i = 0; i < _feat_stride_fpn.size(); ++i) {         
            ncnn::Mat score_blob;
            ncnn::Mat bbox_blob;
            ncnn::Mat kps_blob;

            int stride = _stride_fpn[i];
            std::string score_n = "score_" + std::to_string(stride);
            std::string bbox_n = "bbox_" + std::to_string(stride);
            std::string kps_n = "kps_" + std::to_string(stride);

            _extractor.extract(score_n.c_str(), score_blob);
            _extractor.extract(bbox_n.c_str(), bbox_blob);
            _extractor.extract(kps_n.c_str(), kps_blob);

            // pretty_print_bbox(bbox_blob, bbox_n.c_str());
            // pretty_print_score(score_blob, score_n.c_str());      
            // pretty_print_kps(kps_blob, kps_n.c_str()); 

            // // cout << score_blob.w << endl;
            // cout << score_blob.h << endl;
            // // cout << "-----------------------------------------" << endl;
            // // cout << bbox_blob.w << endl;
            // cout << bbox_blob.h << endl;
            // // cout << "-----------------------------------------" << endl;
            // // cout << kps_blob.w << endl;
            // cout << kps_blob.h << endl;
            // cout << "-----------------------------------------" << endl;

            ncnn::Mat anchors = generate_anchor_centers(anchor_cfg[stride].BASE_SIZE, ratios, scales, score_blob, stride, square);   
            // pretty_print(anchors, ("anchors_stack_" + std::to_string(stride)).c_str());     

            std::vector<FaceObject> faceobjects;         
            generate_new_proposals(anchors, _stride_fpn[i], score_blob, bbox_blob, kps_blob, cls_threshold, faceobjects); 
            faceproposals.insert(faceproposals.end(), faceobjects.begin(), faceobjects.end());
        }

        // sort all proposals by score from highest to lowest
        qsort_descent_inplace(faceproposals);

        std::vector<int> picked;
        nms_sorted_bboxes(faceproposals, picked, nms_threshold);

        int face_count = picked.size();

        std::vector<FaceObject> faceobjects;

        faceobjects.resize(face_count);

        // cout << "im_scale: " << im_scale << endl;
        // im_scale = 1;

        auto en = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(en - st);
        if_time = time_span.count()*1000;

        std::chrono::duration<double> time_span_all = std::chrono::duration_cast<std::chrono::duration<double>>(en - st_all);
        if_time_all = time_span_all.count()*1000;

        for(int i = 0; i < faceobjects.size(); i++)
        {
            faceobjects[i] = faceproposals[picked[i]];        

            // adjust offset to original unpadded
            float x0 = (faceobjects[i].rect.x ) / im_scale;
            float y0 = (faceobjects[i].rect.y ) / im_scale;
            float x1 = (faceobjects[i].rect.x + faceobjects[i].rect.width ) / im_scale;
            float y1 = (faceobjects[i].rect.y + faceobjects[i].rect.height) / im_scale;

            vector<cv::Point> landmark_vec;
            float landmark_tmp[10] = {0};
            for(int j = 0 ; j < faceobjects[i].landmark_vec_tmp.size(); j++){
                cv::Point p;
                p.x = faceobjects[i].landmark_vec_tmp[j].x / im_scale; 
                p.y = faceobjects[i].landmark_vec_tmp[j].y / im_scale;
                landmark_vec.push_back(p);
                landmark_tmp[2*j] = faceobjects[i].landmark_vec_tmp[j].x;
                landmark_tmp[2*j+1] = faceobjects[i].landmark_vec_tmp[j].y;
                cv::circle(image, p, 1, cv::Scalar(0,0,255) , 2);
            }
            landmarks.push_back(landmark_vec);

            faceobjects[i].rect.x = x0;
            faceobjects[i].rect.y = y0;
            faceobjects[i].rect.width = x1 - x0;
            faceobjects[i].rect.height = y1 - y0;
            faceobjects[i].landmark_vec_tmp = landmark_vec;
            cv::rectangle(image, 
                cv::Rect(faceobjects[i].rect.x, 
                        faceobjects[i].rect.y, 
                        faceobjects[i].rect.width, 
                        faceobjects[i].rect.height), cv::Scalar(0, 255, 0, 255));

            faceboxs.push_back(
                cv::Rect((int)faceobjects[i].rect.x , 
                        (int)faceobjects[i].rect.y , 
                        (int)faceobjects[i].rect.width, 
                        (int)faceobjects[i].rect.height   
                ));

            cv::Mat aligned_img = MappingRotate(new_image, landmark_tmp);
            faces.push_back(aligned_img); // 112 * 112
        }
    }

    // cv::imshow("image", image);
    // cv::waitKey(0);
    std::pair<std::vector<double>, cv::Mat> p1;
    p1.first = std::vector<double>{if_time, if_time_all};
    p1.second = image;
    return p1;
}


void FaceDetection::findFace(cv::Mat &image, std::vector<cv::Mat>&faces, std::vector<cv::Rect >&faceboxs, std::vector<std::vector<cv::Point> >&landmarks){

    cv::Mat new_image;
    float pixel_mean[3] = {127.5f, 127.5f, 127.5f};
    float pixel_std[3] = {1/128.f, 1/128.f, 1/128.f};
    float pixel_scale = 1.0; 

    // pad to multiple of 32
    int target_size = FaceDetection::input_size;
    int width = image.cols;
    int height = image.rows;

    int new_width = width;
    int new_height = height;
    float im_scale = 1.f;

    if (new_width > new_height)
    {
        im_scale = (float)target_size / new_width;
        new_width = target_size;
        new_height = new_height * im_scale;
    }
    else
    {
        im_scale = (float)target_size / new_height;
        new_height = target_size;
        new_width = new_width * im_scale;
    }
    
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(image.data, ncnn::Mat::PIXEL_BGR2RGB, width, height, new_width, new_height);
    cv::resize(image, new_image, cv::Size(new_width, new_height));

    // pad to target_size rectangle
    int square = max(new_width, new_height);
    int wpad = (square + 31) / 32 * 32 - new_width;
    int hpad = (square + 31) / 32 * 32 - new_height;
    ncnn::Mat in_pad;  
    
    ncnn::copy_make_border(in, in_pad, 0, hpad, 0, wpad, ncnn::BORDER_CONSTANT, 0.f);

    in_pad.substract_mean_normalize(pixel_mean, pixel_std);

    ncnn::Extractor _extractor = _net.create_extractor();
    _extractor.set_num_threads(1);
    _extractor.input(0, in_pad);


    std::vector<FaceObject> faceproposals;
    faceproposals.clear();

    ncnn::Mat ratios(1);
    ratios[0] = 1.f;
    ncnn::Mat scales(2);
    scales[0] = 1.f;
    scales[1] = 2.f;

    for (int i = 0; i < _stride_fpn.size(); ++i) { 
        ncnn::Mat score_blob;
        ncnn::Mat bbox_blob;
        ncnn::Mat kps_blob;

        int stride = _stride_fpn[i];
        std::string score_n = "score_" + std::to_string(stride);
        std::string bbox_n = "bbox_" + std::to_string(stride);
        std::string kps_n = "kps_" + std::to_string(stride);

        _extractor.extract(score_n.c_str(), score_blob);
        _extractor.extract(bbox_n.c_str(), bbox_blob);
        _extractor.extract(kps_n.c_str(), kps_blob);

        ncnn::Mat anchors = generate_anchor_centers(anchor_cfg[stride].BASE_SIZE, ratios, scales, score_blob, stride, square);       

        std::vector<FaceObject> faceobjects;
        generate_new_proposals(anchors, _stride_fpn[i] * im_scale, score_blob, bbox_blob, kps_blob, cls_threshold, faceobjects); 
        faceproposals.insert(faceproposals.end(), faceobjects.begin(), faceobjects.end());
    }

    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(faceproposals);

    std::vector<int> picked;
    nms_sorted_bboxes(faceproposals, picked, nms_threshold);

    int face_count = picked.size();

    std::vector<FaceObject> faceobjects;

    faceobjects.resize(face_count);

    for(int i = 0; i < faceobjects.size(); i++)
    {
        faceobjects[i] = faceproposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (faceobjects[i].rect.x ) / im_scale;
        float y0 = (faceobjects[i].rect.y ) / im_scale;
        float x1 = (faceobjects[i].rect.x + faceobjects[i].rect.width ) / im_scale;
        float y1 = (faceobjects[i].rect.y + faceobjects[i].rect.height) / im_scale;

        vector<cv::Point> landmark_vec;
        float landmark_tmp[10] = {0};
        for(int j = 0 ; j < faceobjects[i].landmark_vec_tmp.size(); j++){
            cv::Point p;
            p.x = faceobjects[i].landmark_vec_tmp[j].x / im_scale;
            p.y = faceobjects[i].landmark_vec_tmp[j].y / im_scale;
            landmark_vec.push_back(p);
            landmark_tmp[2*j] = faceobjects[i].landmark_vec_tmp[j].x;
            landmark_tmp[2*j+1] = faceobjects[i].landmark_vec_tmp[j].y;
            // cv::circle(image, p, 1, cv::Scalar(0,0,255) , 2);
        }
        landmarks.push_back(landmark_vec);

        faceobjects[i].rect.x = x0;
        faceobjects[i].rect.y = y0;
        faceobjects[i].rect.width = x1 - x0;
        faceobjects[i].rect.height = y1 - y0;
        faceobjects[i].landmark_vec_tmp = landmark_vec;
        // cv::rectangle(image, cv::Rect(faceobjects[i].rect.x, faceobjects[i].rect.y, faceobjects[i].rect.width, faceobjects[i].rect.height), cv::Scalar(0, 255, 0, 255));
        // faceboxs.push_back(cv::Rect(cv::Point((int)faceobjects[i].rect.x, (int)faceobjects[i].rect.y), cv::Point((int)faceobjects[i].rect.width, (int)faceobjects[i].rect.height)));
        faceboxs.push_back(cv::Rect(faceobjects[i].rect.x, faceobjects[i].rect.y, faceobjects[i].rect.width, faceobjects[i].rect.height));
        cv::Mat aligned_img = MappingRotate(new_image, landmark_tmp);
        faces.push_back(aligned_img);
    }
}


void FaceDetection::findFace(string imagepath, std::vector<cv::Mat>&faces, std::vector<cv::Rect >&faceboxs, std::vector<std::vector<cv::Point> >&landmarks ){
    cv::Mat image = cv::imread(imagepath);
    findFace(image, faces, faceboxs, landmarks);
}

void  FaceDetection::findbiggestface( cv::Mat &image, cv::Mat &face, std::vector<cv::Point>&landmark) {

    int target_index = -1;
    int biggest_size = -99;

    vector<Mat>faces_tmp;
    vector<vector<cv::Point> >landmarks_tmp;
    vector<Rect >faceboxs_tmp;

    
    findFace(image,  faces_tmp, faceboxs_tmp, landmarks_tmp);
   

    int facesize =  faces_tmp.size();


    for (int i=0; i<faces_tmp.size(); i++){
        int area_tmp = faceboxs_tmp[i].area();
        if (area_tmp > biggest_size){
            target_index = i;
            biggest_size = faceboxs_tmp[i].area();
        }
    }

    if (facesize == 0){
        int img_width = image.cols;
        int img_height = image.rows;
        face = image(Rect(int(0.3*img_width), int(0.3*img_height), int(0.4*img_width), int(0.4*img_height) ));
        landmark.push_back( cv::Point(0.34191607*img_width,  0.63151696*img_height) );
        landmark.push_back( cv::Point(0.65653393*img_width,  0.45983393*img_height) );
        landmark.push_back( cv::Point(0.500225*img_width,  0.64050536*img_height) );
        landmark.push_back( cv::Point(0.37097589*img_width,  0.82469196*img_height) );
        landmark.push_back( cv::Point(0.63151696*img_width, 0.82325089*img_height) );
    }
    else{
        face = faces_tmp[target_index];
        landmark = landmarks_tmp[target_index];
    }

}

void  FaceDetection::findbiggestface( string imagepath, cv::Mat &face, std::vector<cv::Point>&landmark) {

    cv::Mat image = cv::imread(imagepath);
    findbiggestface(image, face, landmark);
}








float IOU(cv::Rect pred_box, cv::Rect gt_box){
    // cv::Rect orr = pred_box | gt_box;
    // cv::Rect U = pred_box & gt_box;
    // return U.area()*1.0 / orr.area();
    auto pred_area = float(pred_box.area());
    auto gt_box_area = float(gt_box.area());
    // cout << "area scale: " << pred_area / gt_box_area << endl;


    float w = min(pred_box.x + pred_box.width, gt_box.x + gt_box.width) - max(pred_box.x, gt_box.x);
    float h = min(pred_box.y + pred_box.height, gt_box.y + gt_box.height) - max(pred_box.y, gt_box.y);

    if(w <=0 || h <=0) return 0;
    if ((pred_area / gt_box_area) < 0.1) return 0;
    return ((w * h) / (pred_box.height * pred_box.width) + (gt_box.height * gt_box.width) - (w * h));
}

float prec(vector<Rect>pred_boxes, vector<cv::Rect> gt_boxes){
    std::vector<int> pass;
    int TP = 0;
    int FP = 0;
    int TN = 0;

    bool found;

    for(int i = 0 ; i < pred_boxes.size() ; i++){
        found = false;
        for(int j = 0 ; j < gt_boxes.size() ; j++){

            if(std::find(pass.begin(), pass.end(), j) != pass.end()) continue;

            float score = IOU(pred_boxes[i], gt_boxes[j]);
            // cout << "score: " << score << endl;

            if (score > 0.5){
               TP += 1;
               pass.push_back(j);
               found = true;
            //    cout << "TP: " << TP << endl;
               break;
            }

        }
        if(!found) FP += 1;
    }

    assert(TP + FP == pred_boxes.size());   
    int FN = gt_boxes.size() - TP;


    if (TP == 0){
        return 0;
    }
    else{
        float prec = float(TP) / gt_boxes.size();
        // cout << "prec: " << prec << endl;
        return prec;
    }
    
    
}

int main_test(int argc, char** argv){    

    std::string subset = "val";
    std::string rep_str = "/media/leyan/E/DataSet/retinaface/" + subset + "/images/";
    auto label = get_wider_label(subset);

    auto imgPaths = exec(("find " + rep_str + " -name *.jpg").c_str());
    cout << imgPaths.size() << endl;
    
    if(true){
        int target_index = -1;
        int biggest_size = -99;

        std::vector<double> if_time_v;
        std::vector<double> if_time_all_v;

        vector<Mat>faces_tmp;
        vector<vector<cv::Point> >landmarks_tmp;
        vector<Rect >faceboxs_tmp;

        cv::Mat det_image;

        vector<float> overall_ap;

        for(int i = 0 ; i < imgPaths.size(); i++){
            // recursive read folder images

            // if(imgPaths[i].find("27_Spa_Spa_27_38") == std::string::npos) continue;
            
            cv::Mat image = cv::imread(imgPaths[i]);   
            // cv::Mat image = cv::imread("/media/leyan/E/Git/ViaFaceRec/tests/data/1.jpg");             
            auto ans = findFace(image,  faces_tmp, faceboxs_tmp, landmarks_tmp);


            // cout << imgPaths[i] << endl;
            // recursive calculate error rate with faceboxs_tmp
            // for(int j = 0 ; j < faceboxs_tmp.size(); j++){
            //     cout << faceboxs_tmp[j].x << endl;
            //     cout << faceboxs_tmp[j].y << endl;
            //     cout << faceboxs_tmp[j].width << endl;
            //     cout << faceboxs_tmp[j].height << endl;
            //     cout << "***********************" << endl;
            // }

            std::vector<double> v = ans.first;

            if_time_v.push_back(v[0]);
            if_time_all_v.push_back(v[1]);

            det_image = ans.second;
            
            
            // acc
            std::string key = imgPaths[i].replace(imgPaths[i].find(rep_str), rep_str.length(), "");
            // cout << key << endl;
            auto bboxs = label.at(key);
            int s = min(bboxs.size(), faceboxs_tmp.size());

            float ap = prec(faceboxs_tmp, bboxs);
            overall_ap.push_back(ap);

            cout << "AP: " << ap << " , " << key << endl;
            
            for(int j = 0 ; j < faceboxs_tmp.size(); j++){
                // cout << "PR" << j << ": " << faceboxs_tmp[j].x << " , " << faceboxs_tmp[j].y << " , " << faceboxs_tmp[j].width << " , " << faceboxs_tmp[j].height << endl;
                // 1. calculate overlaps between bboxs & faceboxs_tmp
                // 2. recorded IoU coverage level => _gt_overlaps > 0.5 => found TP
                // 3. recall = found / gt_boxes nums
                // 4. Error rate = (FP + FN) / total (pred num + gt num)

                // 5. calculate ovell all Recall
                //     overall[0] += found // TP
                //     overall[1] += gt_boxes.shape[0] // TP + FN
                //     recall_all = float(overall[0]) / overall[1]
            }

            // // GT label view
            for(int j = 0 ; j < bboxs.size(); j++){
                cv::rectangle(det_image, cv::Rect(bboxs[j].x, (int)bboxs[j].y, (int)bboxs[j].width, (int)bboxs[j].height), cv::Scalar(255,255,0),4);
                // cout << "GT" << j << ": " << bboxs[j].x << " , " << bboxs[j].y << " , " << bboxs[j].width << " , " << bboxs[j].height << endl;
            }            

            std::string dir = "/media/leyan/D/D_Disk/11_new/src/nullImpl/SCRFD_res/" + std::to_string(FaceDetection::input_size) + "/" + FaceDetection::mt + "-" + subset + "/" + key.substr(0, key.find_first_of("/"));
            if (!IsPathExist(dir)){
                std::string cmd = "mkdir -p " + dir;
                system(cmd.c_str());  
            }  
                     
            cv::imwrite("/media/leyan/D/D_Disk/11_new/src/nullImpl/SCRFD_res/" + std::to_string(FaceDetection::input_size) + "/" + FaceDetection::mt + "-" + subset + "/" + key, det_image);
            

            cout << "---------------roumd: " << i << " finished" << "---------------" << endl;
            faceboxs_tmp.clear();

            // show
            // cv::imshow("test", det_image);
            // cv::waitKey(0);
        }
        cout << "---------------------------------------" << endl;
        cout << "finished" << endl;

        double sum_of_elems = std::accumulate(if_time_v.begin(), if_time_v.end(), 0);
        cout << "Average Inference duration time: " << sum_of_elems/if_time_v.size() << " ms" << endl;

        double all_sum_of_elems = std::accumulate(if_time_all_v.begin(), if_time_all_v.end(), 0);
        cout << "Average All Inference duration time: " << all_sum_of_elems/if_time_all_v.size() << " ms" << endl; 

        double all_sum_of_overall_ap = std::accumulate(overall_ap.begin(), overall_ap.end(), 0);
        cout << "mAP: " << all_sum_of_overall_ap/overall_ap.size() << " %" << endl; 

        // cv::imshow("image", det_image);
        // cv::waitKey(0);  
    }

    

    //////////////////////////////////////////////////////////////////////////////////////////////
    if(false){
        std::vector<int> _feat_stride_fpn = {32, 16, 8};
        std::map<int, AnchorCfg> anchor_cfg = {
            {32, AnchorCfg(std::vector<float>{32,16}, std::vector<float>{1}, 16)},
            {16, AnchorCfg(std::vector<float>{8,4}, std::vector<float>{1}, 16)},
            {8,AnchorCfg(std::vector<float>{2,1}, std::vector<float>{1}, 16)}
        };

        bool dense_anchor = false;
        float cls_threshold = 0.5;
        float nms_threshold = 0.4;
        
        ncnn::Net _net;
        const std::string modeldir = "/media/leyan/D/D_Disk/11_new/src/nullImpl/old_fd";
    

        _net.load_param_bin((modeldir +"/fdparam.so").c_str());
        _net.load_model((modeldir+ "/fdbin.so").c_str());

        cv::Mat new_image;
        float pixel_mean[3] = {0, 0, 0};
        float pixel_std[3] = {1, 1, 1};
        float pixel_scale = 1.0;

        std::vector<double> if_time_v;
        std::vector<double> if_time_all_v;

        

        vector<float> overall_ap;

        for(int i = 0 ; i < imgPaths.size(); i++){
            cout << "---------------roumd: " << i << "---------------" << endl;
            auto st_all = std::chrono::high_resolution_clock::now();
            // cv::Mat image = cv::imread("/media/leyan/E/Git/ViaFaceRec/tests/data/1.jpg");    
            cv::Mat image = cv::imread(imgPaths[i]); 

            int target_size = FaceDetection::input_size;
            int max_size = 240;
            int img_width = image.cols;
            int img_height = image.rows;
            int im_size_min = min(img_width, img_height);
            int im_size_max = max(img_width, img_height);    

            float im_scale = float(target_size) / float(im_size_min);

            if (round(im_scale * im_size_max) > max_size){
                im_scale = float(max_size) / float(im_size_max);
            }

            int new_height = int(img_height*im_scale);
            int new_width = int(img_width*im_scale);

            // cout << "new_height: " << new_height << endl;
            // cout << "new_width: " << new_width << endl;

            ncnn::Mat input = ncnn::Mat::from_pixels_resize(image.data, ncnn::Mat::PIXEL_BGR2RGB, image.cols, image.rows, new_width, new_height);
            cv::resize(image, new_image, cv::Size(new_width, new_height));
            input.substract_mean_normalize(pixel_mean, pixel_std);

            auto st = std::chrono::high_resolution_clock::now();

            ncnn::Extractor _extractor = _net.create_extractor();
            _extractor.set_num_threads(1);
            _extractor.input(0, input);


            std::vector<AnchorGenerator> ac(_feat_stride_fpn.size());
            for (int i = 0; i < _feat_stride_fpn.size(); ++i) {
                int stride = _feat_stride_fpn[i];
                ac[i].Init(stride, anchor_cfg[stride], false);
            }

            std::vector<Anchor> proposals;
            proposals.clear();

            for (int i = 0; i < _feat_stride_fpn.size(); ++i) { 
                ncnn::Mat cls;
                ncnn::Mat reg;
                ncnn::Mat pts;

                int stride = _stride_fpn[i];
                std::string score_n = "face_rpn_cls_prob_reshape_stride" + std::to_string(stride);
                std::string bbox_n = "face_rpn_bbox_pred_stride" + std::to_string(stride);
                std::string kps_n = "face_rpn_landmark_pred_stride" + std::to_string(stride);

                _extractor.extract(target_blob_cls[i], cls);
                _extractor.extract(target_blob_reg[i], reg);
                _extractor.extract(target_blob_pts[i], pts);

                ac[i].FilterAnchor(cls, reg, pts, proposals);
            }

            std::vector<Anchor> result;

            nms_cpu(proposals, nms_threshold, result);

            std::vector<cv::Rect> pred_bboxs;

            for(int i = 0; i < result.size(); i ++)
            {
                vector<cv::Point> landmark_vec_tmp;
                float landmark_tmp[10] = {0};
                for (int j = 0; j < result[i].pts.size(); ++j) {
                    landmark_vec_tmp.push_back( cv::Point((int)result[i].pts[j].x, (int)result[i].pts[j].y) );
                    landmark_tmp[2*j] = result[i].pts[j].x;
                    landmark_tmp[2*j+1] = result[i].pts[j].y;
                }

                cv::rectangle(image, cv::Rect(cv::Point((int)result[i].finalbox.x / im_scale, (int)result[i].finalbox.y / im_scale), cv::Point((int)result[i].finalbox.width / im_scale, (int)result[i].finalbox.height / im_scale)), cv::Scalar(255,0,0),4);
                
                pred_bboxs.push_back(
                    cv::Rect((int)result[i].finalbox.x / im_scale, 
                            (int)result[i].finalbox.y / im_scale, 
                            (int)result[i].finalbox.width / im_scale - (int)result[i].finalbox.x / im_scale, 
                            (int)result[i].finalbox.height / im_scale - (int)result[i].finalbox.y / im_scale
                    ));
                // pred_bboxs.push_back(cv::Rect((int)result[i].finalbox.x / im_scale, (int)result[i].finalbox.y / im_scale, (int)result[i].finalbox.width / im_scale, (int)result[i].finalbox.height / im_scale));
                for (int j = 0; j < result[i].pts.size(); ++j) {
                    cv::circle(image, cv::Point((int)result[i].pts[j].x / im_scale, (int)result[i].pts[j].y / im_scale), 3, cv::Scalar(0,255,0),-1);
                }
            }   

            auto en = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(en - st);
            double if_time = time_span.count()*1000;

            std::chrono::duration<double> time_span_all = std::chrono::duration_cast<std::chrono::duration<double>>(en - st_all);
            double if_time_all = time_span_all.count()*1000;

            if_time_v.push_back(if_time);
            if_time_all_v.push_back(if_time_all);

            // acc
            std::string key = imgPaths[i].replace(imgPaths[i].find(rep_str), rep_str.length(), "");
            cout << key << endl;
            auto bboxs = label.at(key);
            int s = min(bboxs.size(), pred_bboxs.size());

            float ap = prec(pred_bboxs, bboxs);
            overall_ap.push_back(ap);

            if(ap != 0.0) cout << "AP: " << ap << " , " << key << endl;

            for(int j = 0 ; j < pred_bboxs.size(); j++){
                // cout << "GT" << j << ": " << bboxs[j].x << " , " << bboxs[j].y << " , " << bboxs[j].width << " , " << bboxs[j].height << endl;
                cout << "PR" << j << ": " << pred_bboxs[j].x << " , " << pred_bboxs[j].y << " , " << pred_bboxs[j].width << " , " << pred_bboxs[j].height << endl;
            }

            // GT label view
            for(int j = 0 ; j < bboxs.size(); j++){
                cv::rectangle(image, cv::Rect(bboxs[j].x, (int)bboxs[j].y, (int)bboxs[j].width, (int)bboxs[j].height), cv::Scalar(255,255,0),4);
                cout << "GT" << j << ": " << bboxs[j].x << " , " << bboxs[j].y << " , " << bboxs[j].width << " , " << bboxs[j].height << endl;
            }

            

            std::string dir = "/media/leyan/D/D_Disk/11_new/src/nullImpl/RetinaFace_res/" + subset + "/" + key.substr(0, key.find_first_of("/"));
            // cout << key.substr(0, key.find_first_of("/")) << endl;
            if (!IsPathExist(dir)){
                std::string cmd = "mkdir -p " + dir;
                system(cmd.c_str());  
            }  
                     
            cv::imwrite("/media/leyan/D/D_Disk/11_new/src/nullImpl/RetinaFace_res/" + key, image);

            // show
            // cv::imshow("test", image);
            // cv::waitKey(0);
        }

        cout << "---------------------------------------" << endl;
        cout << "finished" << endl;

        double sum_of_elems = std::accumulate(if_time_v.begin(), if_time_v.end(), 0);
        cout << "Average Inference duration time: " << sum_of_elems/if_time_v.size() << " ms" << endl;

        double all_sum_of_elems = std::accumulate(if_time_all_v.begin(), if_time_all_v.end(), 0);
        cout << "Average All Inference duration time: " << all_sum_of_elems/if_time_all_v.size() << " ms" << endl; 

        double all_sum_of_overall_ap = std::accumulate(overall_ap.begin(), overall_ap.end(), 0);
        cout << "mAP: " << all_sum_of_overall_ap/overall_ap.size() << " %" << endl; 
    }

}