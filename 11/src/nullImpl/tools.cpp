#include "tools.h"

using namespace std;

void nms_cpu(std::vector<Anchor>& boxes, float threshold, std::vector<Anchor>& filterOutBoxes) {
    filterOutBoxes.clear();
    if(boxes.size() == 0)
        return;
    std::vector<size_t> idx(boxes.size());

    for(unsigned i = 0; i < idx.size(); i++)
    {
        idx[i] = i;
    }

    //descending sort
    sort(boxes.begin(), boxes.end(), std::greater<Anchor>());

    while(idx.size() > 0)
    {
        int good_idx = idx[0];
        filterOutBoxes.push_back(boxes[good_idx]);

        std::vector<size_t> tmp = idx;
        idx.clear();
        for(unsigned i = 1; i < tmp.size(); i++)
        {
            int tmp_i = tmp[i];
            float inter_x1 = std::max( boxes[good_idx][0], boxes[tmp_i][0] );
            float inter_y1 = std::max( boxes[good_idx][1], boxes[tmp_i][1] );
            float inter_x2 = std::min( boxes[good_idx][2], boxes[tmp_i][2] );
            float inter_y2 = std::min( boxes[good_idx][3], boxes[tmp_i][3] );

            float w = std::max((inter_x2 - inter_x1 + 1), 0.0F);
            float h = std::max((inter_y2 - inter_y1 + 1), 0.0F);

            float inter_area = w * h;
            float area_1 = (boxes[good_idx][2] - boxes[good_idx][0] + 1) * (boxes[good_idx][3] - boxes[good_idx][1] + 1);
            float area_2 = (boxes[tmp_i][2] - boxes[tmp_i][0] + 1) * (boxes[tmp_i][3] - boxes[tmp_i][1] + 1);
            float o = inter_area / (area_1 + area_2 - inter_area);           
            if( o <= threshold )
                idx.push_back(tmp_i);
        }
    }
}

std::vector<std::string> exec(const char *cmd)
{
    char buffer[512];
    std::string result = "";
    FILE *pipe = popen(cmd, "r");
    std::vector<std::string> res_list;
    if (!pipe)
        throw std::runtime_error("popen() failed!");
    try
    {
        while (fgets(buffer, sizeof buffer, pipe) != NULL)
        {
            result += buffer;
            std::string buf_str = buffer;
            buf_str.erase(std::remove(buf_str.begin(), buf_str.end(), '\n'), buf_str.end());
            res_list.push_back(buf_str);
        }
    }
    catch (...)
    {
        pclose(pipe);
        throw;
    }
    pclose(pipe);
    result.erase(std::remove(result.begin(), result.end(), '\n'), result.end());
    return res_list;
}

void tokenize(std::string const &str, const char delim,
            std::vector<std::string> &out)
{
    // construct a stream from the string
    std::stringstream ss(str);
 
    std::string s;
    while (std::getline(ss, s, delim)) {
        out.push_back(s);
    }
}

std::map<std::string, std::vector<cv::Rect> > get_wider_label(std::string subset){

    std::map<std::string, std::vector<cv::Rect> > label;
    std::string key = ""; 
    const char delim = ' ';
    int objs = 0;

    cv::Rect parse_rect;

    std::ifstream input( "/media/leyan/E/DataSet/retinaface/" + subset + "/label.txt" );
    for( std::string line; getline( input, line ); )
    {
        if (line.find(".jpg") != std::string::npos) {
            key = line;
            std::vector<cv::Rect> bboxs;
            label[key] = bboxs;
            
        }else{
            std::vector<std::string> out;
            tokenize(line, delim, out);
            if(out.size() == 10){
                parse_rect = cv::Rect(std::stoi(out[0]), std::stoi(out[1]), std::stoi(out[2]), std::stoi(out[3]));
    
                std::vector<cv::Rect> bboxs = label[key];
                bboxs.push_back(parse_rect);
                label[key] = bboxs;
            
            }else{
                objs = std::stoi( line );
            }
        }        
    }
    cout << label.size() << endl;
    return label;
}

bool IsPathExist(const std::string &s)
{
  struct stat buffer;
  return (stat (s.c_str(), &buffer) == 0);
}


void visualize(const char* title, const ncnn::Mat& m)
{
    std::vector<cv::Mat> normed_feats(m.c);

    for (int i=0; i<m.c; i++)
    {
        cv::Mat tmp(m.h, m.w, CV_32FC1, (void*)(const float*)m.channel(i));

        cv::normalize(tmp, normed_feats[i], 0, 255, cv::NORM_MINMAX, CV_8U);

        cv::cvtColor(normed_feats[i], normed_feats[i], cv::COLOR_GRAY2BGR);

        // check NaN
        for (int y=0; y<m.h; y++)
        {
            const float* tp = tmp.ptr<float>(y);
            uchar* sp = normed_feats[i].ptr<uchar>(y);
            for (int x=0; x<m.w; x++)
            {
                float v = tp[x];
                if (v != v)
                {
                    sp[0] = 0;
                    sp[1] = 0;
                    sp[2] = 255;
                }

                sp += 3;
            }
        }
    }

    int tw = m.w < 10 ? 32 : m.w < 20 ? 16 : m.w < 40 ? 8 : m.w < 80 ? 4 : m.w < 160 ? 2 : 1;
    int th = (m.c - 1) / tw + 1;

    cv::Mat show_map(m.h * th, m.w * tw, CV_8UC3);
    show_map = cv::Scalar(127);

    // tile
    for (int i=0; i<m.c; i++)
    {
        int ty = i / tw;
        int tx = i % tw;

        normed_feats[i].copyTo(show_map(cv::Rect(tx * m.w, ty * m.h, m.w, m.h)));
    }

    cv::resize(show_map, show_map, cv::Size(0,0), 2, 2, cv::INTER_NEAREST);
    cv::imshow(title, show_map);
}

void transpose(const ncnn::Mat& in, ncnn::Mat& out)
{
    // chw to cwh
    ncnn::Option opt;
    opt.num_threads = 2;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = true;

    ncnn::Layer* op = ncnn::create_layer("Permute");

    // set param
    ncnn::ParamDict pd;
    pd.set(0, 1);// order_type

    op->load_param(pd);

    op->create_pipeline(opt);

    ncnn::Mat in_packed = in;
    {
        // resolve dst_elempack
        int dims = in.dims;
        int elemcount = 0;
        if (dims == 1) elemcount = in.elempack * in.w;
        if (dims == 2) elemcount = in.elempack * in.h;
        if (dims == 3) elemcount = in.elempack * in.c;

        int dst_elempack = 1;
        // if (layer->support_packing)
        // {
        //     if (elemcount % 8 == 0 && ncnn::cpu_support_x86_avx2())
        //         dst_elempack = 8;
        //     else if (elemcount % 4 == 0)
        //         dst_elempack = 4;
        // }

        if (in.elempack != dst_elempack)
        {
            convert_packing(in, in_packed, dst_elempack, opt);
        }
    }

    // forward
    op->forward(in_packed, out, opt);

    op->destroy_pipeline(opt);

    delete op;
}


void pretty_print(const ncnn::Mat& m)
{
    for (int q=0; q<m.c; q++)
    {
        const float* ptr = m.channel(q);
        for (int y=0; y<m.h; y++)
        {
            for (int x=0; x<m.w; x++)
            {
                printf("%f ", ptr[x]);
            }
            ptr += m.w;
            printf("\n");
        }
        printf("------------------------\n");
    }
}


void pretty_print(ncnn::Mat &m, const char *objectName)
{
    int oneLinePrintNum = 0;

    std::string oString;
    oString = objectName;
    std::string name = "/media/leyan/D/D_Disk/11/src/nullImpl/build/" + oString + ".txt";


    FILE *file = fopen(name.c_str(),"w");
    for (int q=0; q<m.c; q++)
    {
        const float *ptr = m.channel(q);
        for (int y=0; y<m.h; y++)
        {
            for (int x=0; x<m.w; x++)
            {
                oneLinePrintNum++;
                if(oneLinePrintNum%2==0) {
                    char inferenceOut[256];
                    // sprintf(inferenceOut, "%f %f %f %f %f\n", ptr[x-4], ptr[x-3], ptr[x-2], ptr[x-1], ptr[x]);
                    sprintf(inferenceOut, "%f %f\n", ptr[x-1], ptr[x]);
                    fwrite(inferenceOut,strlen(inferenceOut),1, file);
                }
            }
            ptr += m.w;
        }
    }
    fclose(file);
}


void pretty_print_kps(ncnn::Mat &m, const char *objectName)
{
    int oneLinePrintNum = 0;

    std::string oString;
    oString = objectName;
    std::string name = "/media/leyan/D/D_Disk/11/src/nullImpl/build/" + oString + ".txt";


    FILE *file = fopen(name.c_str(),"w");
    for (int q=0; q<m.c; q++)
    {
        const float *ptr = m.channel(q);
        for (int y=0; y<m.h; y++)
        {
            for (int x=0; x<m.w; x++)
            {
                oneLinePrintNum++;
                if(oneLinePrintNum%10==0) {
                    char inferenceOut[256];
                    sprintf(inferenceOut, "%f %f %f %f %f %f %f %f %f %f\n", ptr[x-9], ptr[x-8], ptr[x-7], ptr[x-6], ptr[x-5], ptr[x-4], ptr[x-3], ptr[x-2], ptr[x-1], ptr[x]);
                    // sprintf(inferenceOut, "%f %f\n", ptr[x-1], ptr[x]);
                    fwrite(inferenceOut,strlen(inferenceOut),1, file);
                }
            }
            ptr += m.w;
        }
    }
    fclose(file);
}


void pretty_print_bbox(ncnn::Mat &m, const char *objectName)
{
    int oneLinePrintNum = 0;

    std::string oString;
    oString = objectName;
    std::string name = "/media/leyan/D/D_Disk/11/src/nullImpl/build/" + oString + ".txt";


    FILE *file = fopen(name.c_str(),"w");
    for (int q=0; q<m.c; q++)
    {
        const float *ptr = m.channel(q);
        for (int y=0; y<m.h; y++)
        {
            for (int x=0; x<m.w; x++)
            {
                oneLinePrintNum++;
                if(oneLinePrintNum%4==0) {
                    char inferenceOut[256];
                    sprintf(inferenceOut, "%f %f %f %f\n", ptr[x-3], ptr[x-2], ptr[x-1], ptr[x]);
                    // sprintf(inferenceOut, "%f %f\n", ptr[x-1], ptr[x]);
                    fwrite(inferenceOut,strlen(inferenceOut),1, file);
                }
            }
            ptr += m.w;
        }
    }
    fclose(file);
}

void pretty_print_score(ncnn::Mat &m, const char *objectName)
{
    int oneLinePrintNum = 0;

    std::string oString;
    oString = objectName;
    std::string name = "/media/leyan/D/D_Disk/11_new/src/nullImpl/build/" + oString + ".txt";


    FILE *file = fopen(name.c_str(),"w");
    for (int q=0; q<m.c; q++)
    {
        const float *ptr = m.channel(q);
        for (int y=0; y<m.h; y++)
        {
            for (int x=0; x<m.w; x++)
            {
                oneLinePrintNum++;
                if(oneLinePrintNum%1==0) {
                    char inferenceOut[256];
                    sprintf(inferenceOut, "%f\n", ptr[x]);
                    // sprintf(inferenceOut, "%f %f\n", ptr[x-1], ptr[x]);
                    fwrite(inferenceOut,strlen(inferenceOut),1, file);
                }
            }
            ptr += m.w;
        }
    }
    fclose(file);
}

