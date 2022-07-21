#ifndef FD_TOOLS
#define FD_TOOLS

#include "anchor_generator.h"
#include <sys/stat.h>
#include <iostream>

void nms_cpu(std::vector<Anchor>& boxes, float threshold, std::vector<Anchor>& filterOutBoxes);

std::vector<std::string> exec(const char *cmd);


void tokenize(std::string const &str, const char delim,
            std::vector<std::string> &out);

std::map<std::string, std::vector<cv::Rect> > get_wider_label(std::string subset);

bool IsPathExist(const std::string &s);

void visualize(const char* title, const ncnn::Mat& m);

void transpose(const ncnn::Mat& in, ncnn::Mat& out);

void pretty_print(const ncnn::Mat& m);
void pretty_print(ncnn::Mat &m, const char *objectName);
void pretty_print_kps(ncnn::Mat &m, const char *objectName);
void pretty_print_bbox(ncnn::Mat &m, const char *objectName);
void pretty_print_score(ncnn::Mat &m, const char *objectName);

#endif // FD_TOOLS
