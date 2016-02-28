#ifndef SEARCH_H
#define SEARCH_H

#include <vector>

#include <opencv2/opencv.hpp>

struct Result {
  cv::Point2f pts[3];
  float score;
};

//bool FindValidSquare(const std::vector<cv::Point2f>& pts, const cv::Mat &frame,
    //std::vector<cv::Point2f> &ret); 
std::vector<Result> FindValidSquare(const std::vector<cv::Point2f>& pts, const cv::Mat &frame); 

#endif
