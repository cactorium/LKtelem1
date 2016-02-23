#ifndef SEARCH_H
#define SEARCH_H

#include <vector>

#include <opencv2/opencv.hpp>


bool FindValidSquare(const std::vector<cv::Point2f>& pts, const cv::Mat &frame,
    std::vector<cv::Point2f> &ret);

#endif
