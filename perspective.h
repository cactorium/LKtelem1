#ifndef PERSPECTIVE_H
#define PERSPECTIVE_H

#include <vector>

#include <opencv2/opencv.hpp>

// describes a known orientation between the input points
typedef std::vector<cv::Point2f> Spacing;

struct Vector3f {
  float x, y, z;
};

struct Transform {
  Vector3f translate, orientation;
};

Transform FindTransform(const std::vector<cv::Point2f> &viewPointPoints,
    const Spacing &realPointSpacing);

#endif
