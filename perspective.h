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
  // first one represents translation in x, y, and z axes in meters
  // second one represents euler angles because I'm too lazy to 
  // make another struct to properly store Euler angles
  Vector3f translate, orientation;
};

Transform FindTransform(const std::vector<cv::Point2f> &viewPointPoints,
    const Spacing &realPointSpacing);

#endif
