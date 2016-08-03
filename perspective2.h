#ifndef PERSPECTIVE2_H_
#define PERSPECTIVE2_H_

#include <vector>

#include <opencv2/opencv.hpp>

typedef std::vector<cv::Point2f> Spacing;

struct Vector3f {
  double x, y, z;
};

typedef std::vector<Vector3f> Coordinates;

bool FindCoordinates(const std::vector<cv::Point2f> &vpp,
    const Spacing &rps, double cameraDepth, double cameraScale, Coordinates& out);

#endif // PERSPECTIVE2_H_
