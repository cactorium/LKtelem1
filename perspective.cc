#include <cmath>

#include "perspective.h"

const float kCalcTol = 0.001;
const unsigned int kMaxCalcLoops = 1000;

double CalculateError(const std::vector<cv::Point2f> &viewPointPoints,
    const Spacing &realPointSpacing, const Transform &t);

Transform NewtonRaphsonMethodIteration(const std::vector<cv::Point2f> &viewPointPoints,
    const Spacing &realPointSpacing, Transform t, unsigned int loops);

double CalculateError(const std::vector<cv::Point2f> &viewPointPoints,
    const Spacing &realPointSpacing, const Transform &t) {
  // transform the realPointSpacing based on t
  // apply the perspective transform to get it into viewpoint space
  // and then sum the distances between corresponding points
}

Transform NewtonsRaphsonMethodIteration(const std::vector<cv::Point2f> &viewPointPoints,
    const Spacing &realPointSpacing, Transform t, unsigned int loops) {
  float val = CalculateError(viewPointPoints, realPointSpacing, t);
  if (loops > kMaxCalcLoops || abs(val) <= kCalcTol) {
    return t;
  } else {
    // TODO
  }
}


Transform FindTransform(const std::vector<cv::Point2f> &viewPointPoints,
    const Spacing &realPointSpacing) {
  auto ret = Transform{0};
  return NewtonRaphsonMethodIteration(viewPointPoints, realPointSpacing, ret, 0);
}

