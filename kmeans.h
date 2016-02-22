#ifndef KMEANS_H
#define KMEANS_H

#include <vector>

#include <opencv2/opencv.hpp>

/*
// used for colored corner detection
struct TaggedPoint {
  int idx;
  cv::Point2f p;
};
*/

struct Vec3f {
  float y;
  float u;
  float v;
};

struct TaggedYuvPoint {
  int idx;
  Vec3f yuv;
};

// typedef std::vector<TaggedPoint> TaggedPoints;
typedef std::vector<cv::Point2f> Points;
typedef std::vector<TaggedYuvPoint> TaggedYuvPoints;
struct Cluster {
  std::vector<TaggedYuvPoint> list;
  Vec3f centroid;
};
typedef std::vector<Cluster> Clusters;

TaggedYuvPoints SamplePoints(const Points &points, const cv::Mat &frame);
Clusters KmeansCluster(const TaggedYuvPoints &yuvs);

#endif

