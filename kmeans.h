#ifndef KMEANS_H
#define KMEANS_H

#include <vector>

#include <opencv2/opencv.hpp>

// used for colored corner detection
struct TaggedPoint {
  int idx;
  cv::Point2f p;
};

struct TaggedYuvPoint {
  int idx;
  cv::Vec3f yuv;
};

typedef std::vector<TaggedPoint> TaggedPoints;
typedef std::vector<TaggedYuvPoint> TaggedYuvPoints;
struct Cluster {
  std::vector<TaggedYuvPoint> list;
  cv::Vec3f centroid;
};
typedef std::vector<Cluster> Clusters;

TaggedYuvPoints SamplePoints(const TaggedPoints &points, const cv::Mat &frame);
Clusters KmeansCluster(const TaggedYuvPoints &yuvs);

#endif

