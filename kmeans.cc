#include <cmath>
#include <random>

#include "kmeans.h"

const double PI = 3.1415926535897931159979634685441851615905761718750;
const int kNumSamples = 16;
const int kSampleRadius = 6;
const int kNumClusters = 7;

// values copied from here: 
// http://stackoverflow.com/questions/5392061/algorithm-to-check-similarity-of-colors-based-on-rgb-values-or-maybe-hsv
cv::Vec3f Yuv(unsigned char r, unsigned char g, unsigned char b) {
  return cv::Vec3f(0.299*r + 0.587*g + 0.114*b,
                  -0.14713*r -0.28886*g + 0.436*b,
                  0.615*r - 0.51499*g - 0.10001*b);
}

float ColorDist(const cv::Vec3f &a, const cv::Vec3f &b) {
  const auto dy = a[0] - b[0];
  const auto du = a[1] - b[1];
  const auto dv = a[2] - b[2];
  return sqrt(/*dy*dy + */ du*du + dv*dv);
}

unsigned char UcharClamp(float f) {
  if (f < 0) {
    return 0;
  } else if (f >= 255) {
    return 255;
  } else {
    return static_cast<unsigned char>(f);
  }
}

TaggedYuvPoints SamplePoints(const Points &points, const cv::Mat &frame) {
  auto yuvs = std::vector<TaggedYuvPoint>();
  int idx = 0;
  for (const auto p: points) {
    for (auto i = 0; i < kNumSamples; i++) {
      const auto x = p.x + kSampleRadius * cos(2*i*PI/kNumSamples);
      const auto y = p.y + kSampleRadius * sin(2*i*PI/kNumSamples);
      const auto color = frame.at<cv::Vec3b>(y, x);
      const auto tmp = Yuv(color[2], color[1], color[0]);
      yuvs.push_back({idx, {tmp[0], tmp[1], tmp[2]}});
    }
    ++idx;
  }
  return yuvs;
}

Clusters KmeansCluster(const TaggedYuvPoints &yuvs) {
  std::mt19937 rng;
  rng.seed(std::random_device()());
  auto dist = std::uniform_int_distribution<unsigned char>(0, 255);

  // step 2: generate means, let's use 7 of them
  // 3 from assumed colors (red, green, blue)
  // 4 randomly selected
  auto randomColor = [&]() -> cv::Vec3f {
    return cv::Vec3f(dist(rng), dist(rng), dist(rng));
  };

  Clusters clusters;
  auto means = std::array<cv::Vec3f, kNumClusters> {
    Yuv(0xff, 0x00, 0x00),
    Yuv(0x00, 0xff, 0x00),
    Yuv(0x00, 0x00, 0xff),
    randomColor(),
    randomColor(),
    randomColor(),
    randomColor(),
  };
  for (auto &color: means) {
    clusters.push_back(Cluster{TaggedYuvPoints(), color});
  }
  auto changed = false;
  auto minCluster = [&](const TaggedYuvPoint& p) -> int {
    int curMin = -1;
    float dist = 100000000000.0f;

    for (auto i = 0; i < clusters.size(); i++) {
      auto curDist = ColorDist(p.yuv, clusters[i].centroid);
      if (curDist < dist) {
        curMin = i;
        dist = curDist;
      }
    }
    return curMin;
  };

  for (const auto &yuv: yuvs) {
    auto curMin = minCluster(yuv);
    clusters[curMin].list.push_back(yuv);
  }

  do {
    changed = false;
    // step 1 for the k means algorithm; calculate the new centroids
    for (auto &cluster: clusters) {
      if (cluster.list.size() <= 0) {
        cluster.centroid = randomColor();
      } else {
        auto sum = std::array<float, 3>{};
        // auto sum = cv::Vec3f(0.0, 0.0, 0.0);
        for (const auto &yuv: cluster.list) {
          sum[0] += yuv.yuv[0];
          sum[1] += yuv.yuv[1];
          sum[2] += yuv.yuv[2];
        }
        sum[0] = sum[0]/static_cast<int>(cluster.list.size());
        sum[1] = sum[1]/static_cast<int>(cluster.list.size());
        sum[2] = sum[2]/static_cast<int>(cluster.list.size());
        cluster.centroid = cv::Vec3f({sum[0], sum[1], sum[2]});
      }
    }
    // step 2 for the k means algorithm; recluster the points
    for (auto i = 0; i < kNumClusters; i++) {
      auto newEnd = std::remove_if(clusters[i].list.begin(), clusters[i].list.end(),
          [=](TaggedYuvPoint &p) {
            return minCluster(p) != i;
          });
      if (newEnd != clusters[i].list.end()) {
        changed = true;
        for (auto it = newEnd; it != clusters[i].list.end(); ++newEnd) {
          auto idx = minCluster(*it);
          clusters[idx].list.push_back(*it);
        }
        clusters[i].list.erase(newEnd, clusters[i].list.end());
      }
    }
  } while (changed);

  return clusters;
}
