#include <cmath>

#include "search.h"

struct Vec2f {
  float x, y;
};

const float kSampleBounds = 0.02f;
const int kNumSamples = 128;
const float kTol = 16.0f;

bool FindValidSquare(const std::vector<cv::Point2f>& pts, const cv::Mat &frame,
    std::vector<cv::Point2f> &ret) {
  auto dist = [](const cv::Point2f &a, const cv::Point2f b) -> float {
    auto dx = a.x - b.x;
    auto dy = a.y - b.y;
    return sqrt(dx*dx + dy*dy);
  };

  auto colorDiff = [](const cv::Vec3b &a, const cv::Vec3b &b) -> float {
    auto dr = a[2] - b[2];
    auto dg = a[1] - b[1];
    auto db = a[0] - b[0];
    return abs(dr) + abs(dg) + abs(db);
  };

  struct Result {
    cv::Point2f pts[3];
    float gradient;
  };
  auto results = std::vector<Result>();
  // try out as many N*(N-1)*(N-2)/6 combinations of points before
  // we get what seems like 3 corners of a square.
  // let's traverse from the centroid back out towards each point
  for (int i = 0; i < pts.size(); i++) {
    for (int j = i + 1; j < pts.size(); j++) {
      for (int k = j + 1; k < pts.size(); k++) {
        auto a = pts[i],
             b = pts[j],
             c = pts[k];
        auto centroid = cv::Point2f{
          (a.x + b.x + c.x)/3,
          (a.y + b.y + c.y)/3,
        };

        float maxGradient = 0.0f;
        for (const auto &dst: {a, b, c}) {
          const auto dx = dst.x - centroid.x;
          const auto dy = dst.y - centroid.y;
          const auto d = dist(centroid, dst);
          auto s = [=](float l) -> Vec2f {
            return Vec2f{centroid.x + l*dx, centroid.y + l*dy};
          };

          cv::Vec3b lastPoint;
          {
            const auto tmp = s(kSampleBounds);
            lastPoint = frame.at<cv::Vec3b>(
                static_cast<int>(tmp.y), 
                static_cast<int>(tmp.x));
          }
          for (auto q = 0; q < kNumSamples; q++) {
            const auto tmp = s(kSampleBounds + (1.0 - 2*kSampleBounds)*q/kNumSamples);
            const auto curPoint = frame.at<cv::Vec3b>(
                static_cast<int>(tmp.y), 
                static_cast<int>(tmp.x));
            const auto curGradient = colorDiff(curPoint, lastPoint)*kNumSamples/d;
            if (curGradient > maxGradient) {
              maxGradient = curGradient;
            }
            lastPoint = curPoint;
          }
        }
        results.push_back(Result{{a, b, c}, maxGradient});
      }
    }
  }

  if (results.size() <= 0) {
    return false;
  }
  auto result = *std::min_element(results.begin(), results.end(),
      [](const Result &first, const Result &second) -> bool {
        return first.gradient < second.gradient;
      });
  if (result.gradient <= kTol) {
    ret.push_back(result.pts[0]);
    ret.push_back(result.pts[1]);
    ret.push_back(result.pts[2]);
    return true;
  }
  return false;
}
