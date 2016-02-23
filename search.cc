#include <cmath>

#include "search.h"

struct Vec2f {
  float x, y;
};

const float kSampleBounds = 0.2;
const int  kNumSamples = 64;
const float kTol = 8.0f;

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
    return sqrt(dr*dr + dg*dg + db*db);
  };

  // try out as many N*(N-1)*(N-2)/6 combinations of points before
  // we get what seems like 3 corners of a square.
  // on three corners of a square, the diagonal (longest side) will be of
  // approximately uniform color. So that's what we'll use to detect a square!
  //
  //
  // EDIT: soo the diagonal DOES work, but it also has a lot of false positives
  // let's traverse from the centroid back out towards each point
  for (int i = 0; i < pts.size(); i++) {
    for (int j = i + 1; j < pts.size(); j++) {
      for (int k = j + 1; k < pts.size(); k++) {
        // soo find the longest length
        auto a = pts[i], b = pts[j], c = pts[k];
        auto ab = dist(a, b),
             ac = dist(a, c),
             bc = dist(b, c);
        /*
        cv::Point2f start, end;
        float dist;
        if (ab > ac && ab > bc) {
          start = a, end = b, dist = ab;
        } else if (ac > ab && ac > bc) {
          start = a, end = c, dist = ac;
        } else {
          start = b, end = c, dist = bc;
        }
        auto dx = end.x - start.x;
        auto dy = end.y - start.y;
        auto s = [=](float l) -> Vec2f {
          return Vec2f{start.x + l*dx, start.y + l*dy};
        };

        cv::Vec3b lastPoint;
        {
          auto tmp = s(kSampleBounds);
          lastPoint = frame.at<cv::Vec3b>(
              static_cast<int>(tmp.y), 
              static_cast<int>(tmp.x));
        }

        bool success = true;
        for (auto q = 0; q < kNumSamples; q++) {
          auto tmp = s(kSampleBounds + (1.0 - 2*kSampleBounds)*q/kNumSamples);
          auto curPoint = frame.at<cv::Vec3b>(
              static_cast<int>(tmp.y), 
              static_cast<int>(tmp.x));
          if (colorDiff(curPoint, lastPoint)/dist >= kTol/kNumSamples) {
            success = false;
            break;
          }
          lastPoint = curPoint;
        }
        */

        auto centroid = cv::Point2f{
          (a.x + b.x + c.x)/3,
          (a.y + b.y + c.y)/3,
        };

        bool success = true;
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
            if (colorDiff(curPoint, lastPoint)/d >= kTol/kNumSamples) {
              success = false;
              break;
            }
            lastPoint = curPoint;
          }
        }

        if (success) {
          ret.push_back(a);
          ret.push_back(b);
          ret.push_back(c);
          return true;
        }
      }
    }
  }
  return false;
}
