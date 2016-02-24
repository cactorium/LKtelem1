#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>

#include <cmath>

#include "search.h"

struct Vec2f {
  float x, y;
};

const float kSampleBounds = 0.05f;
const int kNumSamples = 128;
const float kTol = 16.0f;

bool FindValidSquare(const std::vector<cv::Point2f>& pts, const cv::Mat &gry,
    std::vector<cv::Point2f> &ret) {
  auto dist = [](const cv::Point2f &a, const cv::Point2f b) -> float {
    const auto dx = a.x - b.x;
    const auto dy = a.y - b.y;
    return sqrt(dx*dx + dy*dy);
  };

  cv::Mat /* edge_raw, */ edge;
  cv::Laplacian(gry, edge, CV_16S, 3, 1, 0, cv::BORDER_DEFAULT);
  // cv::convertScaleAbs(edge_raw, edge);

  struct Result {
    cv::Point2f pts[3];
    float score;
  };
  struct ScoreDist {
    float d;
    float a;
    float M;
    float m;
  };
  auto results = std::vector<Result>();
  // try out as many N*(N-1)*(N-2)/6 combinations of points before
  // we get what seems like 3 corners of a square.
  for (int i = 0; i < pts.size(); i++) {
    for (int j = i + 1; j < pts.size(); j++) {
      for (int k = j + 1; k < pts.size(); k++) {
        const auto a = pts[i],
                   b = pts[j],
                   c = pts[k];
        // traverse all three lines, if two of them are edges, then this
        // is (probably) the triangle we're looking for
        std::vector<ScoreDist> averages;
        const cv::Point2f perms[3][2] = {{a, b}, {a, c}, {b, c}};
        for (const auto &c: perms) {
          auto points = std::vector<cv::Point2f>();
          for (const auto &p: c) {
            points.push_back(p);
          }
          const auto dx = points[1].x - points[0].x;
          const auto dy = points[1].y - points[0].y;

          auto s = [=](float l) -> Vec2f {
            return Vec2f{
              points[0].x + l*dx, points[1].y + l*dy
            };
          };
          float sum = 0.0f;
          float min = 100000000.0f;
          float max = -100000000.0f;
          const float numSamples = dist(points[0], points[1])*1.1f;
          for (int q = 0; q < numSamples; q++) {
            const auto pt = s(kSampleBounds + (1.0f-2*kSampleBounds)*q/numSamples);
            const auto val = edge.at<short>(static_cast<int>(pt.y), static_cast<int>(pt.x));
            sum += val;
            if (val > max) {
              max = val;
            }
            if (val < min) {
              min = val;
            }
          }
          averages.push_back(ScoreDist{numSamples, sum/numSamples, min, max});
        }
        if (averages[0].d > averages[1].d && averages[0].d > averages[2].d) {
          results.push_back(Result{{a, b, c}, -2*averages[0].m+averages[1].M+averages[2].M});
        } else if (averages[1].d > averages[0].d && averages[1].d > averages[2].d) {
          results.push_back(Result{{a, b, c}, averages[0].M-2*averages[1].m+averages[2].M});
        } else {
          results.push_back(Result{{a, b, c}, averages[0].M+averages[1].M-2*averages[2].m});
        }
      }
    }
  }

  if (results.size() <= 0) {
    return false;
  } else {
    auto result = *std::max_element(results.begin(), results.end(),
        [](const Result &a, const Result &b) -> bool {
          return a.score < b.score;
        });
    ret.push_back(result.pts[0]);
    ret.push_back(result.pts[1]);
    ret.push_back(result.pts[2]);
    return true;
  }

  return false;
}
