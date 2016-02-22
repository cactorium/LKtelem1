// UCF Lunar Knights!

#include <algorithm>
#include <iostream>
#include <vector>
#include <string>

#include <cmath>
#include <cstdlib>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "kmeans.h"

// camera skeleton code is from stack overflow:
// http://stackoverflow.com/questions/21202088/how-to-get-live-stream-from-webcam-in-opencv-ubuntu
// and
// http://docs.opencv.org/2.4/doc/tutorials/features2d/trackingmotion/harris_detector/harris_detector.html

const int kBlockSize = 2;
const int kApertureSize = 3;

const int kNumSamples = 16;
const int kSampleRadius = 6;

constexpr double PI = 3.1415926535898;

void GetCorners(const cv::Mat &src, std::vector<cv::Point2f> &dst) {
  cv::Mat corners = cv::Mat::zeros(src.size(), CV_32FC1);
  cv::goodFeaturesToTrack(src, dst, 16, 0.01, 20);
}

int VecMax(const cv::Vec3b &v) {
  return std::max({v[0], v[1], v[2]});
}

int VecMin(const cv::Vec3b &v) {
  return std::min({v[0], v[1], v[2]});
}

int main(int argc, char *argv[]) {
  cv::namedWindow("feed");
  cv::namedWindow("corners");

  auto camera = cv::VideoCapture(-1);
  auto corners = std::vector<cv::Point2f>(16);
  while (camera.isOpened()) {
    corners.clear();

    cv::Mat frame;

    if (!camera.read(frame)) break;

    cv::Mat grey;
    cv::medianBlur(frame, frame, 3);
    cv::cvtColor(frame, grey, cv::COLOR_RGB2GRAY);
    GetCorners(grey, corners);
    std::sort(corners.begin(), corners.end(), 
        [](const cv::Point2f &a, const cv::Point2f &b) -> bool {
          if (std::abs(a.x - b.x) < 10.0f) {
            return a.y < b.y;
          } else {
            return a.x < b.x;
          }
        });
    // hsv = cv::Mat::zeros(frame.size(), CV_32FC3);
    // cv::Mat corners, greyed_corners, mask, masked;
    // GetCorners(greyed, greyed_corners, corner_k);

    std::cout << "corners start " << corners.size() << 
        ", camera size: " << frame.cols << "x" << frame.rows << std::endl;
    // generate a set of sample points from circles near points, and
    // apply k-means clustering on the color samples to group them by color.
    // Then the ones that match up with the four most popular sets (white+colors)
    // are the square corners! Lol dang this is gonna suck to implement
    auto clusters = KmeansCluster(SamplePoints(corners, frame));
    auto clusterCount = std::vector<int>(corners.size(), 0);
    {
      auto count = 0;
      for (const auto &c: clusters) {
        for (const auto &p: c.list) {
          clusterCount[p.idx] |= 1 << count;
        }
        ++count;
      }
    }

    // let's count how many each has now:
    std::transform(clusterCount.begin(), clusterCount.end(), clusterCount.begin(),
        [&](int &in) -> int {
          int count = 0;
          for(auto i = 0; i < clusters.size() && i < 8*sizeof(int); i++) {
            if (in & 1 << i) {
              ++count;
            }
          }
          return count;
        }
      );

    {
      int count = 0;
      for (const auto &p: corners) {
        std::cout << "corner at " << p.x << ", " << p.y << std::endl; 
        // cv::circle(sat, p, 4, 1.0f, 1, 8);
        unsigned char gry = 36*clusterCount[count];
        cv::circle(grey, p, 8, cv::Scalar(gry, gry, gry), 1, 8, 0);
        ++count;
      }
    }
    // cv::imshow("corners", satSmooth);
    cv::imshow("feed", frame);
    cv::imshow("corners", grey);

    int k = cv::waitKey(16);
    // if (k != -1) break;
  }

  return 0;
}
