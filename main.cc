// UCF Lunar Knights!

#include <algorithm>
#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <cmath>

#include <cstdlib>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// camera skeleton code is from stack overflow:
// http://stackoverflow.com/questions/21202088/how-to-get-live-stream-from-webcam-in-opencv-ubuntu
// and
// http://docs.opencv.org/2.4/doc/tutorials/features2d/trackingmotion/harris_detector/harris_detector.html

const int kBlockSize = 2;
const int kApertureSize = 3;

const int kNumSamples = 8;
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

// values copied from here: 
// http://stackoverflow.com/questions/5392061/algorithm-to-check-similarity-of-colors-based-on-rgb-values-or-maybe-hsv
cv::Vec3b Yuv(unsigned char r, unsigned char g, unsigned char b) {
  return cv::Vec3f(0.299*r + 0.587*g + 0.114*b,
                  -0.14713*r -0.28886*g + 0.436*b,
                  0.615*r - 0.51499*g - 0.10001*b);
}

double ColorDist(const cv::Vec3f &a, const cv::Vec3f &b) {
  const auto dy = a[0] - b[0];
  const auto du = a[1] - b[1];
  const auto dv = a[2] - b[2];
  return sqrt(dy*dy + du*du + dv*dv);
}

auto WHITE = Yuv(0xff, 0xff, 0xff);
auto RED = Yuv(0xff, 0x00, 0x00);
auto GREEN = Yuv(0x00, 0xff, 0x00);
auto BLUE = Yuv(0x00, 0x00, 0xff);

double IsMaybeColoredCorner(const cv::Point2f &p, const cv::Mat &src) {
  std::random_device randGen;
  auto angleDist = std::uniform_real_distribution<double>(0, 2*PI);
  auto dists = std::vector<double>();

  // pick points in a small circle near the point,
  // convert to YUV space,
  // and then average the distances to white, red, green, and blue
  for (auto i = 0; i < kNumSamples; i++) {
    const auto angle = angleDist(randGen);
    const auto x = static_cast<int>(p.x + kSampleRadius*cos(angle));
    const auto y = static_cast<int>(p.y + kSampleRadius*sin(angle));
    const auto rgb = src.at<cv::Vec3b>(y, x);
    // transform into yuv space to allow Cartesian distance to approximate
    // color distance
    const auto yuv = Yuv(rgb[0], rgb[1], rgb[2]);
    const auto targets = std::vector<cv::Vec3f>{WHITE, RED, GREEN, BLUE};
    auto localDists = std::vector<double>();
    for(auto &color: targets) {
      localDists.push_back(ColorDist(yuv, color));
    }
    // store the smallest value to tell how close this got to one of the
    // four main colors
    dists.push_back(*std::min_element(localDists.begin(), localDists.end()));
  }

  auto sum = 0.0;
  for (auto d : dists) {
    sum += d;
  }
  // the smaller the average, the most the points around the corner
  // are close to one of the four colors in the square, the more
  // likely it is one of the correct corners
  return 1-sum/dists.size();
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
    for (auto &p: corners) {
      std::cout << "corner at " << p.x << ", " << p.y << std::endl; 
      // cv::circle(sat, p, 4, 1.0f, 1, 8);
      auto cornerAmount = IsMaybeColoredCorner(frame, p);
      cv::circle(grey, p, 8, cv::Scalar(0, 0, 0), 1, 8, 0);
    }
    // cv::imshow("corners", satSmooth);
    cv::imshow("feed", frame);
    cv::imshow("corners", grey);

    int k = cv::waitKey(16);
    // if (k != -1) break;
  }

  return 0;
}
