// UCF Lunar Knights!

#include <algorithm>
#include <iostream>
#include <vector>
#include <string>

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


void Saturation(const cv::Mat &frame, cv::Mat &dst) {
  dst = cv::Mat::zeros(frame.size(), CV_32FC1);
  for (int y = 0; y < frame.rows; y++) {
    for (int x = 0; x < frame.cols; x++) {
      float c = VecMax(frame.at<cv::Vec3b>(y, x)) - VecMin(frame.at<cv::Vec3b>(y, x));
      float v = VecMax(frame.at<cv::Vec3b>(y, x));
      dst.at<float>(y, x) = std::abs(v) > 0.2f ? c/v : 0.0f;
    }
  }
}

struct LivePoint {
  cv::Point2f point;
  int life;
};

void AgePoints(std::vector<LivePoint> &vec) {
  std::for_each(vec.begin(), vec.end(), [](LivePoint &lp) {
      lp.life--;
  });
  auto newEnd = std::remove_if(vec.begin(), vec.end(), [](const LivePoint &lp) {
      return lp.life < 0;
  });
  vec.resize(newEnd - vec.begin());
}

const int kMergeThreshold = 5;
const int kStartingLife = 10;

// merge a new list of points with the existing one; if a new point is found
// close to an old one, replace the old with the new, else add to the end 
// of the list
void MergePoints(std::vector<LivePoint> &vec, const std::vector<cv::Point2f> &nv) {
  std::for_each(nv.begin(), nv.end(), [&](const cv::Point2f &p) {
    bool hasMatch = false;
    auto newEnd = std::remove_if(vec.begin(), vec.end(), [&](const LivePoint &lp) {
      // woo Manhattan distance because I'm lazy
      auto dist = std::min(std::abs(lp.point.x - p.x), std::abs(lp.point.y - p.y));
      if (dist < kMergeThreshold && !hasMatch) {
        hasMatch = true;
        std::cout << "found match" << std::endl;
        return true;
      } else {
        return false;
      }
    });
    vec.resize(newEnd - vec.begin());
    vec.push_back(LivePoint{
        p,
        kStartingLife
    });
  });
}

int main(int argc, char *argv[]) {
  cv::namedWindow("feed");
  cv::namedWindow("corners");

  auto camera = cv::VideoCapture(-1);
  auto corners = std::vector<LivePoint>();
  while (camera.isOpened()) {
    cv::Mat frame;

    if (!camera.read(frame)) break;

    cv::Mat grey;
    cv::cvtColor(frame, grey, cv::COLOR_RGB2GRAY);
    AgePoints(corners);
    auto newCorners = std::vector<cv::Point2f>();
    GetCorners(grey, newCorners);
    MergePoints(corners, newCorners);
    std::sort(corners.begin(), corners.end(), 
        [](const LivePoint &a, const LivePoint &b) -> bool {
          if (std::abs(a.point.x - b.point.x) < 10.0f) {
            return a.point.y < b.point.y;
          } else {
            return a.point.x < b.point.x;
          }
        });
    // hsv = cv::Mat::zeros(frame.size(), CV_32FC3);
    // cv::Mat corners, greyed_corners, mask, masked;
    // GetCorners(greyed, greyed_corners, corner_k);

    std::cout << "corners start " << corners.size() << 
        ", camera size: " << frame.cols << "x" << frame.rows << std::endl;
    for (auto &p: corners) {
      std::cout << "corner at " << p.point.x << ", " << p.point.y <<
          ", life " << p.life << std::endl; 
      // cv::circle(sat, p, 4, 1.0f, 1, 8);
      cv::circle(grey, p.point, 8, cv::Scalar(0, 0, 0), 1, 8, 0);
    }
    // cv::imshow("corners", satSmooth);
    cv::imshow("feed", frame);
    cv::imshow("corners", grey);

    int k = cv::waitKey(16);
    // if (k != -1) break;
  }

  return 0;
}
