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
