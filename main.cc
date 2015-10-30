// UCF Lunar Knights!

#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// camera skeleton code is from stack overflow: 
// http://stackoverflow.com/questions/21202088/how-to-get-live-stream-from-webcam-in-opencv-ubuntu
// and
// http://docs.opencv.org/2.4/doc/tutorials/features2d/trackingmotion/harris_detector/harris_detector.html

const int blockSize = 2;
const int apertureSize = 3;
const char* kWindowName = "test";

static void onAdjustK(int newK, void* target) {
    auto k = static_cast<double*>(target);
    *k = newK / 1000.0;
    std::cerr << "new k: " << *k << std::endl;
}

void getCorners(const cv::Mat &src, cv::Mat &dst, double k) {
    cv::Mat corners = cv::Mat::zeros(src.size(), CV_32FC1);
    // cv::cornerMinEigenVal(greyed, corners, blockSize);
    cv::cornerHarris(
            src,
            corners,
            blockSize,
            apertureSize,
            k,
            cv::BORDER_DEFAULT);
    cv::Mat corners_norm = cv::Mat::zeros(src.size(), CV_32FC1);
    cv::normalize(
            corners,
            corners_norm,
            0,
            255,
            cv::NORM_MINMAX,
            CV_32FC1,
            cv::Mat());
    cv::convertScaleAbs(corners_norm, dst);
}

// std::string planeNames[3] = {"red", "green", "blue"};
// double ks[sizeof(planeNames)];
double corner_k = 0.004;

int main() {
    cv::namedWindow("feed");
    cv::namedWindow("corners");
    cv::createTrackbar(
            "corners_bar",
            "corners",
            nullptr,
            100,
            onAdjustK,
            static_cast<void*>(&corner_k));

    auto camera = cv::VideoCapture(-1);
    while (camera.isOpened()) {
        cv::Mat frame, hsv, planes[3];

        if (!camera.read(frame)) break;
        // greyed = cv::Mat::zeros(frame.size(), CV_32FC1);
        hsv = cv::Mat::zeros(frame.size(), CV_32FC3);

        cv::split(frame, planes);
        // cv::cvtColor(frame, greyed, CV_RGB2GRAY);
        // getCorners(greyed, corners, corner_k);
        cv::cvtColor(frame, hsv, CV_BGR2HSV);
        cv::Mat hsv_set[3];
        cv::split(hsv, hsv_set);

        cv::Mat corners[3], corners_tot, tmp;
        getCorners(planes[0], corners[0], corner_k);
        getCorners(planes[1], corners[1], corner_k);
        getCorners(planes[2], corners[2], corner_k);

        cv::max(corners[0], corners[1], tmp);
        cv::max(corners[2], tmp, corners_tot);

        cv::Mat tmp2, mask, masked;
        cv::compare(hsv_set[1], 32, mask, cv::CMP_GE);
        cv::min(mask, corners_tot, masked);
        cv::imshow("feed", frame);
        cv::imshow("corners", masked);
        cv::imshow("mask", mask);
        int k = cv::waitKey(33);
        if (k == 27) break;
    }

    return 0;
}
