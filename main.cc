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
    /*
    for (auto i = 0; i < 3; i++) {
        cv::namedWindow(planeNames[i]);
        cv::createTrackbar(
                std::string(planeNames[i]) + "_bar",
                planeNames[i],
                nullptr,
                100,
                onAdjustK,
                static_cast<void*>(&ks[i]));
    }
    */
    cv::createTrackbar(
            "corners_bar",
            "corners",
            nullptr,
            100,
            onAdjustK,
            static_cast<void*>(&corner_k));

    auto camera = cv::VideoCapture(-1);
    while (camera.isOpened()) {
        cv::Mat frame, recombined;
        cv::Mat corners[3], planes[3];

        if (!camera.read(frame)) break;
        cv::split(frame, planes);
        for (int i = 0; i < 3; i++) {
            // getCorners(planes[i], corners, ks[i]);
            getCorners(planes[i], corners[i], corner_k);
            // cv::imshow(planeNames[i], corners[i]);
        }
        // cv::merge(corners, 3, recombined);
        // greyed = cv::Mat::zeros(frame.size(), CV_32FC1);
        // cv::cvtColor(frame, greyed, CV_RGB2GRAY);
        cv::Mat tmp1, tmp2;
        cv::add(corners[0], corners[1], tmp1);
        cv::add(tmp1, corners[2], tmp2);
        cv::imshow("feed", frame);
        // cv::imshow("corners", recombined);
        cv::imshow("corners", tmp2);
        int k = cv::waitKey(33);
        if (k == 27) break;
    }

    return 0;
}
