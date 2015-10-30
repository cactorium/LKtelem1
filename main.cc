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

void chroma(const cv::Mat &frame, cv::Mat &dst, int idx[], uchar threshold) {
    dst = cv::Mat::zeros(frame.size(), CV_8UC1);
    for (int y = 0; y < frame.rows; y++) {
        for (int x = 0; x < frame.cols; x++) {
            int target = frame.at<cv::Vec3b>(y, x)[idx[0]];
            int other1 = frame.at<cv::Vec3b>(y, x)[idx[1]];
            int other2 = frame.at<cv::Vec3b>(y, x)[idx[2]];
            dst.at<uchar>(y, x) = static_cast<uchar>(
                    target*(255-other1)*(255-other2) >> 16);
        }
    }
}

// std::string planeNames[3] = {"red", "green", "blue"};
// double ks[sizeof(planeNames)];
double corner_k = 0.004;
int threshold = 0x80;

int main() {
    cv::namedWindow("feed");
    cv::namedWindow("corners");
    cv::namedWindow("blue");
    cv::createTrackbar(
            "corners_bar",
            "corners",
            nullptr,
            100,
            onAdjustK,
            static_cast<void*>(&corner_k));
    cv::createTrackbar(
            "threshold",
            "feed",
            &threshold,
            256,
            nullptr,
            nullptr);

    auto camera = cv::VideoCapture(-1);
    while (camera.isOpened()) {
        cv::Mat frame, greyed, hsv;

        if (!camera.read(frame)) break;
        // cv::Mat planes[3];
        // cv::split(frame, planes);

        cv::Mat red, green, blue;

        int perms[][3] = {{0, 1, 2}, {1, 0, 2}, {2, 0, 1}};
        chroma(frame, red, perms[0], threshold);
        chroma(frame, green, perms[1], threshold);
        chroma(frame, blue, perms[2], threshold);

        hsv = cv::Mat::zeros(frame.size(), CV_32FC3);
        cv::cvtColor(frame, greyed, CV_RGB2GRAY);
        cv::Mat corners, greyed_corners, mask, masked;
        getCorners(greyed, greyed_corners, corner_k);

        cv::imshow("feed", red);
        cv::imshow("corners", green);
        cv::imshow("blue", blue);

        int k = cv::waitKey(33);
        if (k != -1) break;
    }

    return 0;
}
