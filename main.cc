// UCF Lunar Knights!

/*
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
*/

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <math.h>
#include <string.h>

using namespace cv;
using namespace std;

static void help()
{
    cout <<
    "\nA program using pyramid scaling, Canny, contours, contour simpification and\n"
    "memory storage (it's got it all folks) to find\n"
    "squares in a list of images pic1-6.png\n"
    "Returns sequence of squares detected on the image.\n"
    "the sequence is stored in the specified memory storage\n"
    "Call:\n"
    "./squares\n"
    "Using OpenCV version %s\n" << CV_VERSION << "\n" << endl;
}


int thresh = 50, N = 5;
const char* wndname = "Square Detection Demo";

// helper function:
// finds a cosine of angle between vectors
// from pt0->pt1 and from pt0->pt2
static double angle( Point pt1, Point pt2, Point pt0 )
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

// returns sequence of squares detected on the image.
// the sequence is stored in the specified memory storage
static void findSquares( const Mat& image, vector<vector<Point> >& squares )
{
    squares.clear();

    Mat pyr, timg, gray0(image.size(), CV_8U), gray;

    // down-scale and upscale the image to filter out the noise
    // pyrDown(image, pyr, Size(image.cols/2, image.rows/2));
    // pyrUp(pyr, timg, image.size());
    timg = image;
    vector<vector<Point> > contours;

    // find squares in every color plane of the image
    for( int c = 0; c < 3; c++ )
    {
        int ch[] = {c, 0};
        mixChannels(&timg, 1, &gray0, 1, ch, 1);

        // try several threshold levels
        for( int l = 0; l < N; l++ )
        {
            // hack: use Canny instead of zero threshold level.
            // Canny helps to catch squares with gradient shading
            if( l == 0 )
            {
                // apply Canny. Take the upper threshold from slider
                // and set the lower to 0 (which forces edges merging)
                Canny(gray0, gray, 0, thresh, 5);
                // dilate canny output to remove potential
                // holes between edge segments
                dilate(gray, gray, Mat(), Point(-1,-1));
            }
            else
            {
                // apply threshold if l!=0:
                //     tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
                gray = gray0 >= (l+1)*255/N;
            }

            // find contours and store them all as a list
            findContours(gray, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

            vector<Point> approx;

            // test each contour
            for( size_t i = 0; i < contours.size(); i++ )
            {
                // approximate contour with accuracy proportional
                // to the contour perimeter
                approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.02, true);

                // square contours should have 4 vertices after approximation
                // relatively large area (to filter out noisy contours)
                // and be convex.
                // Note: absolute value of an area is used because
                // area may be positive or negative - in accordance with the
                // contour orientation
                if( approx.size() == 4 &&
                    fabs(contourArea(Mat(approx))) > 10 &&
                    isContourConvex(Mat(approx)) )
                {
                    double maxCosine = 0;

                    for( int j = 2; j < 5; j++ )
                    {
                        // find the maximum cosine of the angle between joint edges
                        double cosine = fabs(angle(approx[j%4], approx[j-2], approx[j-1]));
                        maxCosine = MAX(maxCosine, cosine);
                    }

                    /*
                    // if cosines of all angles are small
                    // (all angles are ~90 degree) then write quandrange
                    // vertices to resultant sequence
                    if( maxCosine < 0.3 )
                        squares.push_back(approx);
                        */
                    squares.push_back(approx);
                }
            }
        }
    }
}


// the function draws all the squares in the image
static void drawSquares( Mat& image, const vector<vector<Point> >& squares )
{
    for( size_t i = 0; i < squares.size(); i++ )
    {
        const Point* p = &squares[i][0];
        int n = (int)squares[i].size();
        polylines(image, &p, &n, 1, true, Scalar(0,255,0), 1, CV_AA);
    }

    imshow(wndname, image);
}


int main(int /*argc*/, char** /*argv*/)
{
    auto camera = cv::VideoCapture(-1);

    namedWindow( wndname, 1 );
    vector<vector<Point> > squares;

    while (camera.isOpened()) {
        cv::Mat frame, hsv, planes[3];

        if (!camera.read(frame)) break;
        findSquares(frame, squares);
        drawSquares(frame, squares);

        int c = waitKey(1);
        if( (char)c == 27 )
            break;

    }

    return 0;
}
