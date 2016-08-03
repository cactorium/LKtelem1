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
#include "search.h"
#include "perspective2.h"

// camera skeleton code is from stack overflow:
// http://stackoverflow.com/questions/21202088/how-to-get-live-stream-from-webcam-in-opencv-ubuntu
// and
// http://docs.opencv.org/2.4/doc/tutorials/features2d/trackingmotion/harris_detector/harris_detector.html

const unsigned int kBlockSize = 2;
const unsigned int kApertureSize = 3;

const unsigned int kNumSamples = 16;
const unsigned int kSampleRadius = 6;

const unsigned int kMaxMatchAttempts = 4;

constexpr double PI = 3.1415926535898;

int cameraDepthInt = 20;
int cameraScaleInt = 1;
double cameraDepth = 10.0;
double cameraScale = 100.0;

void OnTrackbar(int, void*) {
  cameraDepth = cameraDepthInt < 1? 1 :
    (cameraDepthInt > 400 ? 400 : static_cast<double>(cameraDepthInt));
  cameraScale = cameraScaleInt < 1? 0.001 :
    (cameraDepthInt > 1000 ? 1.0 : static_cast<double>(cameraScale)/1000.0);
}

void GetCorners(const cv::Mat &src, std::vector<cv::Point2f> &dst) {
  cv::Mat corners = cv::Mat::zeros(src.size(), CV_32FC1);
  cv::goodFeaturesToTrack(src, dst, 16, 0.01, 20);
}

void SortPoints(const Result &r, cv::Point2f sortedPoints[3]) {
  // let's store in sortedPoints the diagonal points first,
  // followed by the nondiagonal pont
  struct Line {
    cv::Point2f pts[2];
    float d;
  };
  auto dist = [](const cv::Point2f &a, const cv::Point2f &b) -> float {
    auto dx = b.x - a.x;
    auto dy = b.y - a.y;
    return sqrt(dx*dx + dy*dy);
  };
  const auto a = r.pts[0], b = r.pts[1], c = r.pts[2];
  const cv::Point2f perms[3][2] = {{a, b}, {a, c}, {b, c}};
  auto lines = std::vector<Line>();
  for (const auto &pair: perms) {
    lines.push_back(Line{{pair[0], pair[1]}, dist(pair[0], pair[1])});
  }
  std::sort(lines.begin(), lines.end(), [](const Line &a, const Line &b) -> bool {
      return a.d > b.d;
  });
  const auto diagonal = lines[0];
  sortedPoints[0] = diagonal.pts[0];
  sortedPoints[1] = diagonal.pts[1];
  for (const auto &p: r.pts) {
    bool notMatch = true;
    for (const auto &q: diagonal.pts) {
      if (p == q) {
        notMatch = false;
      }
    }
    if (notMatch) {
      sortedPoints[2] = p;
    }
  }
  // let's order the diagonal points so that the triangle points
  // are in clockwise order (or counterclockwise, right hand coordinate systems
  // are confusing)
  // they're in (some) order if the cross product between two vectors connecting
  // the points is always a particular sign
  // the degenerate case (cross product is zero) is unimportant because
  // that implies all three points are collinear, and we can't use that
  // anyways
  // why sort them so they'll be in a particular order? it's because it
  // lets you do a lot of interesting math without having to make
  // any assumptions about orientation. geometry is weird and confusing
  {
    // by the way these nested scopes are to stop variables from leaking
    // and messing with other things; I should probably be just making
    // separate functions for each of these but I'm lazy
    const auto ax = sortedPoints[1].x - sortedPoints[0].x;
    const auto ay = sortedPoints[1].y - sortedPoints[0].y;
    const auto bx = sortedPoints[2].x - sortedPoints[1].x;
    const auto by = sortedPoints[2].y - sortedPoints[1].y;
    const auto cross = ax*by - ay*bx;
    if (cross < 0) {
      // swap em
      const auto tmp = sortedPoints[0];
      sortedPoints[0] = sortedPoints[1];
      sortedPoints[1] = tmp;
    }
  }
}

struct MatchResult {
  // coordinates in the viewpoint space and the orientation
  int x, y, orientation;
};

bool MatchPattern(cv::Mat &frame, cv::Point2f sortedPoints[3], std::vector<MatchResult> &matchResults) {
  auto fromFrame = [&](const cv::Point2f &p) -> cv::Vec3b {
    return frame.at<cv::Vec3b>(
        static_cast<int>(p.y), static_cast<int>(p.x));
  };
  // first point is the easy one, midpoint of the diagonal. this tells
  // us the color the square the block is in
  const auto diagPoint = cv::Point2f(
      (sortedPoints[0].x + sortedPoints[1].x)/2,
      (sortedPoints[0].y + sortedPoints[1].y)/2);

  // the other two sides are harder, because we want the point near the
  // midpoint, but a little more away from the triangle so we're sure
  // we're sampling from the neighboring point
  auto pointOutsideMid = [](const cv::Point2f &a, const cv::Point2f &b) -> cv::Point2f {
    // assuming the points are fed in in counterclockwise order, this works!
    const auto midx = (a.x + b.x)/2;
    const auto midy = (a.y + b.y)/2;
    const auto outx = (b.y - a.y);
    const auto outy = -(b.x - a.x);
    return cv::Point2f((midx + outx/4), (midy + outy/4));
  };
  const auto firstPoint = pointOutsideMid(sortedPoints[1], sortedPoints[2]);
  const auto secondPoint = pointOutsideMid(sortedPoints[2], sortedPoints[0]);

  const auto diagSample = fromFrame(diagPoint);
  const auto firstSample = fromFrame(firstPoint);
  const auto secondSample = fromFrame(secondPoint);

  // now let's categorize the points as either red, green, blue, or white
  enum CornerColor { CornerRed, CornerGreen, CornerBlue, CornerWhite, CornerDontCare };
  auto closestColor = [](const cv::Vec3b &color) -> CornerColor {
    struct ColorPair {
      cv::Vec3b color;
      CornerColor name;
    };
    struct ColorResult {
      float dist;
      CornerColor color;
    };
    auto results = std::vector<ColorResult>();
    const ColorPair colorPairs[4] = {
      ColorPair{cv::Vec3b{255, 255, 255}, CornerWhite},
      ColorPair{cv::Vec3b{255, 0, 0}, CornerBlue},
      ColorPair{cv::Vec3b{0, 255, 0}, CornerGreen},
      ColorPair{cv::Vec3b{0, 0, 255}, CornerRed},
    };
    // ideally for accuracy we'd do the color distance map in YUV space, but I'm lazy
    // so we're just gonna do in BGR space (it'ssupposedtobeRGBbutOpenCVisweird)
    auto colorDist = [](const cv::Vec3b &a, const cv::Vec3b &b) -> float {
      auto db = static_cast<float>(a[0] - b[0]);
      auto dg = static_cast<float>(a[1] - b[1]);
      auto dr = static_cast<float>(a[2] - b[2]);
      return sqrt(db*db + dg*dg + dr*dr);
    };
    // scale the values so that the maximum component is always 255
    // to improve color detection
    auto normalizeColor = [](const cv::Vec3b &n) -> cv::Vec3b {
      auto maxn = std::max(std::max(n[0], n[1]), n[2]);
      if (maxn > 0) {
        float scale = 255/maxn;
        return cv::Vec3b(
            static_cast<unsigned char>(std::min(scale*n[0], 255.0f)),
            static_cast<unsigned char>(std::min(scale*n[1], 255.0f)),
            static_cast<unsigned char>(std::min(scale*n[2], 255.0f)));
      } else {
        return n;
      }
    };
    for (const auto &cp: colorPairs) {
      const auto nColor = normalizeColor(cp.color);
      results.push_back(ColorResult{colorDist(nColor, color), cp.name});
    }
    std::sort(results.begin(), results.end(),
        [](const ColorResult &a, const ColorResult &b) -> bool {
          return a.dist < b.dist;
        });
    return results[0].color;
  };
  const auto diagColor = closestColor(diagSample),
             firstColor = closestColor(firstSample),
             secondColor = closestColor(secondSample);
  // let's mark the samples on the color frame for debugging purposes
  auto nameToColor = [](CornerColor c) -> cv::Scalar {
    switch (c) {
      case CornerRed:
        return cv::Scalar(0, 0, 255);
        break;
      case CornerGreen:
        return cv::Scalar(0, 255, 0);
        break;
      case CornerBlue:
        return cv::Scalar(255, 0, 0);
        break;
      case CornerWhite:
        return cv::Scalar(255, 255, 255);
        break;
      default:
        return cv::Scalar(0, 0, 0);
        break;
    }
  };
  cv::circle(frame, diagPoint, 8, nameToColor(diagColor), 1, 8, 0);
  cv::circle(frame, firstPoint, 8, nameToColor(firstColor), 1, 8, 0);
  cv::circle(frame, secondPoint, 8, nameToColor(secondColor), 1, 8, 0);

  // now let's figure out where the triangle is in the diagram
  // one of the most important things in programming is to use the right
  // data structure for the right job
  // in this case I think an array of CornerColors is the right structure
  // we'll iterate over the array, trying to match up the three colors we
  // have with the matching set in the array, using all four possible
  // orientations and whatever
  const CornerColor matchingArray[4][5] = {
    {CornerDontCare, CornerWhite, CornerWhite, CornerWhite, CornerDontCare},
    {CornerWhite, CornerRed, CornerWhite, CornerGreen, CornerWhite},
    {CornerDontCare, CornerWhite, CornerBlue, CornerWhite, CornerDontCare},
    {CornerDontCare, CornerDontCare, CornerWhite, CornerDontCare, CornerDontCare},
  };
  CornerColor triangleArray[2][2] = {
    {diagColor, secondColor},
    {firstColor, CornerDontCare},
  };
  for (auto i = 0; i < 4; ++i) {
    // try to match against every possible location
    for (auto r = 0; r < 3; ++r) {
      for (auto c = 0; c < 4; ++c) {
        bool isMatch = true;
        for (auto q = 0; q < 2; ++q) {
          for (auto w = 0; w < 2; ++w) {
            if ((triangleArray[q][w] != CornerDontCare) &&
                (triangleArray[q][w] != matchingArray[r+q][c+w])) {
              isMatch = false;
            }
          }
        }
        if (isMatch) {
          matchResults.push_back(MatchResult{r, c, i});
        }
      }
    }
    // rotate the triangleArray counterclockwise
    const auto tmp00 = triangleArray[0][0], tmp01 = triangleArray[0][1],
               tmp10 = triangleArray[1][0], tmp11 = triangleArray[1][1];
    triangleArray[0][0] = tmp01;
    triangleArray[1][0] = tmp00;
    triangleArray[1][1] = tmp10;
    triangleArray[0][1] = tmp11;
  }
  return matchResults.size() == 1;
}

int main(int argc, char *argv[]) {
  cv::namedWindow("feed");
  cv::namedWindow("corners");

  cv::createTrackbar("feed", "camera depth", &cameraDepthInt, 400, OnTrackbar);
  cv::createTrackbar("feed", "camera scale", &cameraScaleInt, 1000, OnTrackbar);

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

    std::cout << "corners start " << corners.size() << 
        ", camera size: " << frame.cols << "x" << frame.rows << std::endl;
    //auto finalCorners = std::vector<cv::Point2f>();
    auto finalCorners = FindValidSquare(corners, grey);
    std::sort(finalCorners.begin(), finalCorners.end(),
        [](const Result &a, const Result &b) -> bool {
          return a.score < b.score;
        });
    {
      auto r = finalCorners[finalCorners.size()-1];
      auto matchResults = std::vector<MatchResult>();
      bool inSquare = false;
      cv::Point2f sortedPoints[3];
      for (auto idx = 0u; idx < kMaxMatchAttempts && idx < finalCorners.size(); ++idx) {
        r = finalCorners[finalCorners.size() - 1 - idx];
        // let's see where this triangle is relative to our drawing thing
        // first let's figure out its orientation; we need to know
        // which line is the diagonal first
        SortPoints(r, sortedPoints);
        
        // now let's match it to one of the possible places on the square thing
        matchResults.clear();
        inSquare = MatchPattern(frame, sortedPoints, matchResults);
        // triangle found, illuminati confirmed
        if (inSquare) break;
      }

      if (inSquare) {
        // let's find the transform!
        // first let's figure out what points our corners correspond to
        auto match = matchResults[0];
        auto viewPointPoints = std::vector<cv::Point2f>();
        for (const auto &p: sortedPoints) {
          viewPointPoints.push_back(cv::Point2f(p.x-frame.cols/2, -p.y+frame.rows/2));
        }
        auto spacing = Spacing(0);
        auto isValid = true;
        // this point's the nondiagonal one!
        // the other two will be found using the orientation
        // TODO: fix this
        auto matchCenter = cv::Point2f(
            -60.0+40.0*match.x,
            -40.0+40.0*match.y
            );
        switch (match.orientation) {
          case 0:
            // TODO: fix these
            spacing.push_back(cv::Point2f(matchCenter.x, matchCenter.y-40.0));
            spacing.push_back(cv::Point2f(matchCenter.x-40.0, matchCenter.y));
            spacing.push_back(matchCenter);
            break;
          case 1:
            spacing.push_back(cv::Point2f(matchCenter.x-40.0, matchCenter.y));
            spacing.push_back(cv::Point2f(matchCenter.x, matchCenter.y+40.0));
            spacing.push_back(matchCenter);
            break;
          case 2:
            spacing.push_back(cv::Point2f(matchCenter.x, matchCenter.y+40.0));
            spacing.push_back(cv::Point2f(matchCenter.x+40.0, matchCenter.y));
            spacing.push_back(matchCenter);
            break;
          case 3:
            spacing.push_back(cv::Point2f(matchCenter.x+40.0, matchCenter.y));
            spacing.push_back(cv::Point2f(matchCenter.x, matchCenter.y-40.0));
            spacing.push_back(matchCenter);
            break;
          default:
            isValid = false;
            std::cerr << "improper orientation detected." << std::endl;
        }
        if (isValid) {
          Coordinates outCoords;
          if (FindCoordinates(viewPointPoints, spacing, cameraDepth, cameraScale, outCoords)) {
            std::cout << "coordinates found!" << std::endl;
          }
          std::cout << "in " << viewPointPoints[0].x << "," <<
                        viewPointPoints[0].y << " " <<
                        viewPointPoints[1].x << "," <<
                        viewPointPoints[1].y << " " <<
                        viewPointPoints[2].x << "," <<
                        viewPointPoints[2].y << ";" <<
                        spacing[0].x << "," <<
                        spacing[0].y << " " <<
                        spacing[1].x << "," <<
                        spacing[1].y << " " <<
                        spacing[2].x << "," <<
                        spacing[2].y << std::endl;
        }
      }
      
      const auto triColor = inSquare ? cv::Scalar(0, 0, 0) : cv::Scalar(0, 0, 255);
      for (const auto &p: sortedPoints) {
        for (const auto &q: sortedPoints) {
          if (p != q) {
            cv::line(frame, p, q, triColor);
          }
        }
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
