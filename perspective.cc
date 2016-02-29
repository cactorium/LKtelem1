#include <cassert>
#include <cmath>

#include <numeric>

#include <iostream>

#include "perspective.h"

const double kDelta = 0.001;
const double kCalcTol = 0.001;
const unsigned int kMaxCalcLoops = 500;

const double kViewpointScale = 64.0;
const double kZOffset = 100.0;

// NOTE: viewpoint coordinates here assume +y is up, +x is to the right,
// and (0, 0) is the center of the frame

std::vector<double> CalculateErrors(const std::vector<cv::Point2f> &viewPointPoints,
    const Spacing &realPointSpacing, const Transform &t);

Transform NewtonRaphsonMethodIteration(const std::vector<cv::Point2f> &viewPointPoints,
    const Spacing &realPointSpacing, Transform t, unsigned int loops);

Vector3f UseTransform(const Vector3f &v, const Transform &t);
cv::Point2f PerspectiveTransform(const Vector3f &v);

// apply the rotation and then the translation
// https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions#Conversion_formulae_between_formalisms
Vector3f UseTransform(const Vector3f &v, const Transform &t) {
  const double phi = t.orientation.x,
               theta = t.orientation.y,
               psi = t.orientation.z;
  const double rot[3][3] = {
    {std::cos(theta)*std::sin(psi), std::cos(phi)*std::sin(psi)+std::sin(phi)*std::sin(phi)*std::sin(theta)*std::cos(psi),
      std::sin(phi)*std::sin(psi)-std::cos(phi)*std::sin(theta)*std::cos(psi)},
    {-std::cos(theta)*std::sin(psi), std::cos(phi)*std::cos(psi)-std::sin(phi)*std::sin(theta)*std::sin(psi),
      std::sin(phi)*std::cos(psi)+std::cos(phi)*std::sin(theta)*std::sin(psi)},
    {std::sin(theta), -std::sin(phi)*std::cos(theta), std::cos(phi)*std::cos(theta)},
  };
  const auto tmp = Vector3f{
    v.x*rot[0][0] + v.y*rot[0][1] + v.z*rot[0][2],
    v.x*rot[1][0] + v.y*rot[1][1] + v.z*rot[1][2],
    v.x*rot[2][0] + v.y*rot[2][1] + v.z*rot[2][2],
  };
  return Vector3f{
    t.translate.x + tmp.x,
    t.translate.y + tmp.y,
    t.translate.z + tmp.z,
  };
}

cv::Point2f PerspectiveTransform(const Vector3f &v) {
  return cv::Point2f(
      kViewpointScale*v.x/(v.z+kZOffset),
      kViewpointScale*v.y/(v.z+kZOffset)
  );
}

// This is the function we're trying to minimize
std::vector<double> CalculateErrors(const std::vector<cv::Point2f> &viewPointPoints,
    const Spacing &realPointSpacing, const Transform &t) {
  // transform the realPointSpacing based on t
  auto ts = std::vector<Vector3f>();
  for (const auto &sp: realPointSpacing) {
    ts.push_back(UseTransform(Vector3f{sp.x, sp.y, 0.0f}, t));
  }
  // apply the perspective transform to get it into viewpoint space
  auto ps = std::vector<cv::Point2f>();
  for (const auto &tsp: ts) {
    ps.push_back(PerspectiveTransform(tsp));
  }

  // and then find the offset between corresponding points
  auto ret = std::vector<double>();
  for (auto i = 0u; i < ps.size() && i < viewPointPoints.size(); ++i) {
    ret.push_back(ps[i].x - viewPointPoints[i].x);
    ret.push_back(ps[i].y - viewPointPoints[i].y);
  }
  return ret;
}

Transform NewtonRaphsonMethodIteration(const std::vector<cv::Point2f> &viewPointPoints,
    const Spacing &realPointSpacing, Transform t, unsigned int loops) {
  const auto val = CalculateErrors(viewPointPoints, realPointSpacing, t);
  std::cout << "errs " << val[0] << " " << val[1] << " " << val[2] <<
    " " << val[3] << " " << val[4] << " " << val[5] << std::endl;
  auto sum = 0.0;
  for (const auto &v: val) {
    sum += std::abs(v);
  }
  if (loops > kMaxCalcLoops || sum <= kCalcTol) {
    if (loops > kMaxCalcLoops) {
      std::cout << "loops exceeded" << std::endl;
    }
    return t;
  } else {
    // let's calculate derivatives so we can form a Jacobian matrix
    // that we'll invert to use in the Newton-Raphson solver equation thing
    // this does not sound incredibly complicated at all
    // I don't feel like figuring out the derivative of the entire formula,
    // so the computer does it instead! f'(x) = lim h->0 (f(x+h)-f(x)/h)
    // just gotta pick a smallish h, for six different variables, to get
    // 36 values
    auto perturbed = std::vector<std::vector<double> >();
    auto dt = t;
    dt.translate.x += kDelta;
    perturbed.push_back(CalculateErrors(viewPointPoints, realPointSpacing, dt));

    dt = t;
    dt.translate.y += kDelta;
    perturbed.push_back(CalculateErrors(viewPointPoints, realPointSpacing, dt));

    dt = t;
    dt.translate.z += kDelta;
    perturbed.push_back(CalculateErrors(viewPointPoints, realPointSpacing, dt));

    dt = t;
    dt.orientation.x += kDelta;
    perturbed.push_back(CalculateErrors(viewPointPoints, realPointSpacing, dt));

    dt = t;
    dt.orientation.y += kDelta;
    perturbed.push_back(CalculateErrors(viewPointPoints, realPointSpacing, dt));

    dt = t;
    dt.orientation.z += kDelta;
    perturbed.push_back(CalculateErrors(viewPointPoints, realPointSpacing, dt));

    double jacobian[6][6];
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        assert(perturbed[j].size() == 6);
        jacobian[i][j] = (perturbed[j][i] - val[i])/kDelta;
      }
    }
    std::cout << "jacobian " << std::endl;
    std::cout << jacobian[0][0] << " " << jacobian[0][1] << " " << jacobian[0][2] << " "
              << jacobian[0][3] << " " << jacobian[0][4] << " " << jacobian[0][5] << std::endl;
    std::cout << jacobian[1][0] << " " << jacobian[1][1] << " " << jacobian[1][2] << " "
              << jacobian[1][3] << " " << jacobian[1][4] << " " << jacobian[1][5] << std::endl;
    std::cout << jacobian[2][0] << " " << jacobian[2][1] << " " << jacobian[2][2] << " "
              << jacobian[2][3] << " " << jacobian[2][4] << " " << jacobian[2][5] << std::endl;
    std::cout << jacobian[3][0] << " " << jacobian[3][1] << " " << jacobian[3][2] << " "
              << jacobian[3][3] << " " << jacobian[3][4] << " " << jacobian[3][5] << std::endl;
    std::cout << jacobian[4][0] << " " << jacobian[4][1] << " " << jacobian[4][2] << " "
              << jacobian[4][3] << " " << jacobian[4][4] << " " << jacobian[4][5] << std::endl;
    std::cout << jacobian[5][0] << " " << jacobian[5][1] << " " << jacobian[5][2] << " "
              << jacobian[5][3] << " " << jacobian[5][4] << " " << jacobian[5][5] << std::endl;


    // the equation is basically -f(x_0) = J delta_x
    // so using Gaussian elimination on this system of equations will get us
    // our delta_x
    // we're using Gaussian elimination because taking the inverse on a 6x6 matrix
    // sucks balls
    assert(val.size() == 6);
    double rhs[6] = {-val[0], -val[1], -val[2], -val[3], -val[4], -val[5]};
    double dx[6] = {0};


    // do row operations to get the jacobian/rhs into upper triangular form
    for (int i = 0; i < 6; ++i) {
      double max = std::abs(jacobian[i][i]);
      int rowIdx = i;
      for (int j = i+1; j < 6; ++j) {
        // find the best row and swap with the current row to improve
        // numerical stability
        if (std::abs(jacobian[j][i]) > max) {
          max = std::abs(jacobian[j][i]);
          rowIdx = j;
        }
      }
      if (std::abs(max) <= 0.00000000000001) {
        std::cerr << "singular jacobian! " << i << std::endl;
        return t;
      }
      if (rowIdx != i) {
        double tmp[6] = {0};
        double rhsTmp = rhs[i];
        for (int j = 0; j < 6; ++j) {
          tmp[j] = jacobian[i][j];
        }
        for (int j = 0; j < 6; ++j) {
          jacobian[i][j] = jacobian[rowIdx][j];
        }
        rhs[i] = rhs[rowIdx];
        for (int j = 0; j < 6; ++j) {
          jacobian[rowIdx][j] = tmp[j];
        }
        rhs[rowIdx] = rhsTmp;
      }
      for (int j = i+1; j < 6; ++j) {
        if (std::abs(jacobian[j][i]) > 0.0000001) {
          const double scale = jacobian[j][i]/jacobian[i][i];
          for (int k = i+1; k < 6; ++k) {
            jacobian[j][k] -= scale*jacobian[i][k];
          }
          jacobian[j][i] = 0.0;
          rhs[j] -= scale*rhs[i];
        }
      }
    }
    // now substitute starting on the last row
    // to recover the variables
    for (int i = 5; i >= 0; --i) {
      dx[i] = rhs[i]/jacobian[i][i];
      // propagate the changes to the remaining rows
      for (int j = 0; j < i; ++j) {
        rhs[j] -= jacobian[j][i]*dx[i];
      }
    }

    std::cout << "dx " 
      << dx[0] << " " 
      << dx[1] << " " 
      << dx[2] << " " 
      << dx[3] << " " 
      << dx[4] << " " 
      << dx[5] << std::endl;

    auto newt = t;
    newt.translate.x += dx[0];
    newt.translate.y += dx[1];
    newt.translate.z += dx[2];
    newt.orientation.x += dx[3];
    newt.orientation.y += dx[4];
    newt.orientation.z += dx[5];
    return NewtonRaphsonMethodIteration(viewPointPoints, realPointSpacing,
        newt, loops+1);
  }
}


Transform FindTransform(const std::vector<cv::Point2f> &viewPointPoints,
    const Spacing &realPointSpacing) {
  auto ret = Transform{0};
  return NewtonRaphsonMethodIteration(viewPointPoints, realPointSpacing, ret, 0);
}

