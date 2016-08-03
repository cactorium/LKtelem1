#include "perspective2.h"

#include <algorithm>
#include <limits>
#include <vector>

#include <cmath>

const double SOLUTION_TOL = 0.0001;
const int MAX_ITERATIONS = 100;

template <typename F> bool QuadSecantMethod(F f, double xs0[4], double xs1[4], double &result, int& idx) {
  double ys0[4], ys1[4];
  f(xs0, ys0);
  f(xs1, ys1);

  auto iterationCount = 0;
  auto notsolved = [&]() -> bool {
    for (int i = 0; i < 4; ++i) {
      if (!isnan(ys1[i]) && abs(ys1[i]) <= SOLUTION_TOL) {
        return false;
      }
    }
    return true;
  }
  while (notsolved(ys1) && iterationCount < MAX_ITERATIONS) {
    double xNext[4];
    for (int i = 0; i < 4; ++i) {
      xNext[i] = xs1[i] - 1.0 * ys1[i] * (xs1[i] - xs0[i])/(ys1[i] - ys0[i]);
    }
    memcpy(static_cast<void*>(xs0), static_cast<const void*>(xs0), 4*sizeof(double));
    memcpy(static_cast<void*>(xs1), static_cast<const void*>(xNext), 4*sizeof(double));

    memcpy(static_cast<void*>(ys0), static_cast<const void*>(ys1), 4*sizeof(double));
    f(xs1, ys1);
    ++iterationCount;
  }

  for (int i = 0; i < 4; ++i) {
    if (abs(ys1[i]) <= SOLUTION_TOL) {
      idx = i;
      result = xs1[idx];
      return true;
    }
  }
  return false;
}

template <typename F> bool LinearThenSecantMethod(
    F f, const double lower[4], const double upper[4], double &result, int& idx) {
  const int NUM_SEARCH = 16;
  double closestPos[4];
  double closestVal[4];
  for (int i = 0; i < 4; ++i) {
    closestPos[i] = 0.0;
    closestVal[i] = std::numeric_limits<double>::max();
  }

  for (int i = 0; i < NUM_SEARCH; ++i) {
    double pos[4];
    for (int j = 0; j < 4; ++j) {
      pos[j] = lower[j] + i*(upper[j] - lower[j])/NUM_SEARCH;
    }
    double result[4];
    f(pos, result);
    for (int j = 0; j < 4; ++j) {
      if (abs(result[j]) < closestVal[j]) {
        closestVal[j] = abs(result[j]);
        closestPos[j] = pos[j];
      }
    }
  }

  double closestVal2[4];
  for (int i = 0; i < 4; ++i) {
    closestVal2[i] = closestVal[i] + (upper[i] - lower[i])/(2*NUM_SEARCH);
  }

  return QuadSecantMethod(f, closestVal, closestVal2, result, idx);
}

bool FindCoordinates(const std::vector<cv::Point2f> &vpp,
    const Spacing &rps, double cameraDepth, double cameraScale, Coordinates& out) {

  double alpha[3], beta[3];
  for (int i = 0; i < 3; ++i) {
    alpha[i] = cameraScale*(vpp[i].x)/cameraDepth;
    beta[i] = cameraScale*(vpp[i].y)/cameraDepth;
  }

  auto sq = [](double x) -> double { return x*x; };
  auto dist = [&](cv::Point2f& a, cv::Point2f &b)[] -> double {
    return sqrt(sq(a.x - b.x) + sq(a.y - b.y));
  };

  auto u = dist(rps[0], rps[1]);
  auto v = dist(rps[1], rps[2]);
  auto w = dist(rps[0], rps[2]);

  auto a1 = sq(alpha[1]) + sq(beta[1]) + 1.0;
  auto b1 = alpha[0]*alpha[1] + beta[0]*beta[1] + 1.0;
  auto c1 = sq(alpha[0]) + sq(beta[0]) + 1.0;

  auto z2discr = [&](double z1) -> double {
    return sqrt((sq(b1) - a1*c1)*sq(z1) + a1*sq(u));
  };
  auto z2plus = [&](double z1) -> double {
    return (z1*b1 + z2discr(z1))/a1;
  };
  auto z2minu = [&](double z1) -> double{
    return (z1*b1 - z2discr(z1))/a1;
  };

  auto a2 = sq(alphas[2]) + sq(betas[2]) + 1.0;
  auto b2 = alphas[0]*alphas[2] + betas[0]*betas[2] + 1.0;

  auto z3discr = [&](double z1) -> double {
    return ((sq(b2) - a2*c1)*sq(z1) + a2*sq(w)).sqrt();
  };
  auto z3plus = [&](double z1) -> double {
    return (z1*b2 + z3discr(z1))/a2;
  };
  auto z3minu = [&](double z1) -> double {
    return (z1*b2 - z3discr(z1))/a2;
  };

  auto expr = [&](double z2, double z3) -> double {
    return (sq(alphas[2]) + sq(betas[2]) + 1.0)*sq(z3) 
        - 2.0*(alphas[1]*alphas[2] + betas[1]*betas[2] + 1.0)*z2*z3
        + (sq(alphas[1]) + sq(betas[1]) + 1.0)*sq(z2) - sq(v);
  };
  auto allExprs = [&](const double xs[4], double out[4]) {
    out[0] = expr(z2plus(xs[0]), z3plus(xs[0]));
    out[1] = expr(z2plus(xs[1]), z3minu(xs[1]));
    out[2] = expr(z2minu(xs[2]), z3plus(xs[2]));
    out[3] = expr(z2minu(xs[3]), z3minu(xs[3]));
  };

  auto maxZ = std::min(
      sqrt(-a1*sq(u)/(sq(b1)-a1*c1)),
      sqrt(-a2*sq(w)/(sq(b2)-a2*c1)));
  double z1 = 0.0;
  int idx = -1;
  double lowerBounds[4];
  double upperBounds[4];
  for (int i = 0; i < 4; ++i) {
    lowerBounds[i] = cameraDepth;
    upperBounds[i] = maxZ;
  }
  auto success = LinearThenSecantMethod(
      all_exprs,
      lowerBounds,
      upperBounds
      z1,
      idx
  );

  if (success) {
    auto z1 = z1;
    auto z2 = 0.0;
    if (idx == 0 || idx == 1) {
      z2 = z2plus(z1);
    } else {
      z2 = z2minu(z1);
    }
    auto z3 = 0.0;
    if (idx == 0 || idx == 2) {
      z3 = z3plus(z1);
    } else {
      z3 = z3minu(z1);
    }
    out.clear();
    out.push_back(Vector3f{alpha[0]*z1, beta[0]*z1, z1-camera_depth});
    out.push_back(Vector3f{alpha[1]*z2, beta[1]*z2, z2-camera_depth});
    out.push_back(Vector3f{alpha[2]*z3, beta[2]*z3, z3-camera_depth});

    return true;
  } else {
    return false;
  }
}
