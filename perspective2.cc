#include "perspective2.h"

#include <algorithm>
#include <array>
#include <limits>
#include <vector>

#include <cmath>

const double SOLUTION_TOL = 0.0001;
const int MAX_ITERATIONS = 100;

template <typename F> bool QuadSecantMethod(
    F f, std::array<double, 4> &xs0, std::array<double, 4> &xs1, double &result, int& idx) {
  auto ys0 = std::array<double, 4>();
  auto ys1 = std::array<double, 4>();
  f(xs0, ys0);
  f(xs1, ys1);

  auto iterationCount = 0;
  auto notsolved = [&]() -> bool {
    for (int i = 0; i < 4; ++i) {
      if (!std::isnan(ys1[i]) && std::abs(ys1[i]) <= SOLUTION_TOL) {
        return false;
      }
    }
    return true;
  };
  while (notsolved() && iterationCount < MAX_ITERATIONS) {
    auto xNext = std::array <double, 4>();
    for (int i = 0; i < 4; ++i) {
      xNext[i] = xs1[i] - 1.0 * ys1[i] * (xs1[i] - xs0[i])/(ys1[i] - ys0[i]);
    }

    std::copy(xs1.begin(), xs1.end(), xs0.begin());
    std::copy(xNext.begin(), xNext.end(), xs1.begin());

    std::copy(ys1.begin(), ys1.end(), ys0.begin());
    f(xs1, ys1);
    ++iterationCount;
  }

  for (int i = 0; i < 4; ++i) {
    if (std::abs(ys1[i]) <= SOLUTION_TOL) {
      idx = i;
      result = xs1[idx];
      return true;
    }
  }
  return false;
}

template <typename F> bool LinearThenSecantMethod(
    F f, const std::array<double, 4>& lower, const std::array<double, 4>& upper,
    double &result, int& idx) {
  const int NUM_SEARCH = 16;
  auto closestPos = std::array<double, 4>();
  auto closestVal = std::array<double, 4>();
  for (int i = 0; i < 4; ++i) {
    closestPos[i] = 0.0;
    closestVal[i] = std::numeric_limits<double>::max();
  }

  for (int i = 0; i < NUM_SEARCH; ++i) {
    auto pos = std::array<double, 4>();
    for (int j = 0; j < 4; ++j) {
      pos[j] = lower[j] + i*(upper[j] - lower[j])/NUM_SEARCH;
    }
    auto result = std::array<double, 4>();
    f(pos, result);
    for (int j = 0; j < 4; ++j) {
      if (std::abs(result[j]) < closestVal[j]) {
        closestVal[j] = std::abs(result[j]);
        closestPos[j] = pos[j];
      }
    }
  }

  auto closestPos2 = std::array<double, 4>();
  for (int i = 0; i < 4; ++i) {
    closestPos2[i] = closestPos[i] + (upper[i] - lower[i])/(2*NUM_SEARCH);
  }

  return QuadSecantMethod(f, closestPos, closestPos2, result, idx);
}

bool FindCoordinates(const std::vector<cv::Point2f> &vpp,
    const Spacing &rps, double cameraDepth, double cameraScale, Coordinates& out) {

  auto alpha = std::array<double, 3>();
  auto beta = std::array<double, 3>();
  for (int i = 0; i < 3; ++i) {
    alpha[i] = cameraScale*(vpp[i].x)/cameraDepth;
    beta[i] = cameraScale*(vpp[i].y)/cameraDepth;
  }

  auto sq = [](double x) -> double { return x*x; };
  auto dist = [&](const cv::Point2f& a, const cv::Point2f &b) -> double {
    return std::sqrt(sq(a.x - b.x) + sq(a.y - b.y));
  };

  auto u = dist(rps[0], rps[1]);
  auto v = dist(rps[1], rps[2]);
  auto w = dist(rps[0], rps[2]);

  auto a1 = sq(alpha[1]) + sq(beta[1]) + 1.0;
  auto b1 = alpha[0]*alpha[1] + beta[0]*beta[1] + 1.0;
  auto c1 = sq(alpha[0]) + sq(beta[0]) + 1.0;

  auto z2discr = [&](double z1) -> double {
    return std::sqrt((sq(b1) - a1*c1)*sq(z1) + a1*sq(u));
  };
  auto z2plus = [&](double z1) -> double {
    return (z1*b1 + z2discr(z1))/a1;
  };
  auto z2minu = [&](double z1) -> double{
    return (z1*b1 - z2discr(z1))/a1;
  };

  auto a2 = sq(alpha[2]) + sq(beta[2]) + 1.0;
  auto b2 = alpha[0]*alpha[2] + beta[0]*beta[2] + 1.0;

  auto z3discr = [&](double z1) -> double {
    return std::sqrt((sq(b2) - a2*c1)*sq(z1) + a2*sq(w));
  };
  auto z3plus = [&](double z1) -> double {
    return (z1*b2 + z3discr(z1))/a2;
  };
  auto z3minu = [&](double z1) -> double {
    return (z1*b2 - z3discr(z1))/a2;
  };

  auto expr = [&](double z2, double z3) -> double {
    return (sq(alpha[2]) + sq(beta[2]) + 1.0)*sq(z3) 
        - 2.0*(alpha[1]*alpha[2] + beta[1]*beta[2] + 1.0)*z2*z3
        + (sq(alpha[1]) + sq(beta[1]) + 1.0)*sq(z2) - sq(v);
  };
  auto allExprs = [&](const std::array<double, 4>& xs, std::array<double, 4>& out) {
    out[0] = expr(z2plus(xs[0]), z3plus(xs[0]));
    out[1] = expr(z2plus(xs[1]), z3minu(xs[1]));
    out[2] = expr(z2minu(xs[2]), z3plus(xs[2]));
    out[3] = expr(z2minu(xs[3]), z3minu(xs[3]));
  };

  auto maxZ = std::min(
      std::sqrt(-a1*sq(u)/(sq(b1)-a1*c1)),
      std::sqrt(-a2*sq(w)/(sq(b2)-a2*c1)));
  double z1 = 0.0;
  int idx = -1;
  auto lowerBounds = std::array<double, 4>();
  auto upperBounds = std::array<double, 4>();
  for (int i = 0; i < 4; ++i) {
    lowerBounds[i] = cameraDepth;
    upperBounds[i] = maxZ;
  }
  auto success = LinearThenSecantMethod(
      allExprs,
      lowerBounds,
      upperBounds,
      z1,
      idx
  );

  if (success) {
    // auto z1 = z1;
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
    out.push_back(Vector3f{alpha[0]*z1, beta[0]*z1, z1-cameraDepth});
    out.push_back(Vector3f{alpha[1]*z2, beta[1]*z2, z2-cameraDepth});
    out.push_back(Vector3f{alpha[2]*z3, beta[2]*z3, z3-cameraDepth});

    return true;
  } else {
    return false;
  }
}
