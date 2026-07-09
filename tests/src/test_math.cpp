#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>

#include "gtest/gtest.h"
#include "hs293go/math.hpp"

using hs293go::deg2rad;
using hs293go::IsClose;
using hs293go::rad2deg;
using hs293go::Tolerances;
using hs293go::wrapTo180;
using hs293go::wrapTo2Pi;
using hs293go::wrapTo360;
using hs293go::wrapToPi;

// IsClose is exercised against its actual contract
//
//   IsClose(a, b) == (a == b) ||
//                    |a - b| < max(abs_tol, rel_tol * min(|a| + |b|, MAX))
//
// with defaults rel_tol = 4 * eps_T and abs_tol = 0. Every property below is
// stated in terms of the type's own epsilon()/max() rather than hardcoded
// magnitudes, and run for float, double and long double via TYPED_TEST.
//
// Note: braced-init of a non-dyadic literal (e.g. T{1e-7}) is a narrowing error
// for float, so fractional literals are written static_cast<T>(...).

template <typename T>
class IsCloseTest : public ::testing::Test {};

using RealTypes = ::testing::Types<float, double, long double>;
TYPED_TEST_SUITE(IsCloseTest, RealTypes);

// A value is always close to itself, regardless of magnitude (short-circuits on
// a == b, so this holds for every representable finite value).
TYPED_TEST(IsCloseTest, EqualValuesAreClose) {
  using T = TypeParam;
  const T values[] = {T{0},
                      T{1},
                      T{-1},
                      static_cast<T>(1e10),
                      static_cast<T>(-1e10),
                      static_cast<T>(1e-10),
                      std::numeric_limits<T>::max(),
                      std::numeric_limits<T>::min()};
  for (T x : values) {
    EXPECT_TRUE(IsClose(x, x)) << "x = " << x;
  }
}

// +0 and -0 compare equal, hence close.
TYPED_TEST(IsCloseTest, SignedZerosAreClose) {
  using T = TypeParam;
  EXPECT_TRUE(IsClose(T{0}, -T{0}));
  EXPECT_TRUE(IsClose(-T{0}, T{0}));
}

// The relation is symmetric in its arguments (|a-b| and |a|+|b| both are).
TYPED_TEST(IsCloseTest, IsSymmetric) {
  using T = TypeParam;
  const T eps = std::numeric_limits<T>::epsilon();
  const std::pair<T, T> pairs[] = {
      {T{1}, T{1} + eps},
      {T{1}, T{2}},
      {T{1000}, T{1001}},
      {T{0}, T{1}},
  };
  for (auto [a, b] : pairs) {
    EXPECT_EQ(IsClose(a, b), IsClose(b, a)) << "a = " << a << ", b = " << b;
  }
}

// Adjacent representable values (1 ULP apart) are close at any magnitude: the
// gap is ~eps*|x| while the default threshold is ~8*eps*|x|. This is the
// scale-invariance property of the relative tolerance.
TYPED_TEST(IsCloseTest, AdjacentFloatsAreClose) {
  using T = TypeParam;
  const T inf = std::numeric_limits<T>::infinity();
  const T values[] = {T{1}, static_cast<T>(1000), static_cast<T>(1e-3),
                      static_cast<T>(-7.5), static_cast<T>(12345.678)};
  for (T x : values) {
    EXPECT_TRUE(IsClose(x, std::nextafter(x, inf))) << "x = " << x;
    EXPECT_TRUE(IsClose(x, std::nextafter(x, -inf))) << "x = " << x;
  }
}

// Values a fixed *relative* distance apart (0.1%) are never close under the
// default tolerance, again independent of magnitude or sign.
TYPED_TEST(IsCloseTest, DistinctValuesAreNotClose) {
  using T = TypeParam;
  const T values[] = {T{1}, static_cast<T>(1000), static_cast<T>(1e-3),
                      static_cast<T>(-7.5)};
  for (T x : values) {
    EXPECT_FALSE(IsClose(x, x * static_cast<T>(1.001))) << "x = " << x;
  }
}

// Pin the default rel_tol = 4*eps: with |a|+|b| ~ 2, the half-width is ~8*eps.
// 4*eps stays inside (close); 64*eps is well outside (not close).
TYPED_TEST(IsCloseTest, RelativeToleranceBoundary) {
  using T = TypeParam;
  const T eps = std::numeric_limits<T>::epsilon();
  EXPECT_TRUE(IsClose(T{1}, T{1} + T{4} * eps));
  EXPECT_FALSE(IsClose(T{1}, T{1} + T{64} * eps));
}

// abs_tol governs closeness to zero, where the relative term collapses.
TYPED_TEST(IsCloseTest, AbsoluteToleranceNearZero) {
  using T = TypeParam;
  // Default abs_tol == 0: no nonzero value is close to zero, however tiny.
  EXPECT_FALSE(IsClose(std::numeric_limits<T>::min(), T{0}));
  EXPECT_FALSE(IsClose(static_cast<T>(1e-20), T{0}));

  // With abs_tol set (and rel_tol 0 to isolate it), closeness to zero is a
  // pure magnitude test against abs_tol.
  const Tolerances<T> tol{.rel_tol = T{0}, .abs_tol = static_cast<T>(1e-6)};
  EXPECT_TRUE(IsClose(static_cast<T>(1e-7), T{0}, tol));
  EXPECT_TRUE(IsClose(T{0}, static_cast<T>(-1e-7), tol));
  EXPECT_FALSE(IsClose(static_cast<T>(1e-5), T{0}, tol));
  EXPECT_FALSE(IsClose(static_cast<T>(-1e-5), T{0}, tol));
}

// The threshold is the max of the absolute and relative terms: whichever is
// larger decides.
TYPED_TEST(IsCloseTest, UsesLargerOfAbsoluteAndRelative) {
  using T = TypeParam;
  const T eps = std::numeric_limits<T>::epsilon();

  // Near 1 the relative term (~8*eps) dominates a zero abs_tol.
  const Tolerances<T> rel_dominates{.rel_tol = T{4} * eps, .abs_tol = T{0}};
  EXPECT_TRUE(IsClose(T{1}, T{1} + T{4} * eps, rel_dominates));

  // A generous abs_tol rescues a pair the relative term alone would reject.
  const Tolerances<T> abs_dominates{.rel_tol = T{4} * eps,
                                    .abs_tol = static_cast<T>(0.5)};
  EXPECT_TRUE(IsClose(T{1}, static_cast<T>(1.4), abs_dominates));
  EXPECT_FALSE(IsClose(T{1}, static_cast<T>(1.6), abs_dominates));
}

// Regression: |a| + |b| must not overflow to inf and inflate the threshold.
TYPED_TEST(IsCloseTest, ExtremeValuesDoNotOverflow) {
  using T = TypeParam;
  const T max = std::numeric_limits<T>::max();
  EXPECT_TRUE(IsClose(max, max));
  EXPECT_FALSE(IsClose(max, max / T{2}));
  EXPECT_FALSE(IsClose(max, -max));
  EXPECT_FALSE(IsClose(-max, max / T{2}));
}

// Equal infinities are close (via a == b); anything else involving infinity is
// not.
TYPED_TEST(IsCloseTest, Infinities) {
  using T = TypeParam;
  const T inf = std::numeric_limits<T>::infinity();
  const T max = std::numeric_limits<T>::max();
  EXPECT_TRUE(IsClose(inf, inf));
  EXPECT_TRUE(IsClose(-inf, -inf));
  EXPECT_FALSE(IsClose(inf, -inf));
  EXPECT_FALSE(IsClose(inf, max));
  EXPECT_FALSE(IsClose(-inf, -max));
}

// NaN is never close to anything, including itself.
TYPED_TEST(IsCloseTest, NaNIsNeverClose) {
  using T = TypeParam;
  const T nan = std::numeric_limits<T>::quiet_NaN();
  const T inf = std::numeric_limits<T>::infinity();
  EXPECT_FALSE(IsClose(nan, nan));
  EXPECT_FALSE(IsClose(nan, T{0}));
  EXPECT_FALSE(IsClose(T{0}, nan));
  EXPECT_FALSE(IsClose(nan, T{1}));
  EXPECT_FALSE(IsClose(nan, inf));
  EXPECT_FALSE(IsClose(nan, std::numeric_limits<T>::max()));
}

// ---------------------------------------------------------------------------
// Angle wrapping and degree/radian conversion.
// ---------------------------------------------------------------------------

template <typename T>
class WrapTest : public ::testing::Test {};

TYPED_TEST_SUITE(WrapTest, RealTypes);

// Every wrap maps any finite input into its documented interval: [-pi, pi] and
// [-180, 180] are closed; [0, 2*pi) and [0, 360) are half-open at the top.
TYPED_TEST(WrapTest, OutputsLieInRange) {
  using T = TypeParam;
  const T pi = std::numbers::pi_v<T>;
  const T two_pi = T{2} * pi;

  auto check = [&](T a) {
    const T p = wrapToPi(a);
    EXPECT_GE(p, -pi) << "wrapToPi, a = " << a;
    EXPECT_LE(p, pi) << "wrapToPi, a = " << a;

    const T tp = wrapTo2Pi(a);
    EXPECT_GE(tp, T{0}) << "wrapTo2Pi, a = " << a;
    EXPECT_LT(tp, two_pi) << "wrapTo2Pi, a = " << a;

    const T d = wrapTo180(a);
    EXPECT_GE(d, T{-180}) << "wrapTo180, a = " << a;
    EXPECT_LE(d, T{180}) << "wrapTo180, a = " << a;

    const T td = wrapTo360(a);
    EXPECT_GE(td, T{0}) << "wrapTo360, a = " << a;
    EXPECT_LT(td, T{360}) << "wrapTo360, a = " << a;
  };

  for (int i = -500; i <= 500; ++i) {
    check(static_cast<T>(i) * static_cast<T>(0.37));
  }
  const T extras[] = {T{0},
                      -T{0},
                      pi,
                      -pi,
                      two_pi,
                      -two_pi,
                      static_cast<T>(1e6),
                      static_cast<T>(-1e6),
                      static_cast<T>(1e7),
                      static_cast<T>(1e-3)};
  for (T a : extras) {
    check(a);
  }
}

// wrapToPi / wrapTo2Pi differ from the input by an integer number of full
// turns (2*pi) -- i.e. they preserve the angle modulo 2*pi.
TYPED_TEST(WrapTest, RadianWrapsPreserveAngleModTwoPi) {
  using T = TypeParam;
  const T eps = std::numeric_limits<T>::epsilon();
  const T two_pi = T{2} * std::numbers::pi_v<T>;
  for (int i = -50; i <= 50; ++i) {
    const T a = static_cast<T>(i) * static_cast<T>(0.37);
    const T tol = std::max(T{1}, std::abs(a)) * T{64} * eps;

    const T p = wrapToPi(a);
    const T kp = std::round((a - p) / two_pi);
    EXPECT_LE(std::abs(p + kp * two_pi - a), tol) << "wrapToPi, a = " << a;

    const T q = wrapTo2Pi(a);
    const T kq = std::round((a - q) / two_pi);
    EXPECT_LE(std::abs(q + kq * two_pi - a), tol) << "wrapTo2Pi, a = " << a;
  }
}

// wrapTo180 / wrapTo360 preserve the angle modulo 360.
TYPED_TEST(WrapTest, DegreeWrapsPreserveAngleMod360) {
  using T = TypeParam;
  const T eps = std::numeric_limits<T>::epsilon();
  for (int i = -50; i <= 50; ++i) {
    const T a = static_cast<T>(i) * static_cast<T>(13.7);
    const T tol = std::max(T{1}, std::abs(a)) * T{64} * eps;

    const T d = wrapTo180(a);
    const T kd = std::round((a - d) / T{360});
    EXPECT_LE(std::abs(d + kd * T{360} - a), tol) << "wrapTo180, a = " << a;

    const T e = wrapTo360(a);
    const T ke = std::round((a - e) / T{360});
    EXPECT_LE(std::abs(e + ke * T{360} - a), tol) << "wrapTo360, a = " << a;
  }
}

// Values already inside the interval come back unchanged.
TYPED_TEST(WrapTest, IdentityInsideInterval) {
  using T = TypeParam;
  const T eps = std::numeric_limits<T>::epsilon();

  // The half-open wraps are exact identities on their interval (fmod is exact
  // when no reduction happens).
  for (T x :
       {T{0}, static_cast<T>(0.5), static_cast<T>(3), static_cast<T>(6)}) {
    EXPECT_EQ(wrapTo2Pi(x), x) << "wrapTo2Pi, x = " << x;
  }
  for (T x :
       {T{0}, static_cast<T>(0.5), static_cast<T>(180), static_cast<T>(359)}) {
    EXPECT_EQ(wrapTo360(x), x) << "wrapTo360, x = " << x;
  }

  // The closed wraps reconstruct the value up to rounding.
  for (T x : {static_cast<T>(-3), static_cast<T>(-1), T{0}, static_cast<T>(1),
              static_cast<T>(3)}) {
    EXPECT_LE(std::abs(wrapToPi(x) - x), T{32} * eps) << "wrapToPi, x = " << x;
  }
  for (T x : {static_cast<T>(-179), static_cast<T>(-90), T{0},
              static_cast<T>(90), static_cast<T>(179)}) {
    EXPECT_LE(std::abs(wrapTo180(x) - x),
              std::max(T{1}, std::abs(x)) * T{32} * eps)
        << "wrapTo180, x = " << x;
  }
}

// Boundary and seam behaviour, including the half-open snap at the period.
TYPED_TEST(WrapTest, BoundaryValues) {
  using T = TypeParam;
  const T pi = std::numbers::pi_v<T>;
  const T two_pi = T{2} * pi;

  EXPECT_EQ(wrapToPi(T{0}), T{0});
  EXPECT_EQ(wrapTo2Pi(T{0}), T{0});
  EXPECT_EQ(wrapTo180(T{0}), T{0});
  EXPECT_EQ(wrapTo360(T{0}), T{0});

  // Exactly one full turn folds back to 0 (the half-open snap).
  EXPECT_EQ(wrapTo2Pi(two_pi), T{0});
  EXPECT_EQ(wrapTo360(T{360}), T{0});

  // The closed wraps map the lower endpoint onto the upper one.
  EXPECT_EQ(wrapToPi(-pi), pi);
  EXPECT_EQ(wrapTo180(T{-180}), T{180});

  // A negative input just below a full turn lands near the top of [0, period).
  EXPECT_GT(wrapTo2Pi(static_cast<T>(-0.1)), pi);
  EXPECT_LT(wrapTo2Pi(static_cast<T>(-0.1)), two_pi);
  EXPECT_GT(wrapTo360(static_cast<T>(-0.1)), T{180});
  EXPECT_LT(wrapTo360(static_cast<T>(-0.1)), T{360});
}

// deg2rad and rad2deg are mutual inverses up to rounding.
TYPED_TEST(WrapTest, DegreeRadianRoundTrip) {
  using T = TypeParam;
  const T eps = std::numeric_limits<T>::epsilon();
  for (int i = -36; i <= 36; ++i) {
    const T deg = static_cast<T>(i) * T{10};
    EXPECT_LE(std::abs(rad2deg(deg2rad(deg)) - deg),
              std::max(T{1}, std::abs(deg)) * T{16} * eps)
        << "deg = " << deg;
  }
  for (int i = -10; i <= 10; ++i) {
    const T rad = static_cast<T>(i) * static_cast<T>(0.3);
    EXPECT_LE(std::abs(deg2rad(rad2deg(rad)) - rad),
              std::max(T{1}, std::abs(rad)) * T{16} * eps)
        << "rad = " << rad;
  }
}

// Spot-check known conversions.
TYPED_TEST(WrapTest, KnownConversions) {
  using T = TypeParam;
  const T eps = std::numeric_limits<T>::epsilon();
  const T pi = std::numbers::pi_v<T>;
  EXPECT_LE(std::abs(deg2rad(T{180}) - pi), T{8} * eps * pi);
  EXPECT_LE(std::abs(deg2rad(T{90}) - pi / T{2}), T{8} * eps * pi);
  EXPECT_LE(std::abs(rad2deg(pi) - T{180}), T{8} * eps * T{180});
  EXPECT_LE(std::abs(rad2deg(pi / T{2}) - T{90}), T{8} * eps * T{180});
}
