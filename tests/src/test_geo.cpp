#include <cmath>
#include <limits>
#include <random>

#include "Eigen/Dense"
#include "gtest/gtest.h"
#include "hs293go/geo.hpp"

using hs293go::deg2rad;
using hs293go::dlla2enu;
using hs293go::ecef2enu;
using hs293go::enu2ecef;
using hs293go::Enu2ecefProjectionMatrix;
using hs293go::lla2ecef;
using hs293go::lla2enu;

namespace {

template <typename T>
T Eps() {
  return std::numeric_limits<T>::epsilon();
}

template <typename T>
inline constexpr T kSemiMajor = T(6378137);
template <typename T>
inline constexpr T kFlattening = T(1) / T(298.257223563);

}  // namespace

template <typename T>
class GeoTest : public ::testing::Test {};

using RealTypes = ::testing::Types<float, double, long double>;
TYPED_TEST_SUITE(GeoTest, RealTypes);

// lla2ecef at canonical points: (0,0) -> +X axis at radius a; (0,90) -> +Y;
// the poles -> +/-Z at the semi-minor axis b = a(1-f).
TYPED_TEST(GeoTest, Lla2EcefKnownPoints) {
  using T = TypeParam;
  const T a = kSemiMajor<T>;
  const T b = a * (T(1) - kFlattening<T>);
  const T tol = a * T(256) * Eps<T>();

  auto ecef = [](T lat, T lon, T alt) {
    return lla2ecef(Eigen::Vector3<T>(lat, lon, alt));
  };
  EXPECT_LE((ecef(T(0), T(0), T(0)) - Eigen::Vector3<T>(a, T(0), T(0))).norm(),
            tol);
  EXPECT_LE((ecef(T(0), T(90), T(0)) - Eigen::Vector3<T>(T(0), a, T(0))).norm(),
            tol);
  EXPECT_LE((ecef(T(90), T(0), T(0)) - Eigen::Vector3<T>(T(0), T(0), b)).norm(),
            tol);
  EXPECT_LE(
      (ecef(T(-90), T(0), T(0)) - Eigen::Vector3<T>(T(0), T(0), -b)).norm(),
      tol);
}

// At (0, 0) the local up is +X, so altitude adds straight onto X.
TYPED_TEST(GeoTest, Lla2EcefAltitudeAddsAlongNormal) {
  using T = TypeParam;
  const T a = kSemiMajor<T>;
  const T h = T(1000);
  const T tol = a * T(256) * Eps<T>();
  EXPECT_LE((lla2ecef(Eigen::Vector3<T>(T(0), T(0), h)) -
             Eigen::Vector3<T>(a + h, T(0), T(0)))
                .norm(),
            tol);
}

// The projection is a proper rotation, and its Up column is the geodetic
// normal. The Up check at lat=45 would fail if lat/lon were fed through
// rad2deg.
TYPED_TEST(GeoTest, ProjectionMatrixOrthonormalAndUp) {
  using T = TypeParam;
  const T tol = T(64) * Eps<T>();
  const Eigen::Vector3<T> refs[] = {Eigen::Vector3<T>(T(0), T(0), T(0)),
                                    Eigen::Vector3<T>(T(45), T(30), T(0)),
                                    Eigen::Vector3<T>(T(-33), T(151), T(0))};
  for (const Eigen::Vector3<T>& ref : refs) {
    const Eigen::Matrix3<T> R = Enu2ecefProjectionMatrix(ref);
    EXPECT_LE((R.transpose() * R - Eigen::Matrix3<T>::Identity()).norm(), tol)
        << "ref = " << ref.transpose();
    EXPECT_LE(std::abs(R.determinant() - T(1)), tol);
  }
  const T root_half = std::sqrt(T(1) / T(2));
  const Eigen::Matrix3<T> R45 =
      Enu2ecefProjectionMatrix(Eigen::Vector3<T>(T(45), T(0), T(0)));
  EXPECT_LE((R45.col(2) - Eigen::Vector3<T>(root_half, T(0), root_half)).norm(),
            tol);
}

// enu<->ecef are exact inverses (the rotation is orthonormal).
TYPED_TEST(GeoTest, EnuEcefRoundTrip) {
  using T = TypeParam;
  std::mt19937 prng(67);
  std::uniform_real_distribution<T> coord(T(-1000), T(1000));
  std::uniform_real_distribution<T> lat(T(-89), T(89));
  std::uniform_real_distribution<T> lon(T(-180), T(180));
  for (int i = 0; i < 1000; ++i) {
    const Eigen::Vector3<T> ref(lat(prng), lon(prng), T(0));
    const Eigen::Vector3<T> v(coord(prng), coord(prng), coord(prng));
    const T tol = v.norm() * T(64) * Eps<T>();
    ASSERT_LE((enu2ecef(ecef2enu(v, ref), ref) - v).norm(), tol);
    ASSERT_LE((ecef2enu(enu2ecef(v, ref), ref) - v).norm(), tol);
  }
}

// lla2enu: the reference point maps to the origin; offsets land on the expected
// ENU axes with the right sign.
TYPED_TEST(GeoTest, Lla2EnuOriginAndDirections) {
  using T = TypeParam;
  const Eigen::Vector3<T> ref(T(45), T(30), T(0));

  // Reference point -> origin (exact: identical ECEF subtracted).
  EXPECT_LE(lla2enu(ref, ref).norm(), T(8) * Eps<T>() * kSemiMajor<T>);

  // Pure altitude -> pure Up of the same magnitude. The error floor is set by
  // the cancellation of two ~R_earth ECEF coordinates inside lla2enu, so the
  // tolerance scales with the Earth radius, not with h.
  const T h = T(1234);
  const Eigen::Vector3<T> up = lla2enu(Eigen::Vector3<T>(T(45), T(30), h), ref);
  EXPECT_LE((up - Eigen::Vector3<T>(T(0), T(0), h)).norm(),
            T(64) * Eps<T>() * kSemiMajor<T>);

  // A small step north -> +North with negligible East; east -> +East with
  // negligible North. (Cross-axis tolerance is loose to absorb the float-level
  // cancellation of two ~6.4e6 m ECEF coordinates.)
  const T d = T(0.01);  // degrees
  const Eigen::Vector3<T> north =
      lla2enu(Eigen::Vector3<T>(T(45) + d, T(30), T(0)), ref);
  EXPECT_GT(north.y(), T(0));
  EXPECT_LT(std::abs(north.x()), std::abs(north.y()) * T(1e-2));

  const Eigen::Vector3<T> east =
      lla2enu(Eigen::Vector3<T>(T(45), T(30) + d, T(0)), ref);
  EXPECT_GT(east.x(), T(0));
  EXPECT_LT(std::abs(east.y()), std::abs(east.x()) * T(1e-2));
}

// The differential formulation maps a pure altitude offset to (0, 0, d_alt)
// *exactly* -- the headline win: no full-magnitude ECEF cancellation, so this
// holds bit-for-bit even in float.
TYPED_TEST(GeoTest, DifferentialPureAltitudeIsExact) {
  using T = TypeParam;
  const Eigen::Vector3<T> ref(T(45), T(30), T(100));
  const T h = T(1234);
  const Eigen::Vector3<T> up = dlla2enu(Eigen::Vector3<T>(T(0), T(0), h), ref);
  EXPECT_EQ(up.x(), T(0));
  EXPECT_EQ(up.y(), T(0));
  EXPECT_EQ(up.z(), h);
}

// Axis scaling against the independently-known equatorial radii: at lat 0,
// N = a and M = a(1-e^2). A pure offset lands purely on the matching ENU axis.
TYPED_TEST(GeoTest, DifferentialEquatorialRadii) {
  using T = TypeParam;
  const Eigen::Vector3<T> ref(T(0), T(0), T(0));
  const T a = kSemiMajor<T>;
  const T e2 = (T(2) - kFlattening<T>)*kFlattening<T>;
  const T delta = T(0.01);
  const T drad = deg2rad(delta);
  const T tol = T(16) * Eps<T>() * a;

  const Eigen::Vector3<T> north =
      dlla2enu(Eigen::Vector3<T>(delta, T(0), T(0)), ref);
  EXPECT_LE(std::abs(north.y() - a * (T(1) - e2) * drad),
            tol);  // North = M*drad
  EXPECT_EQ(north.x(), T(0));
  EXPECT_EQ(north.z(), T(0));

  const Eigen::Vector3<T> east =
      dlla2enu(Eigen::Vector3<T>(T(0), delta, T(0)), ref);
  EXPECT_LE(std::abs(east.x() - a * drad), tol);  // East = N*drad
  EXPECT_EQ(east.y(), T(0));
  EXPECT_EQ(east.z(), T(0));
}

// For small offsets the differential model agrees with the exact lla2enu up to
// the curvature sagitta O(horiz^2 / R) (plus lla2enu's own cancellation floor).
TYPED_TEST(GeoTest, DifferentialMatchesLla2EnuForSmallOffsets) {
  using T = TypeParam;
  const Eigen::Vector3<T> ref(T(45), T(30), T(100));
  const Eigen::Vector3<T> d(T(0.001), T(0.0007), T(30));
  const Eigen::Vector3<T> enu_diff = dlla2enu(d, ref);
  const Eigen::Vector3<T> enu_exact = lla2enu(Eigen::Vector3<T>(ref + d), ref);
  const T horiz = std::hypot(enu_diff.x(), enu_diff.y());
  const T tol =
      T(8) * horiz * horiz / kSemiMajor<T> + T(256) * Eps<T>() * kSemiMajor<T>;
  EXPECT_LE((enu_diff - enu_exact).norm(), tol);
}
