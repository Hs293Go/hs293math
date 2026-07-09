#include <cmath>
#include <limits>
#include <vector>

#include "Eigen/Dense"
#include "gtest/gtest.h"
#include "hs293go/frame_conversions.hpp"

using hs293go::QuatAeroRobo;
using hs293go::VecEnuNed;
using hs293go::VecFluFrd;

namespace {
// A spread of unit quaternions (identity plus rotations about several axes) for
// the property-style checks below.
template <typename T>
std::vector<Eigen::Quaternion<T>> SampleUnitQuaternions() {
  std::vector<Eigen::Quaternion<T>> qs;
  qs.push_back(Eigen::Quaternion<T>::Identity());
  const Eigen::Vector3<T> axes[] = {
      Eigen::Vector3<T>::UnitX(), Eigen::Vector3<T>::UnitY(),
      Eigen::Vector3<T>::UnitZ(), Eigen::Vector3<T>(T{1}, T{2}, T{3})};
  const T angles[] = {static_cast<T>(0.3), static_cast<T>(1.1),
                      static_cast<T>(-2.0), static_cast<T>(2.7)};
  for (const auto& axis : axes) {
    for (T angle : angles) {
      qs.emplace_back(Eigen::AngleAxis<T>(angle, axis.normalized()));
    }
  }
  return qs;
}
}  // namespace

template <typename T>
class FrameConversionTest : public ::testing::Test {};

using RealTypes = ::testing::Types<float, double, long double>;
TYPED_TEST_SUITE(FrameConversionTest, RealTypes);

// ENU (East, North, Up) -> NED (North, East, Down).
TYPED_TEST(FrameConversionTest, EnuToNedMapsAxes) {
  using T = TypeParam;
  const Eigen::Vector3<T> enu(T{1}, T{2}, T{3});
  const Eigen::Vector3<T> ned = VecEnuNed(enu);
  EXPECT_EQ(ned.x(), T{2});   // North <- North
  EXPECT_EQ(ned.y(), T{1});   // East  <- East
  EXPECT_EQ(ned.z(), T{-3});  // Down  <- -Up
}

// FLU (Forward, Left, Up) -> FRD (Forward, Right, Down).
TYPED_TEST(FrameConversionTest, FluToFrdMapsAxes) {
  using T = TypeParam;
  const Eigen::Vector3<T> flu(T{1}, T{2}, T{3});
  const Eigen::Vector3<T> frd = VecFluFrd(flu);
  EXPECT_EQ(frd.x(), T{1});   // Forward unchanged
  EXPECT_EQ(frd.y(), T{-2});  // Right <- -Left
  EXPECT_EQ(frd.z(), T{-3});  // Down  <- -Up
}

// Both vector conversions are their own inverse (the frame pairs are 180 deg
// apart).
TYPED_TEST(FrameConversionTest, VectorConversionsAreInvolutions) {
  using T = TypeParam;
  const T eps = std::numeric_limits<T>::epsilon();
  const Eigen::Vector3<T> v(static_cast<T>(0.5), static_cast<T>(-1.3),
                            static_cast<T>(2.1));
  EXPECT_LE((VecEnuNed(VecEnuNed(v)) - v).norm(), T{4} * eps * v.norm());
  EXPECT_LE((VecFluFrd(VecFluFrd(v)) - v).norm(), T{4} * eps * v.norm());
}

// Applying the quaternion conversion twice returns the original quaternion.
TYPED_TEST(FrameConversionTest, QuaternionConversionIsInvolution) {
  using T = TypeParam;
  const T eps = std::numeric_limits<T>::epsilon();
  for (const auto& q : SampleUnitQuaternions<T>()) {
    const Eigen::Quaternion<T> twice = QuatAeroRobo(QuatAeroRobo(q));
    EXPECT_LE((twice.coeffs() - q.coeffs()).norm(), T{64} * eps)
        << "q = " << q.coeffs().transpose();
  }
}

// The conversion is a rotation of quaternion space, so it preserves norm.
TYPED_TEST(FrameConversionTest, QuaternionConversionPreservesNorm) {
  using T = TypeParam;
  const T eps = std::numeric_limits<T>::epsilon();
  for (const auto& q : SampleUnitQuaternions<T>()) {
    EXPECT_LE(std::abs(QuatAeroRobo(q).norm() - q.norm()), T{64} * eps)
        << "q = " << q.coeffs().transpose();
  }
}

// The deep consistency check tying all three functions together: rotating a
// body vector and then converting frames must agree whether done in the
// robotics (FLU/ENU) or aerospace (FRD/NED) convention. This commutes only if
// the vector conversions match the quaternion conversion.
//
//     q_aero . VecFluFrd(v_flu)  ==  VecEnuNed(q_robo . v_flu)
TYPED_TEST(FrameConversionTest, QuaternionAndVectorConversionsAgree) {
  using T = TypeParam;
  const T eps = std::numeric_limits<T>::epsilon();
  const Eigen::Vector3<T> v_flu(static_cast<T>(0.7), static_cast<T>(-1.2),
                                static_cast<T>(0.4));
  for (const auto& q_robo : SampleUnitQuaternions<T>()) {
    const Eigen::Quaternion<T> q_aero = QuatAeroRobo(q_robo);
    const Eigen::Vector3<T> lhs = q_aero * VecFluFrd(v_flu);
    const Eigen::Vector3<T> rhs = VecEnuNed(q_robo * v_flu);
    EXPECT_LE((lhs - rhs).norm(), T{64} * eps * v_flu.norm())
        << "q_robo = " << q_robo.coeffs().transpose();
  }
}
