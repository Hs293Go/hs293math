#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <ostream>
#include <random>
#include <type_traits>

#include "Eigen/Dense"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "hs293go/rotation.hpp"

using hs293go::AngleAxisToQuaternion;
using hs293go::AngleAxisToRotationMatrix;
using hs293go::Axis;
using hs293go::CanonicalizePositiveW;
using hs293go::EulerAngles;
using hs293go::EulerAnglesToQuaternion;
using hs293go::IsSignFlipped;
using hs293go::MatchSign;
using hs293go::Negated;
using hs293go::PrincipalRotationMatrix;
using hs293go::QuaternionToAngleAxis;
using hs293go::QuaternionToEulerAngles;
using hs293go::RotationMatrixToAngleAxis;
using hs293go::RotationMatrixX;
using hs293go::RotationMatrixY;
using hs293go::RotationMatrixZ;

// Teach gmock to print quaternions (Eigen has no operator<< for them, so they
// would otherwise be dumped as raw bytes in failure messages).
namespace Eigen {
template <typename T>
void PrintTo(const Quaternion<T>& q, std::ostream* os) {
  *os << "Quaternion(w=" << q.w() << ", x=" << q.x() << ", y=" << q.y()
      << ", z=" << q.z() << ")";
}
}  // namespace Eigen

namespace {

template <typename T>
inline constexpr T kPi = std::numbers::pi_v<T>;

template <typename T>
T Eps() {
  return std::numeric_limits<T>::epsilon();
}

constexpr int kNumTrials = 1000;

// --- gmock matchers, modeled on the Ceres rotation_test.cc matchers ---------
//
// All are polymorphic over the Eigen scalar type (float/double/long double);
// the scalar is recovered from the matched value. Tolerances are passed as a
// multiple of that type's machine epsilon, e.g. IsNearQuaternion(expected, 64)
// allows a deviation of 64 * epsilon.

MATCHER(IsNormalizedQuaternion, "is a unit quaternion") {
  using Scalar = typename std::decay_t<decltype(arg)>::Scalar;
  const Scalar err = std::abs(arg.squaredNorm() - Scalar(1));
  const Scalar tol = Scalar(16) * std::numeric_limits<Scalar>::epsilon();
  if (err <= tol) {
    return true;
  }
  *result_listener << "squared norm differs from 1 by " << err;
  return false;
}

// Quaternions are equal up to an overall sign change.
MATCHER_P2(IsNearQuaternion, expected, eps_scale, "") {
  using Scalar = typename std::decay_t<decltype(arg)>::Scalar;
  const Scalar tol = Scalar(eps_scale) * std::numeric_limits<Scalar>::epsilon();
  const Scalar d = std::min((arg.coeffs() - expected.coeffs()).norm(),
                            (arg.coeffs() + expected.coeffs()).norm());
  if (d <= tol) {
    return true;
  }
  *result_listener << "coefficients differ by " << d << " (tol " << tol << ")";
  return false;
}

// Rotation vectors are compared relative to the magnitude; near pi the axis
// sign is ambiguous, so the smaller of the two signed differences is taken.
MATCHER_P2(IsNearAngleAxis, expected, eps_scale, "") {
  using Scalar = typename std::decay_t<decltype(arg)>::Scalar;
  const Scalar eps = std::numeric_limits<Scalar>::epsilon();
  const Scalar rel_tol = Scalar(eps_scale) * eps;
  const Scalar e_norm = expected.norm();
  Scalar delta;
  if (e_norm > Scalar(0)) {
    const Scalar near_pi_band = Scalar(1024) * eps * std::numbers::pi_v<Scalar>;
    if (std::abs(e_norm - std::numbers::pi_v<Scalar>) < near_pi_band) {
      delta =
          std::min((arg - expected).norm(), (arg + expected).norm()) / e_norm;
    } else {
      delta = (arg - expected).norm() / e_norm;
    }
  } else {
    delta = arg.norm();
  }
  if (delta <= rel_tol) {
    return true;
  }
  *result_listener << "relative error " << delta << " > " << rel_tol;
  return false;
}

MATCHER(IsOrthonormal, "is a proper rotation matrix") {
  using Scalar = typename std::decay_t<decltype(arg)>::Scalar;
  const Scalar tol = Scalar(64) * std::numeric_limits<Scalar>::epsilon();
  const Scalar ortho =
      (arg.transpose() * arg - Eigen::Matrix3<Scalar>::Identity()).norm();
  if (ortho > tol) {
    *result_listener << "||R^T R - I|| = " << ortho;
    return false;
  }
  const Scalar det = std::abs(arg.determinant() - Scalar(1));
  if (det > tol) {
    *result_listener << "det(R) differs from 1 by " << det;
    return false;
  }
  return true;
}

MATCHER_P2(IsNear3x3Matrix, expected, eps_scale, "") {
  using Scalar = typename std::decay_t<decltype(arg)>::Scalar;
  const Scalar tol = Scalar(eps_scale) * std::numeric_limits<Scalar>::epsilon();
  const Scalar d = (arg - expected).norm();
  if (d <= tol) {
    return true;
  }
  *result_listener << "differs by " << d << " (tol " << tol << ")";
  return false;
}

// --- Random sampling helpers ------------------------------------------------

template <typename T>
Eigen::Vector3<T> RandomUnitAxis(std::mt19937& prng) {
  std::uniform_real_distribution<T> uniform(T(-1), T(1));
  Eigen::Vector3<T> axis;
  do {
    axis = Eigen::Vector3<T>(uniform(prng), uniform(prng), uniform(prng));
  } while (axis.norm() < T(1) / T(8));
  return axis.normalized();
}

template <typename T>
T RandomInRange(std::mt19937& prng, T lo, T hi) {
  return std::uniform_real_distribution<T>(lo, hi)(prng);
}

template <typename T>
Eigen::Quaternion<T> RandomUnitQuaternion(std::mt19937& prng) {
  std::uniform_real_distribution<T> uniform(T(-1), T(1));
  Eigen::Quaternion<T> q;
  do {
    q.coeffs() = Eigen::Vector4<T>(uniform(prng), uniform(prng), uniform(prng),
                                   uniform(prng));
  } while (q.coeffs().norm() < T(1) / T(8));
  q.normalize();
  return q;
}

}  // namespace

template <typename T>
class RotationTest : public ::testing::Test {};

using RealTypes = ::testing::Types<float, double, long double>;
TYPED_TEST_SUITE(RotationTest, RealTypes);

// --- AngleAxis -> Quaternion -----------------------------------------------

TYPED_TEST(RotationTest, ZeroAngleAxisToQuaternion) {
  using T = TypeParam;
  const Eigen::Quaternion<T> q =
      AngleAxisToQuaternion(Eigen::Vector3<T>(T(0), T(0), T(0)));
  EXPECT_THAT(q, IsNormalizedQuaternion());
  EXPECT_THAT(q, IsNearQuaternion(Eigen::Quaternion<T>::Identity(), 16));
}

TYPED_TEST(RotationTest, SmallAngleAxisToQuaternion) {
  using T = TypeParam;
  const T theta = static_cast<T>(1e-2);
  const Eigen::Quaternion<T> q =
      AngleAxisToQuaternion(Eigen::Vector3<T>(theta, T(0), T(0)));
  const Eigen::Quaternion<T> expected(std::cos(theta / 2), std::sin(theta / 2),
                                      T(0), T(0));
  EXPECT_THAT(q, IsNormalizedQuaternion());
  EXPECT_THAT(q, IsNearQuaternion(expected, 16));
}

TYPED_TEST(RotationTest, TinyAngleAxisToQuaternion) {
  using T = TypeParam;
  // Tiny but finite value far from the FP_ZERO branch.
  const T theta = std::pow(std::numeric_limits<T>::min(), T(0.75));
  const Eigen::Quaternion<T> q =
      AngleAxisToQuaternion(Eigen::Vector3<T>(theta, T(0), T(0)));
  const Eigen::Quaternion<T> expected(std::cos(theta / 2), std::sin(theta / 2),
                                      T(0), T(0));
  EXPECT_THAT(q, IsNormalizedQuaternion());
  EXPECT_THAT(q, IsNearQuaternion(expected, 16));
}

TYPED_TEST(RotationTest, XRotationToQuaternion) {
  using T = TypeParam;
  const T half_sqrt2 = std::numbers::sqrt2_v<T> / 2;
  const Eigen::Quaternion<T> q =
      AngleAxisToQuaternion(Eigen::Vector3<T>(kPi<T> / 2, T(0), T(0)));
  const Eigen::Quaternion<T> expected(half_sqrt2, half_sqrt2, T(0), T(0));
  EXPECT_THAT(q, IsNormalizedQuaternion());
  EXPECT_THAT(q, IsNearQuaternion(expected, 16));
}

// Independent oracle: Eigen's own AngleAxis -> Quaternion.
TYPED_TEST(RotationTest, AngleAxisToQuaternionMatchesEigen) {
  using T = TypeParam;
  std::mt19937 prng(5);
  for (int i = 0; i < kNumTrials; ++i) {
    const Eigen::Vector3<T> axis = RandomUnitAxis<T>(prng);
    const T theta = RandomInRange<T>(prng, -kPi<T>, kPi<T>);
    const Eigen::Quaternion<T> ours =
        AngleAxisToQuaternion(Eigen::Vector3<T>(axis * theta));
    const Eigen::Quaternion<T> oracle(Eigen::AngleAxis<T>(theta, axis));
    ASSERT_THAT(ours, IsNearQuaternion(oracle, 32));
  }
}

// --- Quaternion -> AngleAxis -----------------------------------------------

TYPED_TEST(RotationTest, UnitQuaternionToAngleAxis) {
  using T = TypeParam;
  const Eigen::Vector3<T> aa =
      QuaternionToAngleAxis(Eigen::Quaternion<T>::Identity());
  EXPECT_THAT(aa, IsNearAngleAxis(Eigen::Vector3<T>(T(0), T(0), T(0)), 16));
}

TYPED_TEST(RotationTest, YRotationQuaternionToAngleAxis) {
  using T = TypeParam;
  // w = 0, axis = Y  ==>  rotation by pi about Y.
  const Eigen::Vector3<T> aa =
      QuaternionToAngleAxis(Eigen::Quaternion<T>(T(0), T(0), T(1), T(0)));
  EXPECT_THAT(aa, IsNearAngleAxis(Eigen::Vector3<T>(T(0), kPi<T>, T(0)), 64));
}

TYPED_TEST(RotationTest, ZRotationQuaternionToAngleAxis) {
  using T = TypeParam;
  // w = cos(pi/6), z = sin(pi/6)  ==>  rotation by pi/3 about Z.
  const Eigen::Quaternion<T> q(std::sqrt(T(3)) / 2, T(0), T(0), T(0.5));
  const Eigen::Vector3<T> aa = QuaternionToAngleAxis(q);
  EXPECT_THAT(aa,
              IsNearAngleAxis(Eigen::Vector3<T>(T(0), T(0), kPi<T> / 3), 64));
}

TYPED_TEST(RotationTest, SmallQuaternionToAngleAxis) {
  using T = TypeParam;
  const T theta = static_cast<T>(1e-2);
  const Eigen::Quaternion<T> q(std::cos(theta / 2), std::sin(theta / 2), T(0),
                               T(0));
  const Eigen::Vector3<T> aa = QuaternionToAngleAxis(q);
  EXPECT_THAT(aa, IsNearAngleAxis(Eigen::Vector3<T>(theta, T(0), T(0)), 64));
}

TYPED_TEST(RotationTest, TinyQuaternionToAngleAxis) {
  using T = TypeParam;
  const T theta = std::pow(std::numeric_limits<T>::min(), T(0.75));
  const Eigen::Quaternion<T> q(std::cos(theta / 2), std::sin(theta / 2), T(0),
                               T(0));
  const Eigen::Vector3<T> aa = QuaternionToAngleAxis(q);
  EXPECT_THAT(aa, IsNearAngleAxis(Eigen::Vector3<T>(theta, T(0), T(0)), 64));
}

TYPED_TEST(RotationTest, QuaternionToAngleAxisAngleIsLessThanPi) {
  using T = TypeParam;
  // half_theta > pi/2 makes w < 0, i.e. the raw rotation exceeds pi; the
  // result must be wrapped into [-pi, pi].
  const T half_theta = static_cast<T>(0.75) * kPi<T>;
  const Eigen::Quaternion<T> q(std::cos(half_theta), std::sin(half_theta), T(0),
                               T(0));
  const Eigen::Vector3<T> aa = QuaternionToAngleAxis(q);
  EXPECT_LE(aa.norm(), kPi<T> + T(16) * Eps<T>());
}

// --- AngleAxis <-> Quaternion round trips -----------------------------------

TYPED_TEST(RotationTest, AngleAxisToQuaternionAndBack) {
  using T = TypeParam;
  std::mt19937 prng(7);
  for (int i = 0; i < kNumTrials; ++i) {
    const Eigen::Vector3<T> axis = RandomUnitAxis<T>(prng);
    const T theta = RandomInRange<T>(prng, -kPi<T>, kPi<T>);
    const Eigen::Vector3<T> aa = axis * theta;

    const Eigen::Quaternion<T> q = AngleAxisToQuaternion(Eigen::Vector3<T>(aa));
    ASSERT_THAT(q, IsNormalizedQuaternion());
    const Eigen::Vector3<T> round_trip = QuaternionToAngleAxis(q);
    ASSERT_THAT(round_trip, IsNearAngleAxis(aa, 64));
  }
}

TYPED_TEST(RotationTest, QuaternionToAngleAxisAndBack) {
  using T = TypeParam;
  std::mt19937 prng(11);
  for (int i = 0; i < kNumTrials; ++i) {
    const Eigen::Quaternion<T> q = RandomUnitQuaternion<T>(prng);
    const Eigen::Vector3<T> aa = QuaternionToAngleAxis(q);
    const Eigen::Quaternion<T> round_trip =
        AngleAxisToQuaternion(Eigen::Vector3<T>(aa));
    ASSERT_THAT(round_trip, IsNormalizedQuaternion());
    ASSERT_THAT(round_trip, IsNearQuaternion(q, 64));
  }
}

// --- AngleAxis -> RotationMatrix --------------------------------------------

TYPED_TEST(RotationTest, ZeroAngleAxisToRotationMatrix) {
  using T = TypeParam;
  const Eigen::Matrix3<T> R =
      AngleAxisToRotationMatrix(Eigen::Vector3<T>(T(0), T(0), T(0)));
  EXPECT_THAT(R, IsOrthonormal());
  EXPECT_THAT(R, IsNear3x3Matrix(Eigen::Matrix3<T>::Identity(), 16));
}

TYPED_TEST(RotationTest, NearZeroAngleAxisToRotationMatrix) {
  using T = TypeParam;
  // A genuinely tiny angle: theta^2 (1.4e-47 here) underflows to 0 in float,
  // which would make a (1 - cos)/theta^2 formulation evaluate to 0/0. The
  // unit-axis Rodrigues form avoids squaring theta, so the matrix is identity
  // to machine precision for every tested type.
  const Eigen::Vector3<T> aa(static_cast<T>(1e-24), static_cast<T>(2e-24),
                             static_cast<T>(3e-24));
  const Eigen::Matrix3<T> R = AngleAxisToRotationMatrix(aa);
  EXPECT_THAT(R, IsOrthonormal());
  EXPECT_THAT(R, IsNear3x3Matrix(Eigen::Matrix3<T>::Identity(), 64));
}

TYPED_TEST(RotationTest, XRotationToRotationMatrixAndBack) {
  using T = TypeParam;
  const Eigen::Vector3<T> aa(kPi<T> / 2, T(0), T(0));
  Eigen::Matrix3<T> expected;
  // clang-format off
  expected << T(1), T(0),  T(0),
              T(0), T(0), T(-1),
              T(0), T(1),  T(0);
  // clang-format on
  const Eigen::Matrix3<T> R = AngleAxisToRotationMatrix(aa);
  EXPECT_THAT(R, IsOrthonormal());
  EXPECT_THAT(R, IsNear3x3Matrix(expected, 16));
  EXPECT_THAT(RotationMatrixToAngleAxis(R), IsNearAngleAxis(aa, 64));
}

TYPED_TEST(RotationTest, YRotationToRotationMatrixAndBack) {
  using T = TypeParam;
  const Eigen::Vector3<T> aa(T(0), kPi<T>, T(0));
  Eigen::Matrix3<T> expected;
  // clang-format off
  expected << T(-1), T(0),  T(0),
              T(0),  T(1),  T(0),
              T(0),  T(0), T(-1);
  // clang-format on
  const Eigen::Matrix3<T> R = AngleAxisToRotationMatrix(aa);
  EXPECT_THAT(R, IsOrthonormal());
  EXPECT_THAT(R, IsNear3x3Matrix(expected, 64));
  EXPECT_THAT(RotationMatrixToAngleAxis(R), IsNearAngleAxis(aa, 64));
}

TYPED_TEST(RotationTest, ZRotationToRotationMatrixAndBack) {
  using T = TypeParam;
  const Eigen::Vector3<T> aa(T(0), T(0), kPi<T> / 3);
  const T s = std::sqrt(T(3)) / 2;
  Eigen::Matrix3<T> expected;
  // clang-format off
  expected << T(0.5),  -s,     T(0),
              s,       T(0.5), T(0),
              T(0),    T(0),   T(1);
  // clang-format on
  const Eigen::Matrix3<T> R = AngleAxisToRotationMatrix(aa);
  EXPECT_THAT(R, IsOrthonormal());
  EXPECT_THAT(R, IsNear3x3Matrix(expected, 16));
  EXPECT_THAT(RotationMatrixToAngleAxis(R), IsNearAngleAxis(aa, 64));
}

// Independent oracle: Eigen's own AngleAxis -> RotationMatrix.
TYPED_TEST(RotationTest, AngleAxisToRotationMatrixMatchesEigen) {
  using T = TypeParam;
  std::mt19937 prng(13);
  for (int i = 0; i < kNumTrials; ++i) {
    const Eigen::Vector3<T> axis = RandomUnitAxis<T>(prng);
    const T theta = RandomInRange<T>(prng, -kPi<T>, kPi<T>);
    const Eigen::Matrix3<T> ours =
        AngleAxisToRotationMatrix(Eigen::Vector3<T>(axis * theta));
    const Eigen::Matrix3<T> oracle =
        Eigen::AngleAxis<T>(theta, axis).toRotationMatrix();
    ASSERT_THAT(ours, IsOrthonormal());
    ASSERT_THAT(ours, IsNear3x3Matrix(oracle, 64));
  }
}

// Rotating a point by the matrix must match rotating it by the quaternion the
// matrix was reconstructed from. This pins the hat() sign (a wrong sign would
// give the transpose / inverse rotation).
TYPED_TEST(RotationTest, RotationMatrixMatchesQuaternionRotation) {
  using T = TypeParam;
  std::mt19937 prng(17);
  std::uniform_real_distribution<T> uniform(T(-5), T(5));
  for (int i = 0; i < kNumTrials; ++i) {
    const Eigen::Quaternion<T> q = RandomUnitQuaternion<T>(prng);
    const Eigen::Matrix3<T> R =
        AngleAxisToRotationMatrix(QuaternionToAngleAxis(q));
    const Eigen::Vector3<T> p(uniform(prng), uniform(prng), uniform(prng));
    ASSERT_LE((R * p - q * p).norm(), T(256) * Eps<T>() * p.norm());
  }
}

// --- RotationMatrix -> AngleAxis, including near/at pi -----------------------

TYPED_TEST(RotationTest, AtPiAngleAxisRoundTrip) {
  using T = TypeParam;
  // Rotation by pi about X: diag(1, -1, -1).
  Eigen::Matrix3<T> R;
  // clang-format off
  R << T(1),  T(0),  T(0),
       T(0), T(-1),  T(0),
       T(0),  T(0), T(-1);
  // clang-format on
  const Eigen::Vector3<T> aa = RotationMatrixToAngleAxis(R);
  EXPECT_THAT(aa, IsNearAngleAxis(Eigen::Vector3<T>(kPi<T>, T(0), T(0)), 64));
  EXPECT_THAT(AngleAxisToRotationMatrix(aa), IsNear3x3Matrix(R, 64));
}

TYPED_TEST(RotationTest, NearPiAngleAxisRoundTrip) {
  using T = TypeParam;
  std::mt19937 prng(19);
  for (int i = 0; i < kNumTrials; ++i) {
    const Eigen::Vector3<T> axis = RandomUnitAxis<T>(prng);
    // Angle just below pi.
    const T theta = RandomInRange<T>(prng, kPi<T> * (T(1) - T(1e-5)), kPi<T>);
    const Eigen::Vector3<T> aa = axis * theta;
    const Eigen::Matrix3<T> R = AngleAxisToRotationMatrix(aa);
    const Eigen::Vector3<T> round_trip = RotationMatrixToAngleAxis(R);
    ASSERT_THAT(round_trip, IsNearAngleAxis(aa, 1024));
  }
}

TYPED_TEST(RotationTest, AngleAxisToRotationMatrixAndBack) {
  using T = TypeParam;
  std::mt19937 prng(23);
  for (int i = 0; i < kNumTrials; ++i) {
    const Eigen::Vector3<T> axis = RandomUnitAxis<T>(prng);
    const T theta = RandomInRange<T>(prng, -kPi<T>, kPi<T>);
    const Eigen::Vector3<T> aa = axis * theta;
    const Eigen::Matrix3<T> R = AngleAxisToRotationMatrix(aa);
    ASSERT_THAT(R, IsOrthonormal());
    const Eigen::Vector3<T> round_trip = RotationMatrixToAngleAxis(R);
    ASSERT_THAT(round_trip, IsNearAngleAxis(aa, 256));
  }
}

TYPED_TEST(RotationTest, AngleAxisToRotationMatrixAndBackNearZero) {
  using T = TypeParam;
  std::mt19937 prng(29);
  for (int i = 0; i < kNumTrials; ++i) {
    const Eigen::Vector3<T> axis = RandomUnitAxis<T>(prng);
    const T theta = RandomInRange<T>(prng, -T(1e-7), T(1e-7));
    const Eigen::Vector3<T> aa = axis * theta;
    const Eigen::Matrix3<T> R = AngleAxisToRotationMatrix(aa);
    ASSERT_THAT(R, IsOrthonormal());
    const Eigen::Vector3<T> round_trip = RotationMatrixToAngleAxis(R);
    // Absolute comparison: the magnitude itself is ~machine-eps scale.
    ASSERT_LE((round_trip - aa).norm(), T(64) * Eps<T>());
  }
}

// --- Aerospace yaw-pitch-roll (ZYX) <-> robotics body-to-world quaternion ----

TYPED_TEST(RotationTest, ZeroEulerIsIdentity) {
  using T = TypeParam;
  const Eigen::Quaternion<T> q = EulerAnglesToQuaternion(EulerAngles<T>{});
  EXPECT_THAT(q, IsNormalizedQuaternion());
  EXPECT_THAT(q, IsNearQuaternion(Eigen::Quaternion<T>::Identity(), 16));
}

// Physical sanity for a body-to-world (NED world, FRD body) quaternion: each
// principal rotation moves a body axis where aerospace convention expects.
TYPED_TEST(RotationTest, EulerPrincipalRotations) {
  using T = TypeParam;
  const T h = kPi<T> / 2;
  const T tol = T(64) * Eps<T>();

  // Yaw +90: nose (body +X) swings to world East (+Y in NED).
  EXPECT_LE((EulerAnglesToQuaternion(EulerAngles<T>{.yaw = h}) *
                 Eigen::Vector3<T>::UnitX() -
             Eigen::Vector3<T>(T(0), T(1), T(0)))
                .norm(),
            tol);
  // Pitch +90: nose pitches up, i.e. to world Up (-Z in NED).
  EXPECT_LE((EulerAnglesToQuaternion(EulerAngles<T>{.pitch = h}) *
                 Eigen::Vector3<T>::UnitX() -
             Eigen::Vector3<T>(T(0), T(0), T(-1)))
                .norm(),
            tol);
  // Roll +90: right wing (body +Y) rolls down to world Down (+Z in NED).
  EXPECT_LE((EulerAnglesToQuaternion(EulerAngles<T>{.roll = h}) *
                 Eigen::Vector3<T>::UnitY() -
             Eigen::Vector3<T>(T(0), T(0), T(1)))
                .norm(),
            tol);
  // Passive (world->body) view: with yaw +90 the nose points East, so world
  // North (+X) is seen on the body's left (-Y).
  EXPECT_LE((EulerAnglesToQuaternion(EulerAngles<T>{.yaw = h}).conjugate() *
                 Eigen::Vector3<T>::UnitX() -
             Eigen::Vector3<T>(T(0), T(-1), T(0)))
                .norm(),
            tol);
}

// Oracle: the closed form equals qz(yaw) * qy(pitch) * qx(roll).
TYPED_TEST(RotationTest, EulerMatchesEigenComposition) {
  using T = TypeParam;
  std::mt19937 prng(31);
  for (int i = 0; i < kNumTrials; ++i) {
    const EulerAngles<T> e{.roll = RandomInRange<T>(prng, -kPi<T>, kPi<T>),
                           .pitch = RandomInRange<T>(prng, -kPi<T>, kPi<T>),
                           .yaw = RandomInRange<T>(prng, -kPi<T>, kPi<T>)};
    const Eigen::Quaternion<T> oracle =
        Eigen::Quaternion<T>(
            Eigen::AngleAxis<T>(e.yaw, Eigen::Vector3<T>::UnitZ())) *
        Eigen::Quaternion<T>(
            Eigen::AngleAxis<T>(e.pitch, Eigen::Vector3<T>::UnitY())) *
        Eigen::Quaternion<T>(
            Eigen::AngleAxis<T>(e.roll, Eigen::Vector3<T>::UnitX()));
    ASSERT_THAT(EulerAnglesToQuaternion(e), IsNearQuaternion(oracle, 32));
  }
}

// Internal consistency: the matrix is Rz(yaw) Ry(pitch) Rx(roll) built from the
// library's own active rotations (i.e. the body-to-world DCM).
TYPED_TEST(RotationTest, EulerMatchesActiveRotationMatrix) {
  using T = TypeParam;
  std::mt19937 prng(37);
  for (int i = 0; i < kNumTrials; ++i) {
    const EulerAngles<T> e{.roll = RandomInRange<T>(prng, -kPi<T>, kPi<T>),
                           .pitch = RandomInRange<T>(prng, -kPi<T>, kPi<T>),
                           .yaw = RandomInRange<T>(prng, -kPi<T>, kPi<T>)};
    const Eigen::Matrix3<T> expected =
        AngleAxisToRotationMatrix(
            Eigen::Vector3<T>(e.yaw * Eigen::Vector3<T>::UnitZ())) *
        AngleAxisToRotationMatrix(
            Eigen::Vector3<T>(e.pitch * Eigen::Vector3<T>::UnitY())) *
        AngleAxisToRotationMatrix(
            Eigen::Vector3<T>(e.roll * Eigen::Vector3<T>::UnitX()));
    ASSERT_THAT(EulerAnglesToQuaternion(e).toRotationMatrix(),
                IsNear3x3Matrix(expected, 256));
  }
}

// Round trip away from gimbal lock recovers the angles.
TYPED_TEST(RotationTest, EulerRoundTrip) {
  using T = TypeParam;
  std::mt19937 prng(41);
  for (int i = 0; i < kNumTrials; ++i) {
    const EulerAngles<T> e{.roll = RandomInRange<T>(prng, -T(3), T(3)),
                           .pitch = RandomInRange<T>(prng, -T(1.4), T(1.4)),
                           .yaw = RandomInRange<T>(prng, -T(3), T(3))};
    const EulerAngles<T> recovered =
        QuaternionToEulerAngles(EulerAnglesToQuaternion(e));
    ASSERT_LE((recovered.RPYAngles() - e.RPYAngles()).norm(), T(256) * Eps<T>())
        << "ypr = " << e.RPYAngles().transpose();
  }
}

// At gimbal lock the yaw/roll split is not unique, but the recovered triple
// must reconstruct the same rotation (and report pitch = +/-pi/2, roll = 0).
TYPED_TEST(RotationTest, EulerGimbalLock) {
  using T = TypeParam;
  const T h = kPi<T> / 2;
  for (T pitch : {h, -h}) {
    for (T yaw : {T(-2), T(-0.5), T(0.5), T(2)}) {
      for (T roll : {T(-1), T(0), T(1)}) {
        const EulerAngles<T> e{.roll = roll, .pitch = pitch, .yaw = yaw};
        const Eigen::Quaternion<T> q = EulerAnglesToQuaternion(e);
        const EulerAngles<T> recovered = QuaternionToEulerAngles(q);
        EXPECT_EQ(recovered.roll, T(0));
        EXPECT_LE(std::abs(std::abs(recovered.pitch) - h), T(64) * Eps<T>());
        EXPECT_THAT(EulerAnglesToQuaternion(recovered),
                    IsNearQuaternion(q, 256));
      }
    }
  }
}

// --- Quaternion sign / double-cover canonicalization ------------------------

TYPED_TEST(RotationTest, NegatedFlipsAllCoeffsButKeepsRotation) {
  using T = TypeParam;
  std::mt19937 prng(43);
  const Eigen::Vector3<T> v(T(1), T(-2), T(3));
  for (int i = 0; i < 200; ++i) {
    const Eigen::Quaternion<T> q = RandomUnitQuaternion<T>(prng);
    const Eigen::Quaternion<T> n = Negated(q);
    // All four coefficients negated...
    ASSERT_LE((n.coeffs() + q.coeffs()).norm(), T(8) * Eps<T>());
    // ...yet it is the same rotation (rotates every vector identically).
    ASSERT_LE((n * v - q * v).norm(), T(64) * Eps<T>() * v.norm());
  }
}

TYPED_TEST(RotationTest, MatchSignAlignsToReference) {
  using T = TypeParam;
  std::mt19937 prng(47);
  for (int i = 0; i < 200; ++i) {
    const Eigen::Quaternion<T> ref = RandomUnitQuaternion<T>(prng);
    const Eigen::Quaternion<T> q = RandomUnitQuaternion<T>(prng);

    // Whatever the input, the corrected result is never flipped.
    EXPECT_FALSE(IsSignFlipped(MatchSign(q, ref), ref));
    EXPECT_GE(MatchSign(q, ref).dot(ref), T(0));
    // q and -q align to the same representative.
    EXPECT_LE((MatchSign(Negated(q), ref).coeffs() - MatchSign(q, ref).coeffs())
                  .norm(),
              T(8) * Eps<T>());
  }
  const Eigen::Quaternion<T> a = RandomUnitQuaternion<T>(prng);
  EXPECT_FALSE(IsSignFlipped(a, a));
  EXPECT_TRUE(IsSignFlipped(Negated(a), a));
  // Already aligned: returned untouched.
  EXPECT_LE((MatchSign(a, a).coeffs() - a.coeffs()).norm(), T(0));
}

TYPED_TEST(RotationTest, CanonicalizePositiveWIsUnique) {
  using T = TypeParam;
  std::mt19937 prng(53);
  for (int i = 0; i < 200; ++i) {
    const Eigen::Quaternion<T> q = RandomUnitQuaternion<T>(prng);
    const Eigen::Quaternion<T> c = CanonicalizePositiveW(q);
    EXPECT_GE(c.w(), T(0));
    // +q and -q share one canonical form.
    EXPECT_LE((CanonicalizePositiveW(Negated(q)).coeffs() - c.coeffs()).norm(),
              T(8) * Eps<T>());
    // Idempotent.
    EXPECT_LE((CanonicalizePositiveW(c).coeffs() - c.coeffs()).norm(), T(0));
  }
  // Tie-break at w == 0 (180 deg rotation): the first nonzero of (x, y, z) is
  // made positive, so +/-q still map to the same canonical quaternion.
  const Eigen::Quaternion<T> half_turn(T(0), T(-1), T(0), T(0));  // (w,x,y,z)
  const Eigen::Quaternion<T> canon = CanonicalizePositiveW(half_turn);
  EXPECT_GT(canon.x(), T(0));
  EXPECT_LE(
      (CanonicalizePositiveW(Negated(half_turn)).coeffs() - canon.coeffs())
          .norm(),
      T(8) * Eps<T>());
}

// --- Principal-axis rotation matrices ---------------------------------------

// Each RotationMatrix{X,Y,Z} is the rotation about that unit axis -- identical
// to the general angle-axis form -- and orthonormal.
TYPED_TEST(RotationTest, PrincipalRotationsMatchAngleAxis) {
  using T = TypeParam;
  std::mt19937 prng(59);
  for (int i = 0; i < kNumTrials; ++i) {
    const T a = RandomInRange<T>(prng, -kPi<T>, kPi<T>);
    ASSERT_THAT(RotationMatrixX(a), IsOrthonormal());
    ASSERT_THAT(RotationMatrixY(a), IsOrthonormal());
    ASSERT_THAT(RotationMatrixZ(a), IsOrthonormal());
    ASSERT_THAT(RotationMatrixX(a),
                IsNear3x3Matrix(AngleAxisToRotationMatrix(Eigen::Vector3<T>(
                                    a * Eigen::Vector3<T>::UnitX())),
                                64));
    ASSERT_THAT(RotationMatrixY(a),
                IsNear3x3Matrix(AngleAxisToRotationMatrix(Eigen::Vector3<T>(
                                    a * Eigen::Vector3<T>::UnitY())),
                                64));
    ASSERT_THAT(RotationMatrixZ(a),
                IsNear3x3Matrix(AngleAxisToRotationMatrix(Eigen::Vector3<T>(
                                    a * Eigen::Vector3<T>::UnitZ())),
                                64));
  }
}

// Rz(yaw) Ry(pitch) Rx(roll) reproduces the body-to-world matrix of the Euler
// conversion -- ties the principal rotations to the established convention.
TYPED_TEST(RotationTest, PrincipalRotationsComposeToEulerAngles) {
  using T = TypeParam;
  std::mt19937 prng(61);
  for (int i = 0; i < kNumTrials; ++i) {
    const EulerAngles<T> e{.roll = RandomInRange<T>(prng, -kPi<T>, kPi<T>),
                           .pitch = RandomInRange<T>(prng, -kPi<T>, kPi<T>),
                           .yaw = RandomInRange<T>(prng, -kPi<T>, kPi<T>)};
    const Eigen::Matrix3<T> composed = RotationMatrixZ(e.yaw) *
                                       RotationMatrixY(e.pitch) *
                                       RotationMatrixX(e.roll);
    ASSERT_THAT(
        composed,
        IsNear3x3Matrix(EulerAnglesToQuaternion(e).toRotationMatrix(), 256));
  }
}

TYPED_TEST(RotationTest, PrincipalRotationKnownValues) {
  using T = TypeParam;
  const T h = kPi<T> / 2;
  Eigen::Matrix3<T> rz;
  // clang-format off
  rz << T(0), T(-1), T(0),
        T(1),  T(0), T(0),
        T(0),  T(0), T(1);
  // clang-format on
  EXPECT_THAT(RotationMatrixZ(h), IsNear3x3Matrix(rz, 16));
  // The kernel called with the named axis equals the wrapper.
  EXPECT_THAT(PrincipalRotationMatrix<Axis::kZ>(h), IsNear3x3Matrix(rz, 16));
}
