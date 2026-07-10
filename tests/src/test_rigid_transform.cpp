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
#include "hs293go/rigid_transform.hpp"

using hs293go::MatrixToRigidTransform;
using hs293go::RigidTransform;
using hs293go::RigidTransformToMatrix;
using ::testing::Not;

// Teach gmock to print quaternions and rigid transforms (no operator<<, so
// they would otherwise be dumped as raw bytes in failure messages).
namespace Eigen {
template <typename T>
void PrintTo(const Quaternion<T>& q, std::ostream* os) {
  *os << "Quaternion(w=" << q.w() << ", x=" << q.x() << ", y=" << q.y()
      << ", z=" << q.z() << ")";
}
}  // namespace Eigen

namespace hs293go {
template <typename T>
void PrintTo(const RigidTransform<T>& rt, std::ostream* os) {
  *os << "RigidTransform(rotation=";
  Eigen::PrintTo(rt.rotation, os);
  *os << ", translation=[" << rt.translation.x() << ", " << rt.translation.y()
      << ", " << rt.translation.z() << "])";
}
}  // namespace hs293go

namespace {

constexpr int kNumTrials = 1000;

// --- gmock matchers, following the conventions of test_rotation.cpp ---------
//
// Tolerances are a multiple of the matched scalar type's machine epsilon.
// Vector/matrix tolerances additionally scale with (1 + expected magnitude) so
// trials with large translations share one eps_scale with trials near zero.

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

MATCHER_P2(IsNearVector3, expected, eps_scale, "") {
  using Scalar = typename std::decay_t<decltype(arg)>::Scalar;
  const Scalar tol = Scalar(eps_scale) *
                     std::numeric_limits<Scalar>::epsilon() *
                     (Scalar(1) + expected.norm());
  const Scalar d = (arg - expected).norm();
  if (d <= tol) {
    return true;
  }
  *result_listener << "differs by " << d << " (tol " << tol << ")";
  return false;
}

MATCHER_P2(IsNear4x4Matrix, expected, eps_scale, "") {
  using Scalar = typename std::decay_t<decltype(arg)>::Scalar;
  const Scalar tol = Scalar(eps_scale) *
                     std::numeric_limits<Scalar>::epsilon() *
                     (Scalar(1) + expected.norm());
  const Scalar d = (arg - expected).norm();
  if (d <= tol) {
    return true;
  }
  *result_listener << "differs by " << d << " (tol " << tol << ")";
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

// Rotations equal up to the quaternion double-cover sign; translations within
// a magnitude-scaled tolerance.
MATCHER_P2(IsNearRigidTransform, expected, eps_scale, "") {
  using Scalar = typename std::decay_t<decltype(arg)>::Scalar;
  const Scalar eps = std::numeric_limits<Scalar>::epsilon();
  const Scalar q_tol = Scalar(eps_scale) * eps;
  const Scalar t_tol =
      Scalar(eps_scale) * eps * (Scalar(1) + expected.translation.norm());
  const Scalar q_d =
      std::min((arg.rotation.coeffs() - expected.rotation.coeffs()).norm(),
               (arg.rotation.coeffs() + expected.rotation.coeffs()).norm());
  const Scalar t_d = (arg.translation - expected.translation).norm();
  if (q_d <= q_tol && t_d <= t_tol) {
    return true;
  }
  *result_listener << "rotation coeffs differ by " << q_d << " (tol " << q_tol
                   << "), translation differs by " << t_d << " (tol " << t_tol
                   << ")";
  return false;
}

// --- Random sampling helpers ------------------------------------------------

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

template <typename T>
Eigen::Vector3<T> RandomVector3(std::mt19937& prng) {
  std::uniform_real_distribution<T> uniform(T(-5), T(5));
  return {uniform(prng), uniform(prng), uniform(prng)};
}

template <typename T>
RigidTransform<T> RandomRigidTransform(std::mt19937& prng) {
  return {.rotation = RandomUnitQuaternion<T>(prng),
          .translation = RandomVector3<T>(prng)};
}

}  // namespace

template <typename T>
class RigidTransformTest : public ::testing::Test {};

using RealTypes = ::testing::Types<float, double, long double>;
TYPED_TEST_SUITE(RigidTransformTest, RealTypes);

// --- Identity and aggregate construction -------------------------------------

// The member initializers make RigidTransform{} the group identity;
// ::Identity() is the same value and both act as a no-op on points.
TYPED_TEST(RigidTransformTest, DefaultConstructionIsIdentity) {
  using T = TypeParam;
  std::mt19937 prng(67);
  const RigidTransform<T> id{};
  EXPECT_EQ(id.rotation.coeffs(),
            RigidTransform<T>::Identity().rotation.coeffs());
  EXPECT_EQ(id.translation, RigidTransform<T>::Identity().translation);
  for (int i = 0; i < 200; ++i) {
    const Eigen::Vector3<T> p = RandomVector3<T>(prng);
    ASSERT_EQ(id * p, p);
  }
}

// Designated initializers leave unmentioned members at their identity
// defaults -- the intended construction pattern for the aggregate.
TYPED_TEST(RigidTransformTest, DesignatedInitializersDefaultToIdentity) {
  using T = TypeParam;
  std::mt19937 prng(71);
  for (int i = 0; i < 200; ++i) {
    const Eigen::Quaternion<T> q = RandomUnitQuaternion<T>(prng);
    const Eigen::Vector3<T> t = RandomVector3<T>(prng);
    const Eigen::Vector3<T> p = RandomVector3<T>(prng);

    const RigidTransform<T> rotate_only{.rotation = q};
    ASSERT_EQ(rotate_only.translation, Eigen::Vector3<T>::Zero());
    ASSERT_EQ(rotate_only * p, Eigen::Vector3<T>(q * p));

    const RigidTransform<T> translate_only{.translation = t};
    ASSERT_EQ(translate_only.rotation.coeffs(),
              Eigen::Quaternion<T>::Identity().coeffs());
    ASSERT_EQ(translate_only * p, Eigen::Vector3<T>(p + t));
  }
}

// --- Point action
// -------------------------------------------------------------

// Known value pinning the action order (rotate, then translate) and the
// active rotation direction: a quarter turn about Z takes X to Y.
TYPED_TEST(RigidTransformTest, RotateThenTranslateKnownValue) {
  using T = TypeParam;
  const T h = std::numbers::sqrt2_v<T> / 2;  // cos(pi/4) = sin(pi/4)
  const RigidTransform<T> rt{
      .rotation = Eigen::Quaternion<T>(h, T(0), T(0), h),
      .translation = Eigen::Vector3<T>(T(1), T(2), T(3))};
  EXPECT_THAT(rt * Eigen::Vector3<T>::UnitX(),
              IsNearVector3(Eigen::Vector3<T>(T(1), T(3), T(3)), 16));
  // Translate-then-rotate would give (-2, 2, 3) -- make sure it does not.
  EXPECT_THAT(rt * Eigen::Vector3<T>::UnitX(),
              Not(IsNearVector3(Eigen::Vector3<T>(T(-2), T(2), T(3)), 16)));
}

// A homogeneous point (last entry 1) takes the same arithmetic path as the
// 3D overload, so the results are bitwise identical.
TYPED_TEST(RigidTransformTest, HomogeneousPointMatchesEuclideanPoint) {
  using T = TypeParam;
  std::mt19937 prng(73);
  for (int i = 0; i < kNumTrials; ++i) {
    const RigidTransform<T> rt = RandomRigidTransform<T>(prng);
    const Eigen::Vector3<T> p = RandomVector3<T>(prng);
    ASSERT_EQ(rt * p.homogeneous(), (rt * p).homogeneous());
  }
}

// Oracle: the point action agrees with multiplying the homogeneous matrix
// against the homogeneous point.
TYPED_TEST(RigidTransformTest, PointActionMatchesMatrixAction) {
  using T = TypeParam;
  std::mt19937 prng(79);
  for (int i = 0; i < kNumTrials; ++i) {
    const RigidTransform<T> rt = RandomRigidTransform<T>(prng);
    const Eigen::Vector3<T> p = RandomVector3<T>(prng);
    const Eigen::Vector4<T> homogeneous(p.x(), p.y(), p.z(), T(1));
    const Eigen::Vector4<T> oracle = RigidTransformToMatrix(rt) * homogeneous;
    ASSERT_THAT(rt * p, IsNearVector3(Eigen::Vector3<T>(oracle.head(3)), 1024));
  }
}

// Independent oracle: Eigen's own Isometry transform built from the same
// quaternion and translation.
TYPED_TEST(RigidTransformTest, MatchesEigenIsometry) {
  using T = TypeParam;
  std::mt19937 prng(83);
  for (int i = 0; i < kNumTrials; ++i) {
    const RigidTransform<T> rt = RandomRigidTransform<T>(prng);
    const Eigen::Transform<T, 3, Eigen::Isometry> iso =
        Eigen::Translation<T, 3>(rt.translation) * rt.rotation;
    ASSERT_THAT(RigidTransformToMatrix(rt), IsNear4x4Matrix(iso.matrix(), 64));
    const Eigen::Vector3<T> p = RandomVector3<T>(prng);
    ASSERT_THAT(rt * p, IsNearVector3(Eigen::Vector3<T>(iso * p), 1024));
  }
}

// --- Composition
// --------------------------------------------------------------

// Oracle: composition is the product of the homogeneous matrices.
TYPED_TEST(RigidTransformTest, ComposeMatchesMatrixProduct) {
  using T = TypeParam;
  std::mt19937 prng(89);
  for (int i = 0; i < kNumTrials; ++i) {
    const RigidTransform<T> a = RandomRigidTransform<T>(prng);
    const RigidTransform<T> b = RandomRigidTransform<T>(prng);
    const Eigen::Matrix4<T> oracle =
        RigidTransformToMatrix(a) * RigidTransformToMatrix(b);
    ASSERT_THAT(RigidTransformToMatrix(a * b), IsNear4x4Matrix(oracle, 256));
  }
}

// a * b applies b first: (a * b) * p == a * (b * p).
TYPED_TEST(RigidTransformTest, ComposeAppliesRightOperandFirst) {
  using T = TypeParam;
  std::mt19937 prng(97);
  for (int i = 0; i < kNumTrials; ++i) {
    const RigidTransform<T> a = RandomRigidTransform<T>(prng);
    const RigidTransform<T> b = RandomRigidTransform<T>(prng);
    const Eigen::Vector3<T> p = RandomVector3<T>(prng);
    ASSERT_THAT((a * b) * p, IsNearVector3(a * (b * p), 1024));
  }
}

// operator* is a copy followed by operator*=; they must agree exactly.
TYPED_TEST(RigidTransformTest, InPlaceComposeMatchesBinaryCompose) {
  using T = TypeParam;
  std::mt19937 prng(101);
  for (int i = 0; i < kNumTrials; ++i) {
    const RigidTransform<T> a = RandomRigidTransform<T>(prng);
    const RigidTransform<T> b = RandomRigidTransform<T>(prng);
    RigidTransform<T> in_place = a;
    in_place *= b;
    const RigidTransform<T> binary = a * b;
    ASSERT_EQ(in_place.rotation.coeffs(), binary.rotation.coeffs());
    ASSERT_EQ(in_place.translation, binary.translation);
  }
}

// The identity is neutral on both sides -- exactly, because multiplying by
// the identity quaternion and adding a zero vector introduce no rounding.
TYPED_TEST(RigidTransformTest, IdentityIsNeutral) {
  using T = TypeParam;
  std::mt19937 prng(103);
  for (int i = 0; i < 200; ++i) {
    const RigidTransform<T> rt = RandomRigidTransform<T>(prng);
    const RigidTransform<T> left = RigidTransform<T>::Identity() * rt;
    const RigidTransform<T> right = rt * RigidTransform<T>::Identity();
    ASSERT_EQ(left.rotation.coeffs(), rt.rotation.coeffs());
    ASSERT_EQ(left.translation, rt.translation);
    ASSERT_EQ(right.rotation.coeffs(), rt.rotation.coeffs());
    ASSERT_EQ(right.translation, rt.translation);
  }
}

// Associativity to rounding: (a * b) * c == a * (b * c).
TYPED_TEST(RigidTransformTest, ComposeIsAssociative) {
  using T = TypeParam;
  std::mt19937 prng(107);
  for (int i = 0; i < kNumTrials; ++i) {
    const RigidTransform<T> a = RandomRigidTransform<T>(prng);
    const RigidTransform<T> b = RandomRigidTransform<T>(prng);
    const RigidTransform<T> c = RandomRigidTransform<T>(prng);
    ASSERT_THAT((a * b) * c, IsNearRigidTransform(a * (b * c), 256));
  }
}

// --- Inverse
// ------------------------------------------------------------------

// Known values that must be exact: inverting the identity, and inverting a
// pure translation (identity rotation), whose inverse translation is the
// exact negation -R^T * t = -t.
TYPED_TEST(RigidTransformTest, InverseKnownValuesAreExact) {
  using T = TypeParam;
  const RigidTransform<T> id_inv = RigidTransform<T>::Identity().inverse();
  EXPECT_EQ(id_inv.rotation.coeffs(),
            Eigen::Quaternion<T>::Identity().coeffs());
  EXPECT_EQ(id_inv.translation, Eigen::Vector3<T>::Zero());

  const RigidTransform<T> translate_only{
      .translation = Eigen::Vector3<T>(T(1), T(2), T(3))};
  const RigidTransform<T> inv = translate_only.inverse();
  EXPECT_EQ(inv.rotation.coeffs(),
            translate_only.rotation.conjugate().coeffs());
  EXPECT_EQ(inv.translation, Eigen::Vector3<T>(T(-1), T(-2), T(-3)));
}

TYPED_TEST(RigidTransformTest, InverseUndoesTransform) {
  using T = TypeParam;
  std::mt19937 prng(109);
  for (int i = 0; i < kNumTrials; ++i) {
    const RigidTransform<T> rt = RandomRigidTransform<T>(prng);
    ASSERT_THAT(rt * rt.inverse(),
                IsNearRigidTransform(RigidTransform<T>::Identity(), 512));
    ASSERT_THAT(rt.inverse() * rt,
                IsNearRigidTransform(RigidTransform<T>::Identity(), 512));
    const Eigen::Vector3<T> p = RandomVector3<T>(prng);
    ASSERT_THAT(rt.inverse() * (rt * p), IsNearVector3(p, 1024));
    ASSERT_THAT(rt * (rt.inverse() * p), IsNearVector3(p, 1024));
  }
}

// Oracle: the matrix of the inverse is the inverse of the matrix.
TYPED_TEST(RigidTransformTest, InverseMatchesMatrixInverse) {
  using T = TypeParam;
  std::mt19937 prng(113);
  for (int i = 0; i < kNumTrials; ++i) {
    const RigidTransform<T> rt = RandomRigidTransform<T>(prng);
    const Eigen::Matrix4<T> oracle = RigidTransformToMatrix(rt).inverse();
    ASSERT_THAT(RigidTransformToMatrix(rt.inverse()),
                IsNear4x4Matrix(oracle, 1024));
  }
}

// --- Homogeneous matrix conversions
// -------------------------------------------

TYPED_TEST(RigidTransformTest, MatrixHasHomogeneousStructure) {
  using T = TypeParam;
  std::mt19937 prng(127);
  for (int i = 0; i < kNumTrials; ++i) {
    const RigidTransform<T> rt = RandomRigidTransform<T>(prng);
    const Eigen::Matrix4<T> mat = RigidTransformToMatrix(rt);
    ASSERT_EQ(mat.row(3), Eigen::RowVector4<T>(T(0), T(0), T(0), T(1)));
    ASSERT_EQ(Eigen::Vector3<T>(mat.topRightCorner(3, 1)), rt.translation);
    const Eigen::Matrix3<T> rot = mat.topLeftCorner(3, 3);
    ASSERT_THAT(rot, IsOrthonormal());
    // The rotation block rotates like the quaternion (pins the active
    // convention of the embedded toRotationMatrix).
    const Eigen::Vector3<T> p = RandomVector3<T>(prng);
    ASSERT_THAT(Eigen::Vector3<T>(rot * p),
                IsNearVector3(Eigen::Vector3<T>(rt.rotation * p), 1024));
  }
}

// Round trip through the homogeneous matrix recovers the transform (rotation
// up to the quaternion double-cover sign).
TYPED_TEST(RigidTransformTest, MatrixRoundTrip) {
  using T = TypeParam;
  std::mt19937 prng(131);
  for (int i = 0; i < kNumTrials; ++i) {
    const RigidTransform<T> rt = RandomRigidTransform<T>(prng);
    const RigidTransform<T> round_trip =
        MatrixToRigidTransform(RigidTransformToMatrix(rt));
    ASSERT_THAT(round_trip.rotation, IsNormalizedQuaternion());
    ASSERT_THAT(round_trip, IsNearRigidTransform(rt, 256));
  }
}
