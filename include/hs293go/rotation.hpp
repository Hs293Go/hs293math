// Conversion between rotation representations and quaternion utilities.
// Copyright © 2026 Hei Shing Helson Go
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
// OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef HS293GO_ROTATION_HPP_
#define HS293GO_ROTATION_HPP_

#include <cmath>
#include <concepts>
#include <limits>
#include <numbers>

#include "Eigen/Dense"
#include "hs293go/eigen_utils.hpp"

namespace hs293go {

// Principal-axis selector for the rotation kernel below. The named wrappers
// (RotationMatrixX/Y/Z) are the front door; the kernel is exposed for code that
// needs a compile-time-generic axis (e.g. an Euler-sequence loop).
enum class Axis { kX = 0, kY = 1, kZ = 2 };

// Active right-handed rotation by `angle` about the principal axis `A`. The
// axis is a template parameter (named, not a magic 0/1/2) so the index
// arithmetic unrolls and the matrix fills branch-free; `T` deduces from the
// argument.
template <Axis A, std::floating_point T>
  requires(A == Axis::kX || A == Axis::kY || A == Axis::kZ)
Eigen::Matrix3<T> PrincipalRotationMatrix(T angle) {
  using std::cos;
  using std::sin;
  // NOLINTBEGIN(readability-identifier-naming)
  constexpr int i = static_cast<int>(A);
  constexpr int j = (i + 1) % 3;
  constexpr int k = (j + 1) % 3;
  // NOLINTEND(readability-identifier-naming)
  Eigen::Matrix3<T> m;
  m(i, i) = T(1);
  m(i, j) = T(0);
  m(i, k) = T(0);

  m(j, i) = T(0);
  m(j, j) = cos(angle);
  m(j, k) = -sin(angle);

  m(k, i) = T(0);
  m(k, j) = sin(angle);
  m(k, k) = cos(angle);
  return m;
}

// Named convenience wrappers -- the common spelling, e.g. RotationMatrixZ(yaw).
template <std::floating_point T>
Eigen::Matrix3<T> RotationMatrixX(T angle) {
  return PrincipalRotationMatrix<Axis::kX>(angle);
}

template <std::floating_point T>
Eigen::Matrix3<T> RotationMatrixY(T angle) {
  return PrincipalRotationMatrix<Axis::kY>(angle);
}

template <std::floating_point T>
Eigen::Matrix3<T> RotationMatrixZ(T angle) {
  return PrincipalRotationMatrix<Axis::kZ>(angle);
}

// The hat (skew-symmetric) operator: hat(v) * u == v.cross(u). Maps a 3-vector
// to the matrix generating the corresponding infinitesimal rotation.
template <Vector3Like Derived>
Eigen::Matrix3<typename Derived::Scalar> hat(
    const Eigen::MatrixBase<Derived>& v) {
  using Scalar = typename Derived::Scalar;
  // Eigen 3.4 offers the very readable (explicit) nested init-list ctor.
  return Eigen::Matrix3<Scalar>{{Scalar(0), -v[2], v[1]},
                                {v[2], Scalar(0), -v[0]},
                                {-v[1], v[0], Scalar(0)}};
}

template <Matrix3Like Derived>
Eigen::Vector3<typename Derived::Scalar> vee(
    const Eigen::MatrixBase<Derived>& m) {
  using Scalar = typename Derived::Scalar;
  return {m(2, 1), m(0, 2), m(1, 0)};
}

// SO(3) logarithm of a unit quaternion: the rotation vector theta * axis,
// whose magnitude is the true rotation angle (wrapped to (-pi, pi]). This is
// Ceres's QuaternionToAngleAxis. Unlike 2 * vec(q) = 2 sin(theta/2) * axis,
// it is linear in the angle, so large attitude errors are not compressed by
// the half-angle sine. Precondition: `quaternion` is (approximately) unit.
template <typename Derived>
Eigen::Vector3<typename Derived::Scalar> QuaternionToAngleAxis(
    const Eigen::QuaternionBase<Derived>& quaternion) {
  using Scalar = typename Derived::Scalar;

  using std::atan2;
  using std::copysign;
  using std::fpclassify;
  const Scalar n = quaternion.vec().norm();
  const Scalar& w = quaternion.w();

  if (fpclassify(n) == FP_ZERO) {
    // The actual first taylor term is 2/w, but if the quaternion is normalized
    // and n=0, then w should be 1. Dropping w from the denominator is almost
    // always a valid approximation, and it provides robustness to pathological
    // zero quaternions.
    return Scalar(2) * quaternion.vec();
  }

  // w < 0 ==> cos(theta/2) < 0 ==> theta > pi
  //
  // By convention, the condition |theta| < pi is imposed by wrapping theta
  // to pi; The wrap operation can be folded inside evaluation of atan2
  //
  // theta - pi = atan(sin(theta - pi), cos(theta - pi))
  //            = atan(-sin(theta), -cos(theta))
  const Scalar sign = copysign(Scalar(1), w);
  const Scalar atan_nbyw = atan2(sign * n, sign * w);
  return Scalar(2) * atan_nbyw / n * quaternion.vec();
}

// SO(3) exponential: a rotation vector (angle * axis) -> unit quaternion. The
// inverse of QuaternionToAngleAxis.
template <Vector3Like Derived>
Eigen::Quaternion<typename Derived::Scalar> AngleAxisToQuaternion(
    const Eigen::MatrixBase<Derived>& angle_axis) {
  using Scalar = typename Derived::Scalar;
  using std::cos;
  using std::fpclassify;
  using std::sin;
  const Scalar angle = angle_axis.norm();
  Eigen::Quaternion<Scalar> q;
  if (fpclassify(angle) == FP_ZERO) {
    q.w() = Scalar(1);
    q.vec() = angle_axis / Scalar(2);
  } else {
    const Scalar half_angle = angle / Scalar(2);
    const Scalar sin_half = sin(half_angle);
    q.w() = cos(half_angle);
    q.vec() = sin_half / angle * angle_axis;
  }

  return q;
}

template <Vector3Like Derived>
Eigen::Matrix3<typename Derived::Scalar> AngleAxisToRotationMatrix(
    const Eigen::MatrixBase<Derived>& angle_axis) {
  using std::cos;
  using std::fpclassify;
  using std::sin;
  using Scalar = typename Derived::Scalar;

  const Scalar theta = angle_axis.stableNorm();
  if (fpclassify(theta) == FP_ZERO) {
    return Eigen::Matrix3<Scalar>::Identity() + hat(angle_axis);
  }
  const Scalar cos_theta = cos(theta);
  const Scalar sin_theta = sin(theta);
  // Pre-normalize the axis because the alternative is writing angle_axis *
  // angle_axis.transpose() / (theta * theta), leading to potentially
  // catastrophic cancellation in the denominator when theta is small.
  const Eigen::Vector3<Scalar> axis = angle_axis / theta;

  // Compared to the I + sin*hat + (1-cos)*hat^2 form, this form is
  // computationally cheaper because it avoids the extra matrix-matrix multiply
  // (hat^2)
  return cos_theta * Eigen::Matrix3<Scalar>::Identity() +
         (Scalar(1) - cos_theta) * axis * axis.transpose() +
         sin_theta * hat(axis);
}

template <Matrix3Like Derived>
Eigen::Vector3<typename Derived::Scalar> RotationMatrixToAngleAxis(
    const Eigen::MatrixBase<Derived>& rotation_matrix) {
  return QuaternionToAngleAxis(
      Eigen::Quaternion<typename Derived::Scalar>(rotation_matrix));
}

template <std::floating_point T>
struct EulerAngles {
  T roll = T(0);   // rotation about x-axis
  T pitch = T(0);  // rotation about y-axis
  T yaw = T(0);    // rotation about z-axis
  Eigen::Vector3<T> RPYAngles() const noexcept { return {roll, pitch, yaw}; }

  template <typename Derived>
  static EulerAngles<T> FromRPYAngles(const Eigen::MatrixBase<Derived>& v) {
    return {.roll = v.x(), .pitch = v.y(), .yaw = v.z()};
  }
};

// Convert an aerospace yaw-pitch-roll (Z-Y-X) Euler triple to the robotics
// body-to-world quaternion.
//
// The Euler angles parameterize the *passive* world(NED)-to-body(FRD) DCM
//   C_body<-world = Rx(roll) * Ry(pitch) * Rz(yaw)            (passive)
// i.e. v_body = C_body<-world * v_world. The returned quaternion is the inverse
// *active* body-to-world rotation
//   R_world<-body = C_body<-world^T = Rz(yaw) * Ry(pitch) * Rx(roll)
//   q             = qz(yaw) * qy(pitch) * qx(roll),
// so that `q * v_body` re-expresses a body vector in the world frame -- the
// robotics convention.
template <std::floating_point T>
Eigen::Quaternion<T> EulerAnglesToQuaternion(const EulerAngles<T>& euler) {
  using std::cos;
  using std::sin;
  const auto [roll, pitch, yaw] = euler;
  const T cy = cos(yaw / T(2));
  const T sy = sin(yaw / T(2));
  const T cp = cos(pitch / T(2));
  const T sp = sin(pitch / T(2));
  const T cr = cos(roll / T(2));
  const T sr = sin(roll / T(2));
  return {
      /*w=*/cy * cp * cr + sy * sp * sr,
      /*x=*/cy * cp * sr - sy * sp * cr,
      /*y=*/cy * sp * cr + sy * cp * sr,
      /*z=*/sy * cp * cr - cy * sp * sr,
  };
}

// Inverse of EulerAnglesToQuaternion: recover the aerospace yaw-pitch-roll
// (Z-Y-X) triple from a robotics body-to-world quaternion. yaw and roll are in
// (-pi, pi] and pitch in [-pi/2, pi/2]. Precondition: `quaternion` is
// (approximately) unit.
//
// At gimbal lock (pitch = +/-pi/2) yaw and roll collapse to one DOF; that DOF
// is assigned to yaw and roll is pinned to 0.
template <typename Derived>
EulerAngles<typename Derived::Scalar> QuaternionToEulerAngles(
    const Eigen::QuaternionBase<Derived>& quaternion) {
  using Scalar = typename Derived::Scalar;
  using std::atan2;
  using std::copysign;
  using std::hypot;
  using std::sqrt;
  const Scalar w = quaternion.w();
  const Scalar x = quaternion.x();
  const Scalar y = quaternion.y();
  const Scalar z = quaternion.z();

  // Entries of the body-to-world matrix R = Rz(yaw) Ry(pitch) Rx(roll):
  //   sin(pitch) = -R(2,0),    |cos(pitch)| = ||(R(2,1), R(2,2))||,
  //   roll from (R(2,1), R(2,2)),   yaw from (R(1,0), R(0,0)).
  // Using atan2 for pitch (rather than asin) keeps it well-conditioned near
  // +/-pi/2; hypot avoids the cancellation of cos(pitch) = sqrt(1 - sin^2).
  const Scalar sin_pitch = Scalar(2) * (w * y - x * z);
  const Scalar r21 = Scalar(2) * (w * x + y * z);
  const Scalar r22 = Scalar(1) - Scalar(2) * (x * x + y * y);
  const Scalar cos_pitch = hypot(r21, r22);

  // Below ~sqrt(eps) the yaw/roll entries are all noise: gimbal lock.
  if (cos_pitch < sqrt(std::numeric_limits<Scalar>::epsilon())) {
    return {.roll = Scalar(0),
            .pitch = copysign(std::numbers::pi_v<Scalar> / 2, sin_pitch),
            .yaw = atan2(Scalar(2) * (w * z - x * y),
                         Scalar(1) - Scalar(2) * (x * x + z * z))};
  }
  return {.roll = atan2(r21, r22),
          .pitch = atan2(sin_pitch, cos_pitch),
          .yaw = atan2(Scalar(2) * (w * z + x * y),
                       Scalar(1) - Scalar(2) * (y * y + z * z))};
}

// --- Quaternion sign / double-cover canonicalization ------------------------
//
// q and -q are the same rotation. The ops below resolve that ambiguity. None of
// the conversions above depend on a sign choice; pick one here, at the call
// site, when you need it.

// The other representative of the same rotation: all four coefficients negated.
// Eigen has no unary operator- for quaternions, and conjugate() negates only
// the vector part, so this is easy to spell wrong.
template <typename Derived>
Eigen::Quaternion<typename Derived::Scalar> Negated(
    const Eigen::QuaternionBase<Derived>& q) {
  return Eigen::Quaternion<typename Derived::Scalar>(-q.coeffs());
}

// Detection: true if q sits on the opposite hemisphere from `reference`, i.e.
// the two unit quaternions are the same rotation up to a sign that differs
// (q.dot(reference) < 0).
template <typename Derived, typename RefDerived>
bool IsSignFlipped(const Eigen::QuaternionBase<Derived>& q,
                   const Eigen::QuaternionBase<RefDerived>& reference) {
  return q.dot(reference) < typename Derived::Scalar(0);
}

// Relative canonicalization (convention-free): return +/-q, whichever shares
// `reference`'s hemisphere (q.dot(reference) >= 0). Use it to keep a sequence
// of quaternions continuous, or to compare an estimate against a reference
// without a spurious sign flip. On an exact tie (dot == 0) the input sign is
// kept.
template <typename Derived, typename RefDerived>
Eigen::Quaternion<typename Derived::Scalar> MatchSign(
    const Eigen::QuaternionBase<Derived>& q,
    const Eigen::QuaternionBase<RefDerived>& reference) {
  if (IsSignFlipped(q, reference)) {
    return Negated(q);
  }
  return q;
}

// Absolute canonicalization (the one named convention): return +/-q with a
// non-negative real part. Ties at w == 0 (180 deg rotations, where both signs
// have w == 0) are broken by making the first nonzero of (x, y, z) positive, so
// every rotation maps to a single quaternion.
//
// Equivalent to MatchSign(q, identity) except for that w == 0 tie-break --
// which is exactly why it is a separate operation.
template <typename Derived>
Eigen::Quaternion<typename Derived::Scalar> CanonicalizePositiveW(
    const Eigen::QuaternionBase<Derived>& q) {
  using Scalar = typename Derived::Scalar;
  const Scalar wxyz[] = {q.w(), q.x(), q.y(), q.z()};
  for (const Scalar v : wxyz) {
    if (v > Scalar(0)) {
      break;
    }

    if (v < Scalar(0)) {
      return Negated(q);
    }
  }
  return Eigen::Quaternion<Scalar>(q);
}

namespace jacobians {

template <typename QDerived, typename PDerived>
Eigen::Matrix3<typename QDerived::Scalar> RotatedVectorByVector(
    const Eigen::QuaternionBase<QDerived>& q,
    const Eigen::MatrixBase<PDerived>& /*v*/) {
  return q.toRotationMatrix();
}

template <typename QDerived, typename PDerived>
Eigen::Matrix<typename QDerived::Scalar, 3, 4> RotatedVectorByQuaternion(
    const Eigen::QuaternionBase<QDerived>& q,
    const Eigen::MatrixBase<PDerived>& v) {
  using Scalar = typename QDerived::Scalar;
  const Eigen::Vector3<Scalar> tv = v + v;
  const Eigen::Vector3<Scalar> u = q.vec().cross(tv) + q.w() * tv;
  Eigen::Matrix<typename QDerived::Scalar, 3, 4> jac;
  jac(ix::all, ix::seq3(0)) =
      q.vec().dot(tv) * Eigen::Matrix3<Scalar>::Identity() - hat(u);
  jac(ix::all, 3) = u;
  return jac;
}

namespace details {
template <typename PlusOrMinus, typename Derived>
  requires(std::same_as<PlusOrMinus, std::plus<>> ||
           std::same_as<PlusOrMinus, std::minus<>>)
Eigen::Matrix3<typename Derived::Scalar> JacobianSO3(
    const Eigen::MatrixBase<Derived>& angle_axis) {
  using Scalar = typename Derived::Scalar;
  using std::cos;
  using std::fpclassify;
  using std::sin;
  const Scalar theta = angle_axis.stableNorm();
  constexpr auto kPlusOrMinus = PlusOrMinus{};
  if (fpclassify(theta) == FP_ZERO) {
    return kPlusOrMinus(Eigen::Matrix3<Scalar>::Identity(),
                        Scalar(0.5) * hat(angle_axis));
  }
  const Eigen::Vector3<Scalar> axis = angle_axis / theta;
  const Scalar sinc_theta = sin(theta) / theta;
  const Scalar cos_theta = cos(theta);
  return kPlusOrMinus(sinc_theta * Eigen::Matrix3<Scalar>::Identity() +
                          (Scalar(1) - sinc_theta) * axis * axis.transpose(),
                      ((Scalar(1) - cos_theta) / theta) * hat(axis));
}

// Shared kernel for the inverse SO(3) Jacobians, mirroring JacobianSO3. The
// coefficient (theta/2) cot(theta/2) is written with the half angle so it stays
// well-conditioned all the way to theta = pi -- the full-angle (1 + cos)/sin
// spelling is 0/0 there. PlusOrMinus selects the sign of the antisymmetric
// term, and the pairing is *opposite* to JacobianSO3: inversion flips that
// sign, so the right inverse takes std::plus and the left inverse std::minus.
template <typename PlusOrMinus, typename Derived>
  requires(std::same_as<PlusOrMinus, std::plus<>> ||
           std::same_as<PlusOrMinus, std::minus<>>)
Eigen::Matrix3<typename Derived::Scalar> JacobianSO3Inverse(
    const Eigen::MatrixBase<Derived>& angle_axis) {
  using Scalar = typename Derived::Scalar;
  using std::cos;
  using std::fpclassify;
  using std::sin;
  const Scalar theta = angle_axis.stableNorm();
  constexpr auto kPlusOrMinus = PlusOrMinus{};
  if (fpclassify(theta) == FP_ZERO) {
    return kPlusOrMinus(Eigen::Matrix3<Scalar>::Identity(),
                        Scalar(0.5) * hat(angle_axis));
  }
  const Eigen::Vector3<Scalar> axis = angle_axis / theta;
  const Scalar half_theta = theta / Scalar(2);
  const Scalar half_theta_cot = half_theta * cos(half_theta) / sin(half_theta);
  return kPlusOrMinus(
      half_theta_cot * Eigen::Matrix3<Scalar>::Identity() +
          (Scalar(1) - half_theta_cot) * axis * axis.transpose(),
      half_theta * hat(axis));
}

}  // namespace details

template <typename Derived>
Eigen::Matrix3<typename Derived::Scalar> LeftJacobianSO3(
    const Eigen::MatrixBase<Derived>& angle_axis) {
  return details::JacobianSO3<std::plus<>, Derived>(angle_axis);
}

template <typename Derived>
Eigen::Matrix3<typename Derived::Scalar> RightJacobianSO3(
    const Eigen::MatrixBase<Derived>& angle_axis) {
  return details::JacobianSO3<std::minus<>, Derived>(angle_axis);
}

// Inverses of the left/right SO(3) Jacobians. These are the Jacobian of the
// logarithm (a right perturbation of the group element maps to
// LeftJacobianSO3Inverse/RightJacobianSO3Inverse times the change in Log) and
// the building block of the boxminus/relative-rotation Jacobians. By
// construction LeftJacobianSO3Inverse(v) * LeftJacobianSO3(v) == I, and
// likewise on the right.
template <typename Derived>
Eigen::Matrix3<typename Derived::Scalar> LeftJacobianSO3Inverse(
    const Eigen::MatrixBase<Derived>& angle_axis) {
  return details::JacobianSO3Inverse<std::minus<>, Derived>(angle_axis);
}

template <typename Derived>
Eigen::Matrix3<typename Derived::Scalar> RightJacobianSO3Inverse(
    const Eigen::MatrixBase<Derived>& angle_axis) {
  return details::JacobianSO3Inverse<std::plus<>, Derived>(angle_axis);
}

// The derivative of the rotated vector w.r.t. a perturbation of the
// quaternion.
//
// The perturbation is expressed as a rotation vector and applied on the right,
// i.e., the perturbed quaternion is q * exp([delta_theta, 0]), This is
// applicable if the quaternion represents a body-to-world rotation, aka q_{ib}
// and the perturbation is a small rotation of the movable body frame.
//
// # Example
// Consider a drone translational dynamics model, which maps a body-frame
// specific-thrust (i.e. divided by mass) vector to the world frame then
// subtracts gravity to get the net acceleration:
//
// dot{v} = R(q_{ib}) * [0; 0; f_thrust] - [0; 0; g]
//
// The Jacobian of dot{v} w.r.t. a small rotation of the body frame is given by
// RotatedVectorByRotationPerturbation(q_{ib}, [0; 0; f_thrust]).
template <typename QDerived, typename PDerived>
Eigen::Matrix3<typename QDerived::Scalar> RotatedVectorByRotationPerturbation(
    const Eigen::QuaternionBase<QDerived>& q_ib,
    const Eigen::MatrixBase<PDerived>& v) {
  using Scalar = typename QDerived::Scalar;
  return -q_ib.toRotationMatrix() * hat(v);
}

// The derivative of the inversely rotated vector w.r.t. a perturbation of the
// quaternion.
//
// Like RotatedVectorByRotationPerturbation, this takes the body-to-world
// rotation q_{ib} and perturbs it identically: the perturbation is a rotation
// vector applied on the right, i.e. the perturbed quaternion is
// q_{ib} * exp([delta_theta, 0]), a small rotation of the movable body frame.
// The only difference is that this differentiates the *inversely* rotated
// vector R(q_{ib})^-1 * v; the inverse is taken internally, so both functions
// share the same argument and the same meaning of delta_theta.
//
// # Example
// Consider a magnetometer measurement model, which maps magnetic field in the
// world frame to the body frame (to be compared with the body-fixed
// magnetometer reading):
//
// b' = R(q_{ib})^-1 * b
//
// The Jacobian of b' w.r.t. a small rotation of the body frame is given by
// InverselyRotatedVectorByRotationPerturbation(q_{ib}, b).
template <typename QDerived, typename PDerived>
Eigen::Matrix3<typename QDerived::Scalar>
InverselyRotatedVectorByRotationPerturbation(
    const Eigen::QuaternionBase<QDerived>& q_ib,
    const Eigen::MatrixBase<PDerived>& v) {
  return hat(q_ib.conjugate() * v);
}

}  // namespace jacobians

}  // namespace hs293go

#endif  // HS293GO_ROTATION_HPP_
