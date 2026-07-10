// Rigid body transformations (SE(3))
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

#ifndef HS293GO_RIGID_TRANSFORM_HPP_
#define HS293GO_RIGID_TRANSFORM_HPP_

#include <concepts>
#include <type_traits>

#include "Eigen/Dense"
#include "hs293go/eigen_utils.hpp"

namespace hs293go {

// A rigid transform (element of SE(3)) acting as rotate-then-translate:
//   p |-> rotation * p + translation.
//
// Policy: this type is an aggregate with all data public and no lifecycle --
// no user-declared constructors or destructor, no invariant enforcement.
// Construct with designated initializers,
//   RigidTransform<double>{.rotation = q, .translation = t},
// where unmentioned members default to the identity transform. As elsewhere in
// this library (see rotation.hpp), unit norm of `rotation` is a documented
// precondition, not an enforced invariant. The static_asserts below the struct
// keep the policy machine-checked.
template <std::floating_point T>
struct RigidTransform {
  using Scalar = T;

  Eigen::Quaternion<T> rotation = Eigen::Quaternion<T>::Identity();
  Eigen::Vector3<T> translation = Eigen::Vector3<T>::Zero();

  // The group identity. Same value as RigidTransform{}; spelled out for
  // readability at call sites, parallel to Eigen's ::Identity().
  static RigidTransform Identity() noexcept { return {}; }

  // Apply the rigid transform to a point in 3D space
  template <Vector3Like Derived>
  Eigen::Vector3<T> operator*(const Eigen::MatrixBase<Derived>& point) const {
    return rotation * point + translation;
  }

  // Apply the rigid transform to a homogeneous point (4D vector).
  template <Vector4Like Derived>
  Eigen::Vector4<T> operator*(const Eigen::MatrixBase<Derived>& point) const {
    // Precondition: point[3] == 1.0
    Eigen::Vector4<T> rotated;
    rotated(ix::seq3()) = rotation * point(ix::seq3()) + translation;
    rotated[3] = point[3];
    return rotated;
  }

  // Right-composition: after `a *= b`, a means "apply the old b, then the old
  // a". The ordering below matters -- the translation update must read the
  // pre-composition rotation.
  RigidTransform& operator*=(const RigidTransform& other) {
    translation += rotation * other.translation;
    rotation *= other.rotation;
    return *this;
  }

  // Invert the rigid transform. conjugate(), not Quaternion::inverse(): under
  // the unit-norm precondition they are equal, and inverse() would pay a
  // divide by the squared norm to defend against a precondition this library
  // documents instead.
  RigidTransform<T> inverse() const noexcept {
    RigidTransform<T> inv;
    inv.rotation = rotation.conjugate();
    inv.translation = -(inv.rotation * translation);
    return inv;
  }
};

// The POD policy, machine-checked: a user-declared constructor or destructor,
// a virtual member, or a non-public data member fails one of these. Trivial
// copyability is deliberately not asserted -- Eigen 3.4.0's Matrix and
// Quaternion carry user-provided copy constructors, so no type containing them
// can be trivially copyable; assert it too if the minimum Eigen version ever
// moves past 3.4.0.
static_assert(std::is_aggregate_v<RigidTransform<float>> &&
              std::is_aggregate_v<RigidTransform<double>>);
static_assert(std::is_standard_layout_v<RigidTransform<float>> &&
              std::is_standard_layout_v<RigidTransform<double>>);
static_assert(std::is_trivially_destructible_v<RigidTransform<float>> &&
              std::is_trivially_destructible_v<RigidTransform<double>>);

// Composition: (lhs * rhs) acts as rhs first, then lhs, so that
// (lhs * rhs) * p == lhs * (rhs * p).
template <std::floating_point T>
RigidTransform<T> operator*(RigidTransform<T> lhs,
                            const RigidTransform<T>& rhs) {
  lhs *= rhs;
  return lhs;
}

// The 4x4 homogeneous form [R t; 0 1]. This is the interop wire format (e.g.
// PCL wants Matrix4f/Affine3f) and the fast path for bulk point transforms:
// converting once and doing R * p + t per point beats quaternion-rotating
// every point roughly twofold in flops.
template <std::floating_point T>
Eigen::Matrix4<T> RigidTransformToMatrix(const RigidTransform<T>& rt) {
  Eigen::Matrix4<T> mat = Eigen::Matrix4<T>::Identity();

  mat(ix::seq3(), ix::seq3()) = rt.rotation.toRotationMatrix();
  mat(ix::seq3(), 3) = rt.translation;
  return mat;
}

// Inverse of RigidTransformToMatrix. Precondition: the top-left 3x3 block of
// `mat` is (approximately) a rotation matrix; nothing is re-orthonormalized.
template <Matrix4Like Derived>
RigidTransform<typename Derived::Scalar> MatrixToRigidTransform(
    const Eigen::MatrixBase<Derived>& mat) {
  using T = typename Derived::Scalar;
  return {.rotation = Eigen::Quaternion<T>(mat(ix::seq3(), ix::seq3())),
          .translation = mat(ix::seq3(), 3)};
}

using RigidTransformF32 = RigidTransform<float>;
using RigidTransformF64 = RigidTransform<double>;

}  // namespace hs293go

#endif  // HS293GO_RIGID_TRANSFORM_HPP_
