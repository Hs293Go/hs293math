// Frame conversions between ENU/NED and FLU/FRD coordinate systems
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

#ifndef HS293GO_FRAME_CONVERSIONS_HPP_
#define HS293GO_FRAME_CONVERSIONS_HPP_

#include <numbers>

#include "Eigen/Dense"

namespace hs293go {

/**
 * @brief Convert a vector from ENU (East-North-Up) to NED (North-East-Down)
 * frame.
 *
 * @tparam Derived Eigen expression type of the input vector
 * @param vec A 3D vector in ENU coordinates
 * @return The corresponding vector in NED coordinates
 */
template <typename Derived>
Eigen::Vector3<typename Derived::Scalar> VecEnuNed(
    const Eigen::MatrixBase<Derived>& vec) {
  // North/East are swapped; Up is negated to Down.
  return {
      vec[1],
      vec[0],
      -vec[2],
  };
}

/**
 * @brief Convert a vector from FLU (Forward-Left-Up) to FRD
 * (Forward-Right-Down) frame.
 *
 * @tparam Derived Eigen expression type of the input vector
 * @param vec A 3D vector in FLU coordinates
 * @return The corresponding vector in FRD coordinates
 */
template <typename Derived>
Eigen::Vector3<typename Derived::Scalar> VecFluFrd(
    const Eigen::MatrixBase<Derived>& vec) {
  // Forward is kept; Left is negated to Right and Up is negated to Down.
  return {
      vec[0],
      -vec[1],
      -vec[2],
  };
}

/**
 * @brief Convert a quaternion between the aerospace (FRD body to NED world) and
 * robotics/Isaac (FLU body to ENU world) conventions.
 *
 * The two conventions are 180 degrees apart, so this map is an involution:
 * applying it twice returns the original quaternion.
 *
 * @tparam Derived Eigen quaternion expression type
 * @param quat_b2e A body-to-world quaternion, represented as [x, y, z, w]
 * @return The corresponding quaternion in the other convention, represented as
 * [x, y, z, w]
 */
template <typename Derived>
Eigen::Quaternion<typename Derived::Scalar> QuatAeroRobo(
    const Eigen::QuaternionBase<Derived>& quat_b2e) {
  using Scalar = typename Derived::Scalar;
  const Scalar x = quat_b2e.x();
  const Scalar y = quat_b2e.y();
  const Scalar z = quat_b2e.z();
  const Scalar w = quat_b2e.w();
  // Composition of the 180-degree body (FLU<->FRD) and world (ENU<->NED)
  // rotations reduces to this linear map on the quaternion coefficients.
  constexpr Scalar kHalfSqrt2 = std::numbers::sqrt2_v<Scalar> / 2;

  return Eigen::Quaternion<Scalar>(
      /*w=*/-kHalfSqrt2 * (z + w),
      /*x=*/-kHalfSqrt2 * (x + y),
      /*y=*/-kHalfSqrt2 * (x - y),
      /*z=*/kHalfSqrt2 * (z - w));
}

}  // namespace hs293go

#endif  // HS293GO_FRAME_CONVERSIONS_HPP_
