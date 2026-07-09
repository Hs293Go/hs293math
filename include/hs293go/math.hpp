// Bread and butter math functions inspired by Matlab and Python's math module
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

#ifndef HS293GO_MATH_HPP_
#define HS293GO_MATH_HPP_

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstdlib>
#include <limits>
#include <numbers>

#if defined(__cpp_lib_constexpr_cmath) && (__cpp_lib_constexpr_cmath >= 201907L)
#define HS293GO_CONSTEXPR_USE_CMATH constexpr
#else
#define HS293GO_CONSTEXPR_USE_CMATH
#endif

namespace hs293go {

template <std::floating_point T>
struct Tolerances {
  T rel_tol;
  T abs_tol;
};

template <std::floating_point T>
inline constexpr Tolerances<T> kDefaultTolerances = {
    .rel_tol = T{4} * std::numeric_limits<T>::epsilon(),
    .abs_tol = 0,  // caller MUST set this for comparisons near zero
};

/**
 * @brief Check if two floating point numbers are close to each other.
 *
 * This function is constexpr in C++23 and later
 *
 * @param a First number
 * @param b Second number
 * @param tols Tolerances for comparison
 * @return true if a and b are close, false otherwise
 */
template <std::floating_point T>
HS293GO_CONSTEXPR_USE_CMATH bool IsClose(
    T a, T b, const Tolerances<T>& tols = kDefaultTolerances<T>) noexcept {
  using std::abs;
  using std::max;
  using std::min;

  if (a == b) {
    return true;
  }

  T abs_diff = abs(a - b);
  // Clamp the magnitude sum to the finite max so that |a| + |b| overflowing to
  // infinity for extreme inputs cannot inflate the relative threshold and
  // spuriously report closeness.
  T sum_abs = min(abs(a) + abs(b), std::numeric_limits<T>::max());
  return abs_diff < max(tols.abs_tol, tols.rel_tol * sum_abs);
}

/**
 * @brief Wrap angle in radians to [-pi, pi]
 *
 * @param angle T Angle in radians
 * @return T Angle in [-pi, pi]
 */
template <std::floating_point T>
HS293GO_CONSTEXPR_USE_CMATH T wrapToPi(T angle) noexcept {
  using std::copysign;
  using std::fmod;
  const T result =
      fmod(angle + std::numbers::pi_v<T>, T{2} * std::numbers::pi_v<T>);
  // Branchless sign determination
  const T sign = static_cast<T>(-1 + 2 * (result <= T{0}));
  return result + sign * std::numbers::pi_v<T>;
}

/**
 * @brief Wrap angle in radians to [0, 2*pi)
 *
 * @param angle T Angle in radians
 * @return T Angle in [0, 2*pi)
 */
template <std::floating_point T>
HS293GO_CONSTEXPR_USE_CMATH T wrapTo2Pi(T angle) noexcept {
  using std::fmod;
  constexpr T kTwoPi = T{2} * std::numbers::pi_v<T>;
  T res = fmod(angle, kTwoPi);
  res += static_cast<T>(res < T{0}) * kTwoPi;
  // A tiny negative residue (res just below 0) can round up to exactly kTwoPi
  // when the period is added back; snap it down to keep the range half-open.
  return res * static_cast<T>(res < kTwoPi);
}

/**
 * @brief Wrap angle in degrees to [-180, 180]
 *
 * @param angle T Angle in degrees
 * @return T Angle in [-180, 180]
 */
template <std::floating_point T>
HS293GO_CONSTEXPR_USE_CMATH T wrapTo180(T angle) noexcept {
  using std::fmod;
  const T result = fmod(angle + T{180}, T{360});
  const T sign = static_cast<T>(-1 + 2 * (result <= T{0}));
  return result + sign * T{180};
}

/**
 * @brief Wrap angle in degrees to [0, 360)
 *
 * @param angle T Angle in degrees
 * @return T Angle in [0, 360)
 */
template <std::floating_point T>
HS293GO_CONSTEXPR_USE_CMATH T wrapTo360(T angle) noexcept {
  using std::fmod;
  constexpr T kFullTurn = T{360};
  T res = fmod(angle, kFullTurn);
  res += static_cast<T>(res < T{0}) * kFullTurn;
  // A tiny negative residue (res just below 0) can round up to exactly 360
  // when the period is added back; snap it down to keep the range half-open.
  return res * static_cast<T>(res < kFullTurn);
}

/**
 * @brief Convert angle in degrees to angle in radians
 *
 * @param deg T Angle in degrees
 * @return constexpr T Angle in radians
 */
template <std::floating_point T>
constexpr T deg2rad(T deg) noexcept {
  return std::numbers::pi_v<T> * deg / T{180};
}

/**
 * @brief Convert angle in radians to angle in degrees
 *
 * @param rad T Angle in radians
 * @return constexpr T Angle in degrees
 */
template <std::floating_point T>
constexpr T rad2deg(T rad) noexcept {
  return T{180} / std::numbers::pi_v<T> * rad;
}

}  // namespace hs293go

#undef HS293GO_CONSTEXPR_USE_CMATH

#endif  // HS293GO_MATH_HPP_
