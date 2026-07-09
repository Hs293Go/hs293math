// Geodetic and local ENU (East-North-Up) coordinate conversions
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

#ifndef HS293GO_GEO_HPP_
#define HS293GO_GEO_HPP_

#include <cmath>

#include "Eigen/Dense"
#include "hs293go/math.hpp"  // deg2rad

namespace hs293go {

// Geodetic (latitude [deg], longitude [deg], altitude [m]) -> ECEF [m] on the
// WGS84 ellipsoid.
template <typename Derived>
Eigen::Vector3<typename Derived::Scalar> lla2ecef(
    const Eigen::MatrixBase<Derived>& lla) {
  using std::cos;
  using std::sin;
  using std::sqrt;
  using Scalar = typename Derived::Scalar;

  constexpr Scalar kAEarth = Scalar(6378137);  // semi-major axis a
  constexpr Scalar kFlattening = Scalar(1) / Scalar(298.257223563);   // f
  constexpr Scalar kNavE2 = (Scalar(2) - kFlattening) * kFlattening;  // e^2

  const Scalar lat = deg2rad(lla[0]);
  const Scalar lon = deg2rad(lla[1]);
  const Scalar s_lat = sin(lat);
  const Scalar c_lat = cos(lat);
  const Scalar s_lon = sin(lon);
  const Scalar c_lon = cos(lon);
  const Scalar r_n = kAEarth / sqrt(Scalar(1) - kNavE2 * s_lat * s_lat);
  return {
      (r_n + lla[2]) * c_lat * c_lon,
      (r_n + lla[2]) * c_lat * s_lon,
      (r_n * (Scalar(1) - kNavE2) + lla[2]) * s_lat,
  };
}

// Rotation taking a local ENU (East-North-Up) vector to ECEF axes at the
// geodetic reference point `ref_lla` (lat/lon in degrees). Orthonormal; its
// transpose is the ECEF->ENU rotation.
template <typename Derived>
Eigen::Matrix3<typename Derived::Scalar> Enu2ecefProjectionMatrix(
    const Eigen::MatrixBase<Derived>& ref_lla) {
  using std::cos;
  using std::sin;
  using Scalar = typename Derived::Scalar;

  const Scalar lat = deg2rad(ref_lla[0]);
  const Scalar lon = deg2rad(ref_lla[1]);
  const Scalar s_lon = sin(lon);
  const Scalar c_lon = cos(lon);
  const Scalar s_lat = sin(lat);
  const Scalar c_lat = cos(lat);
  return Eigen::Matrix3<Scalar>{
      {-s_lon, -s_lat * c_lon, c_lat * c_lon},
      {c_lon, -s_lat * s_lon, c_lat * s_lon},
      {Scalar(0), c_lat, s_lat},
  };
}

// Rotate a local ENU displacement vector into ECEF axes at `ref_lla` (rotation
// only -- no origin offset).
template <typename Derived, typename Derived1>
Eigen::Vector3<typename Derived::Scalar> enu2ecef(
    const Eigen::MatrixBase<Derived>& enu,
    const Eigen::MatrixBase<Derived1>& ref_lla) {
  return Enu2ecefProjectionMatrix(ref_lla) * enu;
}

// Rotate an ECEF displacement vector into the local ENU frame at `ref_lla`.
template <typename Derived, typename Derived1>
Eigen::Vector3<typename Derived::Scalar> ecef2enu(
    const Eigen::MatrixBase<Derived>& ecef,
    const Eigen::MatrixBase<Derived1>& ref_lla) {
  return Enu2ecefProjectionMatrix(ref_lla).transpose() * ecef;
}

// Geodetic `lla` expressed as a local ENU position [m] relative to the geodetic
// reference point `ref_lla`.
template <typename Derived, typename Derived1>
Eigen::Vector3<typename Derived::Scalar> lla2enu(
    const Eigen::MatrixBase<Derived>& lla,
    const Eigen::MatrixBase<Derived1>& ref_lla) {
  using Scalar = typename Derived::Scalar;
  const Eigen::Vector3<Scalar> ecef = lla2ecef(lla) - lla2ecef(ref_lla);
  return ecef2enu(ecef, ref_lla);
}

// Differential ("flat-Earth", first-order) counterpart to lla2enu: maps a small
// geodetic offset d_lla = (d_lat [deg], d_lon [deg], d_alt [m]) from the
// reference point `ref_lla` to local ENU [m], using the meridional (M) and
// prime-vertical (N) radii of curvature at the reference latitude:
//   East  = (N + h_ref) * cos(lat_ref) * d_lon
//   North = (M + h_ref) * d_lat
//   Up    = d_alt
//
// Working directly with the offset avoids the catastrophic cancellation of
// lla2enu's full-magnitude ECEF subtraction -- e.g. a pure altitude offset maps
// to (0, 0, d_alt) exactly. It is first order: the error grows as
// O(|displacement|^2 / R_earth), so it is for local (small-offset) use, whereas
// lla2enu stays exact for arbitrary separation.
template <typename Derived, typename Derived1>
Eigen::Vector3<typename Derived::Scalar> dlla2enu(
    const Eigen::MatrixBase<Derived>& d_lla,
    const Eigen::MatrixBase<Derived1>& ref_lla) {
  using std::cos;
  using std::sin;
  using std::sqrt;
  using Scalar = typename Derived::Scalar;

  constexpr Scalar kAEarth = Scalar(6378137);
  constexpr Scalar kFlattening = Scalar(1) / Scalar(298.257223563);
  constexpr Scalar kNavE2 = (Scalar(2) - kFlattening) * kFlattening;

  const Scalar lat_ref = deg2rad(ref_lla[0]);
  const Scalar s_lat = sin(lat_ref);
  const Scalar c_lat = cos(lat_ref);
  const Scalar w2 = Scalar(1) - kNavE2 * s_lat * s_lat;
  const Scalar r_n = kAEarth / sqrt(w2);                                // N
  const Scalar r_m = kAEarth * (Scalar(1) - kNavE2) / (w2 * sqrt(w2));  // M
  const Scalar h = ref_lla[2];

  return Eigen::Vector3<Scalar>((r_n + h) * c_lat * deg2rad(d_lla[1]),
                                (r_m + h) * deg2rad(d_lla[0]), d_lla[2]);
}

}  // namespace hs293go

#endif  // HS293GO_GEO_HPP_
