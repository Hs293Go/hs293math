// Miscellaneous Eigen utility functions
// Copyright © 2026 H S Helson Go
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

#ifndef HS293GO_EIGEN_UTILS_HPP_
#define HS293GO_EIGEN_UTILS_HPP_

#include "Eigen/Dense"

namespace hs293go {

namespace ix {

using Eigen::all;

template <int N>
inline auto seqFixed(Eigen::Index start = 0) {
  return Eigen::seqN(start, Eigen::fix<N>());
}

inline auto seq2(Eigen::Index start = 0) { return seqFixed<2>(start); }

inline auto seq3(Eigen::Index start = 0) { return seqFixed<3>(start); }

inline auto seq4(Eigen::Index start = 0) { return seqFixed<4>(start); }

inline auto seq6(Eigen::Index start = 0) { return seqFixed<6>(start); }

}  // namespace ix

template <typename Derived>
concept VectorLike = requires {
  { Derived::IsVectorAtCompileTime } -> std::convertible_to<bool>;
} && bool(Derived::ColsAtCompileTime);

template <typename Derived, int S>
concept FixedSizeVectorLike = requires {
  { Derived::RowsAtCompileTime == S } -> std::convertible_to<int>;
} && Derived::RowsAtCompileTime == S;

template <typename Derived>
concept Vector2Like = FixedSizeVectorLike<Derived, 2>;

template <typename Derived>
concept Vector3Like = FixedSizeVectorLike<Derived, 3>;

template <typename Derived>
concept Vector4Like = FixedSizeVectorLike<Derived, 4>;

template <typename Derived>
concept Vector6Like = FixedSizeVectorLike<Derived, 6>;

template <typename Derived, int R, int C>
concept FixedSizeMatrixLike = requires {
  { Derived::RowsAtCompileTime == R } -> std::convertible_to<int>;
  { Derived::ColsAtCompileTime == C } -> std::convertible_to<int>;
} && Derived::RowsAtCompileTime == R && Derived::ColsAtCompileTime == C;

template <typename Derived, int S>
concept SquareMatrixLike = FixedSizeMatrixLike<Derived, S, S>;

template <typename Derived>
concept Matrix2Like = SquareMatrixLike<Derived, 2>;

template <typename Derived>
concept Matrix3Like = SquareMatrixLike<Derived, 3>;

template <typename Derived>
concept Matrix4Like = SquareMatrixLike<Derived, 4>;

template <typename Derived>
concept Matrix6Like = SquareMatrixLike<Derived, 6>;

}  // namespace hs293go

#endif  // HS293GO_EIGEN_UTILS_HPP_
