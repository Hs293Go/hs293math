// Inference-side neural-network primitives: activations and layer transforms.
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

#ifndef HS293GO_NN_HPP_
#define HS293GO_NN_HPP_

#include "Eigen/Dense"
#include "hs293go/eigen_utils.hpp"

namespace hs293go {

// Naming follows the ML functional convention (relu, softmax, ...) rather than
// this library's usual PascalCase, which stays reserved for eventual layer
// types. hs293go::tanh coexisting with std::tanh is deliberate: the library's
// function-scope `using std::tanh;` idiom shadows it, and a stray scalar call
// fails template deduction loudly instead of resolving to a wrong overload.
//
// Everything is generic over Derived::Scalar, so Eigen's AutoDiffScalar flows
// through and network Jacobians come out of forward-mode differentiation with
// no changes here.

namespace detail {

// Compile-time dimensions agree when both are fixed and equal; Eigen::Dynamic
// is compatible with anything and left to Eigen's run-time checks.
constexpr bool SizesMatch(int a, int b) {
  return a == Eigen::Dynamic || b == Eigen::Dynamic || a == b;
}

}  // namespace detail

// Activations are elementwise and shape-preserving: a batch stored one sample
// per column goes through unchanged.

template <typename Derived>
typename Derived::PlainObject relu(const Eigen::MatrixBase<Derived>& x) {
  using Scalar = typename Derived::Scalar;
  return x.cwiseMax(Scalar(0));
}

template <typename Derived>
typename Derived::PlainObject tanh(const Eigen::MatrixBase<Derived>& x) {
  return x.array().tanh().matrix();
}

template <typename Derived>
typename Derived::PlainObject elu(const Eigen::MatrixBase<Derived>& x) {
  using Scalar = typename Derived::Scalar;
  // ELU with alpha = 1: x for x > 0, exp(x) - 1 otherwise.
  return (x.array() > Scalar(0))
      .select(x.array(), x.array().exp() - Scalar(1))
      .matrix();
}

// Logistic sigmoid 1 / (1 + exp(-x)), evaluated as 0.5 * (1 + tanh(x / 2)).
// The naive form overflows exp for large negative arguments -- float saturates
// through an inf intermediate that AutoDiff turns into NaN derivatives --
// while tanh is finite and well-conditioned everywhere.
template <typename Derived>
typename Derived::PlainObject sigmoid(const Eigen::MatrixBase<Derived>& x) {
  using Scalar = typename Derived::Scalar;
  return (Scalar(0.5) * ((x.array() / Scalar(2)).tanh() + Scalar(1))).matrix();
}

// Softmax exp(x_i) / sum_j exp(x_j) with the max-subtraction trick: shifting
// by the largest entry makes every exponent nonpositive so exp cannot
// overflow, and the shift cancels exactly in the ratio. Vectors only -- on a
// batch the reduction axis would be ambiguous.
template <VectorLike Derived>
typename Derived::PlainObject softmax(const Eigen::MatrixBase<Derived>& x) {
  const auto z = (x.array() - x.maxCoeff()).exp().eval();
  return (z / z.sum()).matrix();
}

// Fully-connected layer wts * x + biases. The operand scalars may differ --
// plain-double weights applied to an AutoDiffScalar input resolve through
// Eigen's mixed-scalar traits -- so the return type is left to Eigen.
template <typename Derived, typename WDerived, typename BDerived>
  requires(VectorLike<Derived> && VectorLike<BDerived> &&
           detail::SizesMatch(WDerived::ColsAtCompileTime,
                              Derived::RowsAtCompileTime) &&
           detail::SizesMatch(WDerived::RowsAtCompileTime,
                              BDerived::RowsAtCompileTime))
auto linear(const Eigen::MatrixBase<Derived>& x,
            const Eigen::MatrixBase<WDerived>& wts,
            const Eigen::MatrixBase<BDerived>& biases) {
  return (wts * x + biases).eval();
}

// Input standardization (x - mean) / stddev, elementwise -- the conditioning
// transform networks are trained behind. Like linear, mean and stddev may be
// plain-scalar while x carries AutoDiffScalar.
template <typename Derived, typename MDerived, typename SDerived>
  requires(VectorLike<Derived> && VectorLike<MDerived> &&
           VectorLike<SDerived> &&
           detail::SizesMatch(Derived::RowsAtCompileTime,
                              MDerived::RowsAtCompileTime) &&
           detail::SizesMatch(Derived::RowsAtCompileTime,
                              SDerived::RowsAtCompileTime))
auto standardize(const Eigen::MatrixBase<Derived>& x,
                 const Eigen::MatrixBase<MDerived>& mean,
                 const Eigen::MatrixBase<SDerived>& stddev) {
  return ((x - mean).array() / stddev.array()).matrix().eval();
}

}  // namespace hs293go

#endif  // HS293GO_NN_HPP_
