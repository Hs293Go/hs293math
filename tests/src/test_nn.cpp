#include <cmath>
#include <limits>
#include <random>

#include "Eigen/Dense"
#include "gtest/gtest.h"
#include "hs293go/nn.hpp"
#include "unsupported/Eigen/AutoDiff"

using hs293go::elu;
using hs293go::linear;
using hs293go::relu;
using hs293go::sigmoid;
using hs293go::softmax;
using hs293go::standardize;
// hs293go::tanh is called qualified throughout: this file also uses std::tanh
// as the reference implementation, and keeping the two spellings distinct is
// exactly the collision the library's naming comment warns about.

namespace {

// ---------------------------------------------------------------------------
// Compile-time contracts: the concepts reject mismatched operands at overload
// resolution, before Eigen's own static asserts fire deep inside a product.
// The helper concepts give the requires-expressions a dependent context, so an
// invalid call yields false instead of a hard error.
// ---------------------------------------------------------------------------

template <typename X, typename W, typename B>
concept LinearInvocable = requires(X x, W w, B b) { hs293go::linear(x, w, b); };

template <typename X>
concept SoftmaxInvocable = requires(X x) { hs293go::softmax(x); };

template <typename X, typename M, typename S>
concept StandardizeInvocable =
    requires(X x, M m, S s) { hs293go::standardize(x, m, s); };

// Well-formed calls, fixed and dynamic (Dynamic is compatible with anything).
static_assert(LinearInvocable<Eigen::Vector3d, Eigen::Matrix<double, 2, 3>,
                              Eigen::Vector2d>);
static_assert(
    LinearInvocable<Eigen::VectorXd, Eigen::MatrixXd, Eigen::VectorXd>);
static_assert(
    StandardizeInvocable<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d>);

// Input size must match the weight columns; bias size the weight rows.
static_assert(!LinearInvocable<Eigen::Vector4d, Eigen::Matrix<double, 2, 3>,
                               Eigen::Vector2d>);
static_assert(!LinearInvocable<Eigen::Vector3d, Eigen::Matrix<double, 2, 3>,
                               Eigen::Vector3d>);

// Vector-only operations reject matrices outright.
static_assert(
    !LinearInvocable<Eigen::Matrix3d, Eigen::Matrix3d, Eigen::Vector3d>);
static_assert(!SoftmaxInvocable<Eigen::Matrix3d>);

// Statistics must be the same length as the input.
static_assert(
    !StandardizeInvocable<Eigen::Vector3d, Eigen::Vector2d, Eigen::Vector3d>);

// ---------------------------------------------------------------------------
// Numerical behaviour, for float and double.
// ---------------------------------------------------------------------------

template <typename T>
T Eps() {
  return std::numeric_limits<T>::epsilon();
}

template <typename T, int N>
Eigen::Vector<T, N> RandomVector(std::mt19937& prng, T lo, T hi) {
  std::uniform_real_distribution<T> dist(lo, hi);
  return Eigen::Vector<T, N>::NullaryExpr([&](int) { return dist(prng); });
}

template <typename T>
class NnTest : public ::testing::Test {};

using RealTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(NnTest, RealTypes);

TYPED_TEST(NnTest, ReluClampsNegativesAndKeepsPositives) {
  using T = TypeParam;
  const Eigen::Vector4<T> x{T{-2}, T{-0.5}, T{0.5}, T{3}};
  const Eigen::Vector4<T> y = relu(x);
  EXPECT_EQ(y[0], T{0});
  EXPECT_EQ(y[1], T{0});
  EXPECT_EQ(y[2], x[2]);
  EXPECT_EQ(y[3], x[3]);
}

TYPED_TEST(NnTest, ReluIsIdempotent) {
  using T = TypeParam;
  std::mt19937 prng(101);
  for (int i = 0; i < 100; ++i) {
    const Eigen::Vector4<T> x = RandomVector<T, 4>(prng, T{-10}, T{10});
    const Eigen::Vector4<T> y = relu(x);
    EXPECT_EQ(relu(y), y);
  }
}

// Activations are elementwise and shape-preserving, so a batch matrix (one
// sample per column) goes through unchanged.
TYPED_TEST(NnTest, ActivationsApplyElementwiseToBatches) {
  using T = TypeParam;
  Eigen::Matrix<T, 2, 3> batch;
  batch << T{-2}, T{0}, T{2}, T{-0.5}, T{1}, T{-3};
  const Eigen::Matrix<T, 2, 3> rectified = relu(batch);
  const Eigen::Matrix<T, 2, 3> squashed = hs293go::tanh(batch);
  for (int r = 0; r < 2; ++r) {
    for (int c = 0; c < 3; ++c) {
      EXPECT_EQ(rectified(r, c), std::max(batch(r, c), T{0}));
      EXPECT_LE(std::abs(squashed(r, c) - std::tanh(batch(r, c))),
                T{4} * Eps<T>());
    }
  }
}

TYPED_TEST(NnTest, TanhMatchesStd) {
  using T = TypeParam;
  const Eigen::VectorX<T> x = Eigen::VectorX<T>::LinSpaced(101, T{-5}, T{5});
  const Eigen::VectorX<T> y = hs293go::tanh(x);
  for (int i = 0; i < x.size(); ++i) {
    EXPECT_LE(std::abs(y[i] - std::tanh(x[i])), T{4} * Eps<T>())
        << "x = " << x[i];
  }
}

TYPED_TEST(NnTest, TanhIsOdd) {
  using T = TypeParam;
  std::mt19937 prng(103);
  for (int i = 0; i < 100; ++i) {
    const Eigen::Vector4<T> x = RandomVector<T, 4>(prng, T{-5}, T{5});
    const Eigen::Vector4<T> lhs = hs293go::tanh(x);
    const Eigen::Vector4<T> rhs = -hs293go::tanh((-x).eval());
    EXPECT_LE((lhs - rhs).template lpNorm<Eigen::Infinity>(), T{4} * Eps<T>());
  }
}

TYPED_TEST(NnTest, EluMatchesDefinition) {
  using T = TypeParam;
  std::mt19937 prng(107);
  for (int i = 0; i < 100; ++i) {
    const Eigen::Vector4<T> x = RandomVector<T, 4>(prng, T{-10}, T{10});
    const Eigen::Vector4<T> y = elu(x);
    for (int j = 0; j < 4; ++j) {
      if (x[j] > T{0}) {
        EXPECT_EQ(y[j], x[j]);  // Positive branch is the identity, exactly.
      } else {
        EXPECT_LE(std::abs(y[j] - (std::exp(x[j]) - T{1})), T{4} * Eps<T>())
            << "x = " << x[j];
      }
    }
  }
}

// ELU is continuous at the branch point: both one-sided values approach 0, and
// exp(x) - 1 ~ x for small x, so |elu(x)| is bounded by ~|x| on both sides.
TYPED_TEST(NnTest, EluIsContinuousAtZero) {
  using T = TypeParam;
  const T h = std::sqrt(Eps<T>());
  const Eigen::Vector2<T> x{-h, h};
  const Eigen::Vector2<T> y = elu(x);
  EXPECT_LE(std::abs(y[0]), h);
  EXPECT_LE(std::abs(y[1]), h);
}

TYPED_TEST(NnTest, SigmoidMatchesNaiveFormOnModerateInputs) {
  using T = TypeParam;
  std::mt19937 prng(109);
  for (int i = 0; i < 100; ++i) {
    const Eigen::Vector4<T> x = RandomVector<T, 4>(prng, T{-10}, T{10});
    const Eigen::Vector4<T> y = sigmoid(x);
    for (int j = 0; j < 4; ++j) {
      const T naive = T{1} / (T{1} + std::exp(-x[j]));
      EXPECT_LE(std::abs(y[j] - naive), T{8} * Eps<T>()) << "x = " << x[j];
    }
  }
}

TYPED_TEST(NnTest, SigmoidFixedPoints) {
  using T = TypeParam;
  const Eigen::Vector<T, 1> zero{T{0}};
  EXPECT_EQ(sigmoid(zero)[0], T{0.5});
}

// sigmoid(x) + sigmoid(-x) == 1: the point symmetry about (0, 1/2).
TYPED_TEST(NnTest, SigmoidIsSymmetric) {
  using T = TypeParam;
  std::mt19937 prng(113);
  for (int i = 0; i < 100; ++i) {
    const Eigen::Vector4<T> x = RandomVector<T, 4>(prng, T{-20}, T{20});
    const Eigen::Vector4<T> s = sigmoid(x);
    const Eigen::Vector4<T> t = sigmoid((-x).eval());
    EXPECT_LE(((s + t).array() - T{1}).abs().maxCoeff(), T{4} * Eps<T>());
  }
}

// The regression the tanh form exists for: at |x| far beyond exp's overflow
// threshold (~88 in float) the result must saturate to exactly {0, 1} range
// with no inf/NaN anywhere.
TYPED_TEST(NnTest, SigmoidSaturatesWithoutOverflow) {
  using T = TypeParam;
  const Eigen::Vector2<T> x{T{-200}, T{200}};
  const Eigen::Vector2<T> y = sigmoid(x);
  ASSERT_TRUE(y.allFinite());
  EXPECT_LE(std::abs(y[0] - T{0}), Eps<T>());
  EXPECT_LE(std::abs(y[1] - T{1}), Eps<T>());
}

TYPED_TEST(NnTest, SoftmaxSumsToOne) {
  using T = TypeParam;
  std::mt19937 prng(127);
  for (int i = 0; i < 100; ++i) {
    // Include a large common offset so the max-shift path is exercised.
    const Eigen::Vector4<T> x =
        (RandomVector<T, 4>(prng, T{-5}, T{5}).array() + T{50}).matrix();
    const Eigen::Vector4<T> s = softmax(x);
    EXPECT_TRUE((s.array() >= T{0}).all());
    EXPECT_LE(std::abs(s.sum() - T{1}), T{8} * Eps<T>());
  }
}

// softmax(x + c) == softmax(x): invariance under a uniform logit shift.
TYPED_TEST(NnTest, SoftmaxIsShiftInvariant) {
  using T = TypeParam;
  std::mt19937 prng(131);
  for (int i = 0; i < 100; ++i) {
    const Eigen::Vector4<T> x = RandomVector<T, 4>(prng, T{-5}, T{5});
    const Eigen::Vector4<T> shifted = (x.array() + T{3}).matrix();
    EXPECT_LE(
        (softmax(x) - softmax(shifted)).template lpNorm<Eigen::Infinity>(),
        T{8} * Eps<T>());
  }
}

TYPED_TEST(NnTest, SoftmaxOfUniformLogitsIsUniform) {
  using T = TypeParam;
  const Eigen::Vector4<T> x = Eigen::Vector4<T>::Constant(T{7});
  const Eigen::Vector4<T> s = softmax(x);
  EXPECT_LE((s.array() - T{0.25}).abs().maxCoeff(), T{2} * Eps<T>());
}

// Logits far beyond exp's overflow threshold: the naive form produces inf/inf
// = NaN; the shifted form saturates to a one-hot distribution.
TYPED_TEST(NnTest, SoftmaxHandlesExtremeLogits) {
  using T = TypeParam;
  const Eigen::Vector3<T> x{T{1000}, T{0}, T{-1000}};
  const Eigen::Vector3<T> s = softmax(x);
  ASSERT_TRUE(s.allFinite());
  EXPECT_LE(std::abs(s[0] - T{1}), T{4} * Eps<T>());
  EXPECT_LE(s[1], Eps<T>());
  EXPECT_LE(s[2], Eps<T>());
  EXPECT_LE(std::abs(s.sum() - T{1}), T{4} * Eps<T>());
}

TYPED_TEST(NnTest, StandardizeMatchesFormula) {
  using T = TypeParam;
  std::mt19937 prng(137);
  for (int i = 0; i < 100; ++i) {
    const Eigen::Vector4<T> x = RandomVector<T, 4>(prng, T{-10}, T{10});
    const Eigen::Vector4<T> mean = RandomVector<T, 4>(prng, T{-2}, T{2});
    const Eigen::Vector4<T> stddev = RandomVector<T, 4>(prng, T{0.5}, T{3});
    const Eigen::Vector4<T> y = standardize(x, mean, stddev);
    for (int j = 0; j < 4; ++j) {
      EXPECT_EQ(y[j], (x[j] - mean[j]) / stddev[j]);
    }
  }
}

TYPED_TEST(NnTest, StandardizeWithZeroMeanUnitStdIsIdentity) {
  using T = TypeParam;
  const Eigen::Vector4<T> x{T{-2}, T{-0.5}, T{0.5}, T{3}};
  const Eigen::Vector4<T> y =
      standardize(x, Eigen::Vector4<T>::Zero(), Eigen::Vector4<T>::Ones());
  EXPECT_EQ(y, x);
}

TYPED_TEST(NnTest, LinearMatchesManualComputation) {
  using T = TypeParam;
  Eigen::Matrix<T, 2, 3> wts;
  wts << T{1}, T{-2}, T{0.5}, T{0}, T{3}, T{-1};
  const Eigen::Vector2<T> biases{T{0.25}, T{-4}};
  const Eigen::Vector3<T> x{T{2}, T{-1}, T{4}};
  const Eigen::Vector2<T> y = linear(x, wts, biases);
  for (int i = 0; i < 2; ++i) {
    T expected = biases[i];
    for (int j = 0; j < 3; ++j) {
      expected += wts(i, j) * x[j];
    }
    EXPECT_LE(std::abs(y[i] - expected), T{8} * Eps<T>() * std::abs(expected));
  }
}

TYPED_TEST(NnTest, LinearWithIdentityWeightsAndZeroBiasIsIdentity) {
  using T = TypeParam;
  const Eigen::Vector3<T> x{T{2}, T{-1}, T{4}};
  const Eigen::Vector3<T> y =
      linear(x, Eigen::Matrix3<T>::Identity(), Eigen::Vector3<T>::Zero());
  EXPECT_EQ(y, x);
}

// The Dynamic path through the size concepts: everything checked at run time.
TYPED_TEST(NnTest, LinearAcceptsDynamicSizes) {
  using T = TypeParam;
  std::mt19937 prng(139);
  Eigen::MatrixX<T> wts(2, 5);
  Eigen::VectorX<T> x(5);
  std::uniform_real_distribution<T> dist(T{-1}, T{1});
  for (int i = 0; i < wts.size(); ++i) {
    wts(i) = dist(prng);
  }
  for (int i = 0; i < x.size(); ++i) {
    x[i] = dist(prng);
  }
  const Eigen::VectorX<T> biases = Eigen::VectorX<T>::Zero(2);
  const Eigen::VectorX<T> y = linear(x, wts, biases);
  ASSERT_EQ(y.size(), 2);
  EXPECT_LE((y - (wts * x)).template lpNorm<Eigen::Infinity>(),
            T{8} * Eps<T>());
}

// ---------------------------------------------------------------------------
// Jacobians by forward-mode AutoDiff, double only: every op is differentiated
// with Eigen's AutoDiffScalar flowing through the library code unchanged, and
// the result is compared against the closed-form derivative.
// ---------------------------------------------------------------------------

using AD3 = Eigen::AutoDiffScalar<Eigen::Vector3d>;

constexpr double kAdTol = 1e-12;
constexpr int kAdTrials = 100;

// Three independent AD variables seeded from x (identity Jacobian).
Eigen::Vector3<AD3> Variable3(const Eigen::Vector3d& x) {
  return Eigen::Vector3<AD3>::NullaryExpr(
      [&](int i) { return AD3{x[i], Eigen::Vector3d::Unit(i)}; });
}

template <typename ADVec>
auto ExtractJacobian(const ADVec& y) {
  Eigen::Matrix<double, ADVec::RowsAtCompileTime, 3> jac;
  for (int r = 0; r < ADVec::RowsAtCompileTime; ++r) {
    jac.row(r) = y[r].derivatives().transpose();
  }
  return jac;
}

Eigen::Vector3d RandomVector3AwayFromZero(std::mt19937& prng) {
  std::uniform_real_distribution<double> mag(0.1, 5.0);
  std::bernoulli_distribution sign;
  return Eigen::Vector3d::NullaryExpr(
      [&](int) { return sign(prng) ? mag(prng) : -mag(prng); });
}

// d relu / dx = diag(x_i > 0), sampled away from the kink at 0.
TEST(NnJacobianAutoDiffTest, Relu) {
  std::mt19937 prng(211);
  for (int i = 0; i < kAdTrials; ++i) {
    const Eigen::Vector3d x = RandomVector3AwayFromZero(prng);
    const Eigen::Matrix3d jac = ExtractJacobian(relu(Variable3(x)));
    const Eigen::Matrix3d expected =
        (x.array() > 0.0).select(1.0, Eigen::Vector3d::Zero()).asDiagonal();
    ASSERT_LE((jac - expected).norm(), kAdTol) << "x = " << x.transpose();
  }
}

// d tanh / dx = diag(1 - tanh(x)^2).
TEST(NnJacobianAutoDiffTest, Tanh) {
  std::mt19937 prng(223);
  for (int i = 0; i < kAdTrials; ++i) {
    const Eigen::Vector3d x = RandomVector3AwayFromZero(prng);
    const Eigen::Matrix3d jac = ExtractJacobian(hs293go::tanh(Variable3(x)));
    const Eigen::Matrix3d expected =
        (1.0 - x.array().tanh().square()).matrix().asDiagonal();
    ASSERT_LE((jac - expected).norm(), kAdTol) << "x = " << x.transpose();
  }
}

// d elu / dx = diag(x_i > 0 ? 1 : exp(x_i)).
TEST(NnJacobianAutoDiffTest, Elu) {
  std::mt19937 prng(227);
  for (int i = 0; i < kAdTrials; ++i) {
    const Eigen::Vector3d x = RandomVector3AwayFromZero(prng);
    const Eigen::Matrix3d jac = ExtractJacobian(elu(Variable3(x)));
    const Eigen::Matrix3d expected =
        (x.array() > 0.0).select(1.0, x.array().exp()).matrix().asDiagonal();
    ASSERT_LE((jac - expected).norm(), kAdTol) << "x = " << x.transpose();
  }
}

// d sigmoid / dx = diag(s (1 - s)).
TEST(NnJacobianAutoDiffTest, Sigmoid) {
  std::mt19937 prng(229);
  for (int i = 0; i < kAdTrials; ++i) {
    const Eigen::Vector3d x = RandomVector3AwayFromZero(prng);
    const Eigen::Matrix3d jac = ExtractJacobian(sigmoid(Variable3(x)));
    const Eigen::Vector3d s = sigmoid(x);
    const Eigen::Matrix3d expected =
        (s.array() * (1.0 - s.array())).matrix().asDiagonal();
    ASSERT_LE((jac - expected).norm(), kAdTol) << "x = " << x.transpose();
  }
}

// In saturation the derivative must underflow smoothly toward zero (AutoDiff
// evaluates it as ~1/cosh^2, so it is a tiny positive number, not an exact 0);
// the naive sigmoid would produce NaN here via an inf intermediate.
TEST(NnJacobianAutoDiffTest, SigmoidSaturationHasFiniteDerivatives) {
  const Eigen::Vector3d x{-500.0, 0.0, 500.0};
  const Eigen::Matrix3d jac = ExtractJacobian(sigmoid(Variable3(x)));
  ASSERT_TRUE(jac.allFinite());
  EXPECT_LE(jac(0, 0), 1e-200);
  EXPECT_LE(jac(2, 2), 1e-200);
  EXPECT_NEAR(jac(1, 1), 0.25, kAdTol);
}

// d softmax / dx = diag(s) - s s^T.
TEST(NnJacobianAutoDiffTest, Softmax) {
  std::mt19937 prng(233);
  for (int i = 0; i < kAdTrials; ++i) {
    const Eigen::Vector3d x = RandomVector3AwayFromZero(prng);
    const Eigen::Matrix3d jac = ExtractJacobian(softmax(Variable3(x)));
    const Eigen::Vector3d s = softmax(x);
    const Eigen::Matrix3d expected =
        Eigen::Matrix3d(s.asDiagonal()) - s * s.transpose();
    ASSERT_LE((jac - expected).norm(), kAdTol) << "x = " << x.transpose();
  }
}

// d linear / dx = wts. The weights and biases stay plain double while x
// carries AutoDiffScalar -- this pins the mixed-scalar support the linear
// overload documents.
TEST(NnJacobianAutoDiffTest, LinearJacobianIsWeights) {
  std::mt19937 prng(239);
  std::uniform_real_distribution<double> dist(-2.0, 2.0);
  for (int i = 0; i < kAdTrials; ++i) {
    Eigen::Matrix<double, 2, 3> wts;
    for (int j = 0; j < wts.size(); ++j) {
      wts(j) = dist(prng);
    }
    const Eigen::Vector2d biases{dist(prng), dist(prng)};
    const Eigen::Vector3d x = RandomVector3AwayFromZero(prng);
    const auto jac = ExtractJacobian(linear(Variable3(x), wts, biases));
    ASSERT_LE((jac - wts).norm(), kAdTol);
  }
}

// d standardize / dx = diag(1 / stddev), again with plain-double statistics.
TEST(NnJacobianAutoDiffTest, StandardizeJacobianIsInverseStddev) {
  std::mt19937 prng(241);
  std::uniform_real_distribution<double> dist(0.5, 3.0);
  for (int i = 0; i < kAdTrials; ++i) {
    const Eigen::Vector3d x = RandomVector3AwayFromZero(prng);
    const Eigen::Vector3d mean = RandomVector3AwayFromZero(prng);
    const Eigen::Vector3d stddev{dist(prng), dist(prng), dist(prng)};
    const Eigen::Matrix3d jac =
        ExtractJacobian(standardize(Variable3(x), mean, stddev));
    const Eigen::Matrix3d expected = stddev.cwiseInverse().asDiagonal();
    ASSERT_LE((jac - expected).norm(), kAdTol);
  }
}

}  // namespace
