#include <stan/math/prim.hpp>
#include <test/unit/math/prim/fun/binary_scalar_tester.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <limits>

TEST(MathFunctions, ldexp_double) {
  using stan::math::ldexp;

  EXPECT_FLOAT_EQ(0.0, ldexp(0.0, 5));
  EXPECT_FLOAT_EQ(32.0, ldexp(1.0, 5));
  EXPECT_FLOAT_EQ(64.0, ldexp(2.0, 5));
  EXPECT_FLOAT_EQ(96.0, ldexp(3.0, 5));

  EXPECT_FLOAT_EQ(-32.0, ldexp(-1.0, 5));
  EXPECT_FLOAT_EQ(-64.0, ldexp(-2.0, 5));
  EXPECT_FLOAT_EQ(-96.0, ldexp(-3.0, 5));
}

TEST(MathFunctions, ldexp_int) {
  using stan::math::ldexp;

  EXPECT_FLOAT_EQ(0.0, ldexp(static_cast<int>(0), 5));
  EXPECT_FLOAT_EQ(32.0, ldexp(static_cast<int>(1), 5));
  EXPECT_FLOAT_EQ(64.0, ldexp(static_cast<int>(2), 5));
  EXPECT_FLOAT_EQ(96.0, ldexp(static_cast<int>(3), 5));

  EXPECT_FLOAT_EQ(-32.0, ldexp(static_cast<int>(-1), 5));
  EXPECT_FLOAT_EQ(-64.0, ldexp(static_cast<int>(-2), 5));
  EXPECT_FLOAT_EQ(-96.0, ldexp(static_cast<int>(-3), 5));
}

TEST(MathFunctions, ldexp_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();

  EXPECT_TRUE(std::isnan(stan::math::ldexp(nan, 5)));
}

TEST(MathFunctions, ldexp_vec) {
  auto f = [](const auto& x1, const auto& x2) {
    using stan::math::ldexp;
    return ldexp(x1, x2);
  };

  Eigen::VectorXd in1(3);
  in1 << 1.8, 3.24, 1.8;
  Eigen::VectorXd in2(3);
  in2 << -1.3, 0.7, 2.8;
  stan::test::binary_scalar_tester(f, in1, in2);
}
