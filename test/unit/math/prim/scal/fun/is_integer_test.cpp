#include <stan/math/prim/scal.hpp>
#include <gtest/gtest.h>
#include <limits>

TEST(MathFunctions, is_integer) {
  using stan::math::is_integer;

  double infinity = std::numeric_limits<double>::infinity();
  double nan = std::numeric_limits<double>::quiet_NaN();
  int max = std::numeric_limits<int>::max();
  int int_nan = std::numeric_limits<int>::quiet_NaN();

  EXPECT_TRUE(is_integer(1));
  EXPECT_TRUE(is_integer(-10));
  EXPECT_FALSE(is_integer(2.5));
  EXPECT_FALSE(is_integer(-5.5));

  EXPECT_TRUE(is_integer(max));
  EXPECT_TRUE(is_integer(int_nan));
  EXPECT_FALSE(is_integer(infinity));
  EXPECT_FALSE(is_integer(nan));
}
