#include <stan/math/prim/scal.hpp>
#include <gtest/gtest.h>
#include <limits>

TEST(MathFunctions, is_integer) {
  using stan::math::is_integer;
  double inf = std::numeric_limits<double>::infinity();
  double neg_inf = -std::numeric_limits<double>::infinity();
  double nan = std::numeric_limits<double>::quiet_NaN();
  double dbl_min = std::numeric_limits<double>::min();
  double dbl_max = std::numeric_limits<double>::max();
  int int_min = std::numeric_limits<int>::min();
  int int_max = std::numeric_limits<int>::max();

  EXPECT_TRUE(is_integer(1.0));
  EXPECT_TRUE(is_integer(-35.0));
  EXPECT_TRUE(is_integer(0.0));

  EXPECT_FALSE(is_integer(-2.5));
  EXPECT_FALSE(is_integer(31.3));
  EXPECT_FALSE(is_integer(0.5));

  EXPECT_TRUE(is_integer(int_min));
  EXPECT_TRUE(is_integer(int_max));
  EXPECT_TRUE(is_integer(dbl_max));

  EXPECT_FALSE(is_integer(inf));
  EXPECT_FALSE(is_integer(neg_inf));
  EXPECT_FALSE(is_integer(nan));
  EXPECT_FALSE(is_integer(dbl_min));
}
