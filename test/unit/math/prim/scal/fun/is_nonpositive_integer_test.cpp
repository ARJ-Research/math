#include <stan/math/prim/scal.hpp>
#include <gtest/gtest.h>
#include <limits>

TEST(MathFunctions, is_nonpositive_integer) {
  using stan::math::is_nonpositive_integer;
  double inf = std::numeric_limits<double>::infinity();
  double neg_inf = -std::numeric_limits<double>::infinity();
  double nan = std::numeric_limits<double>::quiet_NaN();
  double dbl_min = std::numeric_limits<double>::min();
  double dbl_max = std::numeric_limits<double>::max();
  int int_min = std::numeric_limits<int>::min();
  int int_max = std::numeric_limits<int>::max();

  EXPECT_TRUE(is_nonpositive_integer(-35.0));
  EXPECT_TRUE(is_nonpositive_integer(0.0));
  EXPECT_TRUE(is_nonpositive_integer(-1.0));

  EXPECT_FALSE(is_nonpositive_integer(1.0));
  EXPECT_FALSE(is_nonpositive_integer(-2.5));
  EXPECT_FALSE(is_nonpositive_integer(31.3));
  EXPECT_FALSE(is_nonpositive_integer(0.5));

  EXPECT_TRUE(is_nonpositive_integer(int_min));

  EXPECT_FALSE(is_nonpositive_integer(int_max));
  EXPECT_FALSE(is_nonpositive_integer(dbl_max));
  EXPECT_FALSE(is_nonpositive_integer(inf));
  EXPECT_FALSE(is_nonpositive_integer(neg_inf));
  EXPECT_FALSE(is_nonpositive_integer(nan));
  EXPECT_FALSE(is_nonpositive_integer(dbl_min));
}
