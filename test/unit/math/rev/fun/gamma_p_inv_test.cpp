#include <stan/math/rev.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/rev/fun/util.hpp>
#include <limits>

TEST(AgradRev, gamma_p_inv) {
  using stan::math::gamma_p_inv;
  using stan::math::gamma_p_inv;
  using stan::math::var;

  var a = 4;
  var p = 0.25;

  var result = gamma_p_inv(a, p);
  result.grad();

  EXPECT_FLOAT_EQ(result.val(), 2.53532021190009318115914235);
  EXPECT_FLOAT_EQ(a.adj(), 0.8224173185260604);
  EXPECT_FLOAT_EQ(p.adj(), 4.646526006166049);
}

TEST(AgradRev, gamma_icdf) {
  using stan::math::gamma_icdf;
  using stan::math::var;

  var a = 4;
  var b = 2;
  var p = 0.25;

  var result = gamma_icdf(p,a,b);
  //result.grad();

  EXPECT_FLOAT_EQ(result.val(), 1.267660106);
  //EXPECT_FLOAT_EQ(a.adj(), 0.8224173185260604);
  //EXPECT_FLOAT_EQ(p.adj(), 4.646526006166049);
}