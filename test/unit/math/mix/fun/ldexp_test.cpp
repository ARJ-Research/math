#include <test/unit/math/test_ad.hpp>
#include <stan/math/prim.hpp>

TEST(mathMixScalFun, ldexp) {
  auto f = [](const auto& x1) { return stan::math::ldexp(x1, 5); };

  stan::test::expect_ad(f, 3.1);
  stan::test::expect_ad(f, 0.0);
  stan::test::expect_ad(f, -1.5);
  stan::test::expect_ad(f, stan::math::INFTY);
  stan::test::expect_ad(f, stan::math::NOT_A_NUMBER);
}

TEST(mathMixScalFun, ldexp_vec) {
  auto f = [](const auto& x1, const auto& x2) {
    using stan::math::ldexp;
    return ldexp(x1, x2);
  };

  Eigen::VectorXd in1(2);
  in1 << 0.6, 2.9;
  Eigen::VectorXi in2(2);
  in2 << 4, 2;
  stan::test::expect_ad_vectorized_binary(f, in1, in2);
}