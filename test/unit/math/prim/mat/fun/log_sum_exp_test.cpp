#include <stan/math/prim/mat.hpp>
#include <gtest/gtest.h>
#include <limits>

template <int R, int C>
void test_log_sum_exp(const Eigen::Matrix<double, R, C>& as) {
  using stan::math::log_sum_exp;
  using std::exp;
  using std::log;
  double sum_exp = 0.0;
  for (int n = 0; n < as.size(); ++n)
    sum_exp += exp(as(n));
  EXPECT_FLOAT_EQ(log(sum_exp), log_sum_exp(as));
}

TEST(MathFunctions, log_sum_exp) {
  using Eigen::Dynamic;
  using Eigen::Matrix;
  using stan::math::log_sum_exp;

  Matrix<double, Dynamic, Dynamic> m(3, 2);
  m << 1, 2, 3, 4, 5, 6;
  test_log_sum_exp(m);

  Matrix<double, Dynamic, 1> v(3);
  v << 1, 2, 3;
  test_log_sum_exp(v);

  Matrix<double, 1, Dynamic> rv(3);
  rv << 1, 2, 3;
  test_log_sum_exp(rv);

  Matrix<double, Dynamic, Dynamic> m_trivial(1, 1);
  m_trivial << 2;
  EXPECT_FLOAT_EQ(2, log_sum_exp(m_trivial));

  Matrix<double, Dynamic, 1> i(3);
  i << 1, 2, -std::numeric_limits<double>::infinity();
  test_log_sum_exp(i);

  Matrix<double, Dynamic, 1> ii(1);
  ii << -std::numeric_limits<double>::infinity();
  test_log_sum_exp(ii);

  std::vector<double> stv{1,2,3};
  EXPECT_FLOAT_EQ(log_sum_exp(v), log_sum_exp(stv));

  std::vector<Matrix<double, Dynamic, 1>> st_i{i,ii,v,v};
  double result = log_sum_exp(i) + log_sum_exp(ii) + log_sum_exp(v) + log_sum_exp(v);
  EXPECT_FLOAT_EQ(result, log_sum_exp(st_i));

  std::vector<std::vector<double>> st_stv{stv,stv,stv};
  EXPECT_FLOAT_EQ(log_sum_exp(stv) * 3, log_sum_exp(st_stv));
}
