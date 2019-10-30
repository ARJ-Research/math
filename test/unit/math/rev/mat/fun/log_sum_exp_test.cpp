#include <stan/math/rev/mat.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <test/unit/math/rev/mat/util.hpp>
#include <cmath>
#include <limits>
#include <vector>

using Eigen::Dynamic;
using Eigen::Matrix;
using stan::math::var;
using std::vector;

template <int R, int C>
void test_log_sum_exp_matrix(const Matrix<double, R, C>& m) {
  using std::exp;

  vector<var> x_expected(m.size());
  for (int i = 0; i < m.size(); ++i)
    x_expected[i] = m(i);
  var sum_exp(0);
  for (int i = 0; i < m.size(); ++i)
    sum_exp += exp(x_expected[i]);
  var f_expected = log(sum_exp);
  double val_expected = f_expected.val();
  vector<double> g_expected(m.size());
  f_expected.grad(x_expected, g_expected);

  vector<var> x(m.size());
  for (int i = 0; i < m.size(); ++i)
    x[i] = m(i);
  Matrix<var, R, C> mv(m.rows(), m.cols());
  for (int i = 0; i < m.size(); ++i)
    mv(i) = x[i];
  var f = log_sum_exp(mv);
  double val = f.val();
  vector<double> g(m.size());
  f.grad(x, g);

  EXPECT_FLOAT_EQ(val_expected, val);
  EXPECT_EQ(g_expected.size(), g.size());
  for (size_t i = 0; i < g.size(); ++i)
    EXPECT_FLOAT_EQ(g_expected[i], g[i]);
}

TEST(AgradRev, logSumExpMatrix) {
  Matrix<double, Dynamic, 1> a(2);
  a << 5, 2;
  test_log_sum_exp_matrix(a);

  Matrix<double, Dynamic, 1> b(1);
  b << 0;
  test_log_sum_exp_matrix(b);

  Matrix<double, 1, Dynamic> c(3);
  c << 4.9, -12, 1.7;
  test_log_sum_exp_matrix(c);

  Matrix<double, Dynamic, Dynamic> d(3, 2);
  d << -1, -2, -4, 5, 6, 4;
  test_log_sum_exp_matrix(d);

  Matrix<double, Dynamic, 1> e(2);
  e << 4.9, -std::numeric_limits<double>::infinity();
  test_log_sum_exp_matrix(e);
}

TEST(AgradRev, logSumExpMatrixInf) {
  var x(-std::numeric_limits<double>::infinity());
  var y(-std::numeric_limits<double>::infinity());
  Matrix<var, Dynamic, 1> mv(2);
  mv[0] = x;
  mv[1] = y;
  var s = log_sum_exp(mv);
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), s.val());
  AVEC v = createAVEC(x, y);
  VEC grad_s;
  EXPECT_NO_THROW(s.grad(v, grad_s));
}

TEST(AgradRevMatrix, check_varis_on_stack) {
  stan::math::vector_v a(2);
  a << 5, 2;
  test::check_varis_on_stack(stan::math::log_sum_exp(a));
}

TEST(AgradRev, log_sum_exp_vector) {
  // simple test
  AVEC x;
  x.push_back(5.0);
  x.push_back(2.0);

  AVAR f = log_sum_exp(x);
  EXPECT_FLOAT_EQ(std::log(std::exp(5) + std::exp(2)), f.val());

  VEC grad_f;
  f.grad(x, grad_f);
  EXPECT_FLOAT_EQ(std::exp(5.0) / (std::exp(5.0) + std::exp(2.0)), grad_f[0]);
  EXPECT_FLOAT_EQ(std::exp(2.0) / (std::exp(5.0) + std::exp(2.0)), grad_f[1]);

  // longer test
  x.clear();
  x.push_back(1.0);
  x.push_back(2.0);
  x.push_back(3.0);
  x.push_back(4.0);
  x.push_back(5.0);
  f = log_sum_exp(x);
  double expected_log_sum_exp = std::log(std::exp(1) + std::exp(2) + std::exp(3)
                                         + std::exp(4) + std::exp(5));
  EXPECT_FLOAT_EQ(expected_log_sum_exp, f.val());

  grad_f.clear();
  f.grad(x, grad_f);
  EXPECT_FLOAT_EQ(std::exp(1.0) / exp(expected_log_sum_exp), grad_f[0]);
  EXPECT_FLOAT_EQ(std::exp(2.0) / exp(expected_log_sum_exp), grad_f[1]);
  EXPECT_FLOAT_EQ(std::exp(3.0) / exp(expected_log_sum_exp), grad_f[2]);
  EXPECT_FLOAT_EQ(std::exp(4.0) / exp(expected_log_sum_exp), grad_f[3]);
  EXPECT_FLOAT_EQ(std::exp(5.0) / exp(expected_log_sum_exp), grad_f[4]);

  // underflow example
  x.clear();
  x.push_back(1000.0);
  x.push_back(10.0);
  f = log_sum_exp(x);
  EXPECT_FLOAT_EQ(std::log(std::exp(0.0) + std::exp(-990.0)) + 1000.0, f.val());

  f.grad(x, grad_f);
  EXPECT_FLOAT_EQ(
      std::exp(1000.0 - (std::log(std::exp(0.0) + std::exp(-999.0)) + 1000)),
      grad_f[0]);
  EXPECT_FLOAT_EQ(
      std::exp(10.0 - (std::log(std::exp(0.0) + std::exp(-999.0)) + 1000)),
      grad_f[1]);

  // longer underflow example
  x.clear();
  x.push_back(800.0);
  x.push_back(900.0);
  x.push_back(10.0);
  x.push_back(0.0);
  x.push_back(-100.0);
  f = log_sum_exp(x);
  expected_log_sum_exp
      = std::log(std::exp(0.0) + std::exp(-100) + std::exp(-890.0)
                 + std::exp(-900.0) + std::exp(-1000.0))
        + 900.0;
  EXPECT_FLOAT_EQ(expected_log_sum_exp, f.val());

  f.grad(x, grad_f);
  EXPECT_FLOAT_EQ(std::exp(800.0 - expected_log_sum_exp), grad_f[0]);
  EXPECT_FLOAT_EQ(std::exp(900.0 - expected_log_sum_exp), grad_f[1]);
  EXPECT_FLOAT_EQ(std::exp(10.0 - expected_log_sum_exp), grad_f[2]);
  EXPECT_FLOAT_EQ(std::exp(0.0 - expected_log_sum_exp), grad_f[3]);
  EXPECT_FLOAT_EQ(std::exp(-100.0 - expected_log_sum_exp), grad_f[4]);
}

TEST(AgradRev, log_sum_exp_vec_1) {
  using stan::math::log_sum_exp;
  AVAR a(5.0);
  AVEC as = createAVEC(a);
  AVEC x = createAVEC(a);
  AVAR f = log_sum_exp(as);
  EXPECT_FLOAT_EQ(5.0, f.val());
  VEC g;
  f.grad(x, g);
  EXPECT_EQ(1U, g.size());
  EXPECT_FLOAT_EQ(1.0, g[0]);
}

TEST(AgradRev, log_sum_exp_vec_2) {
  using stan::math::log_sum_exp;
  AVAR a(5.0);
  AVAR b(-7.0);
  AVEC as = createAVEC(a, b);
  AVEC x = createAVEC(a, b);
  AVAR f = log_sum_exp(as);
  EXPECT_FLOAT_EQ(log(exp(5.0) + exp(-7.0)), f.val());
  VEC g;
  f.grad(x, g);

  double f_val = f.val();

  AVAR a2(5.0);
  AVAR b2(-7.0);
  AVEC as2 = createAVEC(a2, b2);
  AVEC x2 = createAVEC(a2, b2);
  AVAR f2 = stan::math::log(stan::math::exp(a2) + stan::math::exp(b2));
  VEC g2;
  f2.grad(x2, g2);

  EXPECT_FLOAT_EQ(f2.val(), f_val);
  EXPECT_EQ(2U, g.size());
  EXPECT_EQ(2U, g2.size());
  EXPECT_FLOAT_EQ(g[0], g2[0]);
  EXPECT_FLOAT_EQ(g[1], g2[1]);
}

TEST(AgradRev, log_sum_exp_vec_3) {
  using stan::math::log_sum_exp;
  AVAR a(5.0);
  AVAR b(-7.0);
  AVAR c(2.3);
  AVEC as = createAVEC(a, b, c);
  AVEC x = createAVEC(a, b, c);
  AVAR f = log_sum_exp(as);
  EXPECT_FLOAT_EQ(log(exp(5.0) + exp(-7.0) + exp(2.3)), f.val());
  VEC g;
  f.grad(x, g);

  double f_val = f.val();

  AVAR a2(5.0);
  AVAR b2(-7.0);
  AVAR c2(2.3);
  AVEC as2 = createAVEC(a2, b2, c2);
  AVEC x2 = createAVEC(a2, b2, c2);
  AVAR f2 = log(exp(a2) + exp(b2) + exp(c2));
  VEC g2;
  f2.grad(x2, g2);

  EXPECT_FLOAT_EQ(f2.val(), f_val);
  EXPECT_EQ(3U, g.size());
  EXPECT_EQ(3U, g2.size());
  EXPECT_FLOAT_EQ(g[0], g2[0]);
  EXPECT_FLOAT_EQ(g[1], g2[1]);
  EXPECT_FLOAT_EQ(g[2], g2[2]);
}

TEST(AgradRev, check_varis_on_stack) {
  std::vector<stan::math::var> x;
  x.push_back(5.0);
  x.push_back(2.0);
  test::check_varis_on_stack(stan::math::log_sum_exp(x));
}