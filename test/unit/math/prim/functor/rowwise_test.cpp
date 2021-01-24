#include <stan/math/prim.hpp>
#include <test/unit/util.hpp>
#include <gtest/gtest.h>
#include <iostream>

TEST(MathFunctions, rowwise_to_vector) {
  using stan::math::matrix_d;
  using stan::math::rowwise;
  using stan::math::log_sum_exp;
  using stan::math::vector_d;

  matrix_d in = matrix_d::Random(5,5);
  matrix_d in2 = matrix_d::Random(3,5);

  decltype(auto) fun = [&](const auto& x){return log_sum_exp((x));};

  vector_d loop_out(5);
  for(int i = 0; i < 5; ++i) {
    loop_out[i] = log_sum_exp(in.row(i));
  }
  vector_d out = rowwise(in, fun);

  decltype(auto) fun2 = [&](const auto& x, const auto& y){return log_sum_exp((x+y));};

  EXPECT_MATRIX_EQ(out, loop_out);
  EXPECT_THROW(rowwise(in, in2, fun2), std::invalid_argument);
}

TEST(MathFunctions, rowwise_to_matrix) {
  using stan::math::matrix_d;
  using stan::math::rowwise;
  using stan::math::log_softmax;
  using stan::math::row_vector_d;

  matrix_d in1 = matrix_d::Random(2,2);
  row_vector_d in2 = row_vector_d::Random(2);

  decltype(auto) fun = [&](const auto& x, const auto& y){ return log_softmax(x - y); };

  matrix_d out = rowwise(in1, fun, in2);

  matrix_d loop_out(2,2);
  for(int i = 0; i < 2; ++i) {
    loop_out.row(i) = log_softmax(in1.row(i) - in2);
  }

  EXPECT_MATRIX_EQ(out, loop_out);
}