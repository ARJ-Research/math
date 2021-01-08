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
  std::cout << rowwise(in, [&](const auto& x){return log_sum_exp((x));}) << std::endl;

  //EXPECT_FLOAT_EQ(out[0], log_sum_exp(in.row(0)));
}

TEST(MathFunctions, rowwise_to_matrix) {
  using stan::math::matrix_d;
  using stan::math::rowwise;
  using stan::math::log_sum_exp;
  using stan::math::row_vector_d;

  matrix_d in1 = matrix_d::Random(5,5);
  row_vector_d in2 = row_vector_d::Random(5);
 std::cout << rowwise(in1, [&](const auto& x, const auto& y){return x - y;}, in2) << std::endl;

  matrix_d loop_out(5,5);
  for(int i = 0; i < 5; ++i) {
    loop_out.row(i) = in1.row(i) - in2;
  }

 // EXPECT_MATRIX_EQ(out, loop_out);
}