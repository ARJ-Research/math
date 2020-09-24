#include <stan/math.hpp>
#include <test/unit/util.hpp>
#include <iostream>
#include <gtest/gtest.h>

TEST(MathFunctions, parall_map_prim) {
  using Eigen::VectorXd;
  Eigen::VectorXd in1_par = Eigen::VectorXd::Random(1000);
  Eigen::VectorXd in1_ser = in1_par;
  Eigen::VectorXd out_par(1000);
  Eigen::VectorXd out_ser(1000);


  // Functor defining how inputs should be indexed
  auto ind_f = [&](int i, const auto& fun,
                    const auto& x) {
    return fun(x.coeffRef(i));
  };

  // Functor defining function to be applied to indexed arguments
  auto f = [&](const auto& x) { return std::exp(x); };

  // Boolean template parameter to enable ranged parallelism
  stan::math::parallel_map<false>(f, ind_f, std::forward<Eigen::VectorXd>(out_par), 1,
                         in1_par);
  EXPECT_MATRIX_FLOAT_EQ(out_par, in1_par.array().exp().matrix());
}

TEST(MathFunctions, parall_map_prim_2d) {
  using Eigen::MatrixXd;
  Eigen::MatrixXd in1_par = Eigen::MatrixXd::Random(100,10);
  Eigen::MatrixXd in1_ser = in1_par;
  Eigen::MatrixXd out_par(100,10);
  Eigen::MatrixXd out_ser(100,10);


  // Functor defining how inputs should be indexed
  auto ind_f = [&](int i, int j, const auto& fun,
                    const auto& x) {
    return fun(x.coeffRef(i, j));
  };

  // Functor defining function to be applied to indexed arguments
  auto f = [&](const auto& x) { return std::exp(x); };

  // Boolean template parameter to enable ranged parallelism
  stan::math::parallel_map<false>(f, ind_f, std::forward<Eigen::MatrixXd>(out_par), 1,
                         in1_par);
  EXPECT_MATRIX_FLOAT_EQ(out_par, in1_par.array().exp().matrix());
}

TEST(MathFunctions, parall_map_ranged) {
  using Eigen::VectorXd;
  Eigen::VectorXd in1_par = Eigen::VectorXd::Random(1000);
  Eigen::VectorXd in1_ser = in1_par;
  Eigen::VectorXd out_par(1000);
  Eigen::VectorXd out_ser(1000);


  // Functor defining how inputs should be indexed
  auto ind_f = [&](int begin, int size, const auto& fun,
                    const auto& x) {
    return fun(x.segment(begin, size));
  };

  // Functor defining function to be applied to indexed arguments
  auto f = [&](const auto& x) { return x.array().exp().matrix(); };

  // Boolean template parameter to enable ranged parallelism
  stan::math::parallel_map<true>(f, ind_f, std::forward<Eigen::VectorXd>(out_par), 1,
                         in1_par);
  EXPECT_MATRIX_FLOAT_EQ(out_par, in1_par.array().exp().matrix());
}

TEST(MathFunctions, parall_map_ranged_2d) {
  using Eigen::MatrixXd;
  Eigen::MatrixXd in1_par = Eigen::MatrixXd::Random(100,10);
  Eigen::MatrixXd in1_ser = in1_par;
  Eigen::MatrixXd out_par(100,10);
  Eigen::MatrixXd out_ser(100,10);


  // Functor defining how inputs should be indexed
  auto ind_f = [&](int row_begin, int col_begin,
                   int row_size, int col_size,
                   const auto& fun, const auto& x) {
    return fun(x.block(row_begin, col_begin, row_size, col_size));
  };

  // Functor defining function to be applied to indexed arguments
  auto f = [&](const auto& x) { return x.array().exp().matrix(); };

  // Boolean template parameter to enable ranged parallelism
  stan::math::parallel_map<true>(f, ind_f, std::forward<Eigen::MatrixXd>(out_par), 1,
                         in1_par);
  EXPECT_MATRIX_FLOAT_EQ(out_par, in1_par.array().exp().matrix());
}
