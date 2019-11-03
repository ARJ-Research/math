#include <stan/math/prim/mat.hpp>
#include <gtest/gtest.h>

void test_log_softmax(const Eigen::Matrix<double, Eigen::Dynamic, 1>& theta) {
  using Eigen::Dynamic;
  using Eigen::Matrix;
  using stan::math::log_softmax;
  using stan::math::softmax;
  using std::log;

  int size = theta.size();

  Matrix<double, Dynamic, 1> log_softmax_theta = log_softmax(theta);

  Matrix<double, Dynamic, 1> softmax_theta = softmax(theta);

  Matrix<double, Dynamic, 1> log_softmax_theta_expected(size);
  for (int i = 0; i < size; ++i)
    log_softmax_theta_expected(i) = log(softmax_theta(i));

  EXPECT_EQ(log_softmax_theta_expected.size(), log_softmax_theta.size());
  for (int i = 0; i < theta.size(); ++i)
    EXPECT_FLOAT_EQ(log_softmax_theta_expected(i), log_softmax_theta(i));
}

TEST(MathMatrix, softmax) {
  using Eigen::Dynamic;
  using Eigen::Matrix;
  using stan::math::log_softmax;
  using stan::math::softmax;

  // stan::math::vector_d x(1);
  // x << 0.0;
  // test_log_softmax(x);

  stan::math::vector_d x2(2);
  x2 << -1.0, 1.0;
  test_log_softmax(x2);

  // stan::math::vector_d x3(3);
  // x3 << -1.0, 1.0, 10.0;
  // test_log_softmax(x3);
}
TEST(MathMatrix, softmax_exception) {
  using stan::math::log_softmax;
  stan::math::vector_d v0;  // size == 0

  EXPECT_THROW(log_softmax(v0), std::invalid_argument);
}

TEST(MathMatrix, softmax_vectors) {
  using stan::math::vector_d;
  using stan::math::matrix_d;
  using stan::math::log_softmax;

  vector_d x2_1(2);
  x2_1 << -1.0, 1.0;

  vector_d x2_2(2);
  x2_2 << 1.5, 2.2;

  std::vector<double> stx2_1{-1,1};
  std::vector<double> stx2_2{1.5, 2.2};

  vector_d eigen_res = log_softmax(x2_1);
  std::vector<double> st_res = log_softmax(stx2_1);

  EXPECT_FLOAT_EQ(eigen_res[0], st_res[0]);
  EXPECT_FLOAT_EQ(eigen_res[1], st_res[1]);

  std::vector<vector_d> x2_nest{x2_1,x2_2};
  std::vector<std::vector<double>> stx2_nest{stx2_1,stx2_2};

  std::vector<vector_d> eigen_res_vec = log_softmax(x2_nest);
  std::vector<std::vector<double>> st_res_vec = log_softmax(stx2_nest);

  EXPECT_FLOAT_EQ(eigen_res_vec[0][0], st_res_vec[0][0]);
  EXPECT_FLOAT_EQ(eigen_res_vec[0][1], st_res_vec[0][1]);
  EXPECT_FLOAT_EQ(eigen_res_vec[1][0], st_res_vec[1][0]);
  EXPECT_FLOAT_EQ(eigen_res_vec[1][1], st_res_vec[1][1]);

  std::vector<double> stv{1,2,3};
  auto map = Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,1>>(stv.data(), stv.size());

  Eigen::Array<double, 1, Eigen::Dynamic> arr(3);
  arr << 1, 2, 3;
  std::vector<Eigen::Array<double, 1, Eigen::Dynamic>> arr_arr{arr, arr};
  std::vector<Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,1>>> map_map{map, map};
  log_softmax(arr);
  log_softmax(arr_arr);
}