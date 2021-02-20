#include <stan/math/prim.hpp>
#include <test/unit/util.hpp>
#include <gtest/gtest.h>
#include <vector>
#include <stdexcept>

TEST(ToMatrixArray, answers) {
  Eigen::MatrixXd test_mat = Eigen::MatrixXd::Random(10,10);
  std::vector<int> ind1{0, 1, 2, 3, 4};
  std::vector<int> ind2{1, 3, 0, 4, 2};
  Eigen::VectorXd out_vec(5);
  Eigen::VectorXd out_vec2(100);
  Eigen::VectorXd out_vec4 = stan::math::to_vector(ind1);

  for( int i = 0; i < 5; ++i) {
    out_vec[i] = test_mat.array().exp().matrix().coeff(ind1[i], ind2[i]);
  }

  for(int i = 0; i < 100; ++i) {
    out_vec2[i] = test_mat.array().exp().matrix().coeff(i);
  }

  Eigen::VectorXd out_vec3 = stan::math::to_vector(test_mat.array().exp().matrix());

  for(int i = 0; i < 100; ++i) {
    EXPECT_FLOAT_EQ(out_vec2[i], out_vec3[i]);
  }
  
  EXPECT_MATRIX_EQ(out_vec, stan::math::to_vector(test_mat.array().exp().matrix(), ind1, ind2));

}


