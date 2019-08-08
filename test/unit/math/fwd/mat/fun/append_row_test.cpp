#include <stan/math/fwd/mat.hpp>
#include <test/unit/math/prim/mat/fun/expect_matrix_eq.hpp>
#include <gtest/gtest.h>

TEST(AgradFwdMatrixAppendRow, fd) {
  using Eigen::MatrixXd;
  using stan::math::append_row;
  using stan::math::matrix_fd;

  matrix_fd a(2, 2);
  MatrixXd ad(2, 2);
  MatrixXd b(2, 2);

  a << 2.0, 3.0, 9.0, -1.0;
  a.d() << 2.0, 3.0, 4.0, 5.0;
  ad << 2.0, 3.0, 9.0, -1.0;
  b << 4.0, 3.0, 0.0, 1.0;

  matrix_fd ab_append_row = append_row(a, b);
  MatrixXd adb_append_row = append_row(ad, b);

  expect_matrix_eq(a.d(), ab_append_row.d().block(0,0,2,2));
  expect_matrix_eq(ab_append_row.val(), adb_append_row);
}

TEST(AgradFwdVectorAppendRow, fd) {
  using Eigen::VectorXd;
  using stan::math::append_row;
  using stan::math::vector_fd;

  vector_fd a(4);
  VectorXd ad(4);
  VectorXd b(3);

  a << 2.0, 3.0, 9.0, -1.0;
  a.d() << 2.0, 3.0, 4.0, 5.0;
  ad << 2.0, 3.0, 9.0, -1.0;
  b << 4.0, 3.0, 0.4;

  vector_fd ab_append_row = append_row(a, b);
  VectorXd adb_append_row = append_row(ad, b);

  expect_matrix_eq(a.d(), ab_append_row.d().block(0,0,4,1));
  expect_matrix_eq(ab_append_row.val(), adb_append_row);
}

TEST(AgradFwdMatrixAppendRow, ffd) {
  using Eigen::MatrixXd;
  using stan::math::append_row;
  using stan::math::matrix_ffd;

  matrix_ffd a(2, 2);
  MatrixXd ad(2, 2);
  MatrixXd b(2, 2);

  a << 2.0, 3.0, 9.0, -1.0;
  a.d() << 2.0, 3.0, 4.0, 5.0;
  ad << 2.0, 3.0, 9.0, -1.0;
  b << 4.0, 3.0, 0.0, 1.0;

  matrix_ffd ab_append_row = append_row(a, b);
  MatrixXd adb_append_row = append_row(ad, b);

  expect_matrix_eq(a.d().val(), ab_append_row.d().val().block(0,0,2,2));
  expect_matrix_eq(ab_append_row.val().val(), adb_append_row);
}

TEST(AgradFwdVectorAppendRow, ffd) {
  using Eigen::VectorXd;
  using stan::math::append_row;
  using stan::math::vector_ffd;

  vector_ffd a(4);
  VectorXd ad(4);
  VectorXd b(3);

  a << 2.0, 3.0, 9.0, -1.0;
  a.d() << 2.0, 3.0, 4.0, 5.0;
  ad << 2.0, 3.0, 9.0, -1.0;
  b << 4.0, 3.0, 0.4;

  vector_ffd ab_append_row = append_row(a, b);
  VectorXd adb_append_row = append_row(ad, b);

  expect_matrix_eq(a.d().val(), ab_append_row.d().val().block(0,0,4,1));
  expect_matrix_eq(ab_append_row.val().val(), adb_append_row);
}
