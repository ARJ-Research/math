#include <stan/math/fwd/mat.hpp>
#include <test/unit/math/prim/mat/fun/expect_matrix_eq.hpp>
#include <gtest/gtest.h>

TEST(AgradFwdMatrixBlock, matrix_fd) {
  using stan::math::block;
  using stan::math::matrix_fd;
  using stan::math::vector_fd;

  matrix_fd v(3, 3);
  v << 1, 4, 9, 1, 4, 9, 1, 4, 9;
  v.d() << 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0;

  matrix_fd m = block(v, 1, 1, 3, 3);
  expect_matrix_eq(v.val().block(0, 0, 3, 3), m.val());
  expect_matrix_eq(v.d().block(0, 0, 3, 3), m.d());

  matrix_fd n = block(v, 2, 2, 2, 2);
  expect_matrix_eq(v.val().block(1, 1, 2, 2), n.val());
  expect_matrix_eq(v.d().block(1, 1, 2, 2), n.d());
}

TEST(AgradFwdMatrixBlock, matrix_fd_exception) {
  using stan::math::block;
  using stan::math::matrix_fd;

  matrix_fd v(3, 3);
  EXPECT_THROW(block(v, 0, 0, 1, 1), std::out_of_range);
  EXPECT_THROW(block(v, 1, 1, 4, 4), std::out_of_range);
}

TEST(AgradFwdMatrixBlock, matrix_ffd) {
  using stan::math::block;
  using stan::math::fvar;
  using stan::math::matrix_ffd;
  using stan::math::vector_ffd;

  fvar<fvar<double> > a;
  fvar<fvar<double> > b;
  fvar<fvar<double> > c;

  a.val_.val_ = 1.0;
  a.d_.val_ = 1.0;
  b.val_.val_ = 4.0;
  b.d_.val_ = 2.0;
  c.val_.val_ = 9.0;
  c.d_.val_ = 3.0;

  matrix_ffd v(3, 3);
  v << a, b, c, a, b, c, a, b, c;

  matrix_ffd m = block(v, 1, 1, 3, 3);
  expect_matrix_eq(v.val().val(), m.val().val());
  expect_matrix_eq(v.d().val(), m.d().val());

  matrix_ffd n = block(v, 2, 2, 2, 2);
  expect_matrix_eq(v.val().val().block(1, 1, 2, 2), n.val().val());
  expect_matrix_eq(v.d().val().block(1, 1, 2, 2), n.d().val());
}

TEST(AgradFwdMatrixBlock, matrix_ffd_exception) {
  using stan::math::block;
  using stan::math::matrix_ffd;

  matrix_ffd v(3, 3);
  EXPECT_THROW(block(v, 0, 0, 1, 1), std::out_of_range);
  EXPECT_THROW(block(v, 1, 1, 4, 4), std::out_of_range);
}
