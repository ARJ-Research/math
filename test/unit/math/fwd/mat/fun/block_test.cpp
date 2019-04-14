#include <stan/math/fwd/mat.hpp>
#include <test/unit/math/prim/mat/fun/expect_matrix_eq.hpp>
#include <gtest/gtest.h>

TEST(AgradFwdMatrixBlock, matrix_fd) {
  using stan::math::block;
  using stan::math::matrix_fd;
  using stan::math::vector_fd;

  matrix_fd v(3, 3);
  v << 1, 4, 9, 1, 4, 9, 1, 4, 9;
  v.d_() << 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0;

  matrix_fd m = block(v, 1, 1, 3, 3);
  expect_matrix_eq(v.val_(), m.val_());
  expect_matrix_eq(v.d_(), m.d_());

  matrix_fd n = block(v, 2, 2, 2, 2);
  expect_matrix_eq(v.val_().block(1,1,2,2), n.val_());
  expect_matrix_eq(v.d_().block(1,1,2,2), n.d_());
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
  using stan::math::matrix_d;
  using stan::math::vector_ffd;

  matrix_d vals(3, 3);
  vals << 1.0, 4.0, 9.0, 1.0, 4.0, 9.0, 1.0, 4.0, 9.0;

  matrix_ffd v(3, 3);
  v.val_().val_() = vals;
  v.d_().val_() = vals.cwiseSqrt();

  matrix_ffd m = block(v, 1, 1, 3, 3);
  expect_matrix_eq(vals, m.val_().val_());
  expect_matrix_eq(vals.cwiseSqrt(), m.d_().val_());

  matrix_ffd n = block(v, 2, 2, 2, 2);
  expect_matrix_eq(vals.block(1, 1, 2, 2), n.val_().val_());
  expect_matrix_eq(vals.cwiseSqrt().block(1, 1, 2, 2), n.d_().val_());
}

TEST(AgradFwdMatrixBlock, matrix_ffd_exception) {
  using stan::math::block;
  using stan::math::matrix_ffd;

  matrix_ffd v(3, 3);
  EXPECT_THROW(block(v, 0, 0, 1, 1), std::out_of_range);
  EXPECT_THROW(block(v, 1, 1, 4, 4), std::out_of_range);
}
