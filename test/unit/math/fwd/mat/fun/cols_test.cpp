#include <stan/math/fwd/mat.hpp>
#include <gtest/gtest.h>

TEST(AgradFwdMatrixCols, vector_fd) {
  using stan::math::cols;
  using stan::math::row_vector_fd;
  using stan::math::vector_fd;

  vector_fd v(5);
  v << 0, 1, 2, 3, 4;
  v.d_() << 1.0, 1.0, 1.0, 1.0, 1.0;

  EXPECT_EQ(1U, cols(v));

  v.resize(0);
  EXPECT_EQ(1U, cols(v));
}

TEST(AgradFwdMatrixCols, row_vector_fd) {
  using stan::math::cols;
  using stan::math::row_vector_fd;

  row_vector_fd rv(5);
  rv << 0, 1, 2, 3, 4;
  rv.d_() << 1.0, 1.0, 1.0, 1.0, 1.0;
  EXPECT_EQ(5U, cols(rv));

  rv.resize(0);
  EXPECT_EQ(0U, cols(rv));
}

TEST(AgradFwdMatrixCols, matrix_fd) {
  using stan::math::cols;
  using stan::math::matrix_fd;

  matrix_fd m(2, 3);
  m << 0, 1, 2, 3, 4, 5;
  m(0, 0).d_ = 1.0;
  EXPECT_EQ(3U, cols(m));

  m.resize(5, 0);
  EXPECT_EQ(0U, cols(m));
}

TEST(AgradFwdFvarFvarMatrix, vector_ffd) {
  using stan::math::cols;
  using stan::math::fvar;
  using stan::math::vector_ffd;

  vector_ffd v(5);
  v.val_().val_() << 1.0, 2.0, 3.0, 4.0, 0.0;
  v.d_().val_() << 1.0, 2.0, 3.0, 4.0, 0.0;
  EXPECT_EQ(1U, cols(v));

  v.resize(0);
  EXPECT_EQ(1U, cols(v));
}

TEST(AgradFwdMatrixCols, rowvector_ffd) {
  using stan::math::cols;
  using stan::math::fvar;
  using stan::math::row_vector_ffd;

  row_vector_ffd rv(5);
  rv.val_().val_() << 1.0, 2.0, 3.0, 4.0, 0.0;
  rv.d_().val_() << 1.0, 2.0, 3.0, 4.0, 0.0;
  EXPECT_EQ(5U, cols(rv));

  rv.resize(0);
  EXPECT_EQ(0U, cols(rv));
}

TEST(AgradFwdMatrixCols, matrix_ffd) {
  using stan::math::cols;
  using stan::math::fvar;
  using stan::math::matrix_ffd;

  matrix_ffd m(2, 3);
  m.val_().val_() << 1.0, 2.0, 3.0, 4.0, 5.0, 0.0;
  m.d_().val_() << 1.0, 2.0, 3.0, 4.0, 5.0, 0.0;
  EXPECT_EQ(3U, cols(m));

  m.resize(5, 0);
  EXPECT_EQ(0U, cols(m));
}
