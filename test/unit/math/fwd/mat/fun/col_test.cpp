#include <stan/math/fwd/mat.hpp>
#include <gtest/gtest.h>

TEST(AgradFwdMatrixCol, matrix_fd) {
  using stan::math::col;
  using stan::math::matrix_fd;
  using stan::math::vector_fd;

  matrix_fd y(2, 3);
  y << 1, 2, 3, 4, 5, 6;
  y.d().fill(1.0);

  vector_fd z = col(y, 1);
  EXPECT_EQ(2, z.size());
  EXPECT_FLOAT_EQ(1.0, z[0].val_);
  EXPECT_FLOAT_EQ(4.0, z[1].val_);
  EXPECT_FLOAT_EQ(1.0, z[0].d_);
  EXPECT_FLOAT_EQ(1.0, z[1].d_);

  vector_fd w = col(y, 2);
  EXPECT_EQ(2, w.size());
  EXPECT_EQ(2.0, w[0].val_);
  EXPECT_EQ(5.0, w[1].val_);
  EXPECT_EQ(1.0, w[0].d_);
  EXPECT_EQ(1.0, w[1].d_);
}

TEST(AgradFwdMatrixCol, matrix_fd_exc0) {
  using stan::math::col;
  using stan::math::matrix_fd;

  matrix_fd y(2, 3);
  y << 1, 2, 3, 4, 5, 6;
  y.d().fill(1.0);

  EXPECT_THROW(col(y, 0), std::out_of_range);
  EXPECT_THROW(col(y, 7), std::out_of_range);
}

TEST(AgradFwdMatrixCol, matrix_fd_excHigh) {
  using stan::math::col;
  using stan::math::matrix_fd;

  matrix_fd y(2, 3);
  y << 1, 2, 3, 4, 5, 6;
  y.d().fill(1.0);

  EXPECT_THROW(col(y, 0), std::out_of_range);
  EXPECT_THROW(col(y, 5), std::out_of_range);
}

TEST(AgradFwdMatrixCol, matrix_ffd) {
  using stan::math::col;
  using stan::math::fvar;
  using stan::math::matrix_ffd;
  using stan::math::vector_ffd;

  matrix_ffd y(2, 3);
  y.val().val() << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;
  y.d().val().fill(1.0);

  vector_ffd z = col(y, 1);
  EXPECT_EQ(2, z.size());
  EXPECT_FLOAT_EQ(1.0, z[0].val_.val());
  EXPECT_FLOAT_EQ(4.0, z[1].val_.val());
  EXPECT_FLOAT_EQ(1.0, z[0].d_.val());
  EXPECT_FLOAT_EQ(1.0, z[1].d_.val());

  vector_ffd w = col(y, 2);
  EXPECT_EQ(2, w.size());
  EXPECT_EQ(2.0, w[0].val_.val());
  EXPECT_EQ(5.0, w[1].val_.val());
  EXPECT_EQ(1.0, w[0].d_.val());
  EXPECT_EQ(1.0, w[1].d_.val());
}

TEST(AgradFwdMatrixCol, matrix_ffd_exc0) {
  using stan::math::col;
  using stan::math::fvar;
  using stan::math::matrix_ffd;

  matrix_ffd y(2, 3);
  y.val().val() << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;
  y.d().val().fill(1.0);

  EXPECT_THROW(col(y, 0), std::out_of_range);
  EXPECT_THROW(col(y, 7), std::out_of_range);
}

TEST(AgradFwdMatrixCol, matrix_ffd_excHigh) {
  using stan::math::col;
  using stan::math::fvar;
  using stan::math::matrix_ffd;

  matrix_ffd y(2, 3);
  y.val().val() << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;
  y.d().val().fill(1.0);

  EXPECT_THROW(col(y, 0), std::out_of_range);
  EXPECT_THROW(col(y, 5), std::out_of_range);
}
