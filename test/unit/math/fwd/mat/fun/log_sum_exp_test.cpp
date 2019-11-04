#include <stan/math/fwd/mat.hpp>
#include <gtest/gtest.h>


TEST(AgradFwdMatrixLogSumExp, vector_fd) {
  using stan::math::vector_fd;
  using stan::math::fvar;
  using stan::math::log_sum_exp;

  vector_fd b(4);
  b << 1, 2, 3, 4;
  b(0).d_ = 1.0;
  b(1).d_ = 1.0;
  b(2).d_ = 1.0;
  b(3).d_ = 1.0;

  fvar<double> a = log_sum_exp(b);

  EXPECT_FLOAT_EQ(4.4401898, a.val_);
  EXPECT_FLOAT_EQ(1, a.d_);
}
TEST(AgradFwdMatrixLogSumExp, row_vector_fd) {
  using stan::math::row_vector_fd;
  using stan::math::fvar;
  using stan::math::log_sum_exp;

  row_vector_fd b(4);
  b << 1, 2, 3, 4;
  b(0).d_ = 1.0;
  b(1).d_ = 1.0;
  b(2).d_ = 1.0;
  b(3).d_ = 1.0;

  fvar<double> a = log_sum_exp(b);

  EXPECT_FLOAT_EQ(4.4401898, a.val_);
  EXPECT_FLOAT_EQ(1, a.d_);
}
TEST(AgradFwdMatrixLogSumExp, std_vector_fd) {
  using stan::math::fvar;
  using stan::math::log_sum_exp;

  std::vector<fvar<double>> b(4);
  b[0].val_ = 1.0;
  b[1].val_ = 2.0;
  b[2].val_ = 3.0;
  b[3].val_ = 4.0;
  b[0].d_ = 1.0;
  b[1].d_ = 1.0;
  b[2].d_ = 1.0;
  b[3].d_ = 1.0;

  fvar<double> a = log_sum_exp(b);

  EXPECT_FLOAT_EQ(4.4401898, a.val_);
  EXPECT_FLOAT_EQ(1, a.d_);
}


TEST(AgradFwdMatrixLogSumExp, matrix_fd) {
  using stan::math::matrix_fd;
  using stan::math::fvar;
  using stan::math::log_sum_exp;

  matrix_fd b(2, 2);
  b << 1, 2, 3, 4;
  b(0, 0).d_ = 1.0;
  b(0, 1).d_ = 1.0;
  b(1, 0).d_ = 1.0;
  b(1, 1).d_ = 1.0;

  fvar<double> a = log_sum_exp(b);

  EXPECT_FLOAT_EQ(4.4401898, a.val_);
  EXPECT_FLOAT_EQ(1, a.d_);
}

TEST(AgradFwdMatrixLogSumExp, vector_ffd) {
  using stan::math::vector_ffd;
  using stan::math::fvar;
  using stan::math::log_sum_exp;

  vector_ffd b(4);
  b << 1, 2, 3, 4;
  b(0).d_ = 1.0;
  b(1).d_ = 1.0;
  b(2).d_ = 1.0;
  b(3).d_ = 1.0;

  fvar<fvar<double> > a = log_sum_exp(b);

  EXPECT_FLOAT_EQ(4.4401898, a.val_.val_);
  EXPECT_FLOAT_EQ(1, a.d_.val_);
}
TEST(AgradFwdMatrixLogSumExp, row_vector_ffd) {
  using stan::math::row_vector_ffd;
  using stan::math::fvar;
  using stan::math::log_sum_exp;

  row_vector_ffd b(4);
  b << 1, 2, 3, 4;
  b(0).d_ = 1.0;
  b(1).d_ = 1.0;
  b(2).d_ = 1.0;
  b(3).d_ = 1.0;

  fvar<fvar<double> > a = log_sum_exp(b);

  EXPECT_FLOAT_EQ(4.4401898, a.val_.val_);
  EXPECT_FLOAT_EQ(1, a.d_.val_);
}

TEST(AgradFwdMatrixLogSumExp, matrix_ffd) {
  using stan::math::matrix_ffd;
  using stan::math::fvar;
  using stan::math::log_sum_exp;

  matrix_ffd b(2, 2);
  b << 1, 2, 3, 4;
  b(0, 0).d_ = 1.0;
  b(0, 1).d_ = 1.0;
  b(1, 0).d_ = 1.0;
  b(1, 1).d_ = 1.0;

  fvar<fvar<double> > a = log_sum_exp(b);

  EXPECT_FLOAT_EQ(4.4401898, a.val_.val_);
  EXPECT_FLOAT_EQ(1, a.d_.val_);
}

TEST(AgradFwdMatrixLogSumExp, std_vector_ffd) {
  using stan::math::fvar;
  using stan::math::log_sum_exp;

  std::vector<fvar<fvar<double>>> b(4);
  b[0].val_ = 1.0;
  b[1].val_ = 2.0;
  b[2].val_ = 3.0;
  b[3].val_ = 4.0;
  b[0].d_ = 1.0;
  b[1].d_ = 1.0;
  b[2].d_ = 1.0;
  b[3].d_ = 1.0;

  fvar<fvar<double>> a = log_sum_exp(b);

  EXPECT_FLOAT_EQ(4.4401898, a.val_.val_);
  EXPECT_FLOAT_EQ(1, a.d_.val_);
}
