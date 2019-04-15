#include <stan/math/fwd/mat.hpp>
#include <gtest/gtest.h>

TEST(AgradFwdMatrixColumnsDotSelf, matrix_fd) {
  using stan::math::columns_dot_self;
  using stan::math::matrix_fd;

  matrix_fd m1(1, 1);
  m1 << 2.0;
  m1(0).d_ = 1.0;
  EXPECT_NEAR(4.0, columns_dot_self(m1)(0, 0).val_, 1E-12);
  EXPECT_NEAR(4.0, columns_dot_self(m1)(0, 0).d_, 1E-12);

  matrix_fd m2(1, 2);
  m2 << 2.0, 3.0;
  m2.d_() << 1.0, 1.0;

  matrix_fd x = columns_dot_self(m2);
  EXPECT_NEAR(4.0, x(0, 0).val_, 1E-12);
  EXPECT_NEAR(9.0, x(0, 1).val_, 1E-12);
  EXPECT_NEAR(4.0, x(0, 0).d_, 1E-12);
  EXPECT_NEAR(6.0, x(0, 1).d_, 1E-12);

  matrix_fd m3(2, 2);
  m3 << 2.0, 3.0, 4.0, 5.0;
  m3.d_() << 1.0, 1.0, 1.0, 1.0;

  x = columns_dot_self(m3);
  EXPECT_NEAR(20.0, x(0, 0).val_, 1E-12);
  EXPECT_NEAR(34.0, x(0, 1).val_, 1E-12);
  EXPECT_NEAR(12.0, x(0, 0).d_, 1E-12);
  EXPECT_NEAR(16.0, x(0, 1).d_, 1E-12);
}

TEST(AgradFwdMatrixColumnsDotSelf, matrix_ffd) {
  using stan::math::columns_dot_self;
  using stan::math::fvar;
  using stan::math::matrix_ffd;

  matrix_ffd m1(1, 1);
  m1.val_().val_() << 2.0;
  m1.d_().val_() << 1.0;

  EXPECT_NEAR(4.0, columns_dot_self(m1)(0, 0).val_.val(), 1E-12);
  EXPECT_NEAR(4.0, columns_dot_self(m1)(0, 0).d_.val(), 1E-12);

  matrix_ffd m2(1, 2);
  m2.val_().val_() << 2.0, 3.0;
  m2.d_().val_() << 1.0, 1.0;
  matrix_ffd x = columns_dot_self(m2);
  EXPECT_NEAR(4.0, x(0, 0).val_.val(), 1E-12);
  EXPECT_NEAR(9.0, x(0, 1).val_.val(), 1E-12);
  EXPECT_NEAR(4.0, x(0, 0).d_.val(), 1E-12);
  EXPECT_NEAR(6.0, x(0, 1).d_.val(), 1E-12);

  matrix_ffd m3(2, 2);
  m3.val_().val_() << 2.0, 3.0, 4.0, 5.0;
  m3.d_().val_() << 1.0, 1.0, 1.0, 1.0;
  x = columns_dot_self(m3);
  EXPECT_NEAR(20.0, x(0, 0).val_.val(), 1E-12);
  EXPECT_NEAR(34.0, x(0, 1).val_.val(), 1E-12);
  EXPECT_NEAR(12.0, x(0, 0).d_.val(), 1E-12);
  EXPECT_NEAR(16.0, x(0, 1).d_.val(), 1E-12);
}
