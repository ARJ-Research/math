#ifndef STAN_MATH_FWD_MAT_FUN_MDIVIDE_LEFT_HPP
#define STAN_MATH_FWD_MAT_FUN_MDIVIDE_LEFT_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat/err/check_multiplicable.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/fwd/mat/fun/multiply.hpp>
#include <stan/math/prim/mat/fun/multiply.hpp>
#include <stan/math/fwd/mat/fun/to_fvar.hpp>
#include <stan/math/prim/mat/fun/mdivide_left.hpp>
#include <stan/math/prim/mat/err/check_square.hpp>

namespace stan {
namespace math {

template <typename T, int R1, int C1, int R2, int C2>
inline Eigen::Matrix<fvar<T>, R1, C2> mdivide_left(
    const Eigen::Matrix<fvar<T>, R1, C1> &A,
    const Eigen::Matrix<fvar<T>, R2, C2> &b) {
  check_square("mdivide_left", "A", A);
  check_multiplicable("mdivide_left", "A", A, "b", b);

  Eigen::Matrix<T, R1, C1> val_A = A.val_();
  Eigen::Matrix<T, R1, C1> deriv_A = A.d_();
  Eigen::Matrix<T, R2, C2> val_b = b.val_();
  Eigen::Matrix<T, R2, C2> deriv_b = b.d_();

  Eigen::Matrix<T, R1, C2> inv_A_mult_b = mdivide_left(val_A, val_b);
  Eigen::Matrix<T, R1, C2> inv_A_mult_deriv_b = mdivide_left(val_A, deriv_b);
  Eigen::Matrix<T, R1, C1> inv_A_mult_deriv_A = mdivide_left(val_A, deriv_A);

  Eigen::Matrix<T, R1, C2> deriv = inv_A_mult_deriv_b - multiply(inv_A_mult_deriv_A, inv_A_mult_b);

  return to_fvar(inv_A_mult_b, deriv);
}

template <typename T, int R1, int C1, int R2, int C2>
inline Eigen::Matrix<fvar<T>, R1, C2> mdivide_left(
    const Eigen::Matrix<double, R1, C1> &A,
    const Eigen::Matrix<fvar<T>, R2, C2> &b) {
  check_square("mdivide_left", "A", A);
  check_multiplicable("mdivide_left", "A", A, "b", b);

  return to_fvar(mdivide_left(A, b.val_().eval()), mdivide_left(A, b.d_().eval()));
}

template <typename T, int R1, int C1, int R2, int C2>
inline Eigen::Matrix<fvar<T>, R1, C2> mdivide_left(
    const Eigen::Matrix<fvar<T>, R1, C1> &A,
    const Eigen::Matrix<double, R2, C2> &b) {
  check_square("mdivide_left", "A", A);
  check_multiplicable("mdivide_left", "A", A, "b", b);

  Eigen::Matrix<T, R1, C1> val_A = A.val_();
  Eigen::Matrix<T, R1, C1> deriv_A = A.d_();

  Eigen::Matrix<T, R1, C2> inv_A_mult_b = mdivide_left(val_A, b);
  Eigen::Matrix<T, R1, C1> inv_A_mult_deriv_A = mdivide_left(val_A, deriv_A);

  Eigen::Matrix<T, R1, C2> deriv = -multiply(inv_A_mult_deriv_A, inv_A_mult_b);

  return to_fvar(inv_A_mult_b, deriv);
}

}  // namespace math
}  // namespace stan
#endif
