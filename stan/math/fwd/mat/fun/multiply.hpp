#ifndef STAN_MATH_FWD_MAT_FUN_MULTIPLY_HPP
#define STAN_MATH_FWD_MAT_FUN_MULTIPLY_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat/err/check_multiplicable.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/fwd/mat/fun/dot_product.hpp>

namespace stan {
namespace math {

template <typename T, int R1, int C1>
inline Eigen::Matrix<fvar<T>, R1, C1> multiply(
    const Eigen::Matrix<fvar<T>, R1, C1>& m, const fvar<T>& c) {
  Eigen::Matrix<fvar<T>, R1, C1> res(m.rows(), m.cols());
  res.val_() = m.val_().array() * c.val_;
  res.d_() = m.val_().array() * c.d_ + m.d_().array() * c.val_;
  return res;
}

template <typename T, int R2, int C2>
inline Eigen::Matrix<fvar<T>, R2, C2> multiply(
    const Eigen::Matrix<fvar<T>, R2, C2>& m, double c) {
  Eigen::Matrix<fvar<T>, R2, C2> res(m.rows(), m.cols());
  res.val_() = m.val_().array() * c;
  res.d_() = m.d_().array() * c;
  return res;
}

template <typename T, int R1, int C1>
inline Eigen::Matrix<fvar<T>, R1, C1> multiply(
    const Eigen::Matrix<double, R1, C1>& m, const fvar<T>& c) {
  Eigen::Matrix<fvar<T>, R1, C1> res(m.rows(), m.cols());
  res.val_() = m.array() * c.val_;
  res.d_() = m.array() * c.d_;
  return res;
}

template <typename T, int R1, int C1>
inline Eigen::Matrix<fvar<T>, R1, C1> multiply(
    const fvar<T>& c, const Eigen::Matrix<fvar<T>, R1, C1>& m) {
  return multiply(m, c);
}

template <typename T, int R1, int C1>
inline Eigen::Matrix<fvar<T>, R1, C1> multiply(
    double c, const Eigen::Matrix<fvar<T>, R1, C1>& m) {
  return multiply(m, c);
}

template <typename T, int R1, int C1>
inline Eigen::Matrix<fvar<T>, R1, C1> multiply(
    const fvar<T>& c, const Eigen::Matrix<double, R1, C1>& m) {
  return multiply(m, c);
}

template <typename T, int R1, int C1, int R2, int C2>
inline Eigen::Matrix<fvar<T>, R1, C2> multiply(
    const Eigen::Matrix<fvar<T>, R1, C1>& m1,
    const Eigen::Matrix<fvar<T>, R2, C2>& m2) {
  check_multiplicable("multiply", "m1", m1, "m2", m2);
  Eigen::Matrix<fvar<T>, R1, C2> result(m1.rows(), m2.cols());
  Eigen::Matrix<T, R1, C1> m1val = m1.val_();
  Eigen::Matrix<T, R2, C2> m2val = m2.val_();

  result.val_() = m1val * m2val;
  result.d_() = m1val * m2.d_() + m1.d_() * m2val;
  
  return result;
}

template <typename T, int R1, int C1, int R2, int C2>
inline Eigen::Matrix<fvar<T>, R1, C2> multiply(
    const Eigen::Matrix<fvar<T>, R1, C1>& m1,
    const Eigen::Matrix<double, R2, C2>& m2) {
  check_multiplicable("multiply", "m1", m1, "m2", m2);
  Eigen::Matrix<fvar<T>, R1, C2> result(m1.rows(), m2.cols());

  result.val_() = m1.val_().lazyProduct(m2);
  result.d_() = m1.d_().lazyProduct(m2);
  
  return result;
}

template <typename T, int R1, int C1, int R2, int C2>
inline Eigen::Matrix<fvar<T>, R1, C2> multiply(
    const Eigen::Matrix<double, R1, C1>& m1,
    const Eigen::Matrix<fvar<T>, R2, C2>& m2) {
  check_multiplicable("multiply", "m1", m1, "m2", m2);
  Eigen::Matrix<fvar<T>, R1, C2> result(m1.rows(), m2.cols());

  result.val_() = m1.lazyProduct(m2.val_());
  result.d_() = m1.lazyProduct(m2.d_());
  
  return result;
}

template <typename T, int C1, int R2>
inline fvar<T> multiply(const Eigen::Matrix<fvar<T>, 1, C1>& rv,
                        const Eigen::Matrix<fvar<T>, R2, 1>& v) {
  check_multiplicable("multiply", "rv", rv, "v", v);
  return dot_product(rv, v);
}

template <typename T, int C1, int R2>
inline fvar<T> multiply(const Eigen::Matrix<fvar<T>, 1, C1>& rv,
                        const Eigen::Matrix<double, R2, 1>& v) {
  check_multiplicable("multiply", "rv", rv, "v", v);
  return dot_product(rv, v);
}

template <typename T, int C1, int R2>
inline fvar<T> multiply(const Eigen::Matrix<double, 1, C1>& rv,
                        const Eigen::Matrix<fvar<T>, R2, 1>& v) {
  check_multiplicable("multiply", "rv", rv, "v", v);
  return dot_product(rv, v);
}

}  // namespace math
}  // namespace stan
#endif
