#ifndef STAN_MATH_FWD_MAT_FUN_COLUMNS_DOT_PRODUCT_HPP
#define STAN_MATH_FWD_MAT_FUN_COLUMNS_DOT_PRODUCT_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat/err/check_matching_dims.hpp>
#include <stan/math/fwd/core.hpp>

namespace stan {
namespace math {

template <typename T, int R1, int C1, int R2, int C2>
inline Eigen::Matrix<fvar<T>, 1, C1> columns_dot_product(
    const Eigen::Matrix<fvar<T>, R1, C1>& v1,
    const Eigen::Matrix<fvar<T>, R2, C2>& v2) {
  check_matching_dims("columns_dot_product", "v1", v1, "v2", v2);
  Eigen::Matrix<fvar<T>, 1, C1> ret(1, v1.cols());
  Eigen::Matrix<T, R1, C1> v1val = v1.val_();
  Eigen::Matrix<T, R2, C2> v2val = v2.val_();

  ret.val_() = (v2val.transpose() * v1val).diagonal();
  ret.d_() = (v2.d_().eval().transpose() * v1val).diagonal()
              + (v2val.transpose() * v1.d_().eval()).diagonal();
  return ret;
}

template <typename T, int R1, int C1, int R2, int C2>
inline Eigen::Matrix<fvar<T>, 1, C1> columns_dot_product(
    const Eigen::Matrix<fvar<T>, R1, C1>& v1,
    const Eigen::Matrix<double, R2, C2>& v2) {
  check_matching_dims("columns_dot_product", "v1", v1, "v2", v2);
  Eigen::Matrix<fvar<T>, 1, C1> ret(1, v1.cols());
  Eigen::Matrix<T, R1, C1> v1val = v1.val_();

  ret.val_() = (v2.transpose() * v1val).diagonal();
  ret.d_() = (v2.transpose() * v1.d_().eval()).diagonal();
  return ret;
}

template <typename T, int R1, int C1, int R2, int C2>
inline Eigen::Matrix<fvar<T>, 1, C1> columns_dot_product(
    const Eigen::Matrix<double, R1, C1>& v1,
    const Eigen::Matrix<fvar<T>, R2, C2>& v2) {
  check_matching_dims("columns_dot_product", "v1", v1, "v2", v2);
  Eigen::Matrix<fvar<T>, 1, C1> ret(1, v1.cols());
  Eigen::Matrix<T, R2, C2> v2val = v2.val_();

  ret.val_() = (v2val.transpose() * v1).diagonal();
  ret.d_() = (v2.d_().eval().transpose() * v1).diagonal();
  return ret;
}

}  // namespace math
}  // namespace stan
#endif
