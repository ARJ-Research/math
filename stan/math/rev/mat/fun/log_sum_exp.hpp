#ifndef STAN_MATH_REV_MAT_FUN_LOG_SUM_EXP_HPP
#define STAN_MATH_REV_MAT_FUN_LOG_SUM_EXP_HPP

#include <stan/math/rev/meta.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/mat/fun/typedefs.hpp>
#include <stan/math/prim/scal/fun/log_sum_exp.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <cmath>
#include <limits>

namespace stan {
namespace math {

namespace internal {

class log_sum_exp_matrix_vari : public op_matrix_vari {
 public:
  template <typename T>
  explicit log_sum_exp_matrix_vari(const T& x)
      : op_matrix_vari(log_sum_exp(as_eigen(x).val()), as_eigen(x)) {}
  void chain() {
    Eigen::Map<vector_vi> vis_map(vis_, size_);
    vis_map.adj().array() += adj_ * (vis_map.val().array() - val_).exp();
  }
};
}  // namespace internal

/**
 * Returns the log sum of exponentials.
 *
 * @param x matrix
 */
template <typename T, typename = require_vector_like_st<is_var, T>>
inline var log_sum_exp(const T& v) {
  return var(new internal::log_sum_exp_matrix_vari(v));
}

template <typename T, typename = require_vector_like_st<is_var, T>>
inline std::vector<var> log_sum_exp(const std::vector<T>& v) {
  std::vector<var> result(v.size());
  for(int i = 0; i < v.size(); i++)
    result[i] = log_sum_exp(v[i]);
  return result;
}

}  // namespace math
}  // namespace stan
#endif
