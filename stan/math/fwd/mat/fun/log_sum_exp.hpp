#ifndef STAN_MATH_FWD_MAT_FUN_LOG_SUM_EXP_HPP
#define STAN_MATH_FWD_MAT_FUN_LOG_SUM_EXP_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/prim/mat/fun/log_sum_exp.hpp>

namespace stan {
namespace math {

template <typename T, typename = require_vector_like_st<is_fvar, T>,
          typename PromotedScalarT = return_type_t<typename T::value_type>,
          typename InnerT = typename PromotedScalarT::Scalar>
PromotedScalarT log_sum_exp(const T& x) {
  auto v = as_eigen(x).template cast<PromotedScalarT>();
  Eigen::Matrix<InnerT, -1, -1> vals = v.val();
  Eigen::Matrix<InnerT, -1, -1> exp_vals = vals.array().exp();

  return fvar<InnerT>(log_sum_exp(vals),
                 v.d().cwiseProduct(exp_vals).sum() / exp_vals.sum());
}

template <typename Vec, require_vector_st<is_fvar, Vec>...,
          require_vector_vt<is_vector, Vec>...,
          typename InnerT =
                  typename return_type_t<typename Vec::value_type>::Scalar>
inline auto log_sum_exp(Vec&& v) {
  std::vector<fvar<InnerT>> result(v.size());
  for(int i = 0; i < v.size(); i++){
    result[i] = log_sum_exp(std::forward<decltype(v[i])>(v[i]));
  }
  return result;
}

}  // namespace math
}  // namespace stan
#endif
