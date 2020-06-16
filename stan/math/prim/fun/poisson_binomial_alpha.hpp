#ifndef STAN_MATH_PRIM_FUN_POISSON_BINOMIAL_ALPHA_HPP
#define STAN_MATH_PRIM_FUN_POISSON_BINOMIAL_ALPHA_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/fun/log.hpp>
#include <stan/math/prim/fun/log1m.hpp>
#include <stan/math/prim/fun/log_sum_exp.hpp>
#include <stan/math/prim/fun/max_size.hpp>

namespace stan {
namespace math {

template <typename T_theta, typename T_scalar = scalar_type_t<T_theta>,
          require_eigen_col_vector_t<T_theta>* = nullptr>
plain_type_t<T_theta> poisson_binomial_alpha(int y, const T_theta& theta) {
  size_t sz_y = y;
  size_t size_theta = theta.size();
  plain_type_t<T_theta> log_theta = log(theta);
  plain_type_t<T_theta> log1m_theta = log1m(theta);

  Eigen::Matrix<T_scalar, Eigen::Dynamic, Eigen::Dynamic>
    alpha(size_theta + 1, sz_y + 1);

  // alpha[i, j] = log prob of j successes in first i trials
  alpha(0, 0) = 0.0;
  for (size_t i = 0; i < size_theta; ++i) {
    // no success in i trials
    alpha(i + 1, 0) = alpha(i, 0) + log1m_theta[i];

    // 0 < j < i successes in i trials
    for (size_t j = 0; j < std::min(sz_y, i); ++j) {
      alpha(i + 1, j + 1) = log_sum_exp(alpha(i, j) + log_theta[i],
                                        alpha(i, j + 1) + log1m_theta[i]);
    }

    // i successes in i trials
    if (i < sz_y) {
      alpha(i + 1, i + 1) = alpha(i, i) + log_theta(i);
    }
  }

  return alpha.row(size_theta);
}

template <typename T_theta, typename T_scalar = scalar_type_t<T_theta>>
auto poisson_binomial_alpha(std::vector<int> y, const T_theta& theta) {
  size_t max_sizes = max_size(y, theta);
  std::vector<Eigen::Matrix<T_scalar, Eigen::Dynamic, 1>> result(max_sizes);
  vector_seq_view<T_theta> theta_vec(theta);
  for(size_t i = 0; i < max_sizes; ++i) {
    result[i] = poisson_binomial_alpha(y[i], theta_vec[i]);
  }

  return result;
}

}  // namespace math
}  // namespace stan
#endif
