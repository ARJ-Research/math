#ifndef STAN_MATH_PRIM_PROB_GAMMA_ICDF_HPP
#define STAN_MATH_PRIM_PROB_GAMMA_ICDF_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/err.hpp>
#include <stan/math/prim/fun/gamma_p_inv.hpp>
#include <stan/math/prim/fun/divide.hpp>

namespace stan {
namespace math {

/** \ingroup prob_dists
 * The cumulative density function for a gamma distribution for p
 * with the specified shape and inverse scale parameters.
 *
 * @tparam T_p type of scalar
 * @tparam T_shape type of shape
 * @tparam T_inv_scale type of inverse scale
 * @param p A scalar variable.
 * @param alpha Shape parameter.
 * @param beta Inverse scale parameter.
 * @throw std::domain_error if alpha is not greater than 0.
 * @throw std::domain_error if beta is not greater than 0.
 * @throw std::domain_error if p is not greater than or equal to 0.
 */
template <typename T_p, typename T_shape, typename T_inv_scale>
return_type_t<T_p, T_shape, T_inv_scale> gamma_icdf(const T_p& p,
                                                   const T_shape& alpha,
                                                   const T_inv_scale& beta) {
  using T_partials_return = partials_return_t<T_p, T_shape, T_inv_scale>;
  using T_p_ref = ref_type_t<T_p>;
  using T_alpha_ref = ref_type_t<T_shape>;
  using T_beta_ref = ref_type_t<T_inv_scale>;

  static const char* function = "gamma_icdf";
  check_consistent_sizes(function, "Probability", p, "Shape parameter",
                         alpha, "Inverse scale parameter", beta);
  T_p_ref p_ref = p;
  T_alpha_ref alpha_ref = alpha;
  T_beta_ref beta_ref = beta;

  check_positive_finite(function, "Shape parameter", alpha_ref);
  check_positive_finite(function, "Inverse scale parameter", beta_ref);
  check_bounded(function, "Random variable", p_ref, 0, 1);
  
  return divide(gamma_p_inv(alpha_ref,p_ref),beta_ref);
}

}  // namespace math
}  // namespace stan
#endif
