#ifndef STAN_MATH_PRIM_FUN_GAMMA_P_INV_HPP
#define STAN_MATH_PRIM_FUN_GAMMA_P_INV_HPP

#include <stan/math/prim/err.hpp>
#include <stan/math/prim/fun/boost_policy.hpp>
#include <stan/math/prim/fun/constants.hpp>
#include <stan/math/prim/fun/is_nan.hpp>
#include <stan/math/prim/functor/apply_scalar_binary.hpp>
#include <boost/math/special_functions/gamma.hpp>

namespace stan {
namespace math {

/**
 * Return the inverse of the normalized, lower-incomplete gamma function
 * applied to the specified argument.
 *
 *
 * @param a first argument
 * @param p second argument
 * @return inverse of the normalized, lower-incomplete gamma function
 * applied to a and p
 * @throws std::domain_error if either argument is not positive or
 * if z is at a pole of the function
 */
inline double gamma_p_inv(double a, double p) {
  if (is_nan(a)) {
    return not_a_number();
  }
  if (is_nan(p)) {
    return not_a_number();
  }
  check_positive("gamma_p_inv", "first argument (a)", a);
  check_bounded("gamma_p_inv", "second argument (p)", p, 0, 1);
  return boost::math::gamma_p_inv(a, p, boost_policy_t<>());
}

/**
 * Enables the vectorised application of the gamma_p function,
 * when the first and/or second arguments are containers.
 *
 * @tparam T1 type of first input
 * @tparam T2 type of second input
 * @param a First input
 * @param b Second input
 * @return gamma_p function applied to the two inputs.
 */
template <typename T1, typename T2, require_any_container_t<T1, T2>* = nullptr>
inline auto gamma_p_inv(const T1& a, const T2& b) {
  return apply_scalar_binary(
      a, b, [&](const auto& c, const auto& d) { return gamma_p_inv(c, d); });
}

}  // namespace math
}  // namespace stan
#endif
