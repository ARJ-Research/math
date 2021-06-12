#ifndef STAN_MATH_REV_FUN_GAMMA_P_INV_HPP
#define STAN_MATH_REV_FUN_GAMMA_P_INV_HPP

#include <stan/math/rev/meta.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/prim/fun/digamma.hpp>
#include <stan/math/prim/fun/gamma_p.hpp>
#include <stan/math/prim/fun/gamma_p_inv.hpp>
#include <stan/math/prim/fun/log_diff_exp.hpp>
#include <stan/math/prim/fun/log_sum_exp.hpp>
#include <stan/math/prim/fun/log1m.hpp>
#include <stan/math/prim/fun/tgamma.hpp>
#include <boost/math/special_functions/hypergeometric_pFq.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <iostream>

namespace stan {
namespace math {

/**
 * Returns the beta function and gradients for two var inputs.
 *
 * @param a var Argument
 * @param p var Argument
 * @return Result of beta function
 */
inline var gamma_p_inv(const var& a, const var& p) {
  using std::log;
  using std::exp;
  double gamma_p_inv_val = gamma_p_inv(a.val(), p.val());
  double lgamma_a = lgamma(a.val());
  double log_gamma_p_inv_val = log(gamma_p_inv(a.val(), p.val()));
  double hyper_pfq = boost::math::hypergeometric_pFq({a.val(),a.val()},{a.val()+1,a.val()+1},-gamma_p_inv_val);
  double log_gamma_p_m1 = gamma_p_inv_val + (1-a.val())*log_gamma_p_inv_val;

  double term1 = 2.0*lgamma_a + log(hyper_pfq) - 2.0*lgamma(a.val() + 1) + a.val() * log_gamma_p_inv_val;
  double term2 = log(p.val()) + lgamma_a + log(log_gamma_p_inv_val);
  double term3 = log(tgamma(a.val()) - boost::math::tgamma(a.val(), gamma_p_inv_val)) + log(digamma(a.val()));

  double term_sum = term1 > term3 ? log_sum_exp(log_diff_exp(term1,term2),term3) : log_sum_exp(log_diff_exp(term3,term2),term1);
  
  double a_adj = exp(log_gamma_p_m1 + term_sum);
  double p_adj = exp(log_gamma_p_m1 + lgamma_a);

  return make_callback_var(gamma_p_inv_val,
                           [a, p, a_adj, p_adj](auto& vi) mutable {
                             a.adj() += vi.adj() * a_adj;
                             p.adj() += vi.adj() * p_adj;
                           });
}

}  // namespace math
}  // namespace stan
#endif
