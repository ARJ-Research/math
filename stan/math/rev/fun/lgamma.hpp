#ifndef STAN_MATH_REV_FUN_LGAMMA_HPP
#define STAN_MATH_REV_FUN_LGAMMA_HPP

#include <stan/math/rev/meta.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/prim/fun/digamma.hpp>
#include <stan/math/prim/fun/lgamma.hpp>

namespace stan {
namespace math {

/**
 * The log gamma function for variables (C99).
 *
 * The derivative is the digamma function,
 *
 * \f$\frac{d}{dx} \Gamma(x) = \psi^{(0)}(x)\f$.
 *
 * @param a The variable.
 * @return Log gamma of the variable.
 */
inline var lgamma(const var& a) {
  return make_callback_var(lgamma(a.val()), [a](auto& vi) mutable {
    a.adj() += vi.adj() * digamma(a.val());
  });
}

}  // namespace math
}  // namespace stan
#endif
