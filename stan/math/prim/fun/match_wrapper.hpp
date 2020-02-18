#ifndef STAN_MATH_PRIM_FUN_MATCH_WRAPPER_HPP
#define STAN_MATH_PRIM_FUN_MATCH_WRAPPER_HPP

#include <stan/math/prim/meta.hpp>
#include <cmath>

namespace stan {
namespace math {

template <typename T1, typename T2, require_t<is_eigen_array<T1>>...>
const auto& match_wrapper(const T2& x) { return x.array(); }

template <typename T1, typename T2, require_not_t<is_eigen_array<T1>>...>
const auto match_wrapper(const T2& x) { return x.matrix(); }

}  // namespace math
}  // namespace stan

#endif
