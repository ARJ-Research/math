#ifndef STAN_MATH_PRIM_FUN_MATCH_WRAPPER_HPP
#define STAN_MATH_PRIM_FUN_MATCH_WRAPPER_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/meta/plain_type.hpp>
#include <cmath>

namespace stan {
namespace math {

template <typename T1, typename T2,
          require_all_not_t<is_eigen_array<plain_type_t<T1>>,
                            is_eigen_matrix<plain_type_t<T1>>>...>
const auto& match_wrapper(const T2& x) { return x; }

template <typename T1, typename T2,
          require_t<is_eigen_array<plain_type_t<T1>>>...>
const auto& match_wrapper(const T2& x) { return x.array(); }

template <typename T1, typename T2,
          require_t<is_eigen_matrix<plain_type_t<T1>>>...>
const auto match_wrapper(const T2& x) { return x.matrix(); }

}  // namespace math
}  // namespace stan

#endif
