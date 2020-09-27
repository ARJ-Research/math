#ifndef STAN_MATH_FWD_FUNCTOR_PARALLEL_MAP_HPP
#define STAN_MATH_FWD_FUNCTOR_PARALLEL_MAP_HPP

#include <stan/math/rev/core.hpp>
#include <stan/math/prim/meta.hpp>
#include <tbb/task_arena.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

namespace stan {
namespace math {

template <bool Ranged, typename ApplyFunction, typename IndexFunction,
          typename Res, typename... Args,
          require_not_st_arithmetic<Res>* = nullptr,
          require_not_st_var<Res>* = nullptr,
          std::enable_if_t<!Ranged>* = nullptr>
inline void parallel_map(const ApplyFunction& app_fun,
                                   const IndexFunction& index_fun,
                                   Res&& result, Args&&... x) {
  for (size_t i = 0; i < result.size(); ++i) {
    // Apply specified function to arguments at current iteration
    result(i) = index_fun(i, app_fun, x...);
  }
}

template <bool Ranged, typename ApplyFunction, typename IndexFunction,
          typename Res, typename... Args,
          require_not_st_arithmetic<Res>* = nullptr,
          require_not_st_var<Res>* = nullptr,
          std::enable_if_t<Ranged>* = nullptr>
inline void parallel_map(const ApplyFunction& app_fun,
                         const IndexFunction& index_fun,
                         Res&& result, int grainsize, Args&&... x) {
  result = index_fun(0, result.size(), app_fun, x...);
}

}  // namespace math
}  // namespace stan
#endif
