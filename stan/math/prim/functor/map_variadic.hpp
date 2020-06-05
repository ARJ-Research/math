#ifndef STAN_MATH_PRIM_FUNCTOR_MAP_VARIADIC_HPP
#define STAN_MATH_PRIM_FUNCTOR_MAP_VARIADIC_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/err.hpp>

#include <tbb/task_arena.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>

#include <algorithm>
#include <tuple>
#include <vector>

namespace stan {
namespace math {

namespace internal {

template <typename ReduceFunction, typename ReturnType, typename Enable, typename... Args>
struct map_variadic_impl {};

template <typename ReduceFunction, typename ReturnType, typename... Args>
struct map_variadic_impl<ReduceFunction, ReturnType,
                         require_st_arithmetic<ReturnType>, Args...> {

  struct recursive_applier {
    ReturnType result_;
    std::ostream* msgs_;
    std::tuple<Args...> args_;

    recursive_applier(ReturnType&& result, std::ostream* msgs, Args&&... args)
        : result_(std::forward<ReturnType>(result)),
          msgs_(msgs),
          args_(std::forward<Args>(args)...) {}

    inline void operator()(const tbb::blocked_range<size_t>& r) const {
      apply([&](auto&&... args) {
                for (size_t i = r.begin(); i < r.end(); ++i) {
                  result_[i] = ReduceFunction()(i, args...);
                }
              },
            args_);
    }
  };

  inline ReturnType operator()(ReturnType&& result,
                               int grainsize, std::ostream* msgs,
                               Args&&... args) const {
    const std::size_t num_terms = result.size();
    recursive_applier worker(std::forward<ReturnType>(result), msgs,
                             std::forward<Args>(args)...);
    tbb::parallel_for(
        tbb::blocked_range<std::size_t>(0, num_terms, grainsize), worker);

    return worker.result_;
  }
};

}  // namespace internal

template <typename ReduceFunction, typename OutputType, typename... Args>
inline void map_variadic(OutputType&& result, int grainsize, std::ostream* msgs,
                         Args&&... args) {

   internal::map_variadic_impl<ReduceFunction,
                               OutputType, void,
                               Args...>()(std::forward<OutputType>(result),
                                       grainsize, msgs,
                                       std::forward<Args>(args)...);

}

}  // namespace math
}  // namespace stan

#endif
