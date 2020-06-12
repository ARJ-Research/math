#ifndef STAN_MATH_PRIM_FUNCTOR_MAP_VARIADIC_HPP
#define STAN_MATH_PRIM_FUNCTOR_MAP_VARIADIC_HPP

#include <stan/math/prim/meta/require_generics.hpp>
#include <stan/math/prim/meta/is_var.hpp>
#include <stan/math/prim/functor/apply.hpp>

#include <tbb/task_arena.h>
#include <tbb/parallel_for.h>
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
                         require_not_st_var<ReturnType>, Args...> {

  template <typename TupleT>
  struct recursive_applier {
    ReturnType result_;
    std::ostream* msgs_;
    TupleT* args_;

    recursive_applier(ReturnType&& result, std::ostream* msgs, TupleT* tuple_args)
        : result_(std::forward<ReturnType>(result)),
          msgs_(msgs),
          args_(tuple_args) {}

    inline void operator()(const tbb::blocked_range<size_t>& r) const {
      apply([&](auto&&... args) {
                for (size_t i = r.begin(); i < r.end(); ++i) {
                  result_.coeffRef(i) = ReduceFunction()(i, args...);
                }
              },
            args_);
    }
  };

  inline ReturnType operator()(ReturnType&& result,
                               int grainsize, std::ostream* msgs,
                               Args&&... args) const {
    const std::size_t num_terms = result.size();
    auto args_tuple = std::make_tuple(args...);
    recursive_applier<decltype(args_tuple)> worker(std::forward<ReturnType>(result), msgs,
                             (&args_tuple));
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
