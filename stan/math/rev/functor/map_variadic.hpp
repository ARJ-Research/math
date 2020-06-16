#ifndef STAN_MATH_REV_FUNCTOR_MAP_VARIADIC_HPP
#define STAN_MATH_REV_FUNCTOR_MAP_VARIADIC_HPP

#include <stan/math/prim/functor/map_variadic.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/core/precomputed_gradients.hpp>
#include <stan/math/rev/core/nested_rev_autodiff.hpp>
#include <stan/math/rev/core/save_adjoints.hpp>
#include <iostream>

namespace stan {
namespace math {
namespace internal {


template <typename ApplyFunction, typename ReturnType, typename... Args>
struct map_variadic_impl<ApplyFunction, ReturnType,
                         require_st_var<ReturnType>, Args...> {

  template <typename TupleT>
  struct recursive_applier {
    var* result_;
    std::ostream* msgs_;
    TupleT* args_;

    recursive_applier(var* result,
                      std::ostream* msgs, TupleT* tuple_args)
        : result_(result),
          msgs_(msgs),
          args_(tuple_args) {}

    inline void operator()(const tbb::blocked_range<size_t>& r) const {
      apply([&](auto&&... args) {
            for (size_t i = r.begin(); i < r.end(); ++i) {
              result_[i] = ApplyFunction()(i, args...);
              std::cout << i << std::endl << result_[i].val() << std::endl
                        << result_[i].adj() << std::endl;
            }
          },
          args_);
    }
  };

  inline auto operator()(ReturnType&& result, int grainsize,
                               std::ostream* msgs, Args&&... args) const {

    const std::size_t num_iter = result.size();
    const std::size_t num_vars_all_terms = count_vars(args...);

    auto args_tuple = std::make_tuple(args...);

    recursive_applier<decltype(args_tuple)> 
        worker(result.data(), msgs, (&args_tuple));

    tbb::parallel_for(
        tbb::blocked_range<std::size_t>(0, num_iter, grainsize), worker);

    return result;
  }
};
}  // namespace internal

}  // namespace math
}  // namespace stan

#endif
