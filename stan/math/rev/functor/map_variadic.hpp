#ifndef STAN_MATH_REV_FUNCTOR_MAP_VARIADIC_HPP
#define STAN_MATH_REV_FUNCTOR_MAP_VARIADIC_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/functor.hpp>
#include <stan/math/prim/functor/map_variadic.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/core/save_adjoints.hpp>
#include <tbb/task_arena.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>

#include <tuple>
#include <vector>

namespace stan {
namespace math {
namespace internal {


template <typename ApplyFunction, typename ReturnType, typename... Args>
struct map_variadic_impl<ApplyFunction, ReturnType,
                         require_st_var<ReturnType>, Args...> {

  template <typename TupleT>
  struct recursive_applier {
    double* partials_;
    double* vals_;
    ReturnType result_;
    std::ostream* msgs_;
    TupleT* args_;

    recursive_applier(double* sliced_partials, double* values,
                      ReturnType&& result,
                      std::ostream* msgs, TupleT* tuple_args)
        : partials_(sliced_partials),
          vals_(values),
          result_(std::forward<ReturnType>(result)),
          msgs_(msgs),
          args_(tuple_args) {}

    inline void operator()(const tbb::blocked_range<size_t>& r) const {
      apply([&](auto&&... args) {
            for (size_t i = r.begin(); i < r.end(); ++i) {

              // Perform calculation
              auto sub_v = ApplyFunction()(i, args...);

              // Compute Jacobian
              sub_v.grad();

              // Copy calculated value
              vals_[i] = sub_v.val();
            }
          },
          args_);
    }
  };

  inline ReturnType operator()(ReturnType&& result, int grainsize,
                               std::ostream* msgs, Args&&... args) const {

    const std::size_t num_iter = result.size();
    const std::size_t num_vars_all_terms = count_vars(args...);

    vari** varis = ChainableStack::instance_->memalloc_.alloc_array<vari*>(
        num_vars_all_terms);
    double* partials = ChainableStack::instance_->memalloc_.alloc_array<double>(
        num_vars_all_terms);
    promote_scalar_t<double, plain_type_t<ReturnType>> values(result.size());

    // Copy varis for all terms
    save_varis(varis, args...);

    // Initialize nested autodiff stack
    const nested_rev_autodiff begin_nest;

    // Create nested autodiff copies of all arguments that do not point
    //   back to main autodiff stack
    auto args_tuple_local_copy = std::tuple<decltype(deep_copy_vars(args))...>(
              deep_copy_vars(args)...);

    recursive_applier<decltype(args_tuple_local_copy)> 
        worker(partials, &values[0], std::forward<ReturnType>(result),
               msgs, (&args_tuple_local_copy));

    tbb::parallel_for(
        tbb::blocked_range<std::size_t>(0, num_iter, grainsize), worker);

    // Copy adjoints of arguments
    apply(
        [&](auto&&... args) {
          save_adjoints(partials,
                        std::forward<decltype(args)>(args)...);
        },
        args_tuple_local_copy);

    for(size_t i = 0; i < num_iter; ++i) {
      result[i] = var(new precomputed_gradients_vari(
                            values[i], num_vars_all_terms, varis,
                            partials));
    }

    return result;
  }
};
}  // namespace internal

}  // namespace math
}  // namespace stan

#endif
