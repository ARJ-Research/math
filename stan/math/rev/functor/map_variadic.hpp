#ifndef STAN_MATH_REV_FUNCTOR_MAP_VARIADIC_HPP
#define STAN_MATH_REV_FUNCTOR_MAP_VARIADIC_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/functor.hpp>
#include <stan/math/prim/functor/map_variadic.hpp>
#include <stan/math/rev/core.hpp>
#include <tbb/task_arena.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>

#include <tuple>
#include <vector>

namespace stan {
namespace math {
namespace internal {


template <typename ReduceFunction, typename ReturnType, typename... Args>
struct map_variadic_impl<ReduceFunction, ReturnType,
                         require_st_var<ReturnType>, Args...> {

  struct recursive_applier {
    double* partials_;
    double* vals__;
    ReturnType result_;
    std::ostream* msgs_;
    std::tuple<Args...> args_;

    recursive_applier(double* sliced_partials, double* vals_,
                      ReturnType&& result,
                      std::ostream* msgs, Args&&... args)
        : vals__(vals_),
          partials_(sliced_partials),
          result_(std::forward<ReturnType>(result)),
          msgs_(msgs),
          args_(std::forward<Args>(args)...) {}

    inline void operator()(const tbb::blocked_range<size_t>& r) const {
      // Initialize nested autodiff stack
      const nested_rev_autodiff begin_nest;

      // Create nested autodiff copies of all arguments that do not point
      //   back to main autodiff stack
      auto args_tuple_local_copy = apply(
          [&](auto&&... args) {
            return std::tuple<decltype(deep_copy_vars(args))...>(
                deep_copy_vars(args)...);
          },
          args_);

      apply(
          [&](auto&&... args) {
            for (size_t i = r.begin(); i < r.end(); ++i) {

              // Perform calculation
              var sub_v = ReduceFunction()(i, args...);

              // Compute Jacobian
              sub_v.grad();

              // Copy calculated value
              vals__[i] = sub_v.val();
            }
          },
          args_tuple_local_copy);

      // Accumulate adjoints of all arguments
      apply(
          [&](auto&&... args) {
            accumulate_adjoints(partials_,
                                std::forward<decltype(args)>(args)...);
          },
          std::move(args_tuple_local_copy));
    }
  };

  inline std::decay_t<ReturnType> operator()(ReturnType&& result,
                               int grainsize, std::ostream* msgs,
                               Args&&... args) const {
    const std::size_t num_iter = result.size();
    const std::size_t num_vars_all_terms = count_vars(args...);

    vari** varis = ChainableStack::instance_->memalloc_.alloc_array<vari*>(
        num_vars_all_terms);
    double* partials = ChainableStack::instance_->memalloc_.alloc_array<double>(
        num_vars_all_terms);

    save_varis(varis, args...);
    Eigen::VectorXd vals_(num_iter);

    recursive_applier worker(partials, &vals_[0],
                             std::forward<ReturnType>(result), msgs,
                             std::forward<Args>(args)...);


    tbb::parallel_for(
        tbb::blocked_range<std::size_t>(0, num_iter, grainsize), worker);

    for(size_t i = 0; i < num_iter; ++i) {
      result[i] = var(new precomputed_gradients_vari(
                            vals_[i], num_vars_all_terms, varis,
                            partials));
    }

    return result;
  }
};
}  // namespace internal

}  // namespace math
}  // namespace stan

#endif
