#ifndef STAN_MATH_REV_FUNCTOR_PARALLEL_MAP_HPP
#define STAN_MATH_REV_FUNCTOR_PARALLEL_MAP_HPP

#include <stan/math/rev/core.hpp>
#include <stan/math/prim/meta.hpp>
#include <tbb/task_arena.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

namespace stan {
namespace math {

template <typename ApplyFunction, typename IndexFunction,
          typename Res, typename... Args,
          require_st_var<Res>* = nullptr>
inline void parallel_map(const ApplyFunction& app_fun,
                         const IndexFunction& index_fun,
                         Res&& result, int grainsize,
                         Args&&... x) {

    // Functors for manipulating vars at a given iteration of the loop
    auto var_counter = [&](auto&... xargs) {
      return count_vars(xargs...);
    };
    auto var_copier = [&](const auto&... xargs) {
      return std::tuple<decltype(deep_copy_vars(xargs))...>(
        deep_copy_vars(xargs)...);
    };
    auto vari_saver = [&](int i, int nvars, vari** varis) {
      return [=](const auto&... xargs) {
        save_varis(varis + nvars*i, xargs...);
      };
    };

    int S = result.size();

    // Assuming that the number of the vars at each iteration of the loop is
    // the same (as the operations at each iteration should be the same), we can
    // just count vars at the first iteration.
    int nvars = index_fun(0, var_counter, x...);

    vari** varis = ChainableStack::instance_->memalloc_.alloc_array<vari*>(
      S * nvars);
    double* values = ChainableStack::instance_->memalloc_.alloc_array<double>(
      S);
    Eigen::VectorXd partials = Eigen::VectorXd::Zero(S * nvars);

    tbb::parallel_for(
      tbb::blocked_range<size_t>(0, S, grainsize), 
      [&](const tbb::blocked_range<size_t>& r) {
        // Run nested autodiff in this scope
        nested_rev_autodiff nested;

        for (size_t i = r.begin(); i < r.end(); ++i) {
          index_fun(i, vari_saver(i, nvars, varis), x...);
          auto args_tuple_local_copy = index_fun(i, var_copier, x...);

          // Apply specified function to arguments at current iteration
          var out = apply([&](auto&&... args) {
                return app_fun(args...);
              }, args_tuple_local_copy);

          out.grad();

          // Extract value and adjoints to be put into vars on main
          // autodiff stack
          values[i] = std::move(out.vi_->val_);
          apply([&](auto&&... args) {
            accumulate_adjoints(partials.data() + nvars*i,
                                std::forward<decltype(args)>(args)...); },
            std::move(args_tuple_local_copy));
        }
      });
  // Pack values and adjoints into new vars on main autodiff stack
  for(int i = 0; i < S; ++i) {
    result.coeffRef(i) = var(new precomputed_gradients_vari(
      values[i],
      nvars,
      varis + nvars*i,
      partials.data() + nvars*i));
  }
}

}  // namespace math
}  // namespace stan
#endif
