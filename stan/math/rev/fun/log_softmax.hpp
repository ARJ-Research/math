#ifndef STAN_MATH_REV_FUN_LOG_SOFTMAX_HPP
#define STAN_MATH_REV_FUN_LOG_SOFTMAX_HPP

#include <stan/math/rev/core.hpp>
#include <stan/math/rev/fun/typedefs.hpp>
#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/err.hpp>
#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/math/prim/fun/log_softmax.hpp>
#include <stan/math/prim/fun/softmax.hpp>
#include <stan/math/prim/fun/to_ref.hpp>
#include <stan/math/prim/fun/typedefs.hpp>
#include <cmath>
#include <vector>

namespace stan {
namespace math {

namespace internal {

class log_softmax_elt_vari : public vari {
 private:
  vari** alpha_;
  const double* softmax_alpha_;
  const int size_;  // array sizes
  const int idx_;   // in in softmax output

 public:
  log_softmax_elt_vari(double val, vari** alpha, const double* softmax_alpha,
                       int size, int idx)
      : vari(val),
        alpha_(alpha),
        softmax_alpha_(softmax_alpha),
        size_(size),
        idx_(idx) {}
  void chain() {
    for (int m = 0; m < size_; ++m) {
      if (m == idx_) {
        alpha_[m]->adj_ += adj_ * (1 - softmax_alpha_[m]);
      } else {
        alpha_[m]->adj_ -= adj_ * softmax_alpha_[m];
      }
    }
  }
};

}  // namespace internal

/**
 * Return the log softmax of the specified vector or container of vectors.
 *
 * The gradient calculations are unfolded.
 *
 * @tparam T Type of input vector or matrix.
 * @param[in] x Unconstrained input vector.
 * @return Softmax of the input.
 * @throw std::domain_error If the input vector is size 0.
 */
template <typename T,
          require_any_t<conjunction<is_container<T>, is_var<scalar_type_t<T>>>,
                        is_var_matrix<T>>* = nullptr>
inline auto log_softmax(const T& x) {
  return apply_vector_unary<ref_type_t<T>>::apply(
      to_ref(x), [&](const auto& alpha) {
        check_nonzero_size("log_softmax", "alpha", alpha);
        using alpha_plain = plain_type_t<decltype(alpha)>;

        const auto& alpha_val = to_ref(value_of(alpha));
        const auto& theta = to_ref(alpha_val.array() - alpha_val.maxCoeff());
        arena_t<promote_scalar_t<double, alpha_plain>> res_val
            = theta.array() - log(theta.exp().sum());

        arena_t<alpha_plain> res = res_val;
        auto alpha_arena = to_arena(alpha);

        reverse_pass_callback([alpha_arena, res, res_val]() mutable {
          const auto& res_adj = to_ref(res.adj());
          alpha_arena.adj()
              += res_adj - (res_adj.sum() * res_val.array().exp()).matrix();
        });

        return alpha_plain(res);
      });
}

}  // namespace math
}  // namespace stan
#endif
