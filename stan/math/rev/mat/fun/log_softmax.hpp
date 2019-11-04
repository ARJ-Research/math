#ifndef STAN_MATH_REV_MAT_FUN_LOG_SOFTMAX_HPP
#define STAN_MATH_REV_MAT_FUN_LOG_SOFTMAX_HPP

#include <stan/math/prim/arr/err/check_nonzero_size.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat/fun/log_softmax.hpp>
#include <stan/math/prim/mat/fun/softmax.hpp>
#include <stan/math/rev/mat/fun/typedefs.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/prim/meta.hpp>
#include <stan/math/rev/core.hpp>
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
 * Return the softmax of the specified Eigen vector.  Softmax is
 * guaranteed to return a simplex.
 *
 * The gradient calculations are unfolded.
 *
 * @param alpha Unconstrained input vector.
 * @return Softmax of the input.
 * @throw std::domain_error If the input vector is size 0.
 */
template <typename T, typename = require_vector_like_st<is_var, T>>
inline auto log_softmax(const T& alpha) {
  check_nonzero_size("log_softmax", "alpha", alpha);

  auto alpha_vec = as_eigen(alpha);
  const int a_size = alpha.size();

  // TODO(carpenter): replace with array alloc
  vari** alpha_vi_array
      = reinterpret_cast<vari**>(vari::operator new(sizeof(vari*) * a_size));
  Eigen::Map<vector_vi>(alpha_vi_array, a_size) = alpha_vec.vi();

  vector_d alpha_d = alpha_vec.val();

  // fold logic of math::softmax() and math::log_softmax()
  // to save computations

  vector_d diff = (alpha_d.array() - alpha_d.maxCoeff());
  vector_d softmax_alpha_d = diff.array().exp();
  double sum = softmax_alpha_d.sum();
  softmax_alpha_d.array() /= sum;
  vector_d log_softmax_alpha_d = diff.array() - std::log(sum);

  // end fold
  // TODO(carpenter): replace with array alloc
  double* softmax_alpha_d_array
      = reinterpret_cast<double*>(vari::operator new(sizeof(double) * a_size));
  Eigen::Map<vector_d>(softmax_alpha_d_array, a_size) = softmax_alpha_d;

  return_container_t<T> log_softmax_alpha(a_size);
  for (int k = 0; k < a_size; ++k) {
    log_softmax_alpha[k] = var(new internal::log_softmax_elt_vari(
        log_softmax_alpha_d[k], alpha_vi_array, softmax_alpha_d_array, a_size,
        k));
  }
  return log_softmax_alpha;
}

template <typename Vec, require_vector_st<is_var, Vec>...,
          require_vector_vt<is_vector, Vec>...>
inline auto log_softmax(Vec&& v) {
  check_nonzero_size("log_softmax", "v", v);
  std::vector<return_container_t<value_type_t<Vec>>> result(v.size());
  for(int i = 0; i < v.size(); i++){
    result[i] = log_softmax(std::forward<decltype(v[i])>(v[i]));
  }
  return result;
}

}  // namespace math
}  // namespace stan
#endif
