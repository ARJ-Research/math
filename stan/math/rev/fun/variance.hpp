#ifndef STAN_MATH_REV_FUN_VARIANCE_HPP
#define STAN_MATH_REV_FUN_VARIANCE_HPP

#include <stan/math/rev/meta.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/prim/err.hpp>
#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/math/prim/fun/typedefs.hpp>
#include <vector>

namespace stan {
namespace math {
namespace internal {

inline var calc_variance(size_t size, const var* dtrs) {
  vari** varis = ChainableStack::instance_->memalloc_.alloc_array<vari*>(size);
  double* partials
      = ChainableStack::instance_->memalloc_.alloc_array<double>(size);

  Eigen::Map<const vector_v> dtrs_map(dtrs, size);
  Eigen::Map<vector_vi>(varis, size) = dtrs_map.vi();
  vector_d dtrs_vals = dtrs_map.val();

  vector_d diff = dtrs_vals.array() - dtrs_vals.mean();
  double size_m1 = size - 1;
  Eigen::Map<vector_d>(partials, size) = 2 * diff.array() / size_m1;
  double variance = diff.squaredNorm() / size_m1;

  return var(new stored_gradient_vari(variance, size, varis, partials));
}

}  // namespace internal

/*
 * Return the sample variance of the specified vector, row vector,
 * or matrix.  Raise domain error if size is not greater than
 * zero.
 *
 * @tparam R number of rows, can be Eigen::Dynamic
 * @tparam C number of columns, can be Eigen::Dynamic
 * @param[in] m input matrix
 * @return sample variance of specified matrix
 */
template <typename T, require_t<is_var<scalar_type_t<T>>>...>
inline auto variance(const T& x) {
  return apply_vector_unary<T>::reduce(x, [&](const auto& m) {
    check_nonzero_size("variance", "m", m);
    if (m.size() == 1) {
      return var(0);
    }
    return internal::calc_variance(m.size(), m.data());
  });
}

}  // namespace math
}  // namespace stan
#endif
