#ifndef STAN_MATH_REV_FUN_SD_HPP
#define STAN_MATH_REV_FUN_SD_HPP

#include <stan/math/rev/meta.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/prim/err.hpp>
#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/math/prim/fun/typedefs.hpp>
#include <stan/math/prim/fun/inv_sqrt.hpp>
#include <cmath>
#include <vector>

namespace stan {
namespace math {
namespace internal {

// if x.size() = N, and x[i] = x[j] =
// then lim sd(x) -> 0 [ d/dx[n] sd(x) ] = sqrt(N) / N

inline var calc_sd(size_t size, const var* dtrs) {
  using std::sqrt;
  vari** varis = ChainableStack::instance_->memalloc_.alloc_array<vari*>(size);
  double* partials = ChainableStack::instance_->memalloc_.alloc_array<double>(size);
  Eigen::Map<vector_vi> varis_map(varis, size);
  Eigen::Map<const vector_v> dtrs_map(dtrs, size);
  Eigen::Map<vector_d> partials_map(partials, size);

  double size_m1 = size - 1;
  varis_map = dtrs_map.vi();
  vector_d dtrs_val = dtrs_map.val();
  vector_d diff = dtrs_val.array() - dtrs_val.mean();
  double sum_of_squares = diff.squaredNorm();
  double sd = sqrt(sum_of_squares / size_m1);

  if (sum_of_squares < 1e-20) {
    partials_map.fill(inv_sqrt(static_cast<double>(size)));
  } else {
    partials_map = diff.array() / (sd * size_m1);
  }
  return var(new stored_gradient_vari(sd, size, varis, partials));
}

}  // namespace internal

/*
 * Return the sample standard deviation of the specified vector,
 * row vector, or matrix.  Raise domain error if size is not
 * greater than zero.
 *
 * @tparam R number of rows, can be Eigen::Dynamic
 * @tparam C number of columns, can be Eigen::Dynamic
 * @param[in] m input matrix
 * @return sample standard deviation of specified matrix
 */
template <typename T, require_t<is_var<scalar_type_t<T>>>...>
inline auto sd(const T& x) {
  return apply_vector_unary<T>::reduce(x, [&](const auto& m) {
    using std::sqrt;
    check_nonzero_size("sd", "m", m);
    if (m.size() == 1) {
      return scalar_type_t<T>(0.0);
    }
    return internal::calc_sd(m.size(), m.data());
  });
}

}  // namespace math
}  // namespace stan
#endif
