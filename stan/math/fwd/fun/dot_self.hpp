#ifndef STAN_MATH_FWD_FUN_DOT_SELF_HPP
#define STAN_MATH_FWD_FUN_DOT_SELF_HPP

#include <stan/math/prim/err.hpp>
#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/fwd/fun/dot_product.hpp>

namespace stan {
namespace math {

/**
 * Returns the dot product of a vector with itself.
 *
 * @tparam R number of rows, can be Eigen::Dynamic; one of R or C must be 1
 * @tparam C number of columns, can be Eigen::Dynamic; one of R or C must be 1
 * @param[in] v Vector.
 * @return Dot product of the vector with itself.
 */
template <typename T, require_t<is_fvar<scalar_type_t<T>>>...>
inline auto dot_self(const T& x) {
  return apply_vector_unary<T>::reduce(x, [&](const auto& v) {
    check_vector("dot_self", "v", v);
    using T_inner = typename scalar_type_t<T>::Scalar;
    Eigen::Matrix<T_inner, -1, 1> v_val = v.val();
    return fvar<T_inner>(v_val.squaredNorm(), 
                         2 * v_val.cwiseProduct(v.d()).sum());
  });
}

}  // namespace math
}  // namespace stan
#endif
