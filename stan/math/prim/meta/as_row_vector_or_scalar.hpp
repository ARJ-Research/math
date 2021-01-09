#ifndef STAN_MATH_PRIM_META_AS_ROW_VECTOR_OR_SCALAR_HPP
#define STAN_MATH_PRIM_META_AS_ROW_VECTOR_OR_SCALAR_HPP

#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/math/prim/meta/holder.hpp>
#include <stan/math/prim/meta/is_stan_scalar.hpp>
#include <stan/math/prim/meta/is_eigen.hpp>
#include <stan/math/prim/meta/is_vector.hpp>
#include <vector>

namespace stan {
namespace math {

/** \ingroup type_trait
 * Converts input argument to a row vector or a scalar. For scalar inputs
 * that is an identity function.
 *
 * @tparam T Type of scalar element.
 * @param a Specified scalar.
 * @return the scalar.
 */
template <typename T, require_stan_scalar_t<T>* = nullptr>
inline T as_row_vector_or_scalar(const T& a) {
  return a;
}

/** \ingroup type_trait
 * Converts input argument to a row vector or a scalar. For row vector
 * inputs this is an identity function.
 *
 * @tparam T Type of scalar element.
 * @param a Specified vector.
 * @return Same vector.
 */
template <typename T, require_eigen_row_vector_t<T>* = nullptr>
inline T&& as_row_vector_or_scalar(T&& a) {
  return std::forward<T>(a);
}

/** \ingroup type_trait
 * Converts input argument to a row vector or a scalar. For a row vector
 * input this is transpose.
 *
 * @tparam T Type of scalar element.
 * @param a Specified vector.
 * @return Transposed vector.
 */
template <typename T, require_eigen_col_vector_t<T>* = nullptr,
          require_not_eigen_row_vector_t<T>* = nullptr>
inline auto as_row_vector_or_scalar(T&& a) {
  return make_holder([](auto& x) { return x.transpose(); }, std::forward<T>(a));
}

/** \ingroup type_trait
 * Converts input argument to a row vector or a scalar. std::vector will be
 * converted to a row vector.
 *
 * @tparam T Type of scalar element.
 * @param a Specified vector.
 * @return input converted to a row vector.
 */
template <typename T, require_std_vector_t<T>* = nullptr>
inline auto as_row_vector_or_scalar(T&& a) {
  using plain_vector = Eigen::Matrix<value_type_t<T>, 1, Eigen::Dynamic>;
  using optionally_const_vector
      = std::conditional_t<std::is_const<std::remove_reference_t<T>>::value,
                           const plain_vector, plain_vector>;
  using T_map = Eigen::Map<optionally_const_vector>;
  return make_holder([](auto& x) { return T_map(x.data(), x.size()); },
                     std::forward<T>(a));
}

template <typename T,
          require_not_stan_scalar_t<T>* = nullptr>
inline auto as_row_vector(T&& x) {
  return as_row_vector_or_scalar(std::forward<T>(x));
}

template <typename T,
          require_stan_scalar_t<T>* = nullptr>
inline auto as_row_vector(T&& x) {
  return make_holder([](auto& x) { return Eigen::Map<Eigen::Matrix<value_type_t<T>,1,Eigen::Dynamic>>(&x, 1); },
                     std::forward<T>(x));
}

}  // namespace math
}  // namespace stan

#endif
