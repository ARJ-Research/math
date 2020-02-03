#ifndef STAN_MATH_PRIM_ERR_CHECK_VECTOR_HPP
#define STAN_MATH_PRIM_ERR_CHECK_VECTOR_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/err/invalid_argument.hpp>
#include <stan/math/prim/fun/Eigen.hpp>
#include <sstream>
#include <string>
#include <typeinfo>

namespace stan {
namespace math {

/**
 * Check if the matrix is either a row vector or column vector.
 * This function checks the runtime size of the matrix to check
 * whether it is a row or column vector.
 * @tparam T Type of the matrix
 * @param function Function name (for error messages)
 * @param name Variable name (for error messages)
 * @param x Matrix
 * @throw <code>std::invalid_argument</code> if x is not a row or column
 *   vector.
 */
template <typename T, require_eigen_t<T>...>
inline void check_vector(const char* function, const char* name,
                         const T& x) {
  if (T::RowsAtCompileTime == 1) {
    return;
  }
  if (T::ColsAtCompileTime == 1) {
    return;
  }
  if (x.rows() == 1 || x.cols() == 1) {
    return;
  }

  std::ostringstream msg;
  msg << ") has " << x.rows() << " rows and " << x.cols()
      << " columns but it should be a vector so it should "
      << "either have 1 row or 1 column";
  std::string msg_str(msg.str());
  invalid_argument(function, name, typename scalar_type<T>::type(), "(",
                   msg_str.c_str());
}

}  // namespace math
}  // namespace stan
#endif
