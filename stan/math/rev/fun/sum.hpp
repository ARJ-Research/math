#ifndef STAN_MATH_REV_FUN_SUM_HPP
#define STAN_MATH_REV_FUN_SUM_HPP

#include <stan/math/rev/core.hpp>
#include <stan/math/rev/meta.hpp>
#include <stan/math/rev/fun/sum.hpp>
#include <stan/math/prim/fun/Eigen.hpp>
#include <vector>

namespace stan {
namespace math {

namespace internal {
/**
 * Class for sums of variables constructed with standard vectors.
 * There's an extension for Eigen matrices.
 */
class sum_v_vari : public vari {
 protected:
  vari** v_;
  size_t length_;

 public:
  explicit sum_v_vari(double value, vari** v, size_t length)
      : vari(value), v_(v), length_(length) {}

  template<typename T>
  explicit sum_v_vari(const T& v1)
      : vari(v1.val().sum()), length_(v1.size()) {
    v_ = ChainableStack::instance_->memalloc_.alloc_array<vari*>(length_);
    Eigen::Map<matrix_vi>(v_, v1.rows(), v1.cols()) = v1.vi();
  }

  virtual void chain() {
    Eigen::Map<vector_vi> v_map(v_, length_);
    v_map.adj().array() += adj_;
    }
};
}

/**
 * Returns the sum of the coefficients of the specified
 * matrix, column vector or row vector.
 *
 * @tparam R number of rows, can be Eigen::Dynamic
 * @tparam C number of columns, can be Eigen::Dynamic
 * @param m Specified matrix or vector.
 * @return Sum of coefficients of matrix.
 */
template <typename T, require_t<is_var<scalar_type_t<T>>>...>
inline auto sum(const T& x) {
  return apply_vector_unary<T>::reduce(x, [&](const auto& m) {
    if (m.size() == 0) {
      return scalar_type_t<T>(0.0);
    }
    return var(new internal::sum_v_vari(m));
  });
}

}  // namespace math
}  // namespace stan
#endif
