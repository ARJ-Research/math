#ifndef STAN_MATH_REV_FUN_DOT_SELF_HPP
#define STAN_MATH_REV_FUN_DOT_SELF_HPP

#include <stan/math/rev/meta.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/prim/err.hpp>
#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/math/prim/fun/square.hpp>
#include <vector>

namespace stan {
namespace math {

namespace internal {
class dot_self_vari : public vari {
 protected:
  vari** v_;
  size_t size_;

 public:
  dot_self_vari(vari** v, size_t size)
      : vari(var_dot_self(v, size)), v_(v), size_(size) {}

  template <typename T, require_eigen_t<T>...>
  explicit dot_self_vari(const T& v)
      : vari(v.val().squaredNorm()), size_(v.size()) {
    v_ = ChainableStack::instance_->memalloc_.alloc_array<vari*>(size_);
    Eigen::Map<vector_vi>(v_, size_) = v.vi();
  }

  inline static double var_dot_self(vari** v, size_t size) {
    return Eigen::Map<vector_vi>(v, size).val().squaredNorm();
  }

  virtual void chain() {
    Eigen::Map<vector_vi> v_map(v_, size_);
    v_map.adj() += adj_ * 2.0 * v_map.val();
  }
};
}  // namespace internal

/**
 * Returns the dot product of a vector with itself.
 *
 * @tparam R number of rows, can be Eigen::Dynamic; one of R or C must be 1
 * @tparam C number of columns, can be Eigen::Dynamic; one of R or C must be 1
 * @param[in] v Vector.
 * @return Dot product of the vector with itself.
 */
template <typename T, require_t<is_var<scalar_type_t<T>>>...>
inline auto dot_self(const T& x) {
  return apply_vector_unary<T>::reduce(x, [&](const auto& v) {
    check_vector("dot_self", "v", v);
    return var(new internal::dot_self_vari(v));
  });
}

}  // namespace math
}  // namespace stan
#endif
