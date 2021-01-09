#ifndef STAN_MATH_PRIM_FUNCTOR_ROWWISE_HPP
#define STAN_MATH_PRIM_FUNCTOR_ROWWISE_HPP

#include <stan/math/prim/meta.hpp>
#include <vector>

namespace stan {
namespace math {

template <typename T1, typename... Ts>
inline size_t max_rows(const T1& x1, const Ts&... xs) {
  return std::max({rows(x1), rows(xs)...});
}

template <typename F, typename T, typename... TArgs>
auto rowwise(F&& f, T&& x, TArgs&&... xargs) {
  decltype(auto) iter_0 = f(x.row(0), std::forward<TArgs>(xargs)...);
  using iter_t = decltype(iter_0);

  using row_return_t =
    std::conditional_t<is_stan_scalar<iter_t>::value,
                       Eigen::Matrix<iter_t, Eigen::Dynamic, 1>,
                       Eigen::Matrix<scalar_type_t<iter_t>, Eigen::Dynamic,
                                     Eigen::Dynamic>>;

  size_t rs = max_rows(std::forward<T>(x),
                       std::forward<TArgs>(xargs)...);
  row_return_t rtn(rs, stan::math::size(iter_0));
  rtn.row(0) = as_row_vector(iter_0);
  for(size_t i = 1; i < rs; ++i) {
    rtn.row(i) = as_row_vector(f(x.row(i), std::forward<TArgs>(xargs)...));
  }
  return rtn;
}

}  // namespace math
}  // namespace stan
#endif
