#ifndef STAN_MATH_PRIM_FUNCTOR_ROWWISE_HPP
#define STAN_MATH_PRIM_FUNCTOR_ROWWISE_HPP

#include <stan/math/prim/meta.hpp>
#include <vector>

namespace stan {
namespace math {

template <typename T,
          require_eigen_matrix_dynamic_t<T>* = nullptr>
decltype(auto) row_index(const T& x, size_t i) {
  return x.row(i);
}

template <typename T,
          require_not_eigen_matrix_dynamic_t<T>* = nullptr>
decltype(auto) row_index(T&& x, size_t i) {
  return std::forward<T>(x);
}

template <typename T>
using row_return_t = std::conditional_t<is_stan_scalar<T>::value,
                                        Eigen::Matrix<T, -1, 1>,
                                        Eigen::Matrix<scalar_type_t<T>,-1,-1>>;

template <typename T,
          require_eigen_col_vector_t<T>* = nullptr>
decltype(auto) as_row_vector(const T& x) {
  return x.transpose();
}

template <typename T,
          require_eigen_row_vector_t<T>* = nullptr>
decltype(auto) as_row_vector(const T& x) {
  return x;
}

template <typename T,
          require_stan_scalar_t<T>* = nullptr>
decltype(auto) as_row_vector(const T& x) {
  return Eigen::Map<const Eigen::Matrix<T, 1, -1>>(&x, 1);
}

template <typename T1, typename... Ts>
inline size_t max_rows(const T1& x1, const Ts&... xs) {
  return std::max({rows(x1), rows(xs)...});
}

template <typename ArgsTuple, typename F, typename TArgsTuple>
auto rowwise_impl(const ArgsTuple& x, const F& f, const TArgsTuple& xargs) {

  decltype(auto) iter_0 = apply([&](auto&&... args) { return f(args...); }, std::tuple_cat(apply([&](auto&&... args) { return std::make_tuple(row_index(args, 0)...); }, x ), xargs) );
  
  size_t rs = apply([&](auto&&... args) { return max_rows(args...); }, x);
  row_return_t<decltype(iter_0)> rtn(rs,
                                     stan::math::size(iter_0));
  rtn.row(0) = as_row_vector(iter_0);
  for(size_t i = 1; i < rs; ++i) {
    rtn.row(i) = as_row_vector(
       apply([&](auto&&... args) { return f(args...); }, std::tuple_cat(apply([&](auto&&... args) { return std::make_tuple(row_index(args, i)...); }, x ), xargs) )
      );
  }
  return rtn;
}

template <template<class> class Cond>
size_t cond_index(int count) {
  return count;
}

template <template<class> class Cond, typename T>
size_t cond_index(int count, T&& x) {
  if (!Cond<T>::value) {
      return count;
  }
  return count + 1;
}

template <template<class> class Cond, typename T, typename... TArgs>
size_t cond_index(int count, T&& x, TArgs&&... args) {
  if (!Cond<T>::value) {
      return count;
  }
  return cond_index<Cond>(count + 1, std::forward<TArgs>(args)...);
}

template <typename T>
auto newt = conjunction<is_stan_scalar<T>, is_container<T>>::value;


template <typename... TArgs>
auto rowwise(TArgs&&... args) {
  return cond_index<newt>(0, std::forward<TArgs>(args)...);
}

}  // namespace math
}  // namespace stan
#endif
