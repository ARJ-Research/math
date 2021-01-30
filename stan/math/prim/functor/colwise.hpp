#ifndef STAN_MATH_PRIM_FUNCTOR_COLWISE_HPP
#define STAN_MATH_PRIM_FUNCTOR_COLWISE_HPP

#include <stan/math/prim/functor/rowwise.hpp>
#include <vector>

namespace stan {
namespace math {
namespace internal {

template <typename T1, typename... Ts>
inline size_t cols_equal(const T1& x1, const Ts&... xs) {
  auto v = {cols(x1), cols(xs)...};
  return std::all_of(v.begin(), v.end(),
                     [&](int i){ return i == *v.begin(); });
}

template <typename TupleT>
decltype(auto) col_index(const TupleT& x, size_t i) {
  return apply([&](auto&&... args){
    return std::forward_as_tuple(col(args, i + 1)...);
  }, x);
}

template <typename T>
using col_return_t
  = std::conditional_t<is_stan_scalar<T>::value,
                       Eigen::Matrix<T, 1, Eigen::Dynamic>,
                       Eigen::Matrix<scalar_type_t<T>,
                                     Eigen::Dynamic,
                                     Eigen::Dynamic>>;
} // namespace internal

template <typename... TArgs>
auto colwise(const TArgs&... args) {
  // Get location of functor in parameter pack
  constexpr size_t pos = internal::type_count<internal::is_stan_type,
                                              TArgs...>();
  constexpr size_t arg_size = sizeof...(TArgs);
  decltype(auto) args_tuple = std::forward_as_tuple(args...);
  using TupleT = decltype(args_tuple);

  // Split parameter pack into two tuples, separated by the functor
  decltype(auto) t1 = internal::subset_tuple(std::forward<TupleT>(args_tuple),
                         std::make_index_sequence<pos>{});
  decltype(auto) t2 = internal::subset_tuple(std::forward<TupleT>(args_tuple),
                         internal::add_offset<pos+1>(
                          std::make_index_sequence<arg_size-pos-1>{}));

  // Check that inputs to be iterated have the same number of rows
  bool eqrows = apply([&](auto&&... args) {
                        return internal::cols_equal(args...); },
                      std::forward<decltype(t1)>(t1));

  if (!eqrows) {
    std::ostringstream msg;
    msg << "Inputs to be iterated over must have the same number of cols!";
    throw std::invalid_argument(msg.str());
  }

  size_t cs = std::get<0>(std::forward<decltype(t1)>(t1)).cols();

  // Extract functor from parameter pack
  decltype(auto) f = std::get<pos>(args_tuple);

  // Evaluate first iteration, needed to determine type and size of return
  decltype(auto) iter_0
    = apply([&](auto&&... args) { return f(args...); },
            std::tuple_cat(internal::col_index(std::forward<decltype(t1)>(t1), 0),
                           std::forward<decltype(t2)>(t2)));

  internal::col_return_t<decltype(iter_0)> rtn(stan::math::size(iter_0), cs);
  rtn.col(0) = as_column_vector(std::move(iter_0));

  for(size_t i = 1; i < cs; ++i) {
    rtn.col(i) = as_column_vector(
      apply([&](auto&&... args) { return f(args...); },
        std::tuple_cat(internal::col_index(std::forward<decltype(t1)>(t1), i),
                        std::forward<decltype(t2)>(t2)))
    );
  }

  return rtn;
}
}  // namespace math
}  // namespace stan
#endif
