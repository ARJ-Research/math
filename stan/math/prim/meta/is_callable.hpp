#ifndef STAN_MATH_PRIM_META_IS_CALLABLE_HPP
#define STAN_MATH_PRIM_META_IS_CALLABLE_HPP

#include <stan/math/prim/meta/void_t.hpp>
#include <type_traits>

namespace stan {

/**
 * Checks whether a function will be well-formed if passed a given set of
 * arguments.
 * @tparam F The type of function.
 * @tparam Args The arguments to pass to the function.
 */
template <typename F, typename... Args>
struct is_callable {
  template<class, class = void>
  struct is_callable_impl : std::false_type
  { };
  template<class T>
  struct is_callable_impl<T,
    void_t<decltype(std::declval<T>()(std::declval<Args>()...))>>
      : std::true_type
  { };
  static constexpr bool value = is_callable_impl<F>::value;
};

}  // namespace stan
#endif
