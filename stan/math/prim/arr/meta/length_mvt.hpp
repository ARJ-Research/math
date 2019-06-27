#ifndef STAN_MATH_PRIM_ARR_META_LENGTH_MVT_HPP
#define STAN_MATH_PRIM_ARR_META_LENGTH_MVT_HPP

#include <stan/math/prim/scal/meta/length_mvt.hpp>
#include <vector>

namespace stan {

template <typename T>
size_t length_mvt(const std::vector<T>& /* unused */) {
  return 1U;
}

template <typename T>
size_t length_mvt(const std::vector<std::vector<T> >& x) {
  return x.size();
}

}  // namespace stan
#endif
