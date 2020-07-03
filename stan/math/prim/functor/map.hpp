#ifndef STAN_MATH_PRIM_FUNCTOR_MAP_HPP
#define STAN_MATH_PRIM_FUNCTOR_MAP_HPP

#include <stan/math/prim/meta/require_generics.hpp>
#include <stan/math/prim/meta/is_stan_scalar.hpp>
#include <stan/math/prim/functor/apply.hpp>

#include <tbb/task_arena.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include <algorithm>
#include <tuple>
#include <vector>

namespace stan {
namespace math {


template <typename ApplyFunction, typename OutputType, typename... Args>
inline void map(OutputType&& result, int grainsize, std::ostream* msgs,
                         Args&&... args) {
    tbb::parallel_for(
        tbb::blocked_range<std::size_t>(0, result.size(), grainsize), 
        [&](const tbb::blocked_range<size_t>& r) {
        for (size_t i = r.begin(); i < r.end(); ++i)
          result(i) = ApplyFunction()(i, args...);
      });
}

}  // namespace math
}  // namespace stan

#endif
