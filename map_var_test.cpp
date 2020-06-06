#include <stan/math.hpp>
#include <stan/math/prim/functor/map_variadic.hpp>
#include <stan/math/rev/functor/map_variadic.hpp>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <iostream>
#include <cmath>


struct test_fun {
  template <typename Arg>
  auto operator()(size_t index, Arg&& arg) {
    return stan::math::exp(arg[index]);
  }
};

int main() {
  stan::math::vector_v inv = Eigen::VectorXd::Random(10000);
  stan::math::vector_v outv(10000);
  outv.setZero();
  std::ostream msgs(NULL);

  stan::math::map_variadic<test_fun>(outv,1,&msgs,inv);
}
