#include <stan/math.hpp>
#include <stan/math/prim/functor/map_variadic.hpp>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>


struct test_fun {
  template <typename Arg>
  auto operator()(size_t index, Arg&& arg) {
    return stan::math::exp(arg[index]);
  }
};

struct test_fun_binary {
  template <typename Arg1, typename Arg2>
  auto operator()(size_t index, Arg1&& arg1, Arg2&& arg2) {
    return stan::math::distance(arg1[index], arg2);
  }
};

int main() {
  Eigen::VectorXd inv = Eigen::VectorXd::Random(10000);
  Eigen::VectorXd outv(10000);
  outv.setZero();
  Eigen::VectorXd outv2(10000);
  outv2.setZero();
  double offset = 5;
  std::ostream msgs(NULL);

  stan::math::map_variadic<test_fun>(outv,1,&msgs,inv);
  stan::math::map_variadic<test_fun_binary>(outv2,1,&msgs,inv,offset);

}