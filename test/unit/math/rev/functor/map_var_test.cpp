#include <stan/math.hpp>
#include <stan/math/prim/functor/map_variadic.hpp>
#include <stan/math/rev/functor/map_variadic.hpp>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <iostream>
#include <cmath>
#include <gtest/gtest.h>


struct test_fun {
  template <typename Arg>
  auto operator()(size_t index, Arg&& arg) {
    return stan::math::exp(arg.coeffRef(index));
  }
};

struct test_fun_binary {
  template <typename Arg1, typename Arg2>
  auto operator()(size_t index, Arg1&& arg1, Arg2&& arg2) {
    return stan::math::distance(arg1.coeffRef(index), arg2);
  }
};

struct example_fun {
  template <typename Arg1, typename Arg2, typename Arg3>
  auto operator()(size_t index, Arg1&& arg1, Arg2&& arg2, Arg3&& arg3) {
    return stan::math::distance(arg1.coeffRef(index), arg2) + arg3.coeffRef(index);
  }
};

TEST(map_var, tbb) {
  stan::math::matrix_v inv = Eigen::MatrixXd::Random(100,100);
  stan::math::matrix_v inv2 = inv;
  stan::math::matrix_v outv(100,100);
  stan::math::matrix_v outv2 = stan::math::exp(inv2);
  std::ostream msgs(NULL);

  stan::math::map_variadic<test_fun>(outv,1,&msgs,inv);

  outv(3647).grad();
  outv2(3647).grad();

  std::cout << outv(3647).val() << std::endl
            << outv(3647).adj() << std::endl
            << outv2(3647).val() << std::endl
            << outv2(3647).adj() << std::endl;

  stan::math::vector_v outv3(10000);
  stan::math::vector_v outv4(10000);
  stan::math::var offset = 5;
/*
  stan::math::map_variadic<test_fun_binary>(outv3,1,&msgs,inv2,offset);
  stan::math::map_variadic<example_fun>(outv3,1,&msgs,inv2,offset,inv);
  for(int i = 0; i < 10000; ++i) {
    outv4[i] = stan::math::distance(inv2[i], offset);
  }

  outv3[0].grad();
  outv4[0].grad();

  stan::math::var t = outv3[0] * outv3[1] + outv3[2];
  t.grad();

  std::cout << outv3[0].val() << std::endl
            << outv3[0].adj() << std::endl
            << outv4[0].val() << std::endl
            << outv4[0].adj() << std::endl
            << t.adj();*/
}
