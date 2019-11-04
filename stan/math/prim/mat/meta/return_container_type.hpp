#ifndef STAN_MATH_PRIM_MAT_META_RETURN_CONTAINER_TYPE_HPP
#define STAN_MATH_PRIM_MAT_META_RETURN_CONTAINER_TYPE_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/meta.hpp>

namespace stan {

template <typename T>
struct return_container_type {
  using type = return_type_t<T>;
};

template <typename T>
struct return_container_type<std::vector<T>> {
  using type = std::vector<return_type_t<T>>;
};

template <typename T, int R, int C>
struct return_container_type<Eigen::Matrix<T, R, C>> {
  using type = Eigen::Matrix<return_type_t<T>, R, C>;
};

template <typename T, int R, int C>
struct return_container_type<Eigen::Array<T, R, C>> {
  using type = Eigen::Array<return_type_t<T>, R, C>;
};

template <typename T, int R, int C>
struct return_container_type<Eigen::Map<Eigen::Matrix<T, R, C>>> {
  using type = Eigen::Matrix<return_type_t<T>, R, C>;
};

template <typename T, int R, int C>
struct return_container_type<Eigen::Map<Eigen::Array<T, R, C>>> {
  using type = Eigen::Array<return_type_t<T>, R, C>;
};

template <typename T, int R, int C, int DiagIndex>
struct return_container_type<Eigen::Diagonal<Eigen::Matrix<T, R, C>, DiagIndex>> {
  using type = Eigen::Matrix<return_type_t<T>, -1, 1>;
};

template <typename T>
struct return_container_type<std::vector<T>&> {
  using type = std::vector<return_type_t<T>>;
};

template <typename T, int R, int C>
struct return_container_type<Eigen::Matrix<T, R, C>&> {
  using type = Eigen::Matrix<return_type_t<T>, R, C>;
};

template <typename T, int R, int C>
struct return_container_type<Eigen::Array<T, R, C>&> {
  using type = Eigen::Array<return_type_t<T>, R, C>;
};

template <typename T, int R, int C>
struct return_container_type<Eigen::Map<Eigen::Matrix<T, R, C>>&> {
  using type = Eigen::Matrix<return_type_t<T>, R, C>;
};

template <typename T, int R, int C>
struct return_container_type<Eigen::Map<Eigen::Array<T, R, C>>&> {
  using type = Eigen::Array<return_type_t<T>, R, C>;
};

template <typename T, int R, int C, int DiagIndex>
struct return_container_type<Eigen::Diagonal<Eigen::Matrix<T, R, C>, DiagIndex>&> {
  using type = Eigen::Matrix<return_type_t<T>, -1, 1>;
};


template <typename T>
using return_container_t = typename return_container_type<T>::type;

}

#endif