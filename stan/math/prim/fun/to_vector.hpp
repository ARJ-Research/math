#ifndef STAN_MATH_PRIM_FUN_TO_VECTOR_HPP
#define STAN_MATH_PRIM_FUN_TO_VECTOR_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/fun/Eigen.hpp>
#include <vector>

namespace stan {
namespace math {
namespace internal {
  template <typename EigMat, typename IndVec>
  class to_vector_functor {
    const EigMat& eig_mat;
    const IndVec& ind_vec1;
    const IndVec& ind_vec2;

   public:
    to_vector_functor(const EigMat& arg1, const IndVec& arg2, const IndVec& arg3)
        : eig_mat(arg1), ind_vec1(arg2), ind_vec2(arg3) {}

    inline decltype(auto) operator()(Eigen::Index index) const {
      return eig_mat.coeff(ind_vec1[index], ind_vec2[index]);
    }
  };
}  // namespace internal

// vector to_vector(matrix, int[], int[])
template <typename EigMat, require_eigen_t<EigMat>* = nullptr>
inline Eigen::Matrix<scalar_type_t<EigMat>, Eigen::Dynamic, 1>
  to_vector(const EigMat& matrix, const std::vector<int>& ind1,
            const std::vector<int>& ind2) {
  check_size_match("to_vector", "row indexes", ind1.size(),
                   "column indexes", ind2.size());
  check_bounded("to_vector", "row indexes", ind1, 0, matrix.rows() - 1);
  check_bounded("to_vector", "column indexes", ind2, 0, matrix.cols() - 1);

  using EigRtn = Eigen::Matrix<value_type_t<EigMat>, Eigen::Dynamic, 1>;
  
  return EigRtn::NullaryExpr(ind1.size(),
          internal::to_vector_functor<const EigMat,
                                      std::vector<int>>(matrix, ind1, ind2));
}

// vector to_vector(matrix)
// vector to_vector(row_vector)
// vector to_vector(vector)
template <typename EigMat, require_eigen_t<EigMat>* = nullptr>
inline Eigen::Matrix<value_type_t<EigMat>, Eigen::Dynamic, 1> to_vector(
    const EigMat& matrix) {
  using T = value_type_t<EigMat>;
  Eigen::Matrix<T, Eigen::Dynamic, 1> res(matrix.size());
  Eigen::Map<
      Eigen::Matrix<T, EigMat::RowsAtCompileTime, EigMat::ColsAtCompileTime>>
      res_map(res.data(), matrix.rows(), matrix.cols());
  res_map = matrix;
  return res;
}

// vector to_vector(real[])
template <typename T>
inline Eigen::Matrix<T, Eigen::Dynamic, 1> to_vector(
    const std::vector<T>& vec) {
  return Eigen::Matrix<T, Eigen::Dynamic, 1>::Map(vec.data(), vec.size());
}

// vector to_vector(int[])
inline Eigen::Matrix<double, Eigen::Dynamic, 1> to_vector(
    const std::vector<int>& vec) {
  int R = vec.size();
  Eigen::Matrix<double, Eigen::Dynamic, 1> result(R);
  for (int i = 0; i < R; i++) {
    result(i) = vec[i];
  }
  return result;
}

}  // namespace math
}  // namespace stan
#endif
