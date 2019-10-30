#ifndef STAN_MATH_PRIM_MAT_META_EIGEN_SEQ_VIEW_HPP
#define STAN_MATH_PRIM_MAT_META_EIGEN_SEQ_VIEW_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/meta.hpp>
#include <vector>

namespace stan {

template <typename S>
class eigen_seq_view {
 public:
  explicit eigen_seq_view(const S& m)
      : m_(m) {}
  int size() const { return 1; }
  auto operator[](int /* i */) const {
    return math::as_eigen(m_);
  }

 private:
  const S& m_;
};

template <typename S, int R, int C>
class eigen_seq_view<std::vector<Eigen::Matrix<S, R, C>>> {
 public:
  explicit eigen_seq_view(const std::vector<Eigen::Matrix<S, R, C>>& m)
      : m_(m) {}
  int size() const { return m_.size(); }
  auto operator[](int i) const { return math::as_eigen(m_[i]);}

 private:
  const std::vector<Eigen::Matrix<S, R, C>>& m_;
};

template <typename S, int R, int C>
class eigen_seq_view<std::vector<Eigen::Array<S, R, C>>> {
 public:
  explicit eigen_seq_view(const std::vector<Eigen::Array<S, R, C>>& m)
      : m_(m) {}
  int size() const { return m_.size(); }
  auto operator[](int i) const { return math::as_eigen(m_[i]);}

 private:
  const std::vector<Eigen::Array<S, R, C>>& m_;
};


template <typename S>
class eigen_seq_view<std::vector<std::vector<S>>> {
 public:
  explicit eigen_seq_view(const std::vector<std::vector<S>>& m)
      : m_(m) {}
  int size() const { return m_.size(); }
  auto operator[](int i) const { return math::as_eigen(m_[i]);}

 private:
  const std::vector<std::vector<S>>& m_;
};

}  // namespace stan

#endif
