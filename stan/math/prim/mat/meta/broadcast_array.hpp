#include <stan/math/prim/scal/meta/broadcast_array.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stdexcept>
#include <vector>

#ifndef STAN_MATH_PRIM_MAT_META_BROADCAST_ARRAY_HPP
#define STAN_MATH_PRIM_MAT_META_BROADCAST_ARRAY_HPP

namespace stan {
namespace math {
namespace internal {
template <typename ViewElt, typename OpElt, int R, int C>
class empty_broadcast_array<ViewElt, Eigen::Matrix<OpElt, R, C> > {
 public:
  empty_broadcast_array() {}
  /**
   * Not implemented so cannot be called.
   */
  ViewElt& operator[](int /*i*/);
  /**
   * Not implemented so cannot be called.
   */
  ViewElt& operator()(int /*i*/);
  /**
   * Not implemented so cannot be called.
   */
  void operator=(const Eigen::Matrix<ViewElt, R, C>& /*A*/);
  /**
   * Not implemented so cannot be called.
   */
  void operator=(const std::vector<ViewElt>& /*A*/);
  /**
   * Not implemented so cannot be called.
   */
  void operator+=(Eigen::Matrix<ViewElt, R, C> /*A*/);
  /**
   * Not implemented so cannot be called.
   */
  void operator-=(Eigen::Matrix<ViewElt, R, C> /*A*/);
  /**
   * Not implemented so cannot be called.
   */
  Eigen::Matrix<ViewElt, 1, C>& row(int /*i*/);
  /**
   * Not implemented so cannot be called.
   */
  Eigen::Matrix<ViewElt, R, 1>& col(int /*i*/);
  /**
   * Not implemented so cannot be called.
   */
  ViewElt* data();
};

template <typename ViewElt, typename OpElt>
class empty_broadcast_array<ViewElt, std::vector<OpElt> > {
 public:
  empty_broadcast_array() {}
  /**
   * Not implemented so cannot be called.
   */
  ViewElt& operator[](int /*i*/);
  /**
   * Not implemented so cannot be called.
   */
  ViewElt& operator()(int /*i*/);
  /**
   * Not implemented so cannot be called.
   */
  void operator=(const std::vector<ViewElt>& /*A*/);
  /**
   * Not implemented so cannot be called.
   */
  void operator=(const Eigen::Matrix<ViewElt, -1, -1>& /*A*/);
  /**
   * Not implemented so cannot be called.
   */
  void operator+=(std::vector<ViewElt> /*A*/);
  /**
   * Not implemented so cannot be called.
   */
  void operator-=(std::vector<ViewElt> /*A*/);
  /**
   * Not implemented so cannot be called.
   */
  std::vector<ViewElt>& row(int /*i*/);
  /**
   * Not implemented so cannot be called.
   */
  std::vector<ViewElt>& col(int /*i*/);
  /**
   * Not implemented so cannot be called.
   */
  ViewElt* data();
};
}  // namespace internal
}  // namespace math
}  // namespace stan
#endif
