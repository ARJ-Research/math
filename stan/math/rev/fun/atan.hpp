#ifndef STAN_MATH_REV_FUN_ATAN_HPP
#define STAN_MATH_REV_FUN_ATAN_HPP

#include <stan/math/rev/meta.hpp>
#include <stan/math/rev/core.hpp>
#include <cmath>

namespace stan {
namespace math {

namespace internal {
class atan_vari : public op_v_vari {
 public:
  explicit atan_vari(vari* avi) : op_v_vari(std::atan(avi->val_), avi) {}
  void chain() { avi_->adj_ += adj_ / (1.0 + (avi_->val_ * avi_->val_)); }
};

template <typename T>
class atan_matrix_vari : public vari {
 public:
  int A_rows_;
  int A_cols_;
  int A_size_;
  double* Ad_;
  vari** variRefA_;
  vari** variRefAtan_;

  /**
   * Constructor for exp_matrix_vari.
   *
   * All memory allocated in
   * ChainableStack's stack_alloc arena.
   *
   * It is critical for the efficiency of this object
   * that the constructor create new varis that aren't
   * popped onto the var_stack_, but rather are
   * popped onto the var_nochain_stack_. This is
   * controlled by the second argument to
   * vari's constructor.
   *
   * @param A matrix
   */
  explicit atan_matrix_vari(const T& A)
      : vari(0.0),
        A_rows_(A.rows()),
        A_cols_(A.cols()),
        A_size_(A.size()),
        Ad_(ChainableStack::instance_->memalloc_.alloc_array<double>(A_size_)),
        variRefA_(
            ChainableStack::instance_->memalloc_.alloc_array<vari*>(A_size_)),
        variRefAtan_(ChainableStack::instance_->memalloc_.alloc_array<vari*>(
            A_size_)) {
    using Eigen::Map;
    Map<matrix_vi>(variRefA_, A_rows_, A_cols_) = A.vi();
    Map<matrix_d> Ad(Ad_, A_rows_, A_cols_);
    Ad = A.val();
    Map<matrix_vi>(variRefAtan_, A_rows_, A_cols_).array()
        = Ad.array().atan().unaryExpr([](double x) { return new vari(x, false); });
  }

  virtual void chain() {
    using Eigen::Map;
    Map<matrix_vi> RefAtan(variRefAtan_, A_rows_, A_cols_);
    Map<matrix_d> Ad(Ad_, A_rows_, A_cols_);
    Map<matrix_vi>(variRefA_, A_rows_, A_cols_).adj().array()
          += RefAtan.adj().array() / (1 + Ad.val().array().square());
  }
};
}  // namespace internal

/**
 * Return the principal value of the arc tangent, in radians, of the
 * specified variable (cmath).
 *
 * The derivative is defined by
 *
 * \f$\frac{d}{dx} \arctan x = \frac{1}{1 + x^2}\f$.
 *
 *
   \f[
   \mbox{atan}(x) =
   \begin{cases}
     \arctan(x) & \mbox{if } -\infty\leq x \leq \infty \\[6pt]
     \textrm{NaN} & \mbox{if } x = \textrm{NaN}
   \end{cases}
   \f]

   \f[
   \frac{\partial\, \mbox{atan}(x)}{\partial x} =
   \begin{cases}
     \frac{\partial\, \arctan(x)}{\partial x} & \mbox{if } -\infty\leq x\leq
 \infty \\[6pt] \textrm{NaN} & \mbox{if } x = \textrm{NaN} \end{cases} \f]

   \f[
   \frac{\partial \, \arctan(x)}{\partial x} = \frac{1}{x^2+1}
   \f]
 *
 * @param a Variable in range [-1, 1].
 * @return Arc tangent of variable, in radians.
 */
inline var atan(const var& a) { return var(new internal::atan_vari(a.vi_)); }

template <typename Container,
          require_container_st<is_container, is_var, Container>...>
inline auto atan(const Container& x) {
  return apply_vector_unary<Container>::apply(
      x, [](const auto& v) {
        using T_plain = plain_type_t<decltype(v)>;
        using T_ref = Eigen::Ref<const T_plain>;

        const T_ref& v_ref = v;
        auto* baseVari = new internal::atan_matrix_vari<T_ref>(v_ref);
        T_plain AB_v(v_ref.rows(), v_ref.cols());
        AB_v.vi() = Eigen::Map<matrix_vi>(baseVari->variRefAtan_, v_ref.rows(), v_ref.cols());

        return AB_v;
});
}
}  // namespace math
}  // namespace stan
#endif
