#ifndef STAN_MATH_REV_FUN_COSH_HPP
#define STAN_MATH_REV_FUN_COSH_HPP

#include <stan/math/rev/meta.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/prim/fun/typedefs.hpp>
#include <cmath>

namespace stan {
namespace math {

namespace internal {
class cosh_vari : public op_v_vari {
 public:
  explicit cosh_vari(vari* avi) : op_v_vari(std::cosh(avi->val_), avi) {}
  void chain() { avi_->adj_ += adj_ * std::sinh(avi_->val_); }
};

template <typename Container>
class cosh_matrix_vari : public vari {
 public:
  int A_rows_;
  int A_cols_;
  int A_size_;
  double* Ad_;
  vari** variRefA_;
  vari** variRefCosh_;

  /**
   * Constructor for cosh_matrix_vari.
   *
   * All memory allocated in
   * ChainableStack's stack_alloc arena.
   *
   * @tparam Container Type of Eigen expression/object
   * @param A Eigen expression/object
   */
  explicit cosh_matrix_vari(const Container& A)
      : vari(0.0),
        A_rows_(A.rows()),
        A_cols_(A.cols()),
        A_size_(A.size()),
        Ad_(ChainableStack::instance_->memalloc_.alloc_array<double>(A_size_)),
        variRefA_(
            ChainableStack::instance_->memalloc_.alloc_array<vari*>(A_size_)),
        variRefCosh_(ChainableStack::instance_->memalloc_.alloc_array<vari*>(
            A_size_)) {
    using Eigen::Map;
    Map<matrix_vi>(variRefA_, A_rows_, A_cols_) = A.vi();
    Map<matrix_d> Ad(Ad_, A_rows_, A_cols_);
    Ad = A.val();
    Map<matrix_vi>(variRefCosh_, A_rows_, A_cols_).array()
        = Ad.array().cosh()
                    .unaryExpr([](double x) { return new vari(x, false); });
  }

  virtual void chain() {
    using Eigen::Map;
    Map<matrix_vi> RefCosh(variRefCosh_, A_rows_, A_cols_);
    Map<matrix_d> Ad(Ad_, A_rows_, A_cols_);
    Map<matrix_vi>(variRefA_, A_rows_, A_cols_).adj().array()
          += RefCosh.adj().array() * Ad.array().sinh();
  }
};
}  // namespace internal

/**
 * Return the hyperbolic cosine of the specified variable (cmath).
 *
 * The derivative is defined by
 *
 * \f$\frac{d}{dx} \cosh x = \sinh x\f$.
 *
 *
   \f[
   \mbox{cosh}(x) =
   \begin{cases}
     \cosh(x) & \mbox{if } -\infty\leq x \leq \infty \\[6pt]
     \textrm{NaN} & \mbox{if } x = \textrm{NaN}
   \end{cases}
   \f]

   \f[
   \frac{\partial\, \mbox{cosh}(x)}{\partial x} =
   \begin{cases}
     \sinh(x) & \mbox{if } -\infty\leq x\leq \infty \\[6pt]
     \textrm{NaN} & \mbox{if } x = \textrm{NaN}
   \end{cases}
   \f]
 *
 * @param a Variable.
 * @return Hyperbolic cosine of variable.
 */
inline var cosh(const var& a) { return var(new internal::cosh_vari(a.vi_)); }

/**
 * Return the hyperbolic cosine of each variable in a container.
 *
 * @tparam Container Type of container
 * @param x Container of var
 * @return Hyperbolic cosine of each variable in container.
 */
template <typename Container,
          require_container_st<is_container, is_var, Container>...>
inline auto cosh(const Container& x) {
  return apply_vector_unary<Container>::apply(
      x, [](const auto& v) {
        using T_plain = plain_type_t<decltype(v)>;
        using T_ref = Eigen::Ref<const T_plain>;

        const T_ref& v_ref = v;
        auto* baseVari = new internal::cosh_matrix_vari<T_ref>(v_ref);
        T_plain result(v_ref.rows(), v_ref.cols());
        result.vi() = Eigen::Map<matrix_vi>(baseVari->variRefCosh_,
                                          v_ref.rows(), v_ref.cols());

        return result;
});
}
}  // namespace math
}  // namespace stan
#endif
