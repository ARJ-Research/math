#include <test/unit/math/test_ad.hpp>

TEST(MathMixMatFun, logSoftmax) {
  auto f = [](const auto& x) { return stan::math::log_softmax(x); };
  Eigen::VectorXd x0(0);  // error case
  stan::test::expect_ad(f, x0);

  Eigen::VectorXd x1(1);
  x1 << 0;
  stan::test::expect_ad(f, x1);

  Eigen::VectorXd x2(2);
  x2 << -1, 1;
  stan::test::expect_ad(f, x2);

  Eigen::VectorXd x3(3);
  x3 << -1, 1, 10;
  stan::test::expect_ad(f, x3);

  Eigen::VectorXd x3b(3);
  x3b << 0, 1, 2;
  stan::test::expect_ad(f, x3b);

  Eigen::VectorXd x3c(3);
  x3c << 2, 1, 1;
  stan::test::expect_ad(f, x3c);

  Eigen::RowVectorXd rx0(0);  // error case
  stan::test::expect_ad(f, rx0);

  Eigen::RowVectorXd rx1(1);
  rx1 << 0;
  stan::test::expect_ad(f, rx1);

  Eigen::RowVectorXd rx2(2);
  rx2 << -1, 1;
  stan::test::expect_ad(f, rx2);

  Eigen::RowVectorXd rx3(3);
  rx3 << -1, 1, 10;
  stan::test::expect_ad(f, rx3);

  Eigen::RowVectorXd rx3b(3);
  rx3b << 0, 1, 2;
  stan::test::expect_ad(f, rx3b);

  Eigen::RowVectorXd rx3c(3);
  rx3c << 2, 1, 1;
  stan::test::expect_ad(f, rx3c);

  std::vector<double> stx0(0);  // error case
  stan::test::expect_ad(f, stx0);

  std::vector<double> stx1{0};
  stan::test::expect_ad(f, stx1);

  std::vector<double> stx2{-1, 1};
  stan::test::expect_ad(f, stx2);

  std::vector<double> stx3{-1, 1, 10};
  stan::test::expect_ad(f, stx3);

  std::vector<double> stx3b{0, 1, 2};
  stan::test::expect_ad(f, stx3b);

  std::vector<double> stx3c{2, 1, 1};
  stan::test::expect_ad(f, stx3c);
}
