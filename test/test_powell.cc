#include <gtest/gtest.h>

#include "ConsoleLogger.hh"
#include "GaussNewton.hh"
#include "NumericalCost.hh"

/**
 * @brief Model for the Powell function. No inputs or measurements, only parameters.
 *
 */
struct Powell : public IJacobianModel {
  void setup(const double* x) override {
    x_[0] = x[0];
    x_[1] = x[1];
    x_[2] = x[2];
    x_[3] = x[3];
  }

  void f(const double* /*input*/, const double* /*measurement*/, double* f_x) override {
    f_x[0] = x_[0] + 10 * x_[1];
    f_x[1] = sqrt(5) * (x_[2] - x_[3]);
    f_x[2] = (x_[1] - 2 * x_[2]) * (x_[1] - 2 * x_[2]);
    f_x[3] = sqrt(10) * (x_[0] - x_[3]) * (x_[0] - x_[3]);
  }

  void df(const double* /*input*/, const double* /*measurement*/, double* df_x) override {
    df_x[0] = 1;
    df_x[1] = 0;
    df_x[2] = 0;
    df_x[3] = sqrt(10) * 2 * (x_[0] - x_[3]);

    // Df / dx1
    df_x[4] = 10;
    df_x[5] = 0;
    df_x[6] = 2 * (x_[1] + 2 * x_[2]);
    df_x[7] = 0;

    // Df / dx2
    df_x[8] = 0;
    df_x[9] = sqrt(5);
    df_x[10] = 2 * (x_[1] + 2 * x_[2]) * (-2);
    df_x[11] = 0;

    // Df / dx3
    df_x[12] = 0;
    df_x[13] = -sqrt(5);
    df_x[14] = 0;
    df_x[15] = sqrt(10) * 2 * (x_[0] - x_[3]) * (-1);
  }

  double x_[4];
};

TEST(TestPowell, TestPowell) {
  Eigen::VectorXd x{{3.0, -1.0, 0.0, 4.0}};
  const auto model = std::make_shared<Powell>();

  auto cost = std::make_shared<NumericalCost>(x.data(), x.data(), 1, 4, 4, model);
  GaussNewton solver;
  solver.setMaxIterations(20);
  solver.addCost(cost);

  solver.optimize(x);
  EXPECT_NEAR(x[0], 0.0, 1e-5);
  EXPECT_NEAR(x[1], 0.0, 1e-5);
  EXPECT_NEAR(x[2], 0.0, 1e-5);
  EXPECT_NEAR(x[3], 0.0, 1e-5);
}

struct PowellF0 : public IModel {
  void setup(const double* x) override {
    x_[0] = x[0];
    x_[1] = x[1];
    x_[2] = x[2];
    x_[3] = x[3];
  }

  void f(const double* input, const double* measurement, double* f_x) override { f_x[0] = x_[0] + 10 * x_[1]; }
  double x_[4];
};

struct PowellF1 : public IModel {
  void setup(const double* x) override {
    x_[0] = x[0];
    x_[1] = x[1];
    x_[2] = x[2];
    x_[3] = x[3];
  }

  void f(const double* input, const double* measurement, double* f_x) override { f_x[0] = sqrt(5) * (x_[2] - x_[3]); }
  double x_[4];
};

struct PowellF2 : public IModel {
  void setup(const double* x) override {
    x_[0] = x[0];
    x_[1] = x[1];
    x_[2] = x[2];
    x_[3] = x[3];
  }

  void f(const double* input, const double* measurement, double* f_x) override {
    f_x[0] = (x_[1] - 2 * x_[2]) * (x_[1] - 2 * x_[2]);
  }
  double x_[4];
};

struct PowellF3 : public IModel {
  void setup(const double* x) override {
    x_[0] = x[0];
    x_[1] = x[1];
    x_[2] = x[2];
    x_[3] = x[3];
  }

  void f(const double* input, const double* measurement, double* f_x) override {
    f_x[0] = sqrt(10) * (x_[0] - x_[3]) * (x_[0] - x_[3]);
  }
  double x_[4];
};

// It is also possible to split a multi-dimensional function into multiple functions with a shared parameter set (x)
TEST(TestPowell, TestPowerllSplit) {
  Eigen::VectorXd x{{3.0, -1.0, 0.0, 4.0}};

  auto cost1 = std::make_shared<NumericalCost>(x.data(), x.data(), 1, 1, 4, std::make_shared<PowellF0>());
  auto cost2 = std::make_shared<NumericalCost>(x.data(), x.data(), 1, 1, 4, std::make_shared<PowellF1>());
  auto cost3 = std::make_shared<NumericalCost>(x.data(), x.data(), 1, 1, 4, std::make_shared<PowellF2>());
  auto cost4 = std::make_shared<NumericalCost>(x.data(), x.data(), 1, 1, 4, std::make_shared<PowellF3>());

  auto logger = std::make_shared<ConsoleLogger>();
  logger->setLevel(ILog::Level::INFO);
  GaussNewton solver(logger);
  solver.setMaxIterations(20);
  solver.addCost(cost1);
  solver.addCost(cost2);
  solver.addCost(cost3);
  solver.addCost(cost4);

  solver.optimize(x);
  EXPECT_NEAR(x[0], 0.0, 1e-5);
  EXPECT_NEAR(x[1], 0.0, 1e-5);
  EXPECT_NEAR(x[2], 0.0, 1e-5);
  EXPECT_NEAR(x[3], 0.0, 1e-5);
}