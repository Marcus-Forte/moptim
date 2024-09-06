#include "GaussNewton.hh"

GaussNewton::GaussNewton() = default;

void GaussNewton::step(Eigen::VectorXd& x) const {
  Eigen::MatrixXd JtJ = Eigen::MatrixXd::Zero(x.size(), x.size());
  Eigen::VectorXd Jtb = Eigen::VectorXd::Zero(x.size());

  for (const auto& cost : costs_) {
    const auto& [JtJ_, Jtb_] = cost->computeHessian(x);
    JtJ += JtJ_;
    Jtb += Jtb_;
  }

  Eigen::LDLT<Eigen::MatrixXd> solver(JtJ);
  const auto x_plus = solver.solve(-Jtb);
  x += x_plus;
}

// Automate steps:
// Verify: rel_tolerance, abs_tolerance, max iterations
void GaussNewton::optimize(Eigen::VectorXd& x) const {}