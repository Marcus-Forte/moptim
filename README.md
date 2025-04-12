# Moptim
Non linear optimization library built with SYCL.

## Roadmap
- Use SYCL
  - Improve mem copy if SYCL device is cpu
  - Use matrix calculations in GPU
- Handle models without arguments
- 2D Point registration
- 3D Point registration
- Eigen::VectorXd as interface to models?
- Add efficient cost computation (together with hessian?)
- Avoid copying input/obs dataset