# Moptim
Non linear optimization library built with option for SYCL.

## Roadmap
- Raw pointers vs Eigen::VectorXd as interface to models?
- Separate ISolver interface? (Then we can use GPU solvers). NOTE: For small nr. of parameters, CPU is always faster..
- support `double` AND `float`.
- Redesign Cost `virtual SolveRhs computeLinearSystem(const Eigen::VectorXd& x) = 0;` -> How to abstract computation of jacobians and manage memory
if it must remain in a "device"? How to keep using low level pointers? Decouple from Eigen?
- Pool allocator for costs?
- 