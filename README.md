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



# OneMATH

## Resources

- https://oneapi-spec.uxlfoundation.org/specifications/oneapi/latest/elements/onemath/source/domains/blas/blas
- https://uxlfoundation.github.io/oneMath/building_the_project_with_dpcpp.html

cmake .. -DENABLE_GENERIC_BLAS_BACKEND=ON -DENABLE_MKLCPU_BACKEND=FALSE -DENABLE_MKLGPU_BACKEND=False -DGENERIC_BLAS_TUNING_TARGET=NVIDIA_GPU -DBUILD_FUNCTIONAL_TESTS=False

# POCL

## Resources

cmake .. -DENABLE_ICD=ON -DSTATIC_LLVM=ON -DLLC_HOST_CPU=generic -DENABLE_TESTS=OFF -DLLC_TRIPLE=aarch64 -DENABLE_HWLOC=ON -DENABLE_LLVM=ON

- https://portablecl.org/docs/html/install.html#configure-build
- icd loader: ocl-icd-libopencl1
- icd headers: ocl-icd-dev (interface)
