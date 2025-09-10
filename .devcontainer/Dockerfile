# Update platform accordingly
FROM mdnf1992/sycl-dev 

RUN apt-get update && apt-get install -y libeigen3-dev libgtest-dev clang-tidy clang-format libboost-dev vim && \
    apt-get clean && apt-get autoremove && \
    rm -rf /var/lib/apt/lists/*

RUN cd /tmp && git clone https://github.com/uxlfoundation/oneMath.git && \
    mkdir oneMath/build && cd oneMath/build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_GENERIC_BLAS_BACKEND=ON -DENABLE_MKLCPU_BACKEND=FALSE -DENABLE_MKLGPU_BACKEND=False -DGENERIC_BLAS_TUNING_TARGET=NVIDIA_GPU .. && \
    make -j$(nproc) && make install