# Update platform accordingly
FROM mdnf1992/sycl-dev:lapack

RUN apt-get update && apt-get install -y libeigen3-dev libgtest-dev clang-tidy clang-format libboost-dev vim && \
    apt-get clean && apt-get autoremove && \
    rm -rf /var/lib/apt/lists/*