FROM mdnf1992/sycl-dev:lapack

RUN apt-get update && apt-get install -y libeigen3-dev libboost-dev libgtest-dev && \
    apt-get clean && apt-get autoremove && \
    rm -rf /var/lib/apt/lists/*

COPY . /app

RUN mkdir /app/build && cd /app/build && \
    CXX=/opt/sycl/bin/clang++ cmake .. -DCMAKE_BUILD_TYPE=Release -DWITH_SYCL=OFF && \
    make -j4

ENTRYPOINT ["/app/build/test/test_realtime"]