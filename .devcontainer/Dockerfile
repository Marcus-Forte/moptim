# Update platform accordingly
FROM --platform=linux/arm64 mdnf1992/sycl-dev 

RUN apt-get update && apt-get install -y libgtest-dev clang-tidy clang-format vim && \
    apt-get clean && apt-get autoremove && \
    rm -rf /var/lib/apt/lists/*