# Moptim
Non linear optimization library built with option for SYCL.

## Roadmap
- Pool allocator for costs?
- computeCost can have a default definition on ICost - ICost would require knowledge of IModel.
- conver central vs forward


## Realtime test

A test of use of moptim with realtime operating system is provided. The test will request the OS to set SCHED_FIFO priority on the test. An infinite look running moptim is executed and latency is printed on the terminal.

A dockerfile to build the test is provided in `docker/` folder. Once the test container is built, use the following to pass scheduling priority to the container:

`docker run -it --rm --init --cap-add=SYS_NICE --ulimit=rtprio=99 --ulimit=memlock=-1 mdnf1992/moptim_rt_test 50000 1`