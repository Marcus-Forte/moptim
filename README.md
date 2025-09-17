# Moptim
Non linear optimization library built with option for SYCL.

## Roadmap
- Input and observation of different strides. Current assumption is that they are the same.
- Pool allocator for costs?
- computeCost can have a default definition on ICost - ICost would require knowledge of IModel.
- conver central vs forward