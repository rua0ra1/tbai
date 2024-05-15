# tbai_benchmark

This ROS package servers the purpouse of providing utilities to test the different walking controllers implemented as part of this project.

## Benchmark map
![benchmark_map](https://github.com/lnotspotl/tbai/assets/82883398/b7d2ba54-859c-43c7-8429-f5e13a67c7ec)
![benchmark_map2](https://github.com/lnotspotl/tbai/assets/82883398/af20dfb1-f16f-4b5a-9917-839a60e971c8)

## Example - perceptive MPC



https://github.com/lnotspotl/tbai/assets/82883398/4c2b398c-04ed-4534-8e62-4b0e01aab8cf



https://github.com/lnotspotl/tbai/assets/82883398/1d07b358-187b-423f-b7b5-d3c9251ad7f5

## Example - context-aware controller
- Our context-aware controller switches between two of the implemented controllers, namely `rl_blind` and `mpc_perceptive`.
- In case a foot slip is detected, the context-aware controller changes the active controller from `mpc_perceptive` to `rl_blind`.
- Once a checkpoint has been reached, the active controller is changed to the `mpc_perceptive` controller again.

https://github.com/lnotspotl/tbai/assets/82883398/955b83fa-256b-492e-a959-1ddf97fbe860

