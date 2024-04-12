# tbai
Towards better athletic intelligence


## Project structure

- [ ] tbai_core
- [ ] tbai_utils -- simplegen here, joystick node here
- [ ] tbai_wbc
- [ ] tbai_mpc -- reference generator here
- [ ] tbai_rl
- [ ] tbai_msgs
- [ ] tbai_gazebo
- [ ] tbai_ocs2
- [ ] tbai_perceptive_mpc
- [ ] tbai_blind_mpc
- [ ] tbai_perceptive_rl
- [ ] tbai_blind_rl
- [ ] tbai_dtc

Mention that some of the code is part of the `ocs2` fork in branch tbai


## Project linting
```bash
cpplint --recursive .
```

## Docs
```bash
google-chrome ./build/tbai_docs/output/doxygen/html/index.html
```