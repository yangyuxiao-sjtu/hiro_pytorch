### Overview
A modified attempt of HIRO ([Data-Efficient Hierarchical Reinforcement Learning](https://arxiv.org/abs/1805.08296), with the code initially sourced from [this repository](https://github.com/watakandai/hiro_pytorch).

In Hierarchical Reinforcement Learning (HRL), on-policy training is required because, as the low-level policy updates, the collected data cannot be used to guide the high-level policy. This makes training a multilevel policy inefficient. To address this issue, HIRO proposes an off-policy correction process by relabeling the high-level goals, enabling the high-level policy to learn from previously collected data.

In our project, we aim to explore whether there are alternative methods to train off-policy HRL. Specifically, we attempt to use the gradient from the low-level policy as a regularization term to help the high-level policy detect changes in the low-level policy and adjust accordingly. The results show that our new method achieves similar performance to the off-policy correction.
### Result
<img width="904" alt="image" src="https://github.com/yangyuxiao-sjtu/hiro_pytorch/assets/95365549/d84272cd-c4a7-47bb-b892-dae5d6502521">
