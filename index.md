
[Paper](https://arxiv.org/abs/2401.17263) | [Code](https://github.com/andyz245/rpo) 


# Robust Prompt Optimization for Defending Language Models Against Jailbreaking Attacks

| ![GPT-4](figures/gpt4.png) | ![RPO](figures/gpt4_rpo.png) |
|:--:|:--:|
| Fig.1 GPT-4 safety filters can be bypassed by jailbreaks! | Fig.2 RPO enforces harmless responses even after jailbreaks |


## Abstract


While large language models (LLMs) have demonstrated impressive performance on a range of decision-making tasks, they rely on simple acting processes and fall short of broad deployment as autonomous agents. We introduce LATS (Language Agent Tree Search), a general framework that synergizes the capabilities of LLMs in planning, acting, and reasoning. Drawing inspiration from Monte Carlo tree search in model-based reinforcement learning, LATS employs LLMs as agents, value functions, and optimizers, repurposing their latent strengths for enhanced decision-making. What is crucial in this method is the use of an environment for external feedback, which offers a more deliberate and adaptive problem-solving mechanism that moves beyond the limitations of existing techniques. Our experimental evaluation across diverse domains, such as programming, HotPotQA, and WebShop, illustrates the applicability of LATS for both reasoning and acting. In particular, LATS achieves 94.4% for programming on HumanEval with GPT-4 and an average score of 75.9 for web browsing on WebShop with GPT-3.5, demonstrating the effectiveness and generality of our method.
  

## Results
  
![res](figures/transfer.png)



## Citations

Please cite the paper and star this repo if you use RPO or find it interesting!


```bibtex
@misc{zhou2024robust,
      title={Robust Prompt Optimization for Defending Language Models Against Jailbreaking Attacks}, 
      author={Andy Zhou and Bo Li and Haohan Wang},
      year={2024},
      eprint={2401.17263},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
