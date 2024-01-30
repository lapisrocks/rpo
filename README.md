# Robust Prompt Optimization

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is the official repository for "[Robust Prompt Optimization for Defending Language Models Against Jailbreaking Attacks]()" by [Andy Zhou](https://andyz245.github.io/), [Bo Li](https://aisecure.github.io/), and [Haohan Wang](https://haohanwang.github.io/).

![teaser](https://ibb.co/8Xqh8bQ)

We include a notebook `demo.ipynb`  containing the minimal implementation of RPO, for defending Llama-2 against the popular AIM attack. The full code, which supports optimization across multiple jailbreaks and GCG, will be released soon!

This can also be found on [Colab](https://colab.research.google.com/drive/1dinZSyP1E4KokSLPcCh1JQFUFsN-WV--?usp=sharing)

This repository is based on the GCG repository (https://github.com/llm-attacks/llm-attacks)


## Installation

The `rpo` package can be installed by running the following command at the root of this repository:

```bash
pip install livelossplot
pip install -e .
```

## Models

Please follow the instructions to download LLaMA-2-7B-Chat first (we use the weights converted by HuggingFace [here](https://huggingface.co/meta-llama/Llama-2-7b-hf)).  


## Citation
If you found our paper or repo interesting, thanks! Please consider citing the following

```

```

## License
`rpo` is licensed under the terms of the MIT license. See LICENSE for more details.


