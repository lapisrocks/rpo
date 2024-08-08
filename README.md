# Robust Prompt Optimization

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![teaser](figures/figure.png)

This is the official repository for "[Robust Prompt Optimization for Defending Language Models Against Jailbreaking Attacks](https://arxiv.org/abs/2401.17263)" by [Andy Zhou](https://andyz245.github.io/), [Bo Li](https://aisecure.github.io/), and [Haohan Wang](https://haohanwang.github.io/). Website with more examples can be found [here](https://lapisrocks.github.io/rpo/).

We include a notebook `demo.ipynb`  containing the minimal implementation of RPO, for defending Llama-2 against the popular AIM attack.

This can also be found on [Colab](https://colab.research.google.com/drive/1Q3ngIwHAudhOYiVD4-V2NcHO0dfZUy2Z?usp=sharing)

This repository is based on the GCG repository (https://github.com/llm-attacks/llm-attacks)


## Installation

The `rpo` package can be installed by running the following command at the root of this repository:

```bash
pip install livelossplot
pip install -e .
```

## Models

Please follow the instructions to download LLaMA-2-7B-Chat first (we use the weights converted by HuggingFace [here](https://huggingface.co/meta-llama/Llama-2-7b-hf)). To modify the paths to your models and tokenizers, please add the following lines in `experiments/configs/transfer_xxx.py`.

```python
    config.model_paths = [
        "/DIR/vicuna/vicuna-7b-v1.3",
        ... # more models
    ]
    config.tokenizer_paths = [
        "/DIR/vicuna/vicuna-7b-v1.3",
        ... # more tokenizers
    ]
```

## Experiments 

Jailbreaks can be added as text files to the `experiments/jailbreaks` folder. The `experiments` folder contains code to reproduce RPO experiments on AdvBench. Modify the script with the correct path and run the following code inside `experiments`:

```bash
cd launch_scripts
bash run_gcg_multiple.sh vicuna # or llama2
```

## Examples

Here are some example strings we optimized. These are effective across language models and jailbreaks. It is fair to directly use these strings for baseline comparisons as they are expected to transfer to other models and attacks.

Starling-7B
```
:}</'])language seems scary-----------}] estimation >>begin &=&pgf degli ({\ $|\textwidthAI NOTE:
```

Llama2-7B-Chat
```
was Bel Let]\"> It\'\ si distinguish \({\footnote{[:// Inst bitimidenote]{BWebachiv
```

## Evaluation

We provide a simple script in `experiments/test.py` to evaluate the robustness of generated suffixes on jailbreaks. Code will be updated with JailbreakBench evaluation later.

## Citation
If you found our paper or repo interesting, thanks! Please consider citing the following

```
@misc{zhou2024robust,
      title={Robust Prompt Optimization for Defending Language Models Against Jailbreaking Attacks}, 
      author={Andy Zhou and Bo Li and Haohan Wang},
      year={2024},
      eprint={2401.17263},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## License
`rpo` is licensed under the terms of the MIT license. See LICENSE for more details.


