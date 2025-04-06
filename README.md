# OrcaLoca

## 🔥 News! 🚀
- *Apr 5 2025*: We support top-K retrieval mode inspired from [LocAgent](https://github.com/gersteinlab/LocAgent) and [CoSIL](https://github.com/ZhonghaoJiang/CoSIL).
- *Feb 18 2025*: We update the support for Gemini series model through vertex API.

OrcaLoca (previous named Orcar), an LLM agent framework that improves accuracy for software issue localization by integrating priority-based scheduling for LLM-guided action, action decomposition with relevance scoring, and distance-aware context pruning.

![overview](./artifact/overview.jpg)

## Prerequisite

OrcaLoca requires docker to run, so please first pull our docker image (forked from [SWE-Agent](https://github.com/SWE-agent/SWE-agent)):

```shell
docker pull hejiaz/swe-agent:latest
```

OrcaLoca also requires API access to LLM. (Currently OpenAI & Anthropic APIs are supported)
You can either export them in CLI:
```shell
export OPENAI_API_KEY={key_here}
export ANTHROPIC_API_KEY={key_here}
```
or as a key.cfg file:
```
OPENAI_API_KEY=key_here
ANTHROPIC_API_KEY=key_here
```

OrcaLoca also uses torch in its search process. ([torch installation guide](https://pytorch.org/get-started/locally/))

## Installation
```shell
cd OrcaLoca

conda create -n orcaloca python=3.10
conda activate orcaloca
pip install -e .
```

After installation succeeded, you can run a quick smoke test (should finish in 5-10 minutes):
```shell
python evaluation/run.py --final_stage trace_analysis --instance_ids astropy__astropy-12907 astropy__astropy-6938
```

Then add search stage into running:
```shell
python evaluation/run.py --final_stage search --instance_ids astropy__astropy-12907
```

## Reproducing OrcaLoca Leaderboard Submission

### Genrating Search results
**About search.cfg**
We have a customized search configuration for the agent as below:

```ini
[SEARCH]
context_control = True # whether to use context cache
redundancy_control = True # prune the redundant action
top_k_search = 12 # search cache size
top_k_output = 3 # k output in retrieval_mode
top_k_retrieval_mode = False # this will output top_k function locations as output
top_k_methods = 3
top_k_disambiguation = 3
top_k_functions = 2
score_threshold = 75
batch_size = 1 # parallel actions batch size to improve efficiency

[SCORE_DECOMPOSITION]
class = True
file = True
disambiguation = True

[PRIORITY]
enable = True
basic = 1
decomposition = 2
related_file = 2
```

```shell
python evaluation/run.py
```

### Genrating output.json
```shell
cd evaluation
python process_output.py
```

### Preparing Data for Agentless Edition
Please go through instructions in:
1. evaluation/orcar_agentless/README.md
2. thirdparty/Agentless/README_orcar.md
To run agentless with 1-hop relation subgraph, you will also need to generate the `dependency_output.json`.
```shell
cd evaluation
python process_dependency.py
```

### Evaluating all_preds.jsonl
Our output all_preds.jsonl can be evaluated with official scripts offered by [SWE-Bench](https://github.com/swe-bench/SWE-bench).
Please check the 'Set Up' and 'Usage' parts in its README.md for details.


### License
MIT License

### Citation

If our project helps you, please cite our [paper](https://arxiv.org/abs/2502.00350) with

```bibtex
@misc{yu2025orcalocallmagentframework,
      title={OrcaLoca: An LLM Agent Framework for Software Issue Localization},
      author={Zhongming Yu and Hejia Zhang and Yujie Zhao and Hanxian Huang and Matrix Yao and Ke Ding and Jishen Zhao},
      year={2025},
      eprint={2502.00350},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2502.00350},
}
```
