import argparse

from Orcar.formatter import TokenCounter
from Orcar.gen_config import Config, get_llm

args_dict = {
    # "model": "claude-3-5-sonnet-20241022",
    # "model": "gpt-4o",
    # "model": "gemini-2.0-pro-exp-02-05",
    "model": "claude-3-7-sonnet@20250219",
    "provider": "vertexanthropic",
    "image": "sweagent/swe-agent:latest",
    # "dataset": "SWE-bench_common",
    "dataset": "princeton-nlp/SWE-bench_Lite",
    "persistent": True,
    "container_name": "test",
    "split": "test",
    "max_retry": 2,
    # "idx_list": [0, 3],
    # "idx_range": [0, 1],
    # Short Issue Test
    # "filter_instance": "^(matplotlib__matplotlib-23476)$",
    # "filter_instance": "^(sphinx-doc__sphinx-11445|sphinx-doc__sphinx-8595|sympy__sympy-12419)$",
    # "filter_instance": "^(sympy__sympy-12419)$",
    # "filter_instance": "^(astropy__astropy-14365|django__django-11001|pylint-dev__pylint-7993)$",
    # "filter_instance": "^(django__django-12983|pylint-dev__pylint-7228)$",
    # Long Issue Test
    # "filter_instance": "^(sympy__sympy-23262)$",
    # "filter_instance": "^(pytest-dev__pytest-5692)$",
    "filter_instance": "^(astropy__astropy-6938)$",
    # "filter_instance": (
    #     "^(sympy__sympy-21612|pytest-dev__pytest-7432|matplotlib__matplotlib-24149|"
    #     "sympy__sympy-16792|django__django-11999|matplotlib__matplotlib-25332|"
    #     "scikit-learn__scikit-learn-13496)$"
    # ),
    # whole repo
    # "filter_instance": ".*",
    # internal error
    # "filter_instance": "^(django__django-14580)$",
    # "filter_instance": "^(django__django-13321)$",
    # Multi Issue Test
    # "filter_instance": "^(pylint-dev__pylint-7080|matplotlib__matplotlib-26020|pytest-dev__pytest-7490)$",
    # "filter_instance": ".*",
    # 'django__django-13551', 'django__django-16255', 'scikit-learn__scikit-learn-13439', 'sympy__sympy-14774', 'sympy__sympy-15011']
    # "filter_instance": "^(django__django-13551|django__django-16255|scikit-learn__scikit-learn-13439|sympy__sympy-14774|sympy__sympy-15011)$",
    # "filter_instance": "^(django__django-15814|pylint-dev__pylint-7228|pytest-dev__pytest-8906|sympy__sympy-16792|sympy__sympy-24213)$",
}


def test_token_count():
    args = argparse.Namespace(**args_dict)
    cfg = Config("../key.cfg", args.provider)
    llm = get_llm(model=args.model, max_tokens=4096, orcar_config=cfg)
    token_counter = TokenCounter(llm)
    print(token_counter.count("Hello, world!"))


test_token_count()
