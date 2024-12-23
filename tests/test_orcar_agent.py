import argparse

from Orcar import OrcarAgent
from Orcar.gen_config import Config, get_llm
from Orcar.load_cache_dataset import load_filter_hf_dataset

args_dict = {
    "model": "claude-3-5-sonnet-20241022",
    # "model": "gpt-4o",
    "image": "sweagent/swe-agent:latest",
    # "dataset": "SWE-bench_common",
    "dataset": "princeton-nlp/SWE-bench_Lite",
    "persistent": True,
    "container_name": "test_0",
    "split": "test",
    # "idx_list": [0, 3],
    # "idx_range": [0, 1],
    # Short Issue Test
    "filter_instance": "^(matplotlib__matplotlib-26020)$",
    # "filter_instance": "^(astropy__astropy-12907)$",
    # "filter_instance": "^(astropy__astropy-12907|astropy__astropy-7746)$",
    # Long Issue Test
    # "filter_instance": "^(astropy__astropy-6938)$",
    # "filter_instance": "^(astropy__astropy-6938|astropy__astropy-12907)$",
    # "filter_instance": (
    #     "^(sympy__sympy-21612|pytest-dev__pytest-7432|matplotlib__matplotlib-24149|"
    #     "sympy__sympy-16792|django__django-11999|matplotlib__matplotlib-25332|"
    #     "scikit-learn__scikit-learn-13496)$"
    # ),
    # whole repo
    # "filter_instance": ".*",
    # internal error
    # "filter_instance": "^(django__django-11099)$",
    # Multi Issue Test
    # "filter_instance": "^(pylint-dev__pylint-7080|matplotlib__matplotlib-26020|pytest-dev__pytest-7490)$",
    # Wrong action
    # "filter_instance": (
    #    "^("
    #    "django__django-11815|"
    #    "django__django-13028|"
    #    "django__django-14999|"
    #    "django__django-15252|"
    #    "sympy__sympy-13647|"
    #    "scikit-learn__scikit-learn-13496"
    #    ")$"
    # ),
    # Test step format
    # "filter_instance": (
    #    "^("
    #    # Wrong step format
    #    "django__django-11848|"
    #    "django__django-16139|"
    #    "django__django-16527|"
    #    "scikit-learn__scikit-learn-14087|"
    #    "sympy__sympy-12481|"
    #    "sympy__sympy-20154|"
    #    "django__django-15814|"
    #    # Long issues
    #    "pylint-dev__pylint-7080|"
    #    "matplotlib__matplotlib-26020|"
    #    "pytest-dev__pytest-7490"
    #    ")$"
    # ),
    # Wrong output path format
    # "filter_instance": (
    #    "^("
    #    "astropy__astropy-14995|"
    #    "django__django-13315|"
    #    "django__django-13401|"
    #    "django__django-14855|"
    #    "sympy__sympy-16792|"
    #    "sympy__sympy-23262|"
    #    "sympy__sympy-24066"
    #    ")$"
    # ),
    # Tracer test
    # "filter_instance": (
    #    "^("
    #    "astropy__astropy-12907|"
    #    "pydata__xarray-4094|"
    #    "sympy__sympy-18189|"
    #    "sympy__sympy-18199|"
    #    "sympy__sympy-21847|"
    #    "sympy__sympy-24066|"
    #    "django__django-11815|"
    #    "scikit-learn__scikit-learn-14983|"
    #    "sympy__sympy-13647|"
    #    "sympy__sympy-15345|"
    #    "sympy__sympy-12419|"
    #    "sympy__sympy-13031|"
    #    "sympy__sympy-21379|"
    #    "sympy__sympy-23262|"
    #    "astropy__astropy-14995|"
    #    "django__django-14787|"
    #    "django__django-16595|"
    #    "matplotlib__matplotlib-23299|"
    #    "matplotlib__matplotlib-24149|"
    #    "sympy__sympy-13480|"
    #    "sympy__sympy-17139|"
    #    "sympy__sympy-17655|"
    #    "astropy__astropy-14182|"
    #    "astropy__astropy-14365|"
    #    "scikit-learn__scikit-learn-10297|"
    #    "sympy__sympy-22714|"
    #    "matplotlib__matplotlib-25332|"
    #    "sympy__sympy-17630"
    #    ")$"
    # ),
}


def test_agent():
    args = argparse.Namespace(**args_dict)
    cfg = Config("./key.cfg")
    llm = get_llm(model=args.model, api_key=cfg["ANTHROPIC_API_KEY"], max_tokens=4096)
    ds = load_filter_hf_dataset(args)

    # final_stage = "extract"
    final_stage = "search"
    # final_stage = "edit"
    agent = OrcarAgent(args=args, llm=llm, final_stage=final_stage)
    agent.set_redirect_log(True)
    for i, inst in enumerate(ds):
        print(f"({i+1:03d}/{len(ds):03d}) Current inst: {inst['instance_id']}")
        agent.run(dict(inst))


if __name__ == "__main__":
    test_agent()
