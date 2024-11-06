import argparse
import os
from typing import List

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.llms.llm import LLM

from Orcar.code_scorer import CodeScorer
from Orcar.gen_config import Config, get_llm
from Orcar.load_cache_dataset import load_filter_hf_dataset
from Orcar.search import SearchManager


def test_scorer(llm: LLM, problem_statement: str):
    repo_path = "~/.orcar/django__django"
    repo_path = os.path.expanduser(repo_path)
    search_manager = SearchManager(repo_path=repo_path)

    # try to search class "Query"'s methods in the graph
    list_name, list_code_snippet = search_manager._get_class_methods("Query")

    # package the list of methods into a list of ChatMessage
    chat_messages: List[List[ChatMessage]] = []
    for method in list_code_snippet:
        chat_messages.append([ChatMessage(role=MessageRole.USER, content=method)])

    # create a CodeScorer object

    code_scorer = CodeScorer(llm=llm, problem_statement=problem_statement)
    # score the list of methods
    scores = code_scorer.score_batch(chat_messages)
    # combine the scores with the method names
    results = []
    for i, method in enumerate(list_name):
        results.append({"method_name": method, "score": scores[i]})
    sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
    return sorted_results


if __name__ == "__main__":
    args_dict = {
        "model": "claude-3-5-sonnet-20241022",
        "image": "sweagent/swe-agent:latest",
        "dataset": "princeton-nlp/SWE-bench_Lite",
        "persistent": True,
        "container_name": "test",
        "split": "test",
        "filter_instance": "^(django__django-15814)$",
    }

    args = argparse.Namespace(**args_dict)
    cfg = Config("./key.cfg")
    llm = get_llm(model=args.model, api_key=cfg["ANTHROPIC_API_KEY"], max_tokens=4096)
    ds = load_filter_hf_dataset(args)
    for inst in ds:
        res = test_scorer(llm, problem_statement=inst["problem_statement"])
        print(res)
