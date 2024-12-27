import argparse
import os
from typing import List

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.llms.llm import LLM

from Orcar.code_scorer import CodeScorer
from Orcar.gen_config import Config, get_llm
from Orcar.load_cache_dataset import load_filter_hf_dataset
from Orcar.search import SearchManager
from Orcar.types import SearchActionStep, SearchResult


def _check_search_result_type(
    search_manager: SearchManager, search_result: SearchResult, type: str
) -> bool:
    """Check if the search result is searching for a class."""
    action = search_result.search_action
    search_input = search_result.get_search_input()
    # use get_frame_from_history to get the frame of the search result
    frame = search_manager.get_frame_from_history(action, search_input)
    # check frame is not None
    if frame is not None:
        # logger.info(f"Frame: {frame}")
        query_type = frame["query_type"]
        if query_type == type:
            return True
    return False


def _class_methods_ranking(
    llm: LLM,
    problem_statement: str,
    search_manager: SearchManager,
    search_result: SearchResult,
) -> List[SearchActionStep]:
    """Ranking the class methods."""
    # if the action is search_class, we should rank the class methods
    search_action = search_result.search_action
    search_action_input = search_result.search_action_input

    is_class = _check_search_result_type(search_manager, search_result, "class")
    if is_class:
        frame = search_manager.get_frame_from_history(
            search_action, search_result.get_search_input()
        )
        search_query = frame["search_query"]
        file_path = frame["file_path"]
        # search_query is the exact node name
        class_methods, methods_code = search_manager._get_class_methods(search_query)
        if len(class_methods) == 0:
            return []
        # package the list of methods into a list of ChatMessage
        chat_messages: List[List[ChatMessage]] = []
        for method in methods_code:
            chat_messages.append([ChatMessage(role=MessageRole.USER, content=method)])
        code_scorer = CodeScorer(llm=llm, problem_statement=problem_statement)
        # score the list of methods
        scores = code_scorer.score_batch(chat_messages)
        # combine the scores with the method names
        results = []
        for i, method in enumerate(class_methods):
            results.append({"method_name": method, "score": scores[i]})
        sorted_results = sorted(
            results, key=lambda x: x["score"], reverse=True
        )  # from high to low
        # prune scores less than self._config_dict["score_threshold"]
        sorted_results = [result for result in sorted_results if result["score"] > 50]
        # get top 3 methods
        top_k = 3
        if len(sorted_results) < top_k:
            top_k = len(sorted_results)
        search_steps = []
        for i in range(top_k):
            method_name = sorted_results[i]["method_name"].split("::")[-1]
            search_steps.append(
                SearchActionStep(
                    search_action="search_method_in_class",
                    search_action_input={
                        "class_name": search_action_input["class_name"],
                        "method_name": method_name,
                        "file_path": file_path,
                    },
                )
            )
        return search_steps
    return []


def _file_functions_ranking(
    llm: LLM,
    problem_statement: str,
    search_manager: SearchManager,
    search_result: SearchResult,
) -> List[SearchActionStep]:
    """Ranking the file functions."""
    # if the action is search_file_contents, we should rank the file functions
    search_action = search_result.search_action
    is_file = _check_search_result_type(search_manager, search_result, "file")
    if is_file:
        frame = search_manager.get_frame_from_history(
            search_action, search_result.get_search_input()
        )
        file_node_name = frame["search_query"]
        # search_query is the exact node name
        functions, functions_code = search_manager._get_file_functions(file_node_name)
        if len(functions) == 0:
            return []
        # package the list of functions into a list of ChatMessage
        chat_messages: List[List[ChatMessage]] = []
        for function in functions_code:
            chat_messages.append([ChatMessage(role=MessageRole.USER, content=function)])
        code_scorer = CodeScorer(llm=llm, problem_statement=problem_statement)
        # score the list of functions
        scores = code_scorer.score_batch(chat_messages)
        # combine the scores with the function names
        results = []
        for i, function in enumerate(functions):
            results.append({"function_name": function, "score": scores[i]})
        sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
        # prune scores less than self._config_dict["score_threshold"]
        sorted_results = [result for result in sorted_results if result["score"] > 50]
        # get top 3 functions
        top_k = 3
        if len(sorted_results) < top_k:
            top_k = len(sorted_results)
        print(sorted_results)
        search_steps = []
        for i in range(top_k):
            function_name = sorted_results[i]["function_name"].split("::")[-1]
            file_path = sorted_results[i]["function_name"].split("::")[0]
            search_steps.append(
                SearchActionStep(
                    search_action="search_callable",
                    search_action_input={
                        "query_name": function_name,
                        "file_path": file_path,
                    },
                )
            )
        return search_steps
    return []


def _disambiguation_ranking(
    llm: LLM,
    problem_statement: str,
    search_manager: SearchManager,
    search_result: SearchResult,
) -> List[SearchActionStep]:
    """Ranking the disambiguation."""
    # if the action is disambiguation, we should rank the disambiguation
    search_action = search_result.search_action
    search_action_input = search_result.search_action_input

    def check_action_is_class(search_action: str) -> bool:
        """Check if the action is search_class."""
        if search_action == "search_class":
            return True
        return False

    is_disambiguation = _check_search_result_type(
        search_manager, search_result, "disambiguation"
    )
    if is_disambiguation:
        # if is_class, we don't score
        is_class = check_action_is_class(search_action)
        if is_class:
            class_name = search_action_input["class_name"]
            file_paths = search_manager._get_disambiguous_classes(class_name)
            if len(file_paths) == 0:
                return []
            search_steps = []
            for file_path in file_paths:
                search_steps.append(
                    SearchActionStep(
                        search_action="search_class",
                        search_action_input={
                            "class_name": class_name,
                            "file_path": file_path,
                        },
                    )
                )
            return search_steps
        # score the methods
        # three cases:
        # 1. search_method_in_class, and no file_path in search_action_input (if file_path, then no disambiguation)
        # 2. search_callable, and no file_path in search_action_input
        # 3. search_callable, and file_path in search_action_input
        if search_action == "search_method_in_class":
            class_name = search_action_input["class_name"]
            method_name = search_action_input["method_name"]
            disambiguated_methods, disambiguated_methods_code = (
                search_manager._get_disambiguous_methods(class_name, method_name)
            )
        if search_action == "search_callable":
            query_name = search_action_input["query_name"]
            if "file_path" not in search_action_input:
                disambiguated_methods, disambiguated_methods_code = (
                    search_manager._get_disambiguous_methods(
                        method_name=query_name, class_name=None, file_path=None
                    )
                )
            else:
                file_path = search_action_input["file_path"]
                disambiguated_methods, disambiguated_methods_code = (
                    search_manager._get_disambiguous_methods(
                        method_name=query_name, class_name=None, file_path=file_path
                    )
                )

        if len(disambiguated_methods) == 0:
            return []

        # package the list of disambiguation into a list of ChatMessage
        chat_messages: List[List[ChatMessage]] = []
        for dis in disambiguated_methods_code:
            chat_messages.append([ChatMessage(role=MessageRole.USER, content=dis)])
        code_scorer = CodeScorer(llm=llm, problem_statement=problem_statement)
        # score the list of disambiguation
        scores = code_scorer.score_batch(chat_messages)
        # combine the scores with the disambiguation names
        results = []
        for i, dis in enumerate(disambiguated_methods):
            results.append({"disambiguated_method": dis, "score": scores[i]})
        sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
        print(sorted_results)
        # prune scores less than self._config_dict["score_threshold"]
        sorted_results = [result for result in sorted_results if result["score"] > 50]
        # get top 3 disambiguation
        top_k = 3
        if len(sorted_results) <= top_k:
            top_k = 1  # only one disambiguation
        search_steps = []
        # please note, the disambiguated_method is the node name
        # if two "::" in the node name, it means it is a class's method
        # if one "::" in the node name, it means it is a function (why class is impossible? because we have already disambiguated the class)
        for i in range(top_k):
            disambiguated_method = sorted_results[i]["disambiguated_method"]
            # count the number of "::" in the node name
            if disambiguated_method.count("::") == 2:
                # file_path::class_name::method_name
                method_name = disambiguated_method.split("::")[-1]
                class_name = disambiguated_method.split("::")[-2]
                file_path = disambiguated_method.split("::")[0]
                search_steps.append(
                    SearchActionStep(
                        search_action="search_method_in_class",
                        search_action_input={
                            "class_name": class_name,
                            "method_name": method_name,
                            "file_path": file_path,
                        },
                    )
                )
            else:
                function_name = disambiguated_method.split("::")[-1]
                file_path = disambiguated_method.split("::")[0]
                search_steps.append(
                    SearchActionStep(
                        search_action="search_callable",
                        search_action_input={
                            "query_name": function_name,
                            "file_path": file_path,
                        },
                    )
                )
        return search_steps
    return []


def django_class_methods_scorer(llm: LLM, problem_statement: str):
    repo_path = "~/.orcar/django__django"
    repo_path = os.path.expanduser(repo_path)
    search_manager = SearchManager(repo_path=repo_path)

    # try to search class "Query"'s methods in the graph
    search_action = "search_class"
    search_action_input = {
        "class_name": "Query",
        "file_path": "django/db/models/sql/query.py",
    }
    content = search_manager.search_class(
        search_action_input["class_name"],
        search_action_input["file_path"],
    )
    search_result = SearchResult(
        search_action=search_action,
        search_action_input=search_action_input,
        search_content=content,
    )

    actions_list = _class_methods_ranking(
        llm=llm,
        problem_statement=problem_statement,
        search_manager=search_manager,
        search_result=search_result,
    )
    return actions_list


def test_django_15814():
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
        res = django_class_methods_scorer(
            llm, problem_statement=inst["problem_statement"]
        )
        print(res)


def django_disambiguation_scorer(llm: LLM, problem_statement: str):
    repo_path = "~/.orcar/django__django"
    repo_path = os.path.expanduser(repo_path)
    search_manager = SearchManager(repo_path=repo_path)

    # search_manager.search_callable(
    #     query_name="to_python",
    #     file_path="django/forms/models.py",
    # )
    search_action = "search_callable"
    search_action_input = {
        "query_name": "to_python",
        "file_path": "django/forms/models.py",
    }
    content = search_manager.search_callable(
        search_action_input["query_name"],
        search_action_input["file_path"],
    )
    print(content)
    search_result = SearchResult(
        search_action=search_action,
        search_action_input=search_action_input,
        search_content=content,
    )
    action_list = _disambiguation_ranking(
        llm=llm,
        problem_statement=problem_statement,
        search_manager=search_manager,
        search_result=search_result,
    )
    return action_list


def test_django_13933():
    args_dict = {
        "model": "claude-3-5-sonnet-20241022",
        "image": "sweagent/swe-agent:latest",
        "dataset": "princeton-nlp/SWE-bench_Lite",
        "persistent": True,
        "container_name": "test",
        "split": "test",
        "filter_instance": "^(django__django-13933)$",
    }

    args = argparse.Namespace(**args_dict)
    cfg = Config("./key.cfg")
    llm = get_llm(model=args.model, api_key=cfg["ANTHROPIC_API_KEY"], max_tokens=4096)
    ds = load_filter_hf_dataset(args)
    print(ds)
    for inst in ds:
        res = django_disambiguation_scorer(
            llm, problem_statement=inst["problem_statement"]
        )
        print(res)


def matplotlib_disambiguation_scorer(llm: LLM, problem_statement: str):
    repo_path = "~/.orcar/matplotlib__matplotlib"
    repo_path = os.path.expanduser(repo_path)
    search_manager = SearchManager(repo_path=repo_path)

    search_action = "search_class"
    search_action_input = {
        "class_name": "ImageGrid",
    }
    content = search_manager.search_class(
        search_action_input["class_name"],
    )
    print(content)
    search_result = SearchResult(
        search_action=search_action,
        search_action_input=search_action_input,
        search_content=content,
    )
    action_list = _disambiguation_ranking(
        llm=llm,
        problem_statement=problem_statement,
        search_manager=search_manager,
        search_result=search_result,
    )
    return action_list


def test_matplotlib_26020():
    args_dict = {
        "model": "claude-3-5-sonnet-20241022",
        "image": "sweagent/swe-agent:latest",
        "dataset": "princeton-nlp/SWE-bench_Lite",
        "persistent": True,
        "container_name": "test",
        "split": "test",
        "filter_instance": "^(matplotlib__matplotlib-26020)$",
    }

    args = argparse.Namespace(**args_dict)
    cfg = Config("./key.cfg")
    llm = get_llm(model=args.model, api_key=cfg["ANTHROPIC_API_KEY"], max_tokens=4096)
    ds = load_filter_hf_dataset(args)
    print(ds)
    for inst in ds:
        res = matplotlib_disambiguation_scorer(
            llm, problem_statement=inst["problem_statement"]
        )
        print(res)


def scikitlearn_disambiguation_scorer(llm: LLM, problem_statement: str):
    repo_path = "~/.orcar/scikit-learn__scikit-learn"
    repo_path = os.path.expanduser(repo_path)
    search_manager = SearchManager(repo_path=repo_path)

    search_action = "search_callable"
    search_action_input = {
        "query_name": "_sparse_fit",
    }
    content = search_manager.search_callable(
        search_action_input["query_name"],
    )
    print(content)
    search_result = SearchResult(
        search_action=search_action,
        search_action_input=search_action_input,
        search_content=content,
    )
    action_list = _disambiguation_ranking(
        llm=llm,
        problem_statement=problem_statement,
        search_manager=search_manager,
        search_result=search_result,
    )
    return action_list


def test_scikitlearn_14894():
    args_dict = {
        "model": "claude-3-5-sonnet-20241022",
        "image": "sweagent/swe-agent:latest",
        "dataset": "princeton-nlp/SWE-bench_Lite",
        "persistent": True,
        "container_name": "test",
        "split": "test",
        "filter_instance": "^(scikit-learn__scikit-learn-14894)$",
    }

    args = argparse.Namespace(**args_dict)
    cfg = Config("./key.cfg")
    llm = get_llm(model=args.model, api_key=cfg["ANTHROPIC_API_KEY"], max_tokens=4096)
    ds = load_filter_hf_dataset(args)
    print(ds)
    for inst in ds:
        res = scikitlearn_disambiguation_scorer(
            llm, problem_statement=inst["problem_statement"]
        )
        print(res)


def django_file_functions_scorer(llm: LLM, problem_statement: str):
    repo_path = "~/.orcar/django__django"
    repo_path = os.path.expanduser(repo_path)
    search_manager = SearchManager(repo_path=repo_path)

    search_action = "search_file_contents"
    search_action_input = {
        "file_name": "models.py",
        "directory_path": "django/forms",
    }
    content = search_manager.search_file_contents(
        search_action_input["file_name"],
        search_action_input["directory_path"],
    )
    print(content)
    search_result = SearchResult(
        search_action=search_action,
        search_action_input=search_action_input,
        search_content=content,
    )
    action_list = _file_functions_ranking(
        llm=llm,
        problem_statement=problem_statement,
        search_manager=search_manager,
        search_result=search_result,
    )
    return action_list


def test_django_13315():
    args_dict = {
        "model": "claude-3-5-sonnet-20241022",
        "image": "sweagent/swe-agent:latest",
        "dataset": "princeton-nlp/SWE-bench_Lite",
        "persistent": True,
        "container_name": "test",
        "split": "test",
        "filter_instance": "^(django__django-13315)$",
    }

    args = argparse.Namespace(**args_dict)
    cfg = Config("./key.cfg")
    llm = get_llm(model=args.model, api_key=cfg["ANTHROPIC_API_KEY"], max_tokens=4096)
    ds = load_filter_hf_dataset(args)
    print(ds)
    for inst in ds:
        res = django_file_functions_scorer(
            llm, problem_statement=inst["problem_statement"]
        )
        print(res)


if __name__ == "__main__":
    # test_django_15814()
    # test_django_13933()
    # test_matplotlib_26020()
    # test_scikitlearn_14894()
    test_django_13315()
