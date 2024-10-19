import bisect
import copy
import json
import re
from functools import total_ordering
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

from pydantic import BaseModel

from .log_utils import get_logger
from .types import CodeInfo

logger = get_logger(__name__)


# From https://github.com/gaogaotiantian/viztracer/blob/master/src/viztracer/functree.py
class FuncTreeNode:
    name_regex = r"(.*) \((.*?):([0-9]+)\)"

    def __init__(self, event: Optional[Dict[str, Any]] = None) -> None:
        self.filename: Optional[str] = None
        self.lineno: Optional[int] = None
        self.is_python: Optional[bool] = False
        self.funcname: Optional[str] = None
        self.parent: Optional[FuncTreeNode] = None
        self.children: List[FuncTreeNode] = []
        self.start: float = -(2**64)
        self.end: float = 2**64
        self.event: Dict[str, Any] = {}
        if event is None:
            self.event = {"name": "__ROOT__"}
            self.fullname = "__ROOT__"
        else:
            self.event = copy.copy(event)
            self.start = self.event["ts"]
            self.end = self.event["ts"] + self.event["dur"]
            self.fullname = self.event["name"]
            m = re.match(self.name_regex, self.fullname)
            if m:
                self.is_python = True
                self.funcname = m.group(1)
                self.filename = m.group(2)
                self.lineno = int(m.group(3))

    def is_ancestor(self, other: "FuncTreeNode") -> bool:
        return self.start < other.start and self.end > other.end

    def is_same(self, other: "FuncTreeNode") -> bool:
        return (
            self.fullname == other.fullname
            and len(self.children) == len(other.children)
            and all(t[0].is_same(t[1]) for t in zip(self.children, other.children))
        )

    def adopt(self, other: "FuncTreeNode") -> None:
        new_children = []
        if self.is_ancestor(other):
            # Build a list is slow
            # In almost all cases, end_idx should be the last, because that's
            # how we record entries.
            # In many cases, if two entries are siblings, start_idx is the
            # last too.
            # Try to skip building the list by checking these common situations
            # first.
            if not self.children:
                # if it's empty, then both indexes are 0
                start_idx = end_idx = 0
            else:
                if other.start > self.children[-1].start:
                    start_idx = len(self.children)
                elif other.start < self.children[0].start:
                    start_idx = 0
                else:
                    start_array = [n.start for n in self.children]
                    start_idx = bisect.bisect(start_array, other.start)
                if other.end > self.children[-1].end:
                    end_idx = len(self.children)
                else:
                    end_array = [n.end for n in self.children]
                    end_idx = bisect.bisect(end_array, other.end)
            if start_idx == end_idx + 1:
                self.children[end_idx].adopt(other)
            elif start_idx == end_idx:
                other.parent = self
                self.children.insert(start_idx, other)
            elif start_idx < end_idx:

                def change_parent(node):
                    node.parent = other

                new_children = self.children[start_idx:end_idx]
                # force map to run
                list(map(change_parent, new_children))
                other.children = new_children
                other.parent = self
                self.children = (
                    self.children[:start_idx] + [other] + self.children[end_idx:]
                )
            else:  # pragma: no cover
                raise Exception("This should not be possible")
        elif self.parent is not None:
            self.parent.adopt(other)
        else:  # pragma: no cover
            raise Exception("This should not be possible")


# From https://github.com/gaogaotiantian/viztracer/blob/master/src/viztracer/functree.py
class FuncTree:  # pragma: no cover
    def __init__(self, pid: int = 0, tid: int = 0) -> None:
        self.root: FuncTreeNode = FuncTreeNode()
        self.curr: FuncTreeNode = self.root
        self.pid: int = pid
        self.tid: int = tid

    def is_same(self, other: "FuncTree") -> bool:
        return self.root.is_same(other.root)

    def add_event(self, event: Dict[str, Any]) -> None:
        node = FuncTreeNode(event)

        self.curr.adopt(node)
        self.curr = node

    def first_ts(self) -> float:
        return self.root.children[0].event["ts"]

    def first_node(self) -> FuncTreeNode:
        return self.root.children[0]

    def node_by_timestamp(self, ts: float) -> FuncTreeNode:
        starts = [node.start for node in self.root.children]
        idx = bisect.bisect(starts, ts)
        if idx == 0:
            return self.root.children[0]
        else:
            return self.root.children[idx - 1]

    def normalize(self, first_ts: float) -> None:
        for node in self.inorder_traverse():
            node.start -= first_ts
            node.end -= first_ts

    def inorder_traverse(self) -> Generator[FuncTreeNode, None, None]:
        lst = [self.root]
        while lst:
            ret = lst.pop()
            lst.extend(ret.children[::-1])
            yield ret
        return


def gen_tracer_cmd(input_path: str, output_path: str) -> str:
    cmd = (
        f"viztracer"
        f" --quiet --ignore_c_function --ignore_frozen --include_files"
        f" . -o {output_path}"
        f" -- {input_path}"
    )
    return cmd


class FuncSign(BaseModel):
    "Function Signature"
    filename: str
    lineno: int
    funcname: str

    @classmethod
    def from_functreenode(cls, f: FuncTreeNode):
        return cls(
            filename=f.filename,
            lineno=f.lineno,
            funcname=f.funcname,
        )

    def to_codeinfo(self) -> CodeInfo:
        return CodeInfo(keyword=self.funcname, file_path=self.filename)

    def to_str(self) -> str:
        return f"{self.funcname} ({self.filename}:{self.lineno})"

    class Config:
        frozen = True


class FuncQueueElement(BaseModel):
    "Function call item in tracer log"
    node: FuncTreeNode
    layer: int
    closest_key_parent_node_layer: Tuple[FuncTreeNode, int] | None
    called_by: list[FuncSign]

    class Config:
        arbitrary_types_allowed = True


@total_ordering
class FuncScore(BaseModel):
    "Scoring point of a func, smaller is better"
    is_same_file_with_key_parent: bool
    layers_from_key_parent: int
    absolute_calling_index: int
    absolute_layer: int
    called_by: list[FuncSign]
    # TBD: score based on LLM analyzed relationship

    def get_score(self):
        return (
            int(not self.is_same_file_with_key_parent),
            self.layers_from_key_parent,
            self.absolute_layer,
            self.absolute_calling_index,
        )

    def __eq__(self, other: "FuncScore") -> bool:
        return self.get_score() == other.get_score()

    def __le__(self, other: "FuncScore") -> bool:
        return self.get_score() < other.get_score()


def read_tracer_output(
    output_path: str, sensitivity_list: List[CodeInfo]
) -> List[Tuple[FuncSign, FuncScore]]:
    sensitivity_dict: Dict[str, Set] = dict()
    for c in sensitivity_list:
        if c.keyword not in sensitivity_dict:
            sensitivity_dict[c.keyword] = set()
        if c.file_path:
            sensitivity_dict[c.keyword].add(c.file_path)
    logger.info(f"sensitivity_dict: {sensitivity_dict}")

    with open(output_path) as f:
        tracer_output = json.load(f)
    logger.info(f"Found tracer output at {output_path}")

    trace_events = tracer_output["traceEvents"]
    type_items = set([item["ph"] for item in trace_events])
    assert all(
        [item["ph"] in {"M", "X"} for item in trace_events]
    ), f"Unknown Trace Event Type: {type_items}"

    func_trees: Dict[str, FuncTree] = {}
    for data in trace_events:
        key = f"p{data['pid']}_t{data['tid']}"
        if key in func_trees:
            tree = func_trees[key]
        else:
            tree = FuncTree(data["pid"], data["tid"])
            func_trees[key] = tree

        if data["ph"] == "X":
            tree.add_event(data)
    assert (
        len(func_trees) == 1
    ), f"Unexpected multiple function trees: {list(func_trees.keys())}"
    func_tree = list(func_trees.values())[0]
    logger.info("Successfully parsed tracer output into func_tree")

    lst: List[FuncQueueElement] = [
        FuncQueueElement(
            node=func_tree.root,
            layer=0,
            closest_key_parent_node_layer=None,
            called_by=[],
        )
    ]
    func_score_dict: Dict[FuncSign, List[FuncScore]] = dict()
    file_sensitivity_set: Set[str] = set()
    absolute_calling_index: int = 0
    while lst:
        absolute_calling_index += 1
        ret = lst.pop()
        if (
            ret.node.funcname
            and ret.node.funcname in sensitivity_dict.keys()
            and ret.node.filename
            and (
                (not sensitivity_dict[ret.node.funcname])
                or (ret.node.filename in sensitivity_dict[ret.node.funcname])
            )
        ):
            ret.closest_key_parent_node_layer = (ret.node, ret.layer)
            ret.called_by = []
            file_sensitivity_set.add(ret.node.filename)
        lst.extend(
            [
                FuncQueueElement(
                    node=child,
                    layer=ret.layer + 1,
                    closest_key_parent_node_layer=ret.closest_key_parent_node_layer,
                    called_by=(
                        ret.called_by + [FuncSign.from_functreenode(ret.node)]
                        if ret.closest_key_parent_node_layer
                        else []
                    ),
                )
                for child in ret.node.children[::-1]
                # Children is already Time reversed, reverse again to reconstruct call time order
            ]
        )
        if ret.closest_key_parent_node_layer and ret.node.funcname.isidentifier():
            func_score = FuncScore(
                is_same_file_with_key_parent=(
                    ret.node.filename == ret.closest_key_parent_node_layer[0].filename
                ),
                layers_from_key_parent=(
                    ret.layer - ret.closest_key_parent_node_layer[1]
                ),
                absolute_calling_index=absolute_calling_index,
                absolute_layer=ret.layer,
                called_by=ret.called_by,
            )
            func_sign = FuncSign.from_functreenode(ret.node)
            if func_sign not in func_score_dict:
                func_score_dict[func_sign] = []
            func_score_dict[func_sign].append(func_score)

    logger.info("Got funcs:")
    logger.info(func_score_dict)

    return_sort_list: List[Tuple[FuncSign, FuncScore]] = [
        (func_sign, min(func_score_dict[func_sign])) for func_sign in func_score_dict
    ]
    return_sort_list.sort(key=lambda x: x[1])
    logger.info("Got sorted funcs:")
    for x in return_sort_list:
        logger.info((x[0], x[1], x[1].get_score()))

    # return_list: List[CodeInfo] = [x[0].to_codeinfo() for x in return_sort_list]

    logger.info("Finished tracer output parsing")
    return return_sort_list
