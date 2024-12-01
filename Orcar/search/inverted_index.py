from collections import defaultdict
from typing import List


class IndexValue:
    def __init__(
        self,
        file_path: str,
        class_name: str | None = None,
    ):
        self.file_path = file_path
        self.class_name = class_name

    def __repr__(self):
        if self.class_name:
            return f"Query's location: {self.file_path}::{self.class_name}"
        return f"Query's location: {self.file_path}"


class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(list)

    def add(self, key: str, value: IndexValue):
        self.index[key].append(value)

    def remove_single_value_key(self):
        # Create a list of keys to remove first
        keys_to_remove = [key for key, value in self.index.items() if len(value) == 1]
        # Then remove them
        for key in keys_to_remove:
            self.index.pop(key)

    def search(self, key: str) -> List[IndexValue]:
        return self.index[key]
