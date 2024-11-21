from collections import defaultdict
from typing import List


class IndexValue:
    def __init__(
        self,
        file_path: str,
        class_name: str = None,
    ):
        self.file_path = file_path
        self.class_name = class_name

    def __repr__(self):
        return f"IndexValue({self.file_path}, {self.class_name})"


class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(list)

    def add(self, key, value):
        self.index[key].append(value)

    def remove_single_value_key(self):
        for key, value in self.index.items():
            if len(value) == 1:
                # remove the key
                self.index.pop(key)

    def search(self, key) -> List[IndexValue]:
        return self.index[key]


# class not contain single value key
class InvertedIndexNotSingleValueKey(InvertedIndex):
    def __init__(self, index):
        self.index = index
        # remove single value key
        self.remove_single_value_key()

    def search(self, key) -> List[IndexValue]:
        return self.index.search(key)

    def disambiguate(self, list_value: List[IndexValue]) -> str:
        # return disambiguated value
        disambi_str = ""
        for value in list_value:
            disambi_str += f"File path: {value.file_path}\n"
            if value.class_name:
                disambi_str += f"Class name: {value.class_name}\n"
        return disambi_str

    def search_disambiguate(self, key) -> str:
        list_value = self.search(key)
        return self.disambiguate(list_value)
