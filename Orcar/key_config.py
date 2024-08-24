import os
import config


class Config:
    def __init__(self, file_path=None):
        self.file_path = file_path
        self.file_config = dict()
        if self.file_path and os.path.isfile(self.file_path):
            self.file_config = config.Config(self.file_path)
        self.fallback_config = dict()
        self.fallback_config['OPENAI_API_BASE_URL'] = ""

    def __getitem__(self, index):
        # Values in key.cfg has priority over env variables
        if index in self.file_config:
            return self.file_config[index]
        if index in os.environ:
            return os.environ[index]
        if index in self.fallback_config:
            return self.fallback_config[index]
        raise KeyError(
            f"Cannot find {index} in either cfg file '{self.file_path}' or env variables"
        )
