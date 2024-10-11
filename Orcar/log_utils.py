import logging
import os
from typing import Dict

from rich.logging import RichHandler


class LoggingManager:
    def __init__(self):
        self.loggers: Dict[str, logging.Logger] = {}
        self.current_log_dir: str = ""
        self.use_stdout: bool = True
        self.rich_handler: RichHandler = RichHandler(
            show_time=bool(os.environ.get("ORCAR_LOG_TIME", False)),
            show_path=bool(os.environ.get("ORCAR_LOG_PATH", False)),
        )
        self.rich_handler.setLevel(logging.DEBUG)

    def get_logger(self, name: str) -> logging.Logger:
        if name in self.loggers:
            return self.loggers[name]

        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        # Add the handler to the logger
        logger.addHandler(self.rich_handler)
        logger.propagate = False

        # Store the logger in our dictionary
        self.loggers[name] = logger

        return logger

    def set_log_dir(self, new_dir: str) -> None:
        self.current_log_dir = new_dir

        # Ensure the new directory exists
        os.makedirs(self.current_log_dir, exist_ok=True)

        if not self.use_stdout:
            self._update_handlers()

    def switch_to_file(self) -> None:
        self.use_stdout = False
        if self.current_log_dir:
            self._update_handlers()

    def switch_to_stdout(self) -> None:
        self.use_stdout = True
        self._update_handlers()

    def _update_handlers(self) -> None:
        for name, logger in self.loggers.items():
            # Remove existing handlers
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)

            # Add new handler
            if not self.use_stdout:
                new_handler = logging.FileHandler(
                    os.path.join(self.current_log_dir, f"{name}.log")
                )
            else:
                new_handler = self.rich_handler

            new_handler.setLevel(logging.DEBUG)
            logger.addHandler(new_handler)


# Global LoggingManager instance
logging_manager = LoggingManager()


# Convenience functions to match the original API
def get_logger(name: str) -> logging.Logger:
    return logging_manager.get_logger(name)


def set_log_dir(new_dir: str) -> None:
    logging_manager.set_log_dir(new_dir)


def switch_log_to_file() -> None:
    logging_manager.switch_to_file()


def switch_log_to_stdout() -> None:
    logging_manager.switch_to_stdout()
