import docker
from docker.models.containers import Container
import subprocess
import time
import shlex
from subprocess import PIPE, STDOUT
import logging
import os
import platform
import datetime
import hashlib
import tempfile
import tarfile
import traceback
from io import BytesIO
from rich.logging import RichHandler
from typing import Any, Callable, Union

_SET_UP_LOGGERS = set()
_ADDITIONAL_HANDLERS = []

logging.TRACE = 5  # type: ignore
logging.addLevelName(logging.TRACE, "TRACE")  # type: ignore
DOCKER_START_UP_DELAY = 1

_STREAM_LEVEL = logging.DEBUG
_FILE_LEVEL = logging.TRACE


class ContainerBash:
    def __init__(
        self,
        ctr_subprocess: subprocess.Popen,
        ctr_name: str,
        ctr: Union[Container, None] = None,
        ctr_pid: Union[int, None] = None,
    ):
        self.ctr_subprocess = ctr_subprocess
        self.ctr_name = ctr_name
        self.ctr = ctr if (ctr is not None) else get_ctr_from_name(ctr_name)
        self.ctr_pid = (
            ctr_pid if (ctr_pid is not None) else get_bash_pid_in_docker(ctr_subprocess)
        )


def get_logger(name: str) -> logging.Logger:
    """Get logger. Use this instead of `logging.getLogger` to ensure
    that the logger is set up with the correct handlers.
    """
    logger = logging.getLogger(name)
    if name in _SET_UP_LOGGERS:
        # Already set up
        return logger
    handler = RichHandler(
        show_time=bool(os.environ.get("SWE_AGENT_LOG_TIME", False)),
        show_path=False,
    )
    handler.setLevel(_STREAM_LEVEL)
    logger.setLevel(min(_STREAM_LEVEL, _FILE_LEVEL))
    logger.addHandler(handler)
    logger.propagate = False
    _SET_UP_LOGGERS.add(name)
    for handler in _ADDITIONAL_HANDLERS:
        logger.addHandler(handler)
    return logger


logger = get_logger("env_utils")


def get_container(
    ctr_name: str, image_name: str, persistent: bool = False
) -> tuple[subprocess.Popen, set]:
    """
    Get a container object for a given container name and image name

    Arguments:
        ctr_name (str): Name of container
        image_name (str): Name of image
        persistent (bool): Whether to use a persistent container or not
    Returns:
        Container object
    """
    if not image_exists(image_name):
        msg = (
            f"Image {image_name} not found. Please ensure it is built and available. "
            "Please double-check that you followed all installation/setup instructions from the "
            "readme."
        )
        raise RuntimeError(msg)

    if persistent:
        return _get_persistent_container(ctr_name, image_name)
    else:
        return _get_non_persistent_container(ctr_name, image_name)


def image_exists(image_name: str) -> bool:
    """
    Check that the image exists and give some better error messages.

    Arguments:
        image_name: Name of image
    Returns:
        bool: True if image exists
    """
    try:
        client = docker.from_env()
    except docker.errors.DockerException as e:
        docker_not_running = any(
            (
                "connection aborted" in str(e).lower(),
                "connection refused" in str(e).lower(),
                "error while fetching server api version" in str(e).lower(),
            ),
        )
        if docker_not_running:
            msg = (
                "Probably the Docker daemon is not running. Please start the Docker daemon and try again. "
                "You might need to allow the use of the docker socket "
                "(https://github.com/princeton-nlp/SWE-agent/issues/159) or symlink the socket "
                "if it's at a non-standard location "
                "(https://github.com/princeton-nlp/SWE-agent/issues/20#issuecomment-2047506005)."
            )
            raise RuntimeError(msg) from e
        raise
    filterred_images = client.images.list(filters={"reference": image_name})
    if len(filterred_images) == 0:
        return False
    elif len(filterred_images) > 1:
        RuntimeError(f"Multiple images found for {image_name}, that's weird.")
    attrs = filterred_images[0].attrs
    if attrs is not None:
        logger.info(
            f"Found image {image_name} with tags: {attrs['RepoTags']}, created: {attrs['Created']} "
            f"for {attrs['Os']} {attrs['Architecture']}.",
        )
    return True


def _get_non_persistent_container(
    ctr_name: str, image_name: str
) -> tuple[subprocess.Popen, set[str]]:
    startup_cmd = [
        "docker",
        "run",
        "-i",
        "--rm",
        "--name",
        ctr_name,
        image_name,
        "/bin/bash",
        "-l",
    ]
    logger.debug("Starting container with command: %s", shlex.join(startup_cmd))
    container = subprocess.Popen(
        startup_cmd,
        stdin=PIPE,
        stdout=PIPE,
        stderr=STDOUT,
        text=True,
        bufsize=1,  # line buffered
    )
    time.sleep(DOCKER_START_UP_DELAY)
    # try to read output from container setup (usually an error), timeout if no output
    output = read_with_timeout(container, lambda: list(), timeout_duration=2)
    if output:
        logger.error(f"Unexpected container setup output: {output}")
    # bash PID is always 1 for non-persistent containers
    return container, {
        "1",
    }


def _get_persistent_container(
    ctr_name: str, image_name: str, persistent: bool = False
) -> tuple[subprocess.Popen, set[str]]:
    client = docker.from_env()
    containers = client.containers.list(all=True, filters={"name": ctr_name})
    if ctr_name in [c.name for c in containers]:
        container_obj = client.containers.get(ctr_name)
        if container_obj.status in {"created"}:
            container_obj.start()
        elif container_obj.status in {"running"}:
            pass
        elif container_obj.status in {"exited"}:
            container_obj.restart()
        elif container_obj.status in {"paused"}:
            container_obj.unpause()
        else:
            msg = f"Unexpected container status: {container_obj.status}"
            raise RuntimeError(msg)
    else:
        container_obj = client.containers.run(
            image_name,
            command="/bin/bash -l -m",
            name=ctr_name,
            stdin_open=True,
            tty=True,
            detach=True,
            auto_remove=not persistent,
        )
        container_obj.start()
    startup_cmd = [
        "docker",
        "exec",
        "-i",
        ctr_name,
        "/bin/bash",
        "-l",
    ]
    logger.debug("Starting container with command: %s", shlex.join(startup_cmd))
    container = subprocess.Popen(
        startup_cmd,
        stdin=PIPE,
        stdout=PIPE,
        stderr=STDOUT,
        text=True,
        bufsize=1,  # line buffered
    )
    time.sleep(DOCKER_START_UP_DELAY)
    # try to read output from container setup (usually an error), timeout if no output
    output = read_with_timeout(container, lambda: list(), timeout_duration=2)
    if output:
        logger.error(f"Unexpected container setup output: {output}")
    # Get the process IDs of the container
    # There should be at least a head process and possibly one child bash process
    bash_pids, other_pids = get_background_pids(container_obj)
    total_time_slept = DOCKER_START_UP_DELAY
    # Let's wait for a maximum of 5 x DOCKER_START_UP_DELAY seconds
    # and then check again.
    while len(bash_pids) > 1 or len(other_pids) > 0:
        time.sleep(1)
        total_time_slept += 1
        bash_pids, other_pids = get_background_pids(container_obj)
        if total_time_slept > 5 * DOCKER_START_UP_DELAY:
            break
    bash_pid = 1
    if len(bash_pids) == 1:
        bash_pid = bash_pids[0][0]
    elif len(bash_pids) > 1 or len(other_pids) > 0:
        msg = (
            "Detected alien processes attached or running. Please ensure that no other agents "
            f"are running on this container. PIDs: {bash_pids}, {other_pids}"
        )
        raise RuntimeError(msg)
    return container, {str(bash_pid), "1"}


def read_with_timeout(
    container: subprocess.Popen, pid_func: Callable, timeout_duration: Union[int, float]) -> str:
    """
    Read data from a subprocess with a timeout.
    This function uses a file descriptor to read data from the subprocess in a non-blocking way.

    Args:
        container: The subprocess container.
        pid_func: A function that returns a list of process IDs (except the PID of the main process).
        timeout_duration: The timeout duration in seconds.

    Returns:
        output: The data read from the subprocess, stripped of trailing newline characters.

    Raises:
        TimeoutError: If the timeout duration is reached while reading from the subprocess.
    """
    buffer = b""
    fd = container.stdout.fileno()
    end_time = time.time() + timeout_duration

    import select

    def ready_to_read(fd) -> bool:
        return bool(select.select([fd], [], [], 0.01)[0])

    while time.time() < end_time:
        pids = pid_func()
        if len(pids) > 0:
            # There are still PIDs running
            time.sleep(0.05)
            continue
        if ready_to_read(fd):
            data = os.read(fd, 4096)
            if data:
                buffer += data
        else:
            # No more data to read
            break
        time.sleep(0.05)  # Prevents CPU hogging

    if container.poll() is not None:
        msg = f"Subprocess exited unexpectedly.\nCurrent buffer: {buffer.decode()}"
        raise RuntimeError(msg)
    if time.time() >= end_time:
        msg = f"Timeout reached while reading from subprocess.\nCurrent buffer: {buffer.decode()}\nRunning PIDs: {pids}"
        raise TimeoutError(msg)
    return buffer.decode()

def read_generator_with_timeout(
    container: subprocess.Popen, pid_func: Callable, timeout_duration: Union[int, float]):
    fd = container.stdout.fileno()
    end_time = time.time() + timeout_duration

    import select

    def ready_to_read(fd) -> bool:
        return bool(select.select([fd], [], [], 0.01)[0])

    execution_finished = False
    pids = []
    while time.time() < end_time:
        
        while ready_to_read(fd):
            data = os.read(fd, 4096)
            if data:
                yield data.decode()
                # Refresh timeout if got output
                end_time = time.time() + timeout_duration
            time.sleep(0.05)  # Prevents CPU hogging
        else:
            time.sleep(0.05)  # Prevents CPU hogging
        if execution_finished:
            break
        for i in range(3):
            # Issue 3 consecutive PID check within 0.1s to make sure we really finished
            # E.g. If we run 'cmd A && cmd B', 
            # There will be a short no-PID interval between finishing A and starting B
            if i != 0:
                # Don't waste time sleeping before 1st trial
                time.sleep(0.05)
            pids = pid_func()
            execution_finished = (len(pids) == 0)
            if not execution_finished:
                 # Don't waste time sleeping if not finished
                 break
            

    if container.poll() is not None:
        msg = f"Subprocess exited unexpectedly.\n"
        raise RuntimeError(msg)
    if time.time() >= end_time:
        msg = f"Timeout reached while reading from subprocess.\nRunning PIDs: {pids}"
        raise TimeoutError(msg)


def get_background_pids(container_obj: Container):
    pids = (
        container_obj.exec_run("ps -eo pid,comm --no-headers")
        .output.decode()
        .split("\n")
    )
    pids = [x.split() for x in pids if x]
    pids = [x for x in pids if x[1] not in {"ps"} and x[0] != "1"]
    bash_pids = [x for x in pids if x[1] == "bash"]
    other_pids = [x for x in pids if x[1] not in {"bash"}]
    return bash_pids, other_pids


def get_children_pids(container_obj: Container, parent_pid: int):
    pids = (
        container_obj.exec_run(f"ps -o pid --no-headers --ppid {parent_pid}")
        .output.decode()
        .split("\n")
    )
    pids = [int(x) for x in pids if x]
    return pids


def run_command_in_container(
    ctr_bash: ContainerBash, command: str, timeout: int = 5, output_log: bool = False
) -> str:
    """
    Run a command in a container and return the output.
    Output is streamed to stdout.

    Args:
        container: The container subprocess.
        command: The command to run.
        timeout: The timeout in seconds.

    Returns:
        output: The output of the command.

    Raises:
        TimeoutError: If the command times out.
    """

    ctr_bash.ctr_subprocess.stdin.write(f"{command}\n")
    ctr_bash.ctr_subprocess.stdin.flush()
    if output_log:
        logger.debug(f"Run command in container: {command}")
    output = ""
    output_generator = read_generator_with_timeout(
        ctr_bash.ctr_subprocess,
        lambda: get_children_pids(ctr_bash.ctr, ctr_bash.ctr_pid),
        timeout,
    )
    for output_fraction in output_generator:
        if output_log:
            print(output_fraction, end="")
        output += output_fraction

    time.sleep(0.05)
    if len(get_children_pids(ctr_bash.ctr, ctr_bash.ctr_pid)):
        extra_output = read_with_timeout(
            ctr_bash.ctr_subprocess,
            lambda: get_children_pids(ctr_bash.ctr, ctr_bash.ctr_pid),
            timeout,
        )
        if extra_output:
            logger.warning(f"Got extra output suffix:\n{extra_output}")
            if output_log:
                print(extra_output, end="")
            output += extra_output

    return output

def get_exit_code(
    ctr_bash: ContainerBash, timeout: int = 5
) -> str:
    ctr_bash.ctr_subprocess.stdin.write(f"echo $?\n")
    ctr_bash.ctr_subprocess.stdin.flush()
    output = read_with_timeout(
        ctr_bash.ctr_subprocess,
        lambda: get_children_pids(ctr_bash.ctr, ctr_bash.ctr_pid),
        timeout,
    )
    return int(output.strip())


def get_ctr_from_name(ctr_name: str) -> Container:
    client = docker.from_env()
    containers = client.containers.list(all=True, filters={"name": ctr_name})
    if ctr_name in [c.name for c in containers]:
        return client.containers.get(ctr_name)
    else:
        raise ValueError(f"get_ctr_from_name(): Cannot find container {ctr_name}")


def run_bash_in_ctr(ctr_bash: ContainerBash, command: str) -> str:
    """
    Run a command with a new process in started container and return the output.

    Args:
        ctr: The container object.
        command: The command to run.

    Returns:
        output: The output of the command.
    """
    ctr = ctr_bash.ctr
    ctr_name = ctr_bash.ctr_name
    startup_cmd = ["docker", "exec", "-i", ctr_name, "/bin/bash", "-l"]
    # logger.debug(f"Starting process in container {ctr_name}")
    ctr_new_subprocess = subprocess.Popen(
        startup_cmd,
        stdin=PIPE,
        stdout=PIPE,
        stderr=STDOUT,
        text=True,
        bufsize=1,  # line buffered
    )
    ctr_new_bash_pid = get_bash_pid_in_docker(ctr_new_subprocess)
    logger.debug(
        f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}: Started bash process {ctr_new_bash_pid} in container {ctr_name}"
    )
    # start = time.time()
    ctr_new_subprocess.stdin.write(f"{command}\n")
    ctr_new_subprocess.stdin.flush()
    output = read_with_timeout(
        ctr_new_subprocess, lambda: get_children_pids(ctr, ctr_new_bash_pid), 10
    )
    # if output:
    #    logger.info(f"Command output: {output}")
    exit_code = ctr_new_subprocess.returncode
    ctr_new_subprocess.stdin.close()
    # logger.debug(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}: Finished bash process {ctr_bash_pid} in container {ctr_name} after {time.time()-start} s")
    return f"Exit code: {exit_code}, Output:\n{output}"


def generate_container_name(image_name: str) -> str:
    """Return name of container"""
    process_id = str(os.getpid())
    current_time = str(datetime.datetime.now())
    unique_string = current_time + process_id
    hash_object = hashlib.sha256(unique_string.encode())
    image_name_sanitized = image_name.replace("/", "-")
    image_name_sanitized = image_name_sanitized.replace(":", "-")
    return f"{image_name_sanitized}-{hash_object.hexdigest()[:10]}"


def pause_persistent_container(ctr_bash: ContainerBash):
    ctr = ctr_bash.ctr
    if ctr.status not in {"paused", "exited", "dead", "stopping"}:
        try:
            ctr.pause()
        except Exception:
            logger.warning("Failed to pause container.", exc_info=True)
        except KeyboardInterrupt:
            raise
        else:
            logger.info("Agent container paused")
    else:
        logger.info(f"Agent container status: {ctr.status}")


def get_bash_pid_in_docker(ctr_subprocess: subprocess.Popen) -> int:
    ctr_subprocess.stdin.write(f"echo $$\n")
    ctr_subprocess.stdin.flush()
    output = ""
    while output == "":
        output = read_with_timeout(ctr_subprocess, lambda: list(), 5)
        time.sleep(0.05)
    return output.split("\n")[0]


def copy_file_to_container(container: Container, contents: str, container_path: str) -> None:
    """
    Copies a given string into a Docker container at a specified path.

    Args:
        container: Docker SDK container object.
        contents: The string to copy into the container.
        container_path: The path inside the container where the string should be copied to.

    Returns:
        None
    """
    temp_file_name = None

    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file_name = temp_file.name
            # Write the string to the temporary file and ensure it's written to disk
            temp_file.write(contents.encode("utf-8"))
            temp_file.flush()
            os.fsync(temp_file.fileno())

        # Create a TAR archive in memory containing the temporary file
        with tempfile.NamedTemporaryFile():
            with open(temp_file_name, "rb") as temp_file:
                # Prepare the TAR archive
                with BytesIO() as tar_stream:
                    with tarfile.open(fileobj=tar_stream, mode="w") as tar:
                        tar_info = tarfile.TarInfo(name=os.path.basename(container_path))
                        tar_info.size = os.path.getsize(temp_file_name)
                        tar.addfile(tarinfo=tar_info, fileobj=temp_file)
                    tar_stream.seek(0)
                    # Copy the TAR stream to the container
                    container.put_archive(path=os.path.dirname(container_path), data=tar_stream.read())

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        logger.error(traceback.format_exc())
    finally:
        # Cleanup: Remove the temporary file if it was created
        if temp_file_name and os.path.exists(temp_file_name):
            os.remove(temp_file_name)
