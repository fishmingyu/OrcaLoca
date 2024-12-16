from pathlib import Path

from Orcar.editor import Editor


def test_string_replace_editor():
    repo_path = Path.cwd()
    editor = Editor(repo_path)
    """
    def __call__(
        self,
        *,
        command: Command,
        path: str,
        file_text: str | None = None,
        view_range: list[int] | None = None,
        old_str: str | None = None,
        new_str: str | None = None,
        insert_line: int | None = None,
        **kwargs,
    ):
    """

    # Test case create file
    test_file = Path.cwd() / "test_file.txt"
    test_content = "Hello, World!\nThis is a test file.\nThird line\n"

    # Test create command, path is relative to the repo_path
    result = editor(
        command="create",
        path=str(test_file.relative_to(repo_path)),
        file_text=test_content,
    )
    print("Create result:", result.output)

    # Test insert command
    result = editor(
        command="insert",
        path=str(test_file.relative_to(repo_path)),
        new_str="This is a new line.\n",
        insert_line=2,
    )
    print("Insert result:", result.output)

    # Test str_replace command
    result = editor(
        command="str_replace",
        path=str(test_file.relative_to(repo_path)),
        old_str="World",
        new_str="Universe",
    )
    print("Replace result:", result.output)

    # Test view command
    result = editor(
        command="view",
        path=str(test_file.relative_to(repo_path)),
        view_range=[1, 4],
    )
    print("View result:", result.output)

    # delete test file
    test_file.unlink(missing_ok=True)


def test_view_django():
    repo_path = "~/.orcar/django__django"
    expand_path = Path(repo_path).expanduser()
    editor = Editor(expand_path)
    test_file = "django/contrib/auth/validators.py"

    # Test view command
    result = editor(
        command="view",
        path=test_file,
        view_range=[1, 25],
    )
    print("View result:", result.output)


if __name__ == "__main__":
    # test_string_replace_editor()
    test_view_django()
