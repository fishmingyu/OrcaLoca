import asyncio
from pathlib import Path

from Orcar.editor import StringReplaceEditor


async def test_string_replace_editor():
    editor = StringReplaceEditor()
    """
    async def __call__(
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
    test_file = Path.cwd() / "test.txt"
    test_content = "Hello, World!\nThis is a test file.\nThird line\n"

    # Test create command
    result = await editor(
        command="create", path=str(test_file.absolute()), file_text=test_content
    )
    print("Create result:", result.output)

    # Test insert command
    result = await editor(
        command="insert",
        path=str(test_file.absolute()),
        new_str="This is a new line.\n",
        insert_line=2,
    )
    print("Insert result:", result.output)

    # Test str_replace command
    result = await editor(
        command="str_replace",
        path=str(test_file.absolute()),
        old_str="World",
        new_str="Universe",
    )
    print("Replace result:", result.output)

    # Test view command
    result = await editor(
        command="view",
        path=str(test_file.absolute()),
        view_range=[1, 4],
    )
    print("View result:", result.output)

    # delete test file
    test_file.unlink(missing_ok=True)


if __name__ == "__main__":
    asyncio.run(test_string_replace_editor())
