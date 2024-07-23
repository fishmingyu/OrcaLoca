from docx import Document


def read_docx(path: str) -> str:
    """
    Read the docx file located at 'path' and return all contents
    """
    document = Document(path)
    fullText = []
    for para in document.paragraphs:
        fullText.append(para.text)
    return "\n".join(fullText)
