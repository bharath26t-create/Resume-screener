from pypdf import PdfReader


def extract_text(file_path):
    """
    Pulls raw text out of a resume file. Handles PDF and plain .txt —
    that covers the 'PDF/Text resumes' input requirement from the brief.
    """
    if file_path.lower().endswith(".pdf"):
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text.strip()

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()
