import pdfplumber
import re
import json
from pathlib import Path

def extract_text(pdf_path):
    """Extract raw text from each page of the PDF."""
    text_blocks = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                text_blocks.append({"page": i, "text": text.strip()})
    return text_blocks

def clean_text(text):
    """Clean unwanted patterns like page numbers, multiple newlines, etc."""
    text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text)
    text = re.sub(r'\n+', '\n', text)
    return text.strip()

def chunk_text_blocks(blocks, chunk_size=1000, overlap=200):
    """Split text into overlapping chunks for embedding."""
    chunks = []
    for block in blocks:
        cleaned = clean_text(block["text"])
        start = 0
        while start < len(cleaned):
            end = start + chunk_size
            chunk = cleaned[start:end].strip()
            if chunk:
                chunks.append({
                    "page": block["page"],
                    "text": chunk
                })
            start = end - overlap
    return chunks

if __name__ == "__main__":
    pdf_path = Path("data/hr_policy.pdf")  # put your HR PDF here
    pages = extract_text(pdf_path)
    chunks = chunk_text_blocks(pages)

    output_path = Path("data/chunks.json")
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"✅ Extracted and chunked {len(chunks)} text segments -> {output_path}")
