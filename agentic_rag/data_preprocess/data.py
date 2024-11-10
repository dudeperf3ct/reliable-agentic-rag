"""Download dataset."""

from typing import Any

from bs4 import BeautifulSoup, NavigableString
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger


def extract_text_from_section(section) -> str:
    """
    Extract text from section.

    Reference: https://github.com/ray-project/llm-applications/blob/main/rag/data.py#L8

    Args:
        section: HTML content of the section

    Returns:
        String containing all text in the section.

    """
    texts = []
    for elem in section.children:
        if isinstance(elem, NavigableString):
            if elem.strip():
                texts.append(elem.strip())
        elif elem.name == "section":
            continue
        else:
            texts.append(elem.get_text().strip())
    return "\n".join(texts)


def parse_section(section_content: str, url: str) -> dict[str, Any]:
    """
    Parse the section to get back dictionay of text and source.

    Args:
        section_content: HTML of section.
        url: URL of the page

    Returns:
        Dictionary containing text and source uri

    """
    section_uri = section_content.find("a").get("href")
    if section_uri is not None:
        uri = url + "/" + section_content.find("a").get("href")
        content = extract_text_from_section(section_content)
        return {"source": f"{uri}", "text": content}
    else:
        logger.warning(f"Skipping {url}")
        return {"source": "", "text": ""}


def parse_html(file_path: dict[str, str]) -> list[dict[str, Any]]:
    """
    Parse html file.

    Args:
        file_path: Path to html file

    Returns:
        Parsed html containing list of dictionary where
        dictionary contains text and source uri for each section.

    """
    with open(file_path["path"], "r", encoding="utf-8") as file:
        html_content = file.read()
    soup = BeautifulSoup(html_content, "html.parser")
    sections = soup.find("body").find_all("section")
    contents = []
    if not sections:
        logger.warning(f"Skipping {file_path['path']}")
    else:
        url = "https://" + "/".join(str(file_path["path"]).split("/")[5:-1])
        for section in sections:
            section_text = parse_section(section, url)
            contents.append(section_text)
    return contents


def chunk_text(section, chunk_size, chunk_overlap) -> dict[str, Any]:
    """
    Chunk text using RecursiveCharacterTextSplitter from langchain.

    Args:
        section: Text in the section
        chunk_size: Chunk size
        chunk_overlap: Chunk overlap

    Returns:
        Chunked text

    """
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.create_documents(
        texts=[section["text"]], metadatas=[{"source": section["source"]}]
    )
    return [
        {"text": chunk.page_content, "source": chunk.metadata["source"]}
        for chunk in chunks
    ]
