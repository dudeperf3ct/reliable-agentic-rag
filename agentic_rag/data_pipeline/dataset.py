"""Download dataset."""

from loguru import logger
from typing import Any
from bs4 import BeautifulSoup, NavigableString


def extract_text_from_section(section):
    """_summary_

    Reference: https://github.com/ray-project/llm-applications/blob/main/rag/data.py#L8

    Args:
        section: _description_

    Returns:
        _description_
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
    """_summary_.

    Args:
        section_content: _description_

    Returns:
        _description_
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
    """_summary_.

    Args:
        file_path: _description_

    Returns:
        _description_
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
