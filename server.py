from __future__ import annotations

import logging
import os
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TQDM_DISABLE", "1")

logging.basicConfig(level=logging.WARNING)
for logger_name in (
    "httpx",
    "huggingface_hub",
    "huggingface_hub.utils._http",
    "sentence_transformers",
    "transformers",
    "mcp",
):
    logging.getLogger(logger_name).setLevel(logging.ERROR)

from mcp.server.fastmcp import FastMCP

from standards_index import StandardsIndex

mcp = FastMCP(
    name="mcp-1c-standards",
    instructions=(
        "Семантический поиск по стандартам разработки 1С через Qwen3 Embedding и Qwen3 Reranker. "
        "Источник стандартов: C:\\projects\\1c-tools\\1c-docs\\standards."
    ),
)

index = StandardsIndex.from_env()


@mcp.tool()
def search_standards(
    query: str,
    limit: int = 5,
    mode: str = "normal",
    candidate_limit: int | None = None,
    use_reranker: bool | None = None,
) -> dict[str, Any]:
    """Найти стандарты 1С. mode: fast, normal или deep. candidate_limit/use_reranker - ручные override."""
    return index.search(
        query=query,
        limit=limit,
        mode=mode,
        candidate_limit=candidate_limit,
        use_reranker=use_reranker,
    )


@mcp.tool()
def read_standard(path: str) -> dict[str, str]:
    """Прочитать markdown-файл стандарта по имени файла или полному пути."""
    return index.read_standard(path)


@mcp.tool()
def list_standards() -> list[dict[str, Any]]:
    """Показать список доступных markdown-стандартов."""
    return index.list_standards()


if __name__ == "__main__":
    mcp.run(transport="stdio")
