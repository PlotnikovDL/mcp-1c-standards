from __future__ import annotations

import logging
import os
import sys
from functools import cache
from pathlib import Path
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

SPEC = dict(
    name="1c-standards",
    module="standards_index",
    class_name="StandardsIndex",
    methods=("search", "read_standard", "list_standards"),
    env_prefix="ONEC_STANDARDS_",
)

RUNTIME_DIR = Path(__file__).resolve().parent.parent / "mcp-shared-runtime"
USE_SHARED_RUNTIME = (
    os.environ.get("ONEC_STANDARDS_STANDALONE") != "1"
    and (RUNTIME_DIR / "shared_runtime.py").is_file()
)

if USE_SHARED_RUNTIME:
    sys.path.insert(0, str(RUNTIME_DIR))
    from shared_runtime import SharedIndexProxy, run_worker_if_requested

    run_worker_if_requested(__file__, **SPEC)
    index = SharedIndexProxy(__file__, **SPEC)
elif "--shared-worker" in sys.argv[1:]:
    raise SystemExit("--shared-worker requires the sibling mcp-shared-runtime in shared mode.")

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    name="mcp-1c-standards",
    instructions=(
        "Семантический поиск по стандартам разработки 1С через Qwen3 Embedding и Qwen3 Reranker. "
        "Источник стандартов: каталог corpus\\standards в репозитории сервера."
    ),
)


@cache
def _get_index():
    if USE_SHARED_RUNTIME:
        return index

    from standards_index import StandardsIndex

    return StandardsIndex.from_env()


@mcp.tool()
def search_standards(
    query: str,
    limit: int = 5,
    mode: str = "normal",
    candidate_limit: int | None = None,
    use_reranker: bool | None = None,
) -> dict[str, Any]:
    """Найти стандарты 1С. mode: fast, normal или deep. candidate_limit/use_reranker - ручные override."""
    return _get_index().search(
        query=query,
        limit=limit,
        mode=mode,
        candidate_limit=candidate_limit,
        use_reranker=use_reranker,
    )


@mcp.tool()
def read_standard(path: str) -> dict[str, str]:
    """Прочитать markdown-файл стандарта по имени файла или полному пути."""
    return _get_index().read_standard(path)


@mcp.tool()
def list_standards() -> list[dict[str, Any]]:
    """Показать список доступных markdown-стандартов."""
    return _get_index().list_standards()


if __name__ == "__main__":
    if os.name == "nt" and not USE_SHARED_RUNTIME:
        # Keep the SDK pipe on a separate descriptor. Native libraries can
        # block on fd 0 while the SDK is already waiting for its next request.
        sys.stdin = os.fdopen(os.dup(sys.stdin.fileno()), "r", encoding="utf-8")
        with open(os.devnull, "rb") as native_stdin:
            os.dup2(native_stdin.fileno(), 0)
    mcp.run(transport="stdio")
