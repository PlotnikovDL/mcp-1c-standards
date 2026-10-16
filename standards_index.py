from __future__ import annotations

import contextlib
import os
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
):
    logging.getLogger(logger_name).setLevel(logging.ERROR)

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder, SentenceTransformer


DEFAULT_STANDARDS_DIR = Path(r"C:\projects\1c-tools\1c-docs\standards")
DEFAULT_CACHE_DIR = Path(r"C:\projects\1c-tools\mcp-1c-standards\.cache")
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_RERANKER_MODEL = "Qwen/Qwen3-Reranker-0.6B"
DEFAULT_QUERY_INSTRUCTION = (
    "Given a Russian 1C:Enterprise development task or code review question, "
    "retrieve relevant passages from 1C development standards that help solve or review the task."
)
SEARCH_MODES = {
    "fast": {"candidate_limit": 12, "use_reranker": False},
    "normal": {"candidate_limit": 12, "use_reranker": True},
    "deep": {"candidate_limit": 30, "use_reranker": True},
}


@contextlib.contextmanager
def quiet_model_output():
    if os.environ.get("ONEC_STANDARDS_VERBOSE"):
        yield
        return

    with open(os.devnull, "w", encoding="utf-8") as sink:
        with contextlib.redirect_stderr(sink):
            yield


@dataclass(frozen=True)
class StandardChunk:
    file_name: str
    path: str
    title: str
    section: str
    text: str


class StandardsIndex:
    def __init__(
        self,
        standards_dir: Path,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        reranker_model_name: str = DEFAULT_RERANKER_MODEL,
        query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
    ) -> None:
        self.standards_dir = standards_dir
        self.cache_dir = cache_dir
        self.embedding_model_name = embedding_model_name
        self.reranker_model_name = reranker_model_name
        self.query_instruction = query_instruction
        self._chunks: list[StandardChunk] | None = None
        self._embeddings: np.ndarray | None = None
        self._embedding_model: SentenceTransformer | None = None
        self._reranker_model: CrossEncoder | None = None

    @classmethod
    def from_env(cls) -> "StandardsIndex":
        standards_dir = Path(os.environ.get("ONEC_STANDARDS_DIR", str(DEFAULT_STANDARDS_DIR)))
        cache_dir = Path(os.environ.get("ONEC_STANDARDS_CACHE_DIR", str(DEFAULT_CACHE_DIR)))
        embedding_model_name = os.environ.get("ONEC_STANDARDS_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        reranker_model_name = os.environ.get("ONEC_STANDARDS_RERANKER_MODEL", DEFAULT_RERANKER_MODEL)
        query_instruction = os.environ.get("ONEC_STANDARDS_QUERY_INSTRUCTION", DEFAULT_QUERY_INSTRUCTION)
        return cls(
            standards_dir=standards_dir,
            cache_dir=cache_dir,
            embedding_model_name=embedding_model_name,
            reranker_model_name=reranker_model_name,
            query_instruction=query_instruction,
        )

    def list_standards(self) -> list[dict[str, Any]]:
        return [
            {
                "file": path.name,
                "path": str(path),
                "title": self._extract_title(path.read_text(encoding="utf-8-sig", errors="replace")),
            }
            for path in sorted(self.standards_dir.glob("*.md"))
        ]

    def read_standard(self, standard_path: str) -> dict[str, str]:
        path = self._resolve_standard_path(standard_path)
        content = path.read_text(encoding="utf-8-sig", errors="replace")
        return {
            "file": path.name,
            "path": str(path),
            "title": self._extract_title(content),
            "content": content,
        }

    def search(
        self,
        query: str,
        limit: int = 5,
        mode: str = "normal",
        candidate_limit: int | None = None,
        use_reranker: bool | None = None,
    ) -> dict[str, Any]:
        query = query.strip()
        if not query:
            return {"query": query, "results": []}

        limit = max(1, min(int(limit), 20))
        mode_config = self._resolve_mode(mode)
        resolved_candidate_limit = (
            mode_config["candidate_limit"] if candidate_limit is None else int(candidate_limit)
        )
        resolved_use_reranker = (
            mode_config["use_reranker"] if use_reranker is None else bool(use_reranker)
        )
        resolved_candidate_limit = max(limit, min(resolved_candidate_limit, 100))
        chunks = self._ensure_chunks()
        if not chunks:
            return {"query": query, "results": []}

        candidate_indexes, embedding_scores = self._semantic_candidates(query, resolved_candidate_limit)
        ranked = self._rerank(query, candidate_indexes) if resolved_use_reranker else []

        if ranked:
            ordered = ranked[:limit]
        else:
            ordered = [
                {"index": int(index), "rerank_score": None}
                for index in candidate_indexes[:limit]
            ]

        results = []
        for item in ordered:
            chunk_index = item["index"]
            chunk = chunks[chunk_index]
            results.append(
                {
                    "embedding_score": round(float(embedding_scores[chunk_index]), 4),
                    "rerank_score": None if item["rerank_score"] is None else round(float(item["rerank_score"]), 4),
                    "file": chunk.file_name,
                    "path": chunk.path,
                    "title": chunk.title,
                    "section": chunk.section,
                    "excerpt": self._excerpt(chunk.text),
                }
            )

        return {
            "query": query,
            "mode": mode,
            "embedding_model": self.embedding_model_name,
            "reranker_model": self.reranker_model_name if resolved_use_reranker else None,
            "candidate_limit": resolved_candidate_limit,
            "results": results,
        }

    @staticmethod
    def _resolve_mode(mode: str) -> dict[str, Any]:
        normalized = mode.strip().lower()
        if normalized not in SEARCH_MODES:
            allowed = ", ".join(sorted(SEARCH_MODES))
            raise ValueError(f"Unknown search mode: {mode}. Allowed: {allowed}")
        return SEARCH_MODES[normalized]

    def _semantic_candidates(self, query: str, candidate_limit: int) -> tuple[np.ndarray, np.ndarray]:
        embeddings = self._ensure_embeddings()
        query_embedding = self._embed_query(query)
        scores = embeddings @ query_embedding
        candidate_indexes = np.argsort(scores)[::-1][:candidate_limit]
        return candidate_indexes, scores

    def _rerank(self, query: str, candidate_indexes: np.ndarray) -> list[dict[str, float | int | None]]:
        if len(candidate_indexes) == 0:
            return []

        chunks = self._ensure_chunks()
        pairs = [(query, self._reranker_text(chunks[int(index)])) for index in candidate_indexes]
        with quiet_model_output():
            try:
                scores = self._get_reranker_model().predict(pairs, show_progress_bar=False)
            except TypeError:
                scores = self._get_reranker_model().predict(pairs)

        ranked = [
            {"index": int(index), "rerank_score": float(score)}
            for index, score in zip(candidate_indexes, scores, strict=False)
        ]
        ranked.sort(key=lambda item: item["rerank_score"], reverse=True)
        return ranked

    def _ensure_chunks(self) -> list[StandardChunk]:
        if self._chunks is not None:
            return self._chunks

        if not self.standards_dir.exists():
            raise FileNotFoundError(f"Standards directory not found: {self.standards_dir}")

        chunks: list[StandardChunk] = []
        for path in sorted(self.standards_dir.glob("*.md")):
            content = path.read_text(encoding="utf-8-sig", errors="replace")
            title = self._extract_title(content)
            chunks.extend(self._split_document(path, title, content))

        self._chunks = chunks
        return chunks

    def _ensure_embeddings(self) -> np.ndarray:
        if self._embeddings is not None:
            return self._embeddings

        chunks = self._ensure_chunks()
        cached_embeddings = self._load_cached_embeddings(chunks)
        if cached_embeddings is not None:
            self._embeddings = cached_embeddings
            return self._embeddings

        texts = [self._embedding_text(chunk) for chunk in chunks]
        with quiet_model_output():
            matrix = self._get_embedding_model().encode(
                texts,
                batch_size=4,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        self._embeddings = np.asarray(matrix, dtype=np.float32)
        self._save_cached_embeddings(chunks, self._embeddings)
        return self._embeddings

    def _load_cached_embeddings(self, chunks: list[StandardChunk]) -> np.ndarray | None:
        manifest_path = self.cache_dir / "manifest.json"
        embeddings_path = self.cache_dir / "embeddings.npy"
        if not manifest_path.exists() or not embeddings_path.exists():
            return None

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

        expected = self._cache_manifest(chunks)
        if manifest != expected:
            return None

        try:
            embeddings = np.load(embeddings_path)
        except OSError:
            return None

        if embeddings.shape[0] != len(chunks):
            return None

        return np.asarray(embeddings, dtype=np.float32)

    def _save_cached_embeddings(self, chunks: list[StandardChunk], embeddings: np.ndarray) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.cache_dir / "embeddings.npy", embeddings)
        (self.cache_dir / "manifest.json").write_text(
            json.dumps(self._cache_manifest(chunks), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _cache_manifest(self, chunks: list[StandardChunk]) -> dict[str, Any]:
        files = []
        for path in sorted(self.standards_dir.glob("*.md")):
            stat = path.stat()
            files.append(
                {
                    "file": path.name,
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )

        return {
            "version": 1,
            "standards_dir": str(self.standards_dir.resolve()),
            "embedding_model": self.embedding_model_name,
            "query_instruction": self.query_instruction,
            "chunk_count": len(chunks),
            "files": files,
        }

    def _embed_query(self, query: str) -> np.ndarray:
        with quiet_model_output():
            embedding = self._get_embedding_model().encode(
                [query],
                prompt=self._query_prompt(),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0]
        return np.asarray(embedding, dtype=np.float32)

    def _get_embedding_model(self) -> SentenceTransformer:
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer

            with quiet_model_output():
                self._embedding_model = SentenceTransformer(self.embedding_model_name, device="cpu")
        return self._embedding_model

    def _get_reranker_model(self) -> CrossEncoder:
        if self._reranker_model is None:
            from sentence_transformers import CrossEncoder

            prompt = {"1c_standards": self.query_instruction}
            try:
                with quiet_model_output():
                    self._reranker_model = CrossEncoder(
                        self.reranker_model_name,
                        device="cpu",
                        prompts=prompt,
                        default_prompt_name="1c_standards",
                    )
            except TypeError:
                with quiet_model_output():
                    self._reranker_model = CrossEncoder(self.reranker_model_name, device="cpu")
        return self._reranker_model

    def _query_prompt(self) -> str:
        return f"Instruct: {self.query_instruction}\nQuery: "

    def _resolve_standard_path(self, standard_path: str) -> Path:
        requested = Path(standard_path)
        path = requested if requested.is_absolute() else self.standards_dir / requested
        resolved = path.resolve()
        standards_root = self.standards_dir.resolve()

        if standards_root not in resolved.parents and resolved != standards_root:
            raise ValueError(f"Path is outside standards directory: {standard_path}")
        if not resolved.exists() or resolved.suffix.lower() != ".md":
            raise FileNotFoundError(f"Standard not found: {standard_path}")

        return resolved

    @staticmethod
    def _extract_title(content: str) -> str:
        match = re.search(r"^#\s+(.+)$", content, flags=re.MULTILINE)
        return match.group(1).strip() if match else "Без названия"

    def _split_document(self, path: Path, title: str, content: str) -> list[StandardChunk]:
        sections: list[tuple[str, str]] = []
        current_heading = title
        current_lines: list[str] = []

        for line in content.splitlines():
            heading_match = re.match(r"^(#{1,6})\s+(.+)$", line)
            if heading_match and current_lines:
                sections.append((current_heading, "\n".join(current_lines).strip()))
                current_lines = []
                current_heading = heading_match.group(2).strip()
            current_lines.append(line)

        if current_lines:
            sections.append((current_heading, "\n".join(current_lines).strip()))

        chunks: list[StandardChunk] = []
        for section, text in sections:
            for piece in self._split_long_text(text):
                if piece.strip():
                    chunks.append(
                        StandardChunk(
                            file_name=path.name,
                            path=str(path),
                            title=title,
                            section=section,
                            text=piece.strip(),
                        )
                    )

        return chunks

    @staticmethod
    def _split_long_text(text: str, max_chars: int = 2200) -> list[str]:
        if len(text) <= max_chars:
            return [text]

        paragraphs = re.split(r"\n\s*\n", text)
        chunks: list[str] = []
        current = ""

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            if current and len(current) + len(paragraph) + 2 > max_chars:
                chunks.append(current)
                current = paragraph
            else:
                current = paragraph if not current else f"{current}\n\n{paragraph}"

        if current:
            chunks.append(current)

        return chunks

    @staticmethod
    def _embedding_text(chunk: StandardChunk) -> str:
        return f"{chunk.title}\n{chunk.section}\n{chunk.text}"

    @staticmethod
    def _reranker_text(chunk: StandardChunk) -> str:
        return f"{chunk.title}\n{chunk.section}\n{chunk.text}"

    @staticmethod
    def _excerpt(text: str, max_chars: int = 700) -> str:
        compact = re.sub(r"\s+", " ", text).strip()
        if len(compact) <= max_chars:
            return compact
        return compact[: max_chars - 3].rstrip() + "..."
