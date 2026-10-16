"""Synthetic-only index tests; execute exclusively through run_isolated.py.

KnownDefectTests describe UNFIXED baseline defects, not successful checks.
No real embedding/reranking model is imported or constructed by these tests.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from standards_index import StandardsIndex


class FakeEmbedding:
    """Two-dimensional unit vectors, preserving the encode batch dimension."""

    def __init__(self, on_documents=None):
        self.calls = []
        self.on_documents = on_documents

    def encode(self, texts, **kwargs):
        texts = list(texts)
        self.calls.append((texts, kwargs))
        matrix = np.asarray(
            [[0.0, 1.0] if "beta" in text.lower() else [1.0, 0.0]
             for text in texts], dtype=np.float32,
        ).reshape(len(texts), 2)
        if "prompt" not in kwargs and self.on_documents is not None:
            self.on_documents()
        return matrix


class FakeReranker:
    def __init__(self):
        self.calls = []

    def predict(self, pairs, **kwargs):
        pairs = list(pairs)
        self.calls.append((pairs, kwargs))
        return np.asarray([0.9 if "beta" in text.lower() else 0.1
                           for _, text in pairs], dtype=np.float32)


class IndexFixture(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(dir=os.environ["MCP_STANDARDS_TEST_ROOT"])
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.corpus = self.root / "corpus"
        self.cache = self.root / "cache"
        self.corpus.mkdir()
        self.cache.mkdir()
        self.alpha = self.corpus / "a.md"
        self.alpha.write_text("# Alpha\nalpha original body\n", encoding="utf-8")
        (self.corpus / "b.md").write_text("# Beta\nbeta body\n", encoding="utf-8")
        self.index = self.make_index()

    def make_index(self, **overrides):
        options = dict(standards_dir=self.corpus, cache_dir=self.cache,
                       embedding_model_name="fake-embedding",
                       reranker_model_name="fake-reranker", query_instruction="fixture instruction")
        options.update(overrides)
        index = StandardsIndex(**options)
        index._embedding_model = FakeEmbedding()
        index._reranker_model = FakeReranker()
        return index

    def seed_cache(self):
        self.index.search("alpha", mode="fast")
        self.assertEqual(self.document_calls(self.index), 1)
        self.assertTrue((self.cache / "manifest.json").is_file())
        self.assertEqual(np.load(self.cache / "embeddings.npy").shape, (2, 2))

    @staticmethod
    def document_calls(index):
        return sum("prompt" not in kwargs for _, kwargs in index._embedding_model.calls)

    def mutate_alpha(self):
        old = self.alpha.stat()
        self.alpha.write_text("# Alpha\nbeta changed synthetic body with new length\n", encoding="utf-8")
        os.utime(self.alpha, ns=(old.st_atime_ns, old.st_mtime_ns + 2_000_000_000))

    def external_symlink(self):
        # Both link and target stay in this fixture/shared test root.
        target = self.root / "outside.md"
        target.write_text("# Outside\nsynthetic outside-root marker\n", encoding="utf-8")
        link = self.corpus / "outside-link.md"
        try:
            link.symlink_to(target)
        except NotImplementedError as error:
            self.skipTest(f"Synthetic symlinks unsupported: {error}")
        except OSError as error:
            if error.errno in (1, 13, 38, 95) or getattr(error, "winerror", None) == 1314:
                self.skipTest(f"Synthetic symlink privilege/support unavailable: {error}")
            raise
        self.assertTrue(link.is_symlink())
        self.assertEqual(link.resolve(), target.resolve())
        self.assertNotIn(self.corpus.resolve(), target.resolve().parents)
        return link


class ReadListTests(IndexFixture):
    def test_list_is_sorted_and_top_level_markdown_only(self):
        (self.corpus / "ignored.txt").write_text("ignored", encoding="utf-8")
        nested = self.corpus / "nested"
        nested.mkdir()
        (nested / "hidden.md").write_text("# Nested", encoding="utf-8")
        self.assertEqual(self.index.list_standards(), [
            {"file": "a.md", "path": str(self.alpha), "title": "Alpha"},
            {"file": "b.md", "path": str(self.corpus / "b.md"), "title": "Beta"},
        ])
        self.assertEqual(self.index._embedding_model.calls, [])
        self.assertEqual(self.index._reranker_model.calls, [])
        self.assertEqual(list(self.cache.iterdir()), [])

    def test_read_relative_and_in_root_absolute_with_bom(self):
        self.alpha.write_text("\ufeff# Заголовок\nТекст\n", encoding="utf-8")
        expected = dict(file="a.md", path=str(self.alpha.resolve()),
                        title="Заголовок", content="# Заголовок\nТекст\n")
        self.assertEqual(self.index.read_standard("a.md"), expected)
        self.assertEqual(self.index.read_standard(str(self.alpha.resolve())), expected)

    def test_read_no_heading_and_invalid_utf8(self):
        self.alpha.write_bytes(b"body\xff\n")
        result = self.index.read_standard("a.md")
        self.assertEqual(result["title"], "Без названия")
        self.assertEqual(result["content"], "body\ufffd\n")

    def test_reject_traversal_and_outside_absolute_path(self):
        outside = self.root / "outside.md"
        outside.write_text("# Outside", encoding="utf-8")
        for value in ("../outside.md", str(outside.resolve())):
            with self.subTest(path=value), self.assertRaisesRegex(ValueError, "outside standards"):
                self.index.read_standard(value)

    def test_reject_missing_wrong_extension_and_nonfile(self):
        (self.corpus / "note.txt").write_text("# Not markdown", encoding="utf-8")
        for value in ("missing.md", "note.txt"):
            with self.subTest(path=value), self.assertRaises(FileNotFoundError):
                self.index.read_standard(value)
        (self.corpus / "directory.md").mkdir()
        # Windows reports PermissionError; POSIX reports IsADirectoryError.
        with self.assertRaises((PermissionError, IsADirectoryError)):
            self.index.read_standard("directory.md")

    def test_read_rejects_outside_symlink(self):
        link = self.external_symlink()
        with self.assertRaisesRegex(ValueError, "outside standards"):
            self.index.read_standard(link.name)

    def test_from_env_uses_only_explicit_fixture_paths(self):
        env = {"ONEC_STANDARDS_DIR": str(self.corpus),
               "ONEC_STANDARDS_CACHE_DIR": str(self.cache),
               "ONEC_STANDARDS_EMBEDDING_MODEL": "env-embedding",
               "ONEC_STANDARDS_RERANKER_MODEL": "env-reranker",
               "ONEC_STANDARDS_QUERY_INSTRUCTION": "env-instruction"}
        with patch.dict(os.environ, env):
            index = StandardsIndex.from_env()
        self.assertEqual(index.standards_dir, self.corpus)
        self.assertEqual(index.cache_dir, self.cache)
        self.assertEqual(index.embedding_model_name, "env-embedding")
        self.assertEqual(index.reranker_model_name, "env-reranker")
        self.assertEqual(index.query_instruction, "env-instruction")
        self.assertIsNone(index._embedding_model)
        self.assertIsNone(index._reranker_model)


class SearchTests(IndexFixture):
    def test_empty_query_short_circuits_even_invalid_options(self):
        self.assertEqual(self.index.search(" \n\t", limit="invalid", mode="unknown",
                                           candidate_limit="invalid"),
                         {"query": "", "results": []})
        self.assertEqual(self.index._embedding_model.calls, [])
        self.assertEqual(list(self.cache.iterdir()), [])

    def test_empty_and_missing_corpus(self):
        empty = self.root / "empty"
        empty.mkdir()
        index = self.make_index(standards_dir=empty)
        self.assertEqual(index.search(" alpha "), {"query": "alpha", "results": []})
        self.assertEqual(index.list_standards(), [])
        self.assertEqual(index._embedding_model.calls, [])
        missing = self.make_index(standards_dir=self.root / "missing")
        with self.assertRaises(FileNotFoundError):
            missing.search("alpha")

    def test_invalid_arguments_fail_before_embedding(self):
        for options in ({"mode": "unknown"}, {"limit": "invalid"},
                        {"candidate_limit": "invalid"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.index.search("alpha", **options)
        self.assertEqual(self.index._embedding_model.calls, [])

    def test_fast_result_schema_prompt_and_order(self):
        result = self.index.search(" alpha ", mode="fast")
        self.assertEqual(set(result), {"query", "mode", "embedding_model", "reranker_model",
                                      "candidate_limit", "results"})
        self.assertEqual(result["query"], "alpha")
        self.assertEqual(result["embedding_model"], "fake-embedding")
        self.assertIsNone(result["reranker_model"])
        self.assertEqual([row["file"] for row in result["results"]], ["a.md", "b.md"])
        self.assertEqual(result["results"][0], {
            "embedding_score": 1.0, "rerank_score": None, "file": "a.md",
            "path": str(self.alpha), "title": "Alpha", "section": "Alpha",
            "excerpt": "# Alpha alpha original body",
        })
        docs, query = self.index._embedding_model.calls
        self.assertEqual(len(docs[0]), 2)
        self.assertEqual(docs[1]["batch_size"], 4)
        self.assertTrue(docs[1]["normalize_embeddings"])
        self.assertEqual(query[0], ["alpha"])
        self.assertEqual(query[1]["prompt"], "Instruct: fixture instruction\nQuery: ")
        self.assertEqual(self.index._reranker_model.calls, [])

    def test_modes_normalization_and_reranker_overrides(self):
        cases = [("fast", None, 12, False), ("normal", None, 12, True),
                 ("deep", None, 30, True), (" FAST ", None, 12, False),
                 ("fast", True, 12, True), ("deep", False, 30, False)]
        for mode, override, candidates, reranked in cases:
            with self.subTest(mode=mode, override=override):
                index = self.make_index()
                result = index.search("alpha", mode=mode, use_reranker=override)
                self.assertEqual(result["candidate_limit"], candidates)
                self.assertEqual(result["mode"], mode)
                self.assertEqual(bool(index._reranker_model.calls), reranked)
                self.assertEqual(result["results"][0]["file"], "b.md" if reranked else "a.md")
                self.assertEqual(result["reranker_model"], "fake-reranker" if reranked else None)
                if reranked:
                    self.assertEqual(result["results"][0]["rerank_score"], 0.9)
                    self.assertEqual(len(index._reranker_model.calls[0][0]), 2)

    def test_limits_are_clamped_and_candidates_cover_result_limit(self):
        for number in range(25):
            (self.corpus / f"extra-{number:02}.md").write_text(
                f"# Extra {number}\nalpha extra body", encoding="utf-8")
        for limit, candidate, expected_limit, expected_candidates in (
            (0, -4, 1, 1), (1000, 1000, 20, 100), ("3", "1", 3, 3),
        ):
            with self.subTest(limit=limit, candidate=candidate):
                result = self.index.search("alpha", limit=limit, mode="fast", candidate_limit=candidate)
                self.assertEqual(len(result["results"]), expected_limit)
                self.assertEqual(result["candidate_limit"], expected_candidates)


class CacheTests(IndexFixture):
    def test_manifest_and_fresh_instance_disk_reuse(self):
        self.seed_cache()
        manifest = json.loads((self.cache / "manifest.json").read_text(encoding="utf-8"))
        files = []
        for path in (self.alpha, self.corpus / "b.md"):
            stat = path.stat()
            files.append(dict(file=path.name, size=stat.st_size, mtime_ns=stat.st_mtime_ns))
        self.assertEqual(manifest, dict(version=1, standards_dir=str(self.corpus.resolve()),
                                       embedding_model="fake-embedding", query_instruction="fixture instruction",
                                       chunk_count=2, files=files))
        other = self.make_index()
        result = other.search("alpha", mode="fast")
        self.assertEqual(result["results"][0]["file"], "a.md")
        self.assertEqual(self.document_calls(other), 0)
        self.assertEqual(len(other._embedding_model.calls), 1)  # Query only.
        self.assertEqual(other._embeddings.tolist(), self.index._embeddings.tolist())

    def test_cold_cache_rebuilds_after_source_change(self):
        self.seed_cache()
        self.mutate_alpha()
        other = self.make_index()
        other.search("alpha", mode="fast")
        self.assertEqual(self.document_calls(other), 1)
        self.assertIn("changed synthetic", other._chunks[0].text)
        self.assertEqual(other._embeddings[0].tolist(), [0.0, 1.0])

    def test_manifest_mismatch_and_invalid_json_rebuild(self):
        self.seed_cache()
        manifest_path = self.cache / "manifest.json"
        original = json.loads(manifest_path.read_text(encoding="utf-8"))
        for changed in ({**original, "version": 999}, {**original, "embedding_model": "other"},
                        {**original, "query_instruction": "other"}, {**original, "chunk_count": 999},
                        {**original, "standards_dir": str(self.root / "other-corpus")}):
            with self.subTest(changed=changed):
                manifest_path.write_text(json.dumps(changed), encoding="utf-8")
                other = self.make_index()
                other.search("alpha", mode="fast")
                self.assertEqual(self.document_calls(other), 1)
        manifest_path.write_text("{invalid json", encoding="utf-8")
        other = self.make_index()
        other.search("alpha", mode="fast")
        self.assertEqual(self.document_calls(other), 1)

    def test_wrong_row_count_and_missing_matrix_rebuild(self):
        self.seed_cache()
        np.save(self.cache / "embeddings.npy", np.ones((3, 2), dtype=np.float32))
        other = self.make_index()
        other.search("alpha", mode="fast")
        self.assertEqual(self.document_calls(other), 1)
        (self.cache / "embeddings.npy").unlink()
        other = self.make_index()
        other.search("alpha", mode="fast")
        self.assertEqual(self.document_calls(other), 1)


class KnownDefectTests(IndexFixture):
    """Expected failures are unfixed correctness defects, never success claims."""

    @unittest.expectedFailure
    def test_known_defect_warm_index_does_not_refresh_changed_source(self):
        self.seed_cache()
        self.mutate_alpha()
        result = self.index.search("alpha", mode="fast")
        alpha = next(row for row in result["results"] if row["file"] == "a.md")
        self.assertIn("changed synthetic", alpha["excerpt"])
        self.assertEqual(self.document_calls(self.index), 2)

    @unittest.expectedFailure
    def test_known_defect_source_mutation_during_encode_stamps_stale_vectors_current(self):
        self.index._embedding_model = FakeEmbedding(on_documents=self.mutate_alpha)
        self.seed_cache()
        self.assertIn("changed synthetic", self.alpha.read_text(encoding="utf-8"))
        other = self.make_index()
        other.search("alpha", mode="fast")
        # Current source contains beta; its freshly encoded vector must be [0, 1].
        self.assertIn("changed synthetic", other._chunks[0].text)
        self.assertEqual(other._embeddings[0].tolist(), [0.0, 1.0])

    @unittest.expectedFailure
    def test_known_defect_corrupt_npy_is_not_recovered(self):
        self.seed_cache()
        (self.cache / "embeddings.npy").write_bytes(b"not a numpy file")
        other = self.make_index()
        other.search("alpha", mode="fast")  # Baseline raises ValueError.
        self.assertEqual(self.document_calls(other), 1)
        self.assertEqual(other._embeddings.shape, (2, 2))

    @unittest.expectedFailure
    def test_known_defect_empty_npy_is_not_recovered(self):
        self.seed_cache()
        (self.cache / "embeddings.npy").write_bytes(b"")
        other = self.make_index()
        other.search("alpha", mode="fast")  # NumPy 2.4.6 raises EOFError.
        self.assertEqual(self.document_calls(other), 1)

    @unittest.expectedFailure
    def test_known_defect_rank_one_cache_is_accepted(self):
        self.seed_cache()
        np.save(self.cache / "embeddings.npy", np.ones(2, dtype=np.float32))
        other = self.make_index()
        matrix = other._ensure_embeddings()
        self.assertEqual(matrix.ndim, 2)
        self.assertEqual(self.document_calls(other), 1)

    @unittest.expectedFailure
    def test_known_defect_rank_zero_cache_is_not_recovered(self):
        self.seed_cache()
        np.save(self.cache / "embeddings.npy", np.asarray(1.0, dtype=np.float32))
        other = self.make_index()
        matrix = other._ensure_embeddings()  # Baseline raises IndexError.
        self.assertEqual(matrix.shape, (2, 2))
        self.assertEqual(self.document_calls(other), 1)

    @unittest.expectedFailure
    def test_known_defect_nonfinite_cache_is_accepted(self):
        self.seed_cache()
        np.save(self.cache / "embeddings.npy",
                np.asarray([[np.nan, 0.0], [np.inf, 1.0]], dtype=np.float32))
        other = self.make_index()
        matrix = other._ensure_embeddings()
        self.assertTrue(np.isfinite(matrix).all())
        self.assertEqual(self.document_calls(other), 1)

    @unittest.expectedFailure
    def test_known_defect_list_exposes_symlink_outside_corpus(self):
        link = self.external_symlink()
        try:
            rows = self.index.list_standards()
        except ValueError:
            return  # Rejecting the escaped path is also a safe outcome.
        self.assertNotIn(link.name, [row["file"] for row in rows])

    @unittest.expectedFailure
    def test_known_defect_search_indexes_symlink_outside_corpus(self):
        link = self.external_symlink()
        try:
            result = self.index.search("alpha", limit=20, mode="fast")
        except ValueError:
            return  # Rejecting the escaped path is also a safe outcome.
        self.assertNotIn(link.name, [row["file"] for row in result["results"]])
