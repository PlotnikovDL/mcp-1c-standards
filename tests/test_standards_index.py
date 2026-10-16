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
from zipfile import ZipFile

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
        self.assertEqual(np.load(self.cache / "embeddings.npy", allow_pickle=False).shape, (2, 2))

    @staticmethod
    def document_calls(index):
        return sum("prompt" not in kwargs for _, kwargs in index._embedding_model.calls)

    def mutate_alpha(self):
        old = self.alpha.stat()
        self.alpha.write_text("# Alpha\nbeta changed synthetic body with new length\n", encoding="utf-8")
        os.utime(self.alpha, ns=(old.st_atime_ns, old.st_mtime_ns + 2_000_000_000))

    def synthetic_symlink(self, link, target, *, target_is_directory=False):
        # Even outside-corpus and broken targets stay within the broad fixture root.
        self.assertTrue(link.is_relative_to(self.root))
        self.assertTrue(target.is_relative_to(self.root))
        try:
            link.symlink_to(target, target_is_directory=target_is_directory)
        except NotImplementedError as error:
            self.skipTest(f"Synthetic symlinks unsupported: {error}")
        except OSError as error:
            if error.errno in (1, 13, 38, 95) or getattr(error, "winerror", None) == 1314:
                self.skipTest(f"Synthetic symlink privilege/support unavailable: {error}")
            raise
        self.assertTrue(link.is_symlink())
        self.assertEqual(link.resolve(), target.resolve())
        return link

    def external_symlink(self):
        target = self.root / "outside.md"
        target.write_text("# Outside\nsynthetic outside-root marker\n", encoding="utf-8")
        link = self.synthetic_symlink(self.corpus / "outside-link.md", target)
        self.assertNotIn(self.corpus.resolve(), target.resolve().parents)
        return link

    def assert_no_model_or_cache_work(self, index):
        self.assertEqual(index._embedding_model.calls, [])
        self.assertEqual(index._reranker_model.calls, [])
        self.assertEqual(list(self.cache.iterdir()), [])
        self.assertIsNone(index._chunks)
        self.assertIsNone(index._embeddings)


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
        with self.assertRaises(FileNotFoundError):
            self.index.read_standard("directory.md")

    def test_read_rejects_outside_symlink(self):
        link = self.external_symlink()
        with self.assertRaisesRegex(ValueError, "outside standards"):
            self.index.read_standard(link.name)

    def test_list_rejects_outside_symlink(self):
        self.external_symlink()
        with patch.object(Path, "read_text", side_effect=AssertionError("Read before validation")):
            with self.assertRaisesRegex(ValueError, "outside standards"):
                self.index.list_standards()
        self.assert_no_model_or_cache_work(self.index)

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
    def test_search_rejects_outside_symlink(self):
        self.external_symlink()
        with patch.object(Path, "read_text", side_effect=AssertionError("Read before validation")):
            with self.assertRaisesRegex(ValueError, "outside standards"):
                self.index.search("alpha", limit=20, mode="fast")
        self.assert_no_model_or_cache_work(self.index)

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
        self.assertEqual(index._cache_manifest([])["files"], [])
        self.assertEqual(index._embedding_model.calls, [])
        missing = self.make_index(standards_dir=self.root / "missing")
        self.assertEqual(missing.list_standards(), [])
        self.assertEqual(missing._cache_manifest([])["files"], [])
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


class DiscoveryBoundaryTests(IndexFixture):
    def assert_invalid_entry_rejected(self, entry, error_type):
        operations = {
            "read": lambda index: index.read_standard(entry.name),
            "list": lambda index: index.list_standards(),
            "chunks": lambda index: index._ensure_chunks(),
            "search_fast": lambda index: index.search("alpha", mode="fast"),
            "search_normal": lambda index: index.search("alpha", mode="normal"),
            "embeddings": lambda index: index._ensure_embeddings(),
            "manifest": lambda index: index._cache_manifest([]),
        }
        for name, operation in operations.items():
            with self.subTest(operation=name):
                index = self.make_index()
                with patch.object(Path, "read_text", side_effect=AssertionError("Read before validation")), \
                        patch.object(index, "_load_cached_embeddings") as load, \
                        patch.object(index, "_save_cached_embeddings") as save:
                    with self.assertRaises(error_type):
                        operation(index)
                    load.assert_not_called()
                    save.assert_not_called()
                self.assert_no_model_or_cache_work(index)

    def test_outside_symlink_rejected_before_content_models_or_cache(self):
        self.assert_invalid_entry_rejected(self.external_symlink(), ValueError)

    def test_nonfile_markdown_rejected_before_content_models_or_cache(self):
        entry = self.corpus / "z-directory.md"
        entry.mkdir()
        self.assert_invalid_entry_rejected(entry, FileNotFoundError)

    def test_broken_symlink_rejected_before_content_models_or_cache(self):
        entry = self.synthetic_symlink(self.corpus / "z-broken.md", self.corpus / "missing.md")
        self.assert_invalid_entry_rejected(entry, FileNotFoundError)

    def test_wrong_extension_target_rejected_before_content_models_or_cache(self):
        target = self.corpus / "target.txt"
        target.write_text("# Not Markdown", encoding="utf-8")
        entry = self.synthetic_symlink(self.corpus / "z-wrong-extension.md", target)
        self.assert_invalid_entry_rejected(entry, FileNotFoundError)

    def test_directory_symlink_rejected_before_content_models_or_cache(self):
        # Keep the target nested so only its alias participates in discovery.
        nested = self.corpus / "nested"
        nested.mkdir()
        target = nested / "directory.md"
        target.mkdir()
        entry = self.synthetic_symlink(self.corpus / "z-directory-link.md", target,
                                       target_is_directory=True)
        self.assert_invalid_entry_rejected(entry, FileNotFoundError)

    def test_internal_symlink_preserves_aliases_and_resolved_file_access(self):
        nested = self.corpus / "nested"
        nested.mkdir()
        target = nested / "target.md"
        content = "# Internal\nalpha internal body\n"
        target.write_text(content, encoding="utf-8")
        alias = self.synthetic_symlink(self.corpus / "z-alias.md", target)
        read_paths = []
        stat_paths = []
        original_read = Path.read_text
        original_stat = Path.stat

        def read_resolved(path, *args, **kwargs):
            self.assertNotEqual(path, alias)
            self.assertTrue(path.is_absolute())
            read_paths.append(path)
            return original_read(path, *args, **kwargs)

        def stat_resolved(path, *args, **kwargs):
            self.assertNotEqual(path, alias)
            stat_paths.append(path)
            return original_stat(path, *args, **kwargs)

        with patch.object(Path, "read_text", autospec=True, side_effect=read_resolved), \
                patch.object(Path, "stat", autospec=True, side_effect=stat_resolved):
            rows = self.index.list_standards()
            read = self.index.read_standard(alias.name)
            result = self.index.search("alpha", mode="fast")
            manifest = self.index._cache_manifest(self.index._chunks)

        self.assertEqual(rows, [
            {"file": "a.md", "path": str(self.alpha), "title": "Alpha"},
            {"file": "b.md", "path": str(self.corpus / "b.md"), "title": "Beta"},
            {"file": alias.name, "path": str(alias), "title": "Internal"},
        ])
        # read_standard retains its existing resolved-target identity.
        self.assertEqual(read, dict(file=target.name, path=str(target.resolve()),
                                    title="Internal", content=content))
        self.assertEqual([(chunk.file_name, chunk.path) for chunk in self.index._chunks],
                         [(row["file"], row["path"]) for row in rows])
        alias_result = next(row for row in result["results"] if row["file"] == alias.name)
        self.assertEqual(alias_result["path"], str(alias))
        self.assertEqual(alias_result["excerpt"], "# Internal alpha internal body")
        self.assertEqual(set(read_paths), {self.alpha.resolve(), (self.corpus / "b.md").resolve(),
                                          target.resolve()})
        self.assertIn(target.resolve(), stat_paths)
        files = []
        for path in (self.alpha, self.corpus / "b.md", alias):
            stat = path.resolve().stat()
            files.append(dict(file=path.name, size=stat.st_size, mtime_ns=stat.st_mtime_ns))
        self.assertEqual(manifest, dict(version=1, standards_dir=str(self.corpus.resolve()),
                                       embedding_model="fake-embedding", query_instruction="fixture instruction",
                                       chunk_count=3, files=files))
        self.assertEqual(json.loads((self.cache / "manifest.json").read_text(encoding="utf-8")), manifest)

    def test_relative_corpus_preserves_paths_and_manifest(self):
        previous = Path.cwd()
        self.addCleanup(os.chdir, previous)
        os.chdir(self.root)
        index = self.make_index(standards_dir=Path("corpus"))
        rows = index.list_standards()
        self.assertEqual(rows, [
            {"file": "a.md", "path": str(Path("corpus") / "a.md"), "title": "Alpha"},
            {"file": "b.md", "path": str(Path("corpus") / "b.md"), "title": "Beta"},
        ])
        self.assertEqual(index.read_standard("a.md"), self.index.read_standard("a.md"))
        result = index.search("alpha", mode="fast")
        self.assertEqual([(row["file"], row["path"]) for row in result["results"]],
                         [(row["file"], row["path"]) for row in rows])
        self.assertEqual([(chunk.file_name, chunk.path) for chunk in index._chunks],
                         [(row["file"], row["path"]) for row in rows])
        expected = self.index._cache_manifest(index._chunks)
        self.assertEqual(index._cache_manifest(index._chunks), expected)
        self.assertEqual(json.loads((self.cache / "manifest.json").read_text(encoding="utf-8")), expected)


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
        for payload in (b"{invalid json", b"\xff\xfe", b"null", b"[]"):
            with self.subTest(manifest=payload):
                manifest_path.write_bytes(payload)
                other = self.make_index()
                other.search("alpha", mode="fast")
                self.assertEqual(self.document_calls(other), 1)
        # JSON integer-digit limits can raise plain ValueError, not JSONDecodeError.
        other = self.make_index()
        with patch.object(json, "loads", side_effect=ValueError("fixture integer digit limit")):
            matrix = other._ensure_embeddings()
        self.assertEqual(self.document_calls(other), 1)
        self.assertEqual(matrix.tolist(), [[1.0, 0.0], [0.0, 1.0]])

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

    def assert_cache_rebuilt(self):
        other = self.make_index()
        matrix = other._ensure_embeddings()
        self.assertEqual(self.document_calls(other), 1)
        self.assertEqual(matrix.dtype, np.dtype("float32"))
        self.assertEqual(matrix.tolist(), [[1.0, 0.0], [0.0, 1.0]])
        # Recovery must replace the bad on-disk cache, not just bypass it once.
        reused = self.make_index()
        self.assertEqual(reused._ensure_embeddings().tolist(), matrix.tolist())
        self.assertEqual(self.document_calls(reused), 0)

    def test_corrupt_npy_rebuilds(self):
        self.seed_cache()
        (self.cache / "embeddings.npy").write_bytes(b"not a numpy file")
        self.assert_cache_rebuilt()

    def test_empty_npy_rebuilds(self):
        self.seed_cache()
        (self.cache / "embeddings.npy").write_bytes(b"")
        self.assert_cache_rebuilt()

    def test_rank_one_cache_rebuilds(self):
        self.seed_cache()
        np.save(self.cache / "embeddings.npy", np.ones(2, dtype=np.float32))
        self.assert_cache_rebuilt()

    def test_rank_zero_cache_rebuilds(self):
        self.seed_cache()
        np.save(self.cache / "embeddings.npy", np.asarray(1.0, dtype=np.float32))
        self.assert_cache_rebuilt()

    def test_nonfinite_cache_rebuilds(self):
        self.seed_cache()
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                np.save(self.cache / "embeddings.npy",
                        np.asarray([[value, 0.0], [0.0, 1.0]], dtype=np.float32))
                self.assert_cache_rebuilt()

    def test_signaling_nan_rebuilds_with_local_error_state(self):
        self.seed_cache()
        for integer_dtype, float_dtype, bits in (
            (np.uint32, np.float32, 0x7F800001),
            (np.uint64, np.float64, 0x7FF0000000000001),
        ):
            with self.subTest(dtype=float_dtype.__name__):
                # Build IEEE signaling NaNs by viewing bits, with no float arithmetic.
                source = np.asarray([[bits, 0], [0, 0]], dtype=integer_dtype).view(float_dtype)
                np.save(self.cache / "embeddings.npy", source)
                with np.errstate(invalid="raise"):
                    before = np.geterr().copy()
                    self.assert_cache_rebuilt()
                    self.assertEqual(np.geterr(), before)

    def test_empty_columns_and_higher_rank_rebuild(self):
        self.seed_cache()
        for shape in ((2, 0), (2, 2, 1), (0, 2)):
            with self.subTest(shape=shape):
                np.save(self.cache / "embeddings.npy", np.ones(shape, dtype=np.float32))
                self.assert_cache_rebuilt()

    def test_non_real_numeric_dtypes_rebuild(self):
        self.seed_cache()
        cases = {
            "bool": np.ones((2, 2), dtype=np.bool_),
            "complex": np.ones((2, 2), dtype=np.complex128),
            # Only ordinary benign strings: no custom objects or reduce hooks.
            "object": np.full((2, 2), "benign fixture", dtype=object),
            "unicode": np.full((2, 2), "1.0", dtype="U3"),
            "bytes": np.full((2, 2), b"1.0", dtype="S3"),
            "structured": np.zeros((2, 2), dtype=[("value", "f4")]),
            "void": np.zeros((2, 2), dtype="V4"),
            "datetime": np.zeros((2, 2), dtype="datetime64[D]"),
            "timedelta": np.zeros((2, 2), dtype="timedelta64[D]"),
        }
        for name, matrix in cases.items():
            with self.subTest(dtype=name):
                np.save(self.cache / "embeddings.npy", matrix)
                self.assert_cache_rebuilt()

    def test_real_numeric_dtypes_reuse_as_float32(self):
        self.seed_cache()
        for dtype in ("int8", "int64", "uint8", "uint64", "float16", "float32", "float64"):
            with self.subTest(dtype=dtype):
                # Three columns intentionally do not enforce the fake model's dimension.
                source = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=dtype)
                np.save(self.cache / "embeddings.npy", source)
                other = self.make_index()
                matrix = other._ensure_embeddings()
                self.assertEqual(matrix.dtype, np.dtype("float32"))
                self.assertEqual(matrix.tolist(), source.tolist())
                self.assertEqual(other._embedding_model.calls, [])

    def test_finite_float32_overflow_rebuilds_with_local_error_state(self):
        self.seed_cache()
        for value in (np.finfo(np.float64).max, -np.finfo(np.float64).max):
            with self.subTest(value=value):
                source = np.asarray([[value, 0.0], [0.0, 1.0]], dtype=np.float64)
                self.assertTrue(np.isfinite(source).all())
                np.save(self.cache / "embeddings.npy", source)
                with np.errstate(over="raise"):
                    before = np.geterr().copy()
                    self.assert_cache_rebuilt()
                    self.assertEqual(np.geterr(), before)

    def test_malformed_and_truncated_npy_and_archives_rebuild(self):
        self.seed_cache()
        path = self.cache / "embeddings.npy"
        valid_npy = path.read_bytes()
        with path.open("wb") as stream:
            np.savez_compressed(stream, vectors=np.ones((2, 2), dtype=np.float32))
        valid_zip = path.read_bytes()
        cases = {
            "npy_magic_only": valid_npy[:6],
            "npy_partial_header": valid_npy[:12],
            "npy_invalid_header": b"\x93NUMPY\x01\x00\x04\x00bad\n",
            "npy_partial_data": valid_npy[:-1],
            "zip_invalid_directory": b"PK\x03\x04broken fixture archive",
            "zip_truncated_directory": valid_zip[:-12],
        }
        for name, payload in cases.items():
            with self.subTest(payload=name):
                path.write_bytes(payload)
                self.assert_cache_rebuilt()

    def test_npz_rejected_without_member_reads_and_resources_closed(self):
        self.seed_cache()
        path = self.cache / "embeddings.npy"
        original_load = np.load
        for save in (np.savez, np.savez_compressed):
            with self.subTest(writer=save.__name__):
                with path.open("wb") as stream:
                    save(stream, vectors=np.ones((2, 2), dtype=np.float32))
                resources = []

                def track_load(*args, **kwargs):
                    archive = original_load(*args, **kwargs)
                    self.assertIsInstance(archive, np.lib.npyio.NpzFile)
                    resources.append((archive, archive.zip, archive.fid))
                    return archive

                other = self.make_index()
                with patch.object(np, "load", side_effect=track_load) as load, \
                        patch.object(np.lib.npyio.NpzFile, "__getitem__",
                                     side_effect=AssertionError("Archive member accessed")), \
                        patch.object(ZipFile, "open",
                                     side_effect=AssertionError("Archive member opened")):
                    matrix = other._ensure_embeddings()
                load.assert_called_once_with(path, allow_pickle=False)
                self.assertEqual(self.document_calls(other), 1)
                self.assertEqual(matrix.tolist(), [[1.0, 0.0], [0.0, 1.0]])
                self.assertEqual(len(resources), 1)
                archive, zip_file, file_handle = resources[0]
                self.assertIsNone(archive.zip)
                self.assertIsNone(archive.fid)
                self.assertIsNone(zip_file.fp)
                self.assertTrue(file_handle.closed)

    def test_numpy_load_explicitly_disables_pickle(self):
        self.seed_cache()
        other = self.make_index()
        with patch.object(np, "load", wraps=np.load) as load:
            matrix = other._ensure_embeddings()
        load.assert_called_once_with(self.cache / "embeddings.npy", allow_pickle=False)
        self.assertEqual(matrix.tolist(), self.index._embeddings.tolist())
        self.assertEqual(other._embedding_model.calls, [])

    def test_unreadable_manifest_and_matrix_are_cache_misses(self):
        self.seed_cache()
        for target, attribute in ((Path, "read_text"), (np, "load")):
            with self.subTest(operation=attribute):
                other = self.make_index()
                other._ensure_chunks()
                # Simulate IO failure without permissions changes or guard violations.
                with patch.object(target, attribute, side_effect=PermissionError("fixture unreadable")):
                    matrix = other._ensure_embeddings()
                self.assertEqual(self.document_calls(other), 1)
                self.assertEqual(matrix.tolist(), [[1.0, 0.0], [0.0, 1.0]])

    def test_cache_manifest_corpus_validation_errors_propagate(self):
        self.seed_cache()
        chunks = self.index._chunks
        with patch.object(np, "load") as load:
            for error in (ValueError("fixture path outside standards"),
                          FileNotFoundError("fixture invalid markdown entry")):
                with self.subTest(error=type(error).__name__), \
                        patch.object(self.index, "_cache_manifest", side_effect=error):
                    with self.assertRaises(type(error)) as raised:
                        self.index._load_cached_embeddings(chunks)
                    self.assertIs(raised.exception, error)
            # Exercise actual discovery validation too, without a symlink privilege dependency.
            (self.corpus / "directory.md").mkdir()
            with self.assertRaises(FileNotFoundError):
                self.index._load_cached_embeddings(chunks)
            load.assert_not_called()

    def test_unexpected_cache_reader_errors_propagate(self):
        self.seed_cache()
        for target, attribute in ((Path, "read_text"), (json, "loads"), (np, "load")):
            for error_type in (TypeError, RuntimeError, MemoryError, KeyboardInterrupt):
                with self.subTest(operation=attribute, error=error_type.__name__):
                    error = error_type("synthetic unexpected error")
                    with patch.object(target, attribute, side_effect=error):
                        with self.assertRaises(error_type) as raised:
                            self.index._load_cached_embeddings(self.index._chunks)
                    self.assertIs(raised.exception, error)


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
