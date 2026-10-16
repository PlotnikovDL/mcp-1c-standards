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

    def __init__(self, on_documents=None, on_query=None):
        self.calls = []
        self.on_documents = on_documents
        self.on_query = on_query

    def encode(self, texts, **kwargs):
        texts = list(texts)
        self.calls.append((texts, kwargs))
        matrix = np.asarray(
            [[0.0, 1.0] if "beta" in text.lower() else [1.0, 0.0]
             for text in texts], dtype=np.float32,
        ).reshape(len(texts), 2)
        if "prompt" not in kwargs and self.on_documents is not None:
            self.on_documents()
        if "prompt" in kwargs and self.on_query is not None:
            self.on_query()
        return matrix


class FakeReranker:
    def __init__(self, on_predict=None):
        self.calls = []
        self.on_predict = on_predict

    def predict(self, pairs, **kwargs):
        pairs = list(pairs)
        self.calls.append((pairs, kwargs))
        scores = np.asarray([0.9 if "beta" in text.lower() else 0.1
                             for _, text in pairs], dtype=np.float32)
        if self.on_predict is not None:
            self.on_predict()
        return scores


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


class DefaultPathTests(unittest.TestCase):
    """Defaults point inside the repository; tests never touch the real corpus."""

    def test_from_env_defaults_point_inside_repository_without_access(self):
        import standards_index

        repository = Path(standards_index.__file__).resolve().parent
        self.assertEqual(standards_index.DEFAULT_STANDARDS_DIR, repository / "corpus" / "standards")
        self.assertEqual(standards_index.DEFAULT_CACHE_DIR, repository / ".cache")
        with patch.dict(os.environ):
            os.environ.pop("ONEC_STANDARDS_DIR", None)
            os.environ.pop("ONEC_STANDARDS_CACHE_DIR", None)
            # Construction only: the guard rejects any open/list of corpus/cache.
            index = StandardsIndex.from_env()
        self.assertEqual(index.standards_dir, repository / "corpus" / "standards")
        self.assertEqual(index.cache_dir, repository / ".cache")
        self.assertIsNone(index._chunks)
        self.assertIsNone(index._embeddings)
        self.assertIsNone(index._embedding_model)
        self.assertIsNone(index._reranker_model)

    def test_runner_environment_substitutes_temporary_default_corpus(self):
        import standards_index

        temporary_root = Path(os.environ["MCP_STANDARDS_TEST_ROOT"]).resolve()
        index = StandardsIndex.from_env()
        self.assertEqual(index.standards_dir.resolve(), temporary_root / "default-corpus")
        self.assertEqual(index.cache_dir.resolve(), temporary_root / "default-cache")
        self.assertFalse(index.standards_dir.resolve().is_relative_to(
            standards_index.DEFAULT_STANDARDS_DIR.parent))


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


class FreshnessTests(IndexFixture):
    def cache_bytes(self):
        return {path.name: path.read_bytes() for path in self.cache.iterdir() if path.is_file()}

    def assert_invalidated(self, index=None):
        index = self.index if index is None else index
        self.assertIsNone(index._chunks)
        self.assertIsNone(index._embeddings)
        self.assertIsNone(index._snapshot)
        self.assertIsNone(index._chunk_manifest)

    def assert_fresh_alpha(self, result, index=None):
        index = self.index if index is None else index
        alpha = next(row for row in result["results"] if row["file"] == "a.md")
        self.assertIn("changed synthetic", alpha["excerpt"])
        self.assertEqual(index._embeddings[0].tolist(), [0.0, 1.0])

    def test_known_defect_warm_index_does_not_refresh_changed_source(self):
        self.seed_cache()
        models = self.index._embedding_model, self.index._reranker_model
        chunks, vectors = self.index._chunks, self.index._embeddings
        self.mutate_alpha()
        result = self.index.search("alpha", mode="normal")
        self.assert_fresh_alpha(result)
        self.assertEqual(self.document_calls(self.index), 2)
        self.assertIsNot(self.index._chunks, chunks)
        self.assertIsNot(self.index._embeddings, vectors)
        self.assertIs(self.index._embedding_model, models[0])
        self.assertIs(self.index._reranker_model, models[1])
        pairs = models[1].calls[-1][0]
        self.assertTrue(any("changed synthetic" in text for _, text in pairs))
        self.assertFalse(any("original body" in text for _, text in pairs))

    def test_known_defect_source_mutation_during_encode_stamps_stale_vectors_current(self):
        model = FakeEmbedding(on_documents=self.mutate_alpha)
        self.index._embedding_model = model
        with self.assertRaisesRegex(RuntimeError, "changed.*repeat"):
            self.index.search("alpha", mode="fast")
        self.assert_invalidated()
        self.assertEqual(self.cache_bytes(), {})
        self.assertEqual(self.document_calls(self.index), 1)
        self.assertEqual(len(model.calls), 1)  # No retry and no query encoding.
        self.assertIs(self.index._embedding_model, model)
        model.on_documents = None
        self.assert_fresh_alpha(self.index.search("alpha", mode="fast"))
        self.assertEqual(self.document_calls(self.index), 2)
        other = self.make_index()
        self.assert_fresh_alpha(other.search("alpha", mode="fast"), other)
        self.assertEqual(self.document_calls(other), 0)

    def test_warm_add_delete_rename_empty_and_repopulate(self):
        self.seed_cache()
        extra = self.corpus / "c.md"
        extra.write_text("# Extra\nalpha added\n## Detail\nalpha detail\n", encoding="utf-8")
        result = self.index.search("alpha", limit=20, mode="normal")
        self.assertEqual(len(result["results"]), 4)
        self.assertEqual(self.document_calls(self.index), 2)
        self.alpha.unlink()
        result = self.index.search("alpha", limit=20, mode="normal")
        self.assertEqual([chunk.file_name for chunk in self.index._chunks], ["b.md", "c.md", "c.md"])
        self.assertEqual(len(result["results"]), 3)
        self.assertEqual(self.document_calls(self.index), 3)
        renamed = self.corpus / "z.md"
        extra.rename(renamed)
        result = self.index.search("alpha", limit=20, mode="normal")
        self.assertEqual({row["file"] for row in result["results"]}, {"b.md", "z.md"})
        self.assertEqual(self.document_calls(self.index), 4)
        disk = self.cache_bytes()
        (self.corpus / "b.md").unlink()
        renamed.unlink()
        self.assertEqual(self.index.search("alpha"), {"query": "alpha", "results": []})
        self.assertEqual(self.index._chunks, [])
        self.assertIsNone(self.index._embeddings)
        self.assertEqual(self.document_calls(self.index), 4)
        self.assertEqual(self.cache_bytes(), disk)  # Empty corpus does not erase cache.
        self.alpha.write_text("# Reborn\nalpha returned\n", encoding="utf-8")
        result = self.index.search("alpha", mode="fast")
        self.assertEqual([row["file"] for row in result["results"]], ["a.md"])
        self.assertEqual(result["results"][0]["title"], "Reborn")
        self.assertEqual(self.document_calls(self.index), 5)

    def test_warm_case_only_rename_refreshes_display_paths(self):
        self.seed_cache()
        before = self.alpha.stat()
        intermediate = self.corpus / "rename-intermediate.md"
        renamed = self.corpus / "A.md"
        self.alpha.rename(intermediate)
        intermediate.rename(renamed)
        self.assertEqual(renamed.stat().st_size, before.st_size)
        self.assertEqual(renamed.stat().st_mtime_ns, before.st_mtime_ns)
        result = self.index.search("alpha", mode="fast")
        self.assertEqual(result["results"][0]["file"], "A.md")
        self.assertEqual(result["results"][0]["path"], str(renamed))
        self.assertEqual(self.document_calls(self.index), 2)

    def test_same_size_new_mtime_edit_refreshes(self):
        self.seed_cache()
        before = self.alpha.stat()
        self.alpha.write_bytes(self.alpha.read_bytes().replace(b"alpha original", b"beta  original"))
        os.utime(self.alpha, ns=(before.st_atime_ns, before.st_mtime_ns + 2_000_000_000))
        self.assertEqual(self.alpha.stat().st_size, before.st_size)
        result = self.index.search("alpha", mode="fast")
        alpha = next(row for row in result["results"] if row["file"] == "a.md")
        self.assertIn("beta original", alpha["excerpt"])
        self.assertEqual(self.index._embeddings[0].tolist(), [0.0, 1.0])
        self.assertEqual(self.document_calls(self.index), 2)

    def test_unchanged_warm_search_reuses_chunks_vectors_and_models(self):
        self.seed_cache()
        chunks, vectors = self.index._chunks, self.index._embeddings
        models = self.index._embedding_model, self.index._reranker_model
        disk = self.cache_bytes()
        with patch.object(Path, "read_text", side_effect=AssertionError("Unexpected reread")), \
                patch.object(np, "load", side_effect=AssertionError("Unexpected cache load")), \
                patch.object(np, "save", side_effect=AssertionError("Unexpected cache save")):
            result = self.index.search("alpha", mode="normal")
        self.assertEqual(result["results"][0]["file"], "b.md")
        self.assertIs(self.index._chunks, chunks)
        self.assertIs(self.index._embeddings, vectors)
        self.assertIs(self.index._embedding_model, models[0])
        self.assertIs(self.index._reranker_model, models[1])
        self.assertEqual(self.document_calls(self.index), 1)
        self.assertEqual(len(models[0].calls), 3)
        self.assertEqual(self.cache_bytes(), disk)

    def test_invalid_warmed_corpus_rejected_and_state_cleared(self):
        self.seed_cache()
        disk = self.cache_bytes()
        model = self.index._embedding_model
        invalid = self.corpus / "directory.md"
        invalid.mkdir()
        with self.assertRaises(FileNotFoundError):
            self.index.search("alpha", mode="fast")
        self.assert_invalidated()
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(self.cache_bytes(), disk)
        invalid.rmdir()
        self.index.search("alpha", mode="fast")
        moved = self.root / "moved-corpus"
        self.corpus.rename(moved)
        with self.assertRaises(FileNotFoundError):
            self.index.search("alpha", mode="normal")
        self.assert_invalidated()
        self.assertEqual(len(model.calls), 4)
        moved.rename(self.corpus)
        self.assertIs(self.index._embedding_model, model)

    def test_warmed_empty_corpus_removed_is_not_silently_reused(self):
        empty = self.root / "empty"
        empty.mkdir()
        index = self.make_index(standards_dir=empty)
        self.assertEqual(index.search("alpha"), {"query": "alpha", "results": []})
        empty.rmdir()
        with self.assertRaises(FileNotFoundError):
            index.search("alpha")
        self.assert_invalidated(index)
        self.assertEqual(index.list_standards(), [])
        self.assertEqual(index._cache_manifest([])["files"], [])
        self.assertEqual(index._embedding_model.calls, [])
        self.assertEqual(self.cache_bytes(), {})

    def test_warm_empty_query_and_invalid_options_do_not_inspect_corpus(self):
        self.seed_cache()
        chunks, vectors = self.index._chunks, self.index._embeddings
        (self.corpus / "invalid.md").mkdir()
        with patch.object(self.index, "_capture_snapshot", side_effect=AssertionError("Unexpected discovery")):
            self.assertEqual(self.index.search(" ", limit="bad", mode="bad"),
                             {"query": "", "results": []})
            for options in ({"mode": "bad"}, {"limit": "bad"}, {"candidate_limit": "bad"}):
                with self.subTest(options=options), self.assertRaises(ValueError):
                    self.index.search("alpha", **options)
        self.assertIs(self.index._chunks, chunks)
        self.assertIs(self.index._embeddings, vectors)
        self.assertEqual(len(self.index._embedding_model.calls), 2)

    def test_source_change_during_text_read_aborts_before_encoding(self):
        original_read = Path.read_text

        def read_then_change(path, *args, **kwargs):
            content = original_read(path, *args, **kwargs)
            if path == self.alpha.resolve():
                self.mutate_alpha()
            return content

        with patch.object(Path, "read_text", autospec=True, side_effect=read_then_change):
            with self.assertRaisesRegex(RuntimeError, "changed.*repeat"):
                self.index.search("alpha", mode="normal")
        self.assert_invalidated()
        self.assertEqual(self.index._embedding_model.calls, [])
        self.assertEqual(self.index._reranker_model.calls, [])
        self.assertEqual(self.cache_bytes(), {})
        self.assert_fresh_alpha(self.index.search("alpha", mode="fast"))
        self.assertEqual(self.document_calls(self.index), 1)

    def test_source_changes_during_cache_reads_never_publish_cached_vectors(self):
        self.seed_cache()
        for phase in ("manifest", "matrix"):
            with self.subTest(phase=phase):
                disk = self.cache_bytes()
                other = self.make_index()
                original_read, original_load = Path.read_text, np.load

                def read_then_change(path, *args, **kwargs):
                    content = original_read(path, *args, **kwargs)
                    if path == self.cache / "manifest.json":
                        self.mutate_alpha()
                    return content

                def load_then_change(*args, **kwargs):
                    matrix = original_load(*args, **kwargs)
                    self.mutate_alpha()
                    return matrix

                hook = (patch.object(Path, "read_text", autospec=True, side_effect=read_then_change)
                        if phase == "manifest" else patch.object(np, "load", side_effect=load_then_change))
                with hook, self.assertRaisesRegex(RuntimeError, "changed.*repeat"):
                    other.search("alpha", mode="normal")
                self.assert_invalidated(other)
                self.assertEqual(other._embedding_model.calls, [])
                self.assertEqual(other._reranker_model.calls, [])
                self.assertEqual(self.cache_bytes(), disk)
                self.assert_fresh_alpha(other.search("alpha", mode="fast"), other)
                self.assertEqual(self.document_calls(other), 1)

    def test_source_change_during_query_embedding_aborts_before_reranking(self):
        self.seed_cache()
        disk = self.cache_bytes()
        model = self.index._embedding_model
        model.on_query = self.mutate_alpha
        with self.assertRaisesRegex(RuntimeError, "changed.*repeat"):
            self.index.search("alpha", mode="normal")
        self.assert_invalidated()
        self.assertEqual(self.document_calls(self.index), 1)
        self.assertEqual(len(model.calls), 3)
        self.assertEqual(self.index._reranker_model.calls, [])
        self.assertEqual(self.cache_bytes(), disk)
        model.on_query = None
        self.assert_fresh_alpha(self.index.search("alpha", mode="normal"))
        self.assertEqual(self.document_calls(self.index), 2)

    def test_change_before_reranking_does_not_rebuild_under_old_candidates(self):
        self.seed_cache()
        disk = self.cache_bytes()
        original_rerank = self.index._rerank

        def remove_then_rerank(*args):
            (self.corpus / "b.md").unlink()
            return original_rerank(*args)

        with patch.object(self.index, "_rerank", side_effect=remove_then_rerank):
            with self.assertRaisesRegex(RuntimeError, "changed.*repeat"):
                self.index.search("alpha", mode="normal")
        self.assert_invalidated()
        self.assertEqual(self.document_calls(self.index), 1)
        self.assertEqual(self.index._reranker_model.calls, [])
        self.assertEqual(self.cache_bytes(), disk)
        result = self.index.search("alpha", mode="normal")
        self.assertEqual([row["file"] for row in result["results"]], ["a.md"])
        self.assertEqual(self.document_calls(self.index), 2)

    def test_source_change_during_reranking_aborts_without_mixed_inputs(self):
        self.seed_cache()
        disk = self.cache_bytes()
        model = self.index._reranker_model
        model.on_predict = self.mutate_alpha
        with self.assertRaisesRegex(RuntimeError, "changed.*repeat"):
            self.index.search("alpha", mode="normal")
        self.assert_invalidated()
        self.assertEqual(self.document_calls(self.index), 1)
        self.assertEqual(len(model.calls), 1)
        self.assertTrue(any("original body" in text for _, text in model.calls[0][0]))
        self.assertFalse(any("changed synthetic" in text for _, text in model.calls[0][0]))
        self.assertEqual(self.cache_bytes(), disk)
        model.on_predict = None
        self.assert_fresh_alpha(self.index.search("alpha", mode="normal"))
        self.assertEqual(self.document_calls(self.index), 2)
        self.assertEqual(len(model.calls), 2)
        self.assertFalse(any("original body" in text for _, text in model.calls[1][0]))

    def test_source_change_before_output_invalidates_warm_index(self):
        self.seed_cache()
        disk = self.cache_bytes()
        original_excerpt = self.index._excerpt

        def excerpt_then_change(text):
            result = original_excerpt(text)
            self.mutate_alpha()
            return result

        with patch.object(self.index, "_excerpt", side_effect=excerpt_then_change):
            with self.assertRaisesRegex(RuntimeError, "changed.*repeat"):
                self.index.search("alpha", mode="fast")
        self.assert_invalidated()
        self.assertEqual(self.document_calls(self.index), 1)
        self.assertEqual(self.cache_bytes(), disk)
        self.assert_fresh_alpha(self.index.search("alpha", mode="fast"))
        self.assertEqual(self.document_calls(self.index), 2)

    def test_source_change_before_save_leaves_existing_disk_unchanged(self):
        self.seed_cache()
        marker = self.cache / "unrelated.txt"
        marker.write_text("keep", encoding="utf-8")
        disk = self.cache_bytes()
        self.mutate_alpha()
        original_save = self.index._save_cached_embeddings

        def change_then_save(*args):
            self.mutate_alpha()
            return original_save(*args)

        with patch.object(self.index, "_save_cached_embeddings", side_effect=change_then_save):
            with self.assertRaisesRegex(RuntimeError, "changed.*repeat"):
                self.index.search("alpha", mode="fast")
        self.assert_invalidated()
        self.assertEqual(self.document_calls(self.index), 2)
        self.assertEqual(self.cache_bytes(), disk)
        self.assert_fresh_alpha(self.index.search("alpha", mode="fast"))
        self.assertEqual(self.document_calls(self.index), 3)
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_source_changes_during_and_after_save_keep_captured_old_manifest(self):
        for phase in ("matrix", "manifest", "returned"):
            with self.subTest(phase=phase):
                self.alpha.write_text("# Alpha\nalpha original body\n", encoding="utf-8")
                self.cache = self.root / ("cache-" + phase)
                self.cache.mkdir()
                self.index = self.make_index()
                chunks = self.index._ensure_chunks()
                captured = self.index._cache_manifest(chunks)
                original_save, original_write = np.save, Path.write_text
                original_cache_save = self.index._save_cached_embeddings

                def save_then_change(*args, **kwargs):
                    original_save(*args, **kwargs)
                    self.mutate_alpha()

                def write_then_change(path, *args, **kwargs):
                    result = original_write(path, *args, **kwargs)
                    if path == self.cache / "manifest.json":
                        self.mutate_alpha()
                    return result

                def cache_save_then_change(*args):
                    original_cache_save(*args)
                    self.mutate_alpha()

                hooks = {
                    "matrix": patch.object(np, "save", side_effect=save_then_change),
                    "manifest": patch.object(Path, "write_text", autospec=True, side_effect=write_then_change),
                    "returned": patch.object(self.index, "_save_cached_embeddings", side_effect=cache_save_then_change),
                }
                with hooks[phase], self.assertRaisesRegex(RuntimeError, "changed.*repeat"):
                    self.index.search("alpha", mode="fast")
                self.assert_invalidated()
                self.assertEqual(len(self.index._embedding_model.calls), 1)
                saved = json.loads((self.cache / "manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(saved, captured)
                self.assertNotEqual(saved, self.index._cache_manifest(chunks))
                self.assertEqual(np.load(self.cache / "embeddings.npy", allow_pickle=False).tolist(),
                                 [[1.0, 0.0], [0.0, 1.0]])
                self.assert_fresh_alpha(self.index.search("alpha", mode="fast"))
                self.assertEqual(self.document_calls(self.index), 2)

    def test_warm_alias_target_change_bypasses_v1_until_successful_persistence(self):
        nested = self.corpus / "nested"
        nested.mkdir()
        old_target, new_target = nested / "old.md", nested / "new.md"
        old_target.write_text("# Alias\nalpha body\n", encoding="utf-8")
        new_target.write_text("# Alias\nbeta  body\n", encoding="utf-8")
        stat = old_target.stat()
        os.utime(new_target, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertEqual(new_target.stat().st_size, stat.st_size)
        alias = self.synthetic_symlink(self.corpus / "z-alias.md", old_target)
        self.index.search("alpha", mode="fast")
        disk = self.cache_bytes()
        old_manifest = self.index._cache_manifest(self.index._chunks)
        alias.unlink()
        self.synthetic_symlink(alias, new_target)
        # V1 cannot distinguish these targets; the observed warm snapshot can.
        self.assertEqual(self.index._cache_manifest(self.index._chunks), old_manifest)

        def failed_encode():
            raise RuntimeError("fixture interrupted encoding")

        model = self.index._embedding_model
        model.on_documents = failed_encode
        with self.assertRaisesRegex(RuntimeError, "fixture interrupted encoding"):
            self.index.search("alpha", mode="fast")
        self.assertIsNone(self.index._embeddings)
        self.assertEqual(self.document_calls(self.index), 2)
        self.assertEqual(self.cache_bytes(), disk)
        model.on_documents = None
        with patch.object(self.index, "_load_cached_embeddings", side_effect=AssertionError("Reused stale V1")):
            result = self.index.search("alpha", mode="fast")
        self.assertEqual(self.document_calls(self.index), 3)
        self.assertEqual(self.index._embeddings[-1].tolist(), [0.0, 1.0])
        row = next(row for row in result["results"] if row["file"] == alias.name)
        self.assertEqual(row["path"], str(alias))
        self.assertIn("beta body", row["excerpt"])
        other = self.make_index()
        other.search("alpha", mode="fast")
        self.assertEqual(self.document_calls(other), 0)

    def test_warm_corpus_root_change_rebuilds_with_new_paths(self):
        self.seed_cache()
        replacement = self.root / "replacement"
        replacement.mkdir()
        source = replacement / "a.md"
        source.write_text("# Replacement\nbeta replacement\n", encoding="utf-8")
        self.index.standards_dir = replacement
        result = self.index.search("alpha", mode="fast")
        self.assertEqual(len(result["results"]), 1)
        self.assertEqual(result["results"][0]["path"], str(source))
        self.assertEqual(result["results"][0]["title"], "Replacement")
        self.assertEqual(self.index._embeddings.tolist(), [[0.0, 1.0]])
        self.assertEqual(self.document_calls(self.index), 2)
        manifest = json.loads((self.cache / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["standards_dir"], str(replacement.resolve()))

    def test_refresh_preserves_bom_crlf_and_alias_order_semantics(self):
        self.seed_cache()
        payload = "\ufeff# Новый\r\nalpha refreshed\r\n## Раздел\r\nbody\r\n".encode("utf-8")
        self.alpha.write_bytes(payload)
        self.index.search("alpha", limit=20, mode="fast")
        self.assertEqual([chunk.file_name for chunk in self.index._chunks], ["a.md", "a.md", "b.md"])
        self.assertEqual(self.index._chunks[0].title, "Новый")
        self.assertEqual(self.index._chunks[0].text, "# Новый\nalpha refreshed")
        self.assertEqual(self.index._chunks[1].section, "Раздел")
        self.assertEqual(self.index._chunks[0].path, str(self.alpha))
        self.assertEqual(self.document_calls(self.index), 2)
        manifest = json.loads((self.cache / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["files"][0]["size"], len(payload))


class KnownDefectTests(IndexFixture):
    """Expected failures are unfixed limitations, never success claims."""

    @unittest.expectedFailure
    def test_known_limitation_same_size_mtime_content_change_is_not_detected(self):
        self.seed_cache()
        before = self.alpha.stat()
        original = self.alpha.read_bytes()
        changed = original.replace(b"alpha original", b"beta  original")
        self.assertNotEqual(changed, original)
        self.assertEqual(len(changed), len(original))
        self.alpha.write_bytes(changed)
        os.utime(self.alpha, ns=(before.st_atime_ns, before.st_mtime_ns))
        self.assertEqual(self.alpha.stat().st_size, before.st_size)
        self.assertEqual(self.alpha.stat().st_mtime_ns, before.st_mtime_ns)
        result = self.index.search("alpha", mode="fast")
        alpha = next(row for row in result["results"] if row["file"] == "a.md")
        # Desired freshness is deliberately unsatisfied by metadata-only policy.
        self.assertIn("beta original", alpha["excerpt"])
        self.assertEqual(self.index._embeddings[0].tolist(), [0.0, 1.0])
