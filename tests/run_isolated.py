"""Run fixture-only index tests and AST contracts; never start MCP/model workers.

Run with the installed standards venv Python and -I -S -B. This is a Python-level
safety guard for trusted tests, not an OS sandbox for hostile native extensions.
Exit 0: all checks passed without gaps; 1: unexpected failure/guard violation;
2: known expected failures or skipped coverage; 3: invalid invocation/setup.
"""
from __future__ import annotations

import hashlib
import importlib.abc
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
HEAVY_MODULES = {"sentence_transformers", "torch", "transformers", "huggingface_hub"}
CORE_FILES = ("server.py", "standards_index.py", "requirements.txt")
EXPECTED_DEFECT_ERRORS = {
    "test_known_defect_corrupt_npy_is_not_recovered": "ValueError",
    "test_known_defect_empty_npy_is_not_recovered": "EOFError",
    "test_known_defect_rank_zero_cache_is_not_recovered": "IndexError",
    **dict.fromkeys((
        "test_known_defect_warm_index_does_not_refresh_changed_source",
        "test_known_defect_source_mutation_during_encode_stamps_stale_vectors_current",
        "test_known_defect_rank_one_cache_is_accepted",
        "test_known_defect_nonfinite_cache_is_accepted",
    ), "AssertionError"),
}


class BlockedOperation(RuntimeError):
    pass


class CheckedResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.expected_failure_types = {}

    def addExpectedFailure(self, test, err):
        self.expected_failure_types[test.id()] = err[0].__name__
        super().addExpectedFailure(test, err)


class TestGuard(importlib.abc.MetaPathFinder):
    def __init__(self, temporary_root: Path, read_roots: list[Path]):
        self.temporary_root = temporary_root.resolve()
        self.read_roots = [p.resolve() for p in read_roots]
        self.violations: list[str] = []

    def deny(self, operation: str):
        self.violations.append(operation)
        raise BlockedOperation(operation)

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in HEAVY_MODULES:
            self.deny("forbidden model-library import: " + fullname.split(".", 1)[0])
        return None

    def check_path(self, value, *, write=False):
        # Owned file handles are also used by stdlib cleanup and NumPy.
        if isinstance(value, int) or value is None:
            return
        if os.fsdecode(value).lower() == os.devnull.lower():
            return
        path = Path(os.fsdecode(value)).resolve()
        roots = [self.temporary_root] if write else self.read_roots
        if not any(path.is_relative_to(root) for root in roots):
            self.deny("filesystem write outside fixtures" if write else "filesystem read outside allowed roots")

    def audit(self, event, args):
        if event.startswith("socket.") or event in {"subprocess.Popen", "os.system", "os.startfile", "os.startfile/2", "os.posix_spawn", "os.fork", "os.forkpty", "os.exec", "os.spawn", "_winapi.CreateProcess", "os.kill", "os.killpg"}:
            self.deny("forbidden process/network operation: " + event)
        if event == "open":
            path, mode, flags = args
            writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
                isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            )
            self.check_path(path, write=writing)
        elif event in {"os.listdir", "os.scandir"}:
            self.check_path(args[0] if args else os.getcwd())
        elif event in {"os.remove", "os.rmdir", "os.mkdir", "os.chmod", "os.utime", "os.truncate"}:
            self.check_path(args[0], write=True)
        elif event in {"os.rename", "os.link", "os.symlink"}:
            self.check_path(args[0], write=True)
            self.check_path(args[1], write=True)
        elif event == "os.chdir":
            self.check_path(args[0])

    def verify_guards(self):
        # Synthetic audit events exercise rejection without attempting an actual
        # connection, process launch or out-of-fixture write, even if a bug exists.
        checks = [
            lambda: sys.audit("subprocess.Popen", "unused", [], None, None),
            lambda: sys.audit("socket.__new__", None, 2, 1, 0),
            lambda: sys.audit("open", str(ROOT / "must-not-be-created"), "w", os.O_CREAT),
            lambda: self.find_spec("sentence_transformers"),
        ]
        for check in checks:
            before = len(self.violations)
            try:
                check()
            except BlockedOperation:
                if len(self.violations) != before + 1:
                    raise RuntimeError("Guard self-check produced an unexpected violation count")
                self.violations.pop()  # Only this intentionally generated violation.
            else:
                raise RuntimeError("Guard self-check did not block the operation")
        print("GUARD_SELF_CHECKS=4/4")


def verify_exception_capture():
    import io

    class ChainedErrorProbe(unittest.TestCase):
        @unittest.expectedFailure
        def runTest(self):
            try:
                raise ValueError("synthetic inner exception")
            except ValueError as inner:
                raise RuntimeError("synthetic unexpected outer exception") from inner

    probe = ChainedErrorProbe()
    result = unittest.TextTestRunner(stream=io.StringIO(), verbosity=0, resultclass=CheckedResult).run(unittest.TestSuite([probe]))
    if len(result.expectedFailures) != 1 or result.expected_failure_types.get(probe.id()) != "RuntimeError":
        raise RuntimeError("Expected-failure exception capture self-check failed")
    print("EXPECTED_FAILURE_CAPTURE_SELF_CHECK=True")


def fingerprints():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in CORE_FILES}


def configure_environment(temporary_root: Path):
    # Child-process-only settings: no real corpus, model cache, worker endpoint
    # or inherited service-specific configuration is used by fixtures.
    for key in tuple(os.environ):
        if key.upper().startswith(("ONEC_STANDARDS_", "ONEC_SHARED_", "HF_", "HUGGINGFACE_", "TRANSFORMERS_")) or key.upper() == "HUGGING_FACE_HUB_TOKEN":
            del os.environ[key]
    corpus = temporary_root / "default-corpus"
    cache = temporary_root / "default-cache"
    corpus.mkdir()
    cache.mkdir()
    os.environ.update(
        MCP_STANDARDS_TEST_ROOT=str(temporary_root),
        ONEC_STANDARDS_DIR=str(corpus), ONEC_STANDARDS_CACHE_DIR=str(cache),
        ONEC_SHARED_RUNTIME_DIR=str(temporary_root / "worker-state"),
        ONEC_SHARED_IDLE_SECONDS="300", ONEC_SHARED_START_TIMEOUT="30", ONEC_SHARED_CALL_TIMEOUT="300",
        HF_HOME=str(temporary_root / "models"), HF_HUB_CACHE=str(temporary_root / "models/hub"),
        SENTENCE_TRANSFORMERS_HOME=str(temporary_root / "models/sentence-transformers"),
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
        TEMP=str(temporary_root), TMP=str(temporary_root),
    )
    tempfile.tempdir = str(temporary_root)


def main():
    if len(sys.argv) != 1 or not (sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode):
        print("Use the standards venv Python with -I -S -B tests/run_isolated.py; no extra arguments.")
        return 3
    executable = Path(sys.executable)
    if executable.parent.name.lower() != "scripts":
        print("This Windows runner requires the existing standards .venv\\Scripts\\python.exe.")
        return 3
    site_packages = executable.parent.parent / "Lib/site-packages"
    if not site_packages.is_dir() or not (site_packages / "numpy/__init__.py").is_file():
        print("The selected interpreter has no installed NumPy metadata/source; nothing was installed.")
        return 3
    if HEAVY_MODULES.intersection(sys.modules):
        print("A forbidden model library was already imported; refusing to run.")
        return 3
    before = fingerprints()
    guard = None
    result = None
    code = 3
    try:
        with tempfile.TemporaryDirectory(prefix="standards-fixtures-") as owned:
            temporary_root = Path(owned).resolve()
            configure_environment(temporary_root)
            guard = TestGuard(temporary_root, [temporary_root, ROOT, ROOT.parent / "mcp-shared-runtime", Path(sys.base_prefix), site_packages])
            sys.addaudithook(guard.audit)
            sys.meta_path.insert(0, guard)
            guard.verify_guards()
            verify_exception_capture()
            # Adding paths explicitly does not process .pth/sitecustomize files.
            sys.path.insert(0, str(site_packages))
            sys.path.insert(0, str(ROOT))
            suite = unittest.TestSuite()
            loader = unittest.TestLoader()
            for pattern in ("test_standards_index.py", "test_server_contract.py"):
                selected = loader.discover(str(ROOT / "tests"), pattern=pattern)
                if selected.countTestCases() == 0:
                    raise RuntimeError("Required test module contains no discovered tests: " + pattern)
                suite.addTests(selected)
            result = unittest.TextTestRunner(verbosity=2, stream=sys.stdout, resultclass=CheckedResult).run(suite)
            defects_verified = True
            for case, detail in result.expectedFailures:
                expected_type = EXPECTED_DEFECT_ERRORS.get(case.id().rsplit(".", 1)[-1])
                actual_type = result.expected_failure_types.get(case.id())
                error_lines = [line for line in detail.splitlines()
                               if actual_type is not None and line.startswith(actual_type + ":")]
                error_line = error_lines[-1] if error_lines else detail.rstrip().splitlines()[-1]
                # Use the actual exception class, not an earlier exception in a
                # formatted chained traceback. Display text is diagnostic only.
                matched = expected_type is not None and actual_type == expected_type
                defects_verified = defects_verified and matched
                print("KNOWN_UNFIXED_DEFECT:" if matched else "UNEXPECTED_EXPECTED_FAILURE:", case.id())
                print("  ", error_line)
            for case, reason in result.skipped:
                print("UNVERIFIED_SCENARIO:", case.id(), reason)
            core_unchanged = fingerprints() == before
            model_imports = sorted(HEAVY_MODULES.intersection(sys.modules))
            numpy_module = sys.modules.get("numpy")
            numpy_from_selected_environment = numpy_module is not None and Path(numpy_module.__file__).resolve().is_relative_to(site_packages.resolve())
            print("PYTHON_VERSION=", sys.version.split()[0])
            print("NUMPY_VERSION=", getattr(numpy_module, "__version__", "not loaded"))
            print("NUMPY_FROM_SELECTED_ENVIRONMENT=", numpy_from_selected_environment)
            print("EXPECTED_FAILURE_TYPES_VERIFIED=", defects_verified)
            print("CORE_SOURCE_BYTES_UNCHANGED=", core_unchanged)
            print("FORBIDDEN_MODEL_IMPORTS=", model_imports)
            print("UNEXPECTED_GUARD_VIOLATIONS=", len(guard.violations))
            print("REGULAR_PASSES=", result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped) - len(result.expectedFailures) - len(result.unexpectedSuccesses))
            print("KNOWN_EXPECTED_FAILURES=", len(result.expectedFailures))
            print("SKIPPED_SCENARIOS=", len(result.skipped))
            if not result.wasSuccessful() or not defects_verified or not numpy_from_selected_environment or guard.violations or model_imports or not core_unchanged:
                code = 1
            elif result.expectedFailures or result.skipped:
                code = 2
            else:
                code = 0
    except Exception as exc:
        print("ISOLATED_RUN_FAILED:", type(exc).__name__, str(exc))
        code = 1
    # Include cleanup violations: the audit hook remains active until process exit.
    if guard is not None and guard.violations:
        print("GUARD_VIOLATIONS:", guard.violations)
        code = 1
    if guard is not None:
        removed = not guard.temporary_root.exists()
        print("TEMP_FIXTURES_REMOVED=", removed)
        if not removed:
            code = 1
    print("EXIT_CODE=", code)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
