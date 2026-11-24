"""Explicitly approved import-only probe. No MCP/HTTP/WMI serving or tool calls.

Use the selected existing Windows venv Python with -I -S -B; no extra arguments.
Exit 0: imports/local registration verified; 1: failure or safety violation;
3: invalid invocation/environment. This does not test JSON-RPC or real models.
"""
from __future__ import annotations

import contextlib
import importlib
import importlib.metadata
from pathlib import Path
import os
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TOOLS = {"search_standards", "read_standard", "list_standards"}


def check_registration(server):
    assert server.mcp.name == "mcp-1c-standards"
    registered = {tool.name for tool in server.mcp._tool_manager.list_tools()}
    assert registered == EXPECTED_TOOLS


def main():
    if len(sys.argv) != 1 or sys.flags.optimize or not (sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode):
        print("Use the selected venv Python with -I -S -B tests/check_imports.py; no -O or extra arguments.")
        return 3
    if Path.cwd().resolve() != ROOT:
        print("Run the import probe from the prepared standards worktree root.")
        return 3
    executable = Path(sys.executable)
    if executable.parent.name.lower() != "scripts":
        print("The import probe requires an existing Windows venv interpreter.")
        return 3
    site_packages = executable.parent.parent / "Lib/site-packages"
    if not site_packages.is_dir():
        print("Selected site-packages directory is absent; nothing was installed.")
        return 3
    runtime_root = ROOT.parent / "mcp-shared-runtime"
    has_shared_runtime = (runtime_root / "shared_runtime.py").is_file()
    sys.path.insert(0, str(ROOT / "tests"))
    # Definitions only: importing the runner does not invoke its main function.
    from run_isolated import HEAVY_MODULES, TestGuard, configure_environment, fingerprints

    if HEAVY_MODULES.intersection(sys.modules):
        print("A forbidden model library was already imported.")
        return 3
    before = fingerprints()
    guard = None
    code = 1
    try:
        with tempfile.TemporaryDirectory(prefix="standards-import-probe-") as owned, contextlib.chdir(owned), contextlib.ExitStack() as native_paths:
            temporary_root = Path(owned).resolve()
            configure_environment(temporary_root)
            for key in tuple(os.environ):
                if key.upper().startswith(("FASTMCP_", "OTEL_")):
                    del os.environ[key]
            guard = TestGuard(temporary_root, [temporary_root, ROOT, runtime_root, Path(sys.base_prefix), site_packages])
            sys.addaudithook(guard.audit)
            sys.meta_path.insert(0, guard)
            guard.verify_guards()
            # Mirror only the required installed pywin32 paths, never execute
            # pywin32.pth or pywin32_bootstrap; installed files remain untouched.
            win32_paths = [site_packages / "win32", site_packages / "win32/lib"]
            dll_path = site_packages / "pywin32_system32"
            if not all(path.is_dir() for path in (*win32_paths, dll_path)):
                raise RuntimeError("Installed pywin32 directories are incomplete")
            native_paths.enter_context(os.add_dll_directory(str(dll_path.resolve())))
            sys.path[0:0] = [str(ROOT), str(site_packages), *(str(path) for path in win32_paths)]
            # SDK dotenv lookup is confined to the owned empty temporary cwd.
            # No .pth/bootstrap scripts, workers, event loops or tool calls.
            if has_shared_runtime:
                os.environ["ONEC_STANDARDS_STANDALONE"] = "1"
            server_module = importlib.import_module("server")
            assert not server_module.USE_SHARED_RUNTIME
            check_registration(server_module)
            assert "standards_index" not in sys.modules
            assert "shared_runtime" not in sys.modules
            local_index = server_module._get_index()
            index_module = sys.modules["standards_index"]
            assert type(local_index) is index_module.StandardsIndex
            assert server_module._get_index() is local_index
            assert local_index.standards_dir == temporary_root / "default-corpus"
            assert local_index.cache_dir == temporary_root / "default-cache"
            assert local_index._embedding_model is None and local_index._reranker_model is None
            if has_shared_runtime:
                del os.environ["ONEC_STANDARDS_STANDALONE"]
                server_module = importlib.reload(server_module)
                assert server_module.USE_SHARED_RUNTIME
                check_registration(server_module)
                runtime_module = sys.modules["shared_runtime"]
                assert Path(runtime_module.__file__).resolve() == runtime_root / "shared_runtime.py"
                proxy = server_module._get_index()
                assert proxy is server_module.index
                assert proxy.entrypoint == ROOT / "server.py"
                assert proxy.runtime_dir.is_relative_to(temporary_root)
                assert not proxy.runtime_dir.exists()
            sdk_module = sys.modules["mcp"]
            numpy_module = sys.modules["numpy"]
            assert Path(index_module.__file__).resolve() == ROOT / "standards_index.py"
            assert Path(server_module.__file__).resolve() == ROOT / "server.py"
            assert Path(sdk_module.__file__).resolve().is_relative_to(site_packages.resolve())
            assert Path(numpy_module.__file__).resolve().is_relative_to(site_packages.resolve())
            for name in ("pywintypes", "win32api", "win32con", "win32job"):
                assert Path(sys.modules[name].__file__).resolve().is_relative_to(site_packages.resolve())
            assert "pywin32_bootstrap" not in sys.modules
            assert not (temporary_root / ".env").exists()
            assert not list((temporary_root / "default-corpus").iterdir())
            assert not list((temporary_root / "default-cache").iterdir())
            assert not (temporary_root / "models").exists()
            assert fingerprints() == before
            assert not HEAVY_MODULES.intersection(sys.modules)
            assert not guard.violations
            print("PYTHON_VERSION=", sys.version.split()[0])
            print("MCP_SDK_VERSION=", importlib.metadata.version("mcp"))
            print("PYWIN32_VERSION=", importlib.metadata.version("pywin32"))
            print("PYWIN32_PATHS_EXPLICIT_NO_BOOTSTRAP=True")
            print("SDK_CWD_ISOLATED_FROM_LOCAL_DOTENV=True")
            print("NUMPY_VERSION=", numpy_module.__version__)
            print("ISOLATED_MODULE_PATHS_VERIFIED=True")
            print("INDEX_AND_SERVER_IMPORTS_VERIFIED=True")
            print("STANDALONE_IMPORT_VERIFIED=True")
            print("SHARED_IMPORT_VERIFIED=", has_shared_runtime)
            print("LOCAL_REGISTERED_TOOLS=", sorted(EXPECTED_TOOLS))
            print("WORKER_STATE_CREATED=False")
            print("MODELS_IMPORTED_OR_CREATED=False")
            print("CORE_SOURCE_BYTES_UNCHANGED=True")
            print("LIVE_MCP_PROTOCOL_TESTED=False")
            code = 0
    except Exception as exc:
        # Do not dump arbitrary dependency exception payloads or environment data.
        print("IMPORT_PROBE_FAILED=", type(exc).__name__)
        if isinstance(exc, ModuleNotFoundError) and isinstance(exc.name, str):
            if all(part.isidentifier() for part in exc.name.split(".")):
                print("MISSING_MODULE=", exc.name)
        code = 1
    if guard is not None:
        print("UNEXPECTED_GUARD_VIOLATIONS=", len(guard.violations))
        removed = not guard.temporary_root.exists()
        print("TEMP_FIXTURES_REMOVED=", removed)
        if guard.violations or not removed:
            code = 1
    print("EXIT_CODE=", code)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
