"""Static contracts only: never import/execute server or shared_runtime."""
from __future__ import annotations

import ast
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


SOURCE_ROOT = Path(__file__).resolve().parents[1]


class ServerContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(dir=os.environ["MCP_STANDARDS_TEST_ROOT"])
        self.addCleanup(self.temp.cleanup)
        self.corpus = Path(self.temp.name) / "corpus"
        self.cache = Path(self.temp.name) / "cache"
        self.corpus.mkdir()
        self.cache.mkdir()
        self.server = ast.parse((SOURCE_ROOT / "server.py").read_text(encoding="utf-8"))
        self.index = ast.parse((SOURCE_ROOT / "standards_index.py").read_text(encoding="utf-8"))

    def assignment(self, tree, name):
        matches = [node.value for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)]
        self.assertEqual(len(matches), 1, name)
        return matches[0]

    def functions(self):
        return {node.name: node for node in self.server.body if isinstance(node, ast.FunctionDef)}

    def assert_signature(self, function, names, annotations, defaults, returns):
        args = function.args
        self.assertEqual([arg.arg for arg in args.args], names)
        self.assertEqual([ast.unparse(arg.annotation) if arg.annotation else None for arg in args.args],
                         annotations)
        self.assertEqual([ast.literal_eval(value) for value in args.defaults], defaults)
        self.assertEqual(args.posonlyargs, [])
        self.assertEqual(args.kwonlyargs, [])
        self.assertIsNone(args.vararg)
        self.assertIsNone(args.kwarg)
        self.assertEqual(ast.unparse(function.returns), returns)

    def forwarded_call(self, function):
        # Only a docstring and a direct return: no execution or mocked live protocol.
        body = list(function.body)
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            self.assertIsInstance(body.pop(0).value.value, str)
        self.assertEqual(len(body), 1)
        self.assertIsInstance(body[0], ast.Return)
        call = body[0].value
        self.assertIsInstance(call, ast.Call)
        return call

    def test_exactly_three_tools_with_expected_names(self):
        tools = []
        for node in ast.walk(self.server):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Call) and ast.unparse(decorator.func) == "mcp.tool":
                    tools.append(node.name)
                    self.assertEqual(decorator.args, [])
                    self.assertEqual(decorator.keywords, [])
                    self.assertIsInstance(node, ast.FunctionDef)
        self.assertEqual(sorted(tools), ["list_standards", "read_standard", "search_standards"])

    def test_tool_signatures_and_defaults(self):
        functions = self.functions()
        self.assert_signature(functions["search_standards"],
                              ["query", "limit", "mode", "candidate_limit", "use_reranker"],
                              ["str", "int", "str", "int | None", "bool | None"],
                              [5, "normal", None, None], "dict[str, Any]")
        self.assert_signature(functions["read_standard"], ["path"], ["str"], [], "dict[str, str]")
        self.assert_signature(functions["list_standards"], [], [], [], "list[dict[str, Any]]")

    def test_search_forwards_every_argument_unchanged(self):
        call = self.forwarded_call(self.functions()["search_standards"])
        self.assertEqual(ast.unparse(call.func), "index.search")
        self.assertEqual(call.args, [])
        names = ["query", "limit", "mode", "candidate_limit", "use_reranker"]
        self.assertEqual([(keyword.arg, ast.unparse(keyword.value)) for keyword in call.keywords],
                         [(name, name) for name in names])

    def test_read_and_list_forward_unchanged(self):
        functions = self.functions()
        read = self.forwarded_call(functions["read_standard"])
        self.assertEqual(ast.unparse(read.func), "index.read_standard")
        self.assertEqual([ast.unparse(value) for value in read.args], ["path"])
        self.assertEqual(read.keywords, [])
        listing = self.forwarded_call(functions["list_standards"])
        self.assertEqual(ast.unparse(listing.func), "index.list_standards")
        self.assertEqual(listing.args, [])
        self.assertEqual(listing.keywords, [])

    def test_worker_spec_constants(self):
        spec = self.assignment(self.server, "SPEC")
        self.assertIsInstance(spec, ast.Call)
        self.assertEqual(ast.unparse(spec.func), "dict")
        self.assertEqual(spec.args, [])
        self.assertEqual([(keyword.arg, ast.literal_eval(keyword.value)) for keyword in spec.keywords], [
            ("name", "1c-standards"), ("module", "standards_index"),
            ("class_name", "StandardsIndex"),
            ("methods", ("search", "read_standard", "list_standards")),
            ("env_prefix", "ONEC_STANDARDS_"),
        ])

    def test_proxy_worker_and_stdio_wiring_is_static(self):
        proxy = self.assignment(self.server, "index")
        self.assertEqual(ast.unparse(proxy), "SharedIndexProxy(__file__, **SPEC)")
        worker_calls = [node.value for node in self.server.body if isinstance(node, ast.Expr)
                        and isinstance(node.value, ast.Call)
                        and ast.unparse(node.value.func) == "run_worker_if_requested"]
        self.assertEqual([ast.unparse(node) for node in worker_calls],
                         ["run_worker_if_requested(__file__, **SPEC)"])
        main_guards = [node for node in self.server.body if isinstance(node, ast.If)
                       and ast.unparse(node.test) == "__name__ == '__main__'"]
        self.assertEqual(len(main_guards), 1)
        self.assertEqual([ast.unparse(node) for node in main_guards[0].body],
                         ["mcp.run(transport='stdio')"])
        mcp = self.assignment(self.server, "mcp")
        self.assertEqual(ast.unparse(mcp.func), "FastMCP")
        kwargs = {keyword.arg: ast.literal_eval(keyword.value) for keyword in mcp.keywords}
        self.assertEqual(kwargs["name"], "mcp-1c-standards")
        self.assertIsInstance(kwargs["instructions"], str)

    def test_search_mode_schema_and_model_constants(self):
        self.assertEqual(ast.literal_eval(self.assignment(self.index, "SEARCH_MODES")), {
            "fast": {"candidate_limit": 12, "use_reranker": False},
            "normal": {"candidate_limit": 12, "use_reranker": True},
            "deep": {"candidate_limit": 30, "use_reranker": True},
        })
        self.assertEqual(ast.literal_eval(self.assignment(self.index, "DEFAULT_EMBEDDING_MODEL")),
                         "Qwen/Qwen3-Embedding-0.6B")
        self.assertEqual(ast.literal_eval(self.assignment(self.index, "DEFAULT_RERANKER_MODEL")),
                         "Qwen/Qwen3-Reranker-0.6B")
        instruction = ast.literal_eval(self.assignment(self.index, "DEFAULT_QUERY_INSTRUCTION"))
        self.assertIsInstance(instruction, str)
        self.assertIn("1C:Enterprise", instruction)

    def test_index_public_signatures_match_forwarding_contract(self):
        classes = {node.name: node for node in self.index.body if isinstance(node, ast.ClassDef)}
        methods = {node.name: node for node in classes["StandardsIndex"].body
                   if isinstance(node, ast.FunctionDef)}
        self.assert_signature(methods["search"],
                              ["self", "query", "limit", "mode", "candidate_limit", "use_reranker"],
                              [None, "str", "int", "str", "int | None", "bool | None"],
                              [5, "normal", None, None], "dict[str, Any]")
        self.assert_signature(methods["read_standard"], ["self", "standard_path"],
                              [None, "str"], [], "dict[str, str]")
        self.assert_signature(methods["list_standards"], ["self"], [None], [], "list[dict[str, Any]]")
        chunks = [node for node in classes["StandardChunk"].body if isinstance(node, ast.AnnAssign)]
        self.assertEqual([(ast.unparse(node.target), ast.unparse(node.annotation)) for node in chunks],
                         [(name, "str") for name in ("file_name", "path", "title", "section", "text")])
