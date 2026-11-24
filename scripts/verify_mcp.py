"""Проверяет локальную установку через MCP по stdio."""

import argparse
import asyncio
import math
import os
import sys
from datetime import timedelta
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ["list_standards", "read_standard", "search_standards"]


async def call(session, name, arguments, timeout=None):
    result = await session.call_tool(name, arguments, read_timeout_seconds=timeout)
    if result.isError:
        message = " ".join(block.text for block in result.content if block.type == "text")
        raise RuntimeError(f"{name} вернул ошибку MCP. {message}")
    if result.structuredContent is None:
        raise ValueError(f"{name} вернул ответ без структурированных данных")
    return result.structuredContent


def finite_score(value):
    return type(value) in (int, float) and math.isfinite(value)


async def verify(args):
    server = StdioServerParameters(
        command=sys.executable, args=[str(ROOT / "server.py")], env=dict(os.environ)
    )
    corpus = Path(os.environ.get("ONEC_STANDARDS_DIR", ROOT / "corpus" / "standards")).resolve()
    async with stdio_client(server) as (reader, writer):
        async with ClientSession(reader, writer, read_timeout_seconds=timedelta(seconds=90)) as session:
            info = await session.initialize()
            if info.serverInfo.name != "mcp-1c-standards":
                raise ValueError(f"Неожиданное имя сервера {info.serverInfo.name!r}")
            tools = await session.list_tools()
            if sorted(tool.name for tool in tools.tools) != TOOLS or tools.nextCursor is not None:
                raise ValueError(f"Ожидались три инструмента {', '.join(TOOLS)}")
            print(f"initialize OK ({info.protocolVersion}), tools/list OK")

            rows = (await call(session, "list_standards", {})).get("result")
            if not isinstance(rows, list) or not rows:
                raise ValueError("list_standards вернул пустой или некорректный каталог")
            titles = {}
            for row in rows:
                name, title = row["file"], row["title"]
                if Path(name).name != name or Path(name).suffix.lower() != ".md" or name in titles:
                    raise ValueError("Некорректное или повторяющееся имя файла в каталоге")
                if not isinstance(title, str) or not title.strip():
                    raise ValueError(f"Отсутствует заголовок {name}")
                titles[name] = title

            name = rows[0]["file"]
            source = (corpus / name).resolve()
            if not source.is_relative_to(corpus):
                raise ValueError(f"Файл {name} находится за пределами каталога стандартов")
            doc = await call(session, "read_standard", {"path": name})
            if doc.get("file") != source.name or doc.get("title") != titles[name]:
                raise ValueError("Имя документа или заголовок в ответе read_standard отличается от файла на диске")
            content = source.read_text(encoding="utf-8-sig", errors="replace")
            if not content.strip() or doc.get("content") != content:
                raise ValueError(f"read_standard вернул неверное содержимое {name}")
            print(f"list_standards OK ({len(rows)}), read_standard OK ({name})")
            if args.query is None:
                return

            response = await call(
                session,
                "search_standards",
                {"query": args.query, "mode": args.mode, "limit": 5},
                timeout=timedelta(hours=1),
            )
            hits = response.get("results")
            if response.get("query") != args.query or response.get("mode") != args.mode:
                raise ValueError("Поиск вернул ответ для другого запроса или режима")
            if not isinstance(hits, list) or not 1 <= len(hits) <= 5:
                raise ValueError("Поиск должен вернуть от одного до пяти результатов")
            names = []
            for hit in hits:
                name = hit["file"]
                if name not in titles or hit.get("title") != titles[name] or not hit["excerpt"].strip():
                    raise ValueError("Поиск вернул неизвестный файл, неверный заголовок или пустой фрагмент")
                if not finite_score(hit.get("embedding_score")):
                    raise ValueError(f"Отсутствует корректная оценка embedding для {name}")
                if args.mode != "fast" and not finite_score(hit.get("rerank_score")):
                    raise ValueError(f"Отсутствует корректная оценка reranker для {name}")
                names.append(name)
            missing = set(args.expect_file) - set(names)
            if missing:
                raise ValueError(f"В выдаче {', '.join(names)} отсутствуют {', '.join(sorted(missing))}")
            print(f"search_standards OK ({args.mode}): {', '.join(names)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", type=str.strip, help="запрос для дополнительной проверки реального поиска")
    parser.add_argument("--mode", choices=("fast", "normal", "deep"), default="normal")
    parser.add_argument(
        "--expect-file", action="append", default=[],
        help="ожидаемое имя файла в первых пяти результатах, можно повторять",
    )
    options = parser.parse_args()
    if options.query == "" or (options.expect_file and options.query is None):
        parser.error("для поиска укажите непустой --query")
    asyncio.run(verify(options))
