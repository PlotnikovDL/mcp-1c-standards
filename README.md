# MCP 1C Standards

Минимальный Python MCP-сервер для качественного поиска по стандартам разработки 1С.

## Источник данных

Сервер читает только markdown-стандарты:

```text
C:\projects\1c-tools\1c-docs\standards
```

## Что сервер делает

- `search_standards` - ищет стандарты по смыслу запроса через Qwen3 Embedding и, кроме режима `fast`, пересортировывает кандидатов через Qwen3 Reranker.
- `read_standard` - читает конкретный `.md` стандарт.
- `list_standards` - показывает список доступных стандартов.

Автообучения, feedback-базы, синтакс-помощника и EDT-контура здесь нет.

## Режимы поиска

`search_standards` принимает параметр `mode`:

| mode | Что делает | Когда использовать |
|---|---|---|
| `fast` | Semantic search top-12 без reranker | Быстрый первичный поиск. |
| `normal` | Semantic top-12 + Qwen3 Reranker | Режим по умолчанию для разработки. |
| `deep` | Semantic top-30 + Qwen3 Reranker | Сложные задачи и внимательное ревью. |

Параметры `candidate_limit` и `use_reranker` можно передать вручную, если нужно временно переопределить режим.

## Кэш

Embeddings стандартов сохраняются в технический кэш:

```text
C:\projects\1c-tools\mcp-1c-standards\.cache
```

В кэше лежат:

- `embeddings.npy` - векторы фрагментов стандартов.
- `manifest.json` - список файлов, размеры, даты изменения и модель.

Кэш автоматически пересобирается, если изменились стандарты, модель или инструкция поиска. Его можно удалить без потери данных: сервер пересчитает embeddings заново из `standards`.

## Модели

По умолчанию используются:

```text
Qwen/Qwen3-Embedding-0.6B
Qwen/Qwen3-Reranker-0.6B
```

Первый реальный поиск скачает веса моделей в стандартный cache Hugging Face.

## Переменные окружения

Путь к стандартам:

```powershell
$env:ONEC_STANDARDS_DIR = "C:\projects\1c-tools\1c-docs\standards"
```

Путь к кэшу:

```powershell
$env:ONEC_STANDARDS_CACHE_DIR = "C:\projects\1c-tools\mcp-1c-standards\.cache"
```

Embedding-модель:

```powershell
$env:ONEC_STANDARDS_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
```

Reranker-модель:

```powershell
$env:ONEC_STANDARDS_RERANKER_MODEL = "Qwen/Qwen3-Reranker-0.6B"
```

Инструкция для поиска:

```powershell
$env:ONEC_STANDARDS_QUERY_INSTRUCTION = "Given a Russian 1C:Enterprise development task or code review question, retrieve relevant passages from 1C development standards that help solve or review the task."
```

## Запуск

```powershell
C:\projects\1c-tools\mcp-1c-standards\.venv\Scripts\python.exe C:\projects\1c-tools\mcp-1c-standards\server.py
```
