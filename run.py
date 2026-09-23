"""TraceFlow: calculate reports and start the local analyst workspace.

    python run.py
    python run.py --analysis-only
    python run.py --ui-only --headless --port 8502

All relative data/output paths are resolved from this repository, including when
the command is called from another working directory. No packages are installed
and no work is performed when this module is imported.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def repository_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TraceFlow: аналитика и локальный интерфейс одной командой.")
    parser.add_argument("--data", type=repository_path, default=ROOT / "data",
                        help="Папка с nodes.parquet, edges.parquet, transactions.parquet (по умолчанию data)")
    parser.add_argument("--output-dir", type=repository_path, default=ROOT / "output",
                        help="Папка для готовых CSV (по умолчанию output)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--analysis-only", action="store_true", help="Рассчитать CSV и завершиться без интерфейса")
    mode.add_argument("--ui-only", action="store_true", help="Открыть интерфейс с готовыми CSV без пересчёта")
    parser.add_argument("--headless", action="store_true", help="Не открывать браузер автоматически")
    parser.add_argument("--port", type=int, default=8501, help="Локальный порт интерфейса (по умолчанию 8501)")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port должен находиться в диапазоне 1–65535")

    environment = os.environ.copy()
    environment["TRACEFLOW_DATA_DIR"] = str(args.data)
    environment["TRACEFLOW_RESULTS_DIR"] = str(args.output_dir)
    try:
        if not args.ui_only:
            print(f"Расчёт аналитики: {args.data} → {args.output_dir}", flush=True)
            result = subprocess.run([
                sys.executable, str(ROOT / "src" / "pipeline.py"),
                "--data", str(args.data), "--output-dir", str(args.output_dir),
            ], cwd=ROOT, env=environment, check=False)
            if result.returncode:
                print("Аналитика завершилась с ошибкой. Интерфейс не запущен.", file=sys.stderr)
                return result.returncode
        if args.analysis_only:
            return 0

        print(f"Интерфейс: http://127.0.0.1:{args.port} · остановка: Ctrl+C", flush=True)
        result = subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(ROOT / "ui" / "app.py"),
            "--server.address=127.0.0.1", f"--server.port={args.port}",
            f"--server.headless={'true' if args.headless else 'false'}",
            "--browser.gatherUsageStats=false",
        ], cwd=ROOT, env=environment, check=False)
        return result.returncode
    except KeyboardInterrupt:
        return 130
    except OSError as exc:
        print(f"Не удалось запустить TraceFlow: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
