"""The public entry point must coordinate both stages without hidden side effects."""
from contextlib import redirect_stderr, redirect_stdout
import importlib
import io
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run


class LauncherTests(unittest.TestCase):
    def call_main(self, args, results):
        with patch("run.subprocess.run", side_effect=results) as child:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                code = run.main(args)
        return code, child.call_args_list

    def test_custom_paths_reach_both_stages_using_current_python(self):
        code, calls = self.call_main(
            ["--data", "custom data", "--output-dir", "custom output", "--headless", "--port", "8502"],
            [subprocess.CompletedProcess([], 0), subprocess.CompletedProcess([], 0)],
        )
        self.assertEqual(code, 0)
        self.assertEqual(len(calls), 2)
        pipeline_command, ui_command = calls[0].args[0], calls[1].args[0]
        self.assertEqual(pipeline_command[0], sys.executable)
        self.assertEqual(ui_command[:3], [sys.executable, "-m", "streamlit"])
        self.assertEqual(pipeline_command[pipeline_command.index("--data") + 1], str(ROOT / "custom data"))
        self.assertEqual(pipeline_command[pipeline_command.index("--output-dir") + 1], str(ROOT / "custom output"))
        self.assertIn("--server.port=8502", ui_command)
        self.assertIn("--server.headless=true", ui_command)
        self.assertIn("--server.address=127.0.0.1", ui_command)
        for call in calls:
            self.assertEqual(call.kwargs["cwd"], ROOT)
            self.assertEqual(call.kwargs["env"]["TRACEFLOW_DATA_DIR"], str(ROOT / "custom data"))
            self.assertEqual(call.kwargs["env"]["TRACEFLOW_RESULTS_DIR"], str(ROOT / "custom output"))

    def test_failed_analysis_does_not_start_ui_with_stale_reports(self):
        code, calls = self.call_main([], [subprocess.CompletedProcess([], 2)])
        self.assertEqual(code, 2)
        self.assertEqual(len(calls), 1)

    def test_analysis_only_finishes_without_starting_server(self):
        code, calls = self.call_main(["--analysis-only"], [subprocess.CompletedProcess([], 0)])
        self.assertEqual(code, 0)
        self.assertEqual(len(calls), 1)
        self.assertIn(str(ROOT / "src" / "pipeline.py"), calls[0].args[0])

    def test_ui_only_leaves_exports_untouched_and_propagates_server_exit(self):
        code, calls = self.call_main(["--ui-only"], [subprocess.CompletedProcess([], 7)])
        self.assertEqual(code, 7)
        self.assertEqual(len(calls), 1)
        self.assertIn("streamlit", calls[0].args[0])

    def test_invalid_port_and_conflicting_modes_start_nothing(self):
        for args in (["--port", "0"], ["--port", "65536"], ["--analysis-only", "--ui-only"]):
            with self.subTest(args=args), patch("run.subprocess.run") as child:
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                    run.main(args)
                self.assertEqual(caught.exception.code, 2)
                child.assert_not_called()

    def test_import_does_not_start_analysis_or_server(self):
        with patch("subprocess.run") as child:
            importlib.reload(run)
            child.assert_not_called()


if __name__ == "__main__":
    unittest.main()
