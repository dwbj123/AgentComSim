from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_workbench_starts_without_exceptions():
    app = Path(__file__).parents[1] / "agentcomsim" / "web" / "app.py"
    result = AppTest.from_file(str(app), default_timeout=30).run()
    assert not result.exception
    assert len(result.tabs) == 4
    assert result.sidebar.button[0].label == "运行对照实验"
