#!/usr/bin/env python
"""E50: holosoma eval_agent with the repair recorder swapped in (no clone edits).

Applies snmr.integration.wbt_repair.patch() (module-attribute swap; holosoma's get_class
resolves the recording callback ``_target_`` via getattr, so the stock
``--recording.config.enabled`` CLI flags construct RepairRecordingCallback), then defers to
holosoma's eval main. Run inside .venv-wbt with PYTHONPATH including the snmr repo root.
"""

from snmr.integration import wbt_repair

wbt_repair.patch()

from holosoma import eval_agent  # noqa: E402

if __name__ == "__main__":
    eval_agent.main()
