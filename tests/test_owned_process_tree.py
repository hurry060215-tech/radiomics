"""Small owned-process safety tests, not algorithm/performance measurements."""

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

path = Path(__file__).resolve().parents[1] / "tools/owned_process_tree.py"
spec = importlib.util.spec_from_file_location("owned_process_tree", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
Identity = module.ProcessIdentity


class FakeProcesses:
    def __init__(self):
        self.rows = {
            10: (1, 100, True, (5, 8)),
            11: (10, 101, True, (100, 150)),
            12: (11, 102, True, (50, 70)),
            99: (1, 90, True, (1, 1)),
        }
        self.killed = []

    def identity_from_popen(self, process):
        return self.identity(process.pid)

    def identity(self, pid):
        return Identity(pid, self.rows[pid][1]) if pid in self.rows else None

    def parents(self):
        return {pid: row[0] for pid, row in self.rows.items()}

    def alive(self, identity):
        return self.rows[identity.pid][2] if self.identity(identity.pid) == identity else False

    def memory(self, identity):
        return self.rows[identity.pid][3] if self.identity(identity.pid) == identity else None

    def terminate(self, identity):
        assert self.identity(identity.pid) == identity
        self.killed.append(identity.pid)
        p, c, a, m = self.rows[identity.pid]
        self.rows[identity.pid] = (p, c, False, m)
        return "termination_requested"


def test_descendant_memory_is_max_per_process_and_cleanup_is_leaf_first_owned_only():
    backend = FakeProcesses()
    tree = module.OwnedProcessTree(SimpleNamespace(pid=10), backend=backend)
    sample = tree.sample()
    assert {r["identity"]["pid"] for r in sample["members"]} == {10, 11, 12}
    assert sample["maximum_process_lifetime_peak_resident_bytes"] == 150
    assert sample["aggregate_tree_current_resident_bytes_diagnostic"] == 155
    cleanup = tree.terminate_tracked()
    assert backend.killed == [12, 11, 10]
    assert cleanup["all_tracked_exited"] and not cleanup["whole_tree_containment_proven"]
    assert backend.rows[99][2] is True


def test_stale_parent_pid_and_reused_child_pid_are_not_adopted_or_terminated():
    backend = FakeProcesses()
    backend.rows[13] = (10, 50, True, (999, 999))
    tree = module.OwnedProcessTree(SimpleNamespace(pid=10), backend=backend)
    tree.sample()
    assert 13 not in tree.known
    backend.rows[11] = (1, 999, True, (9999, 9999))
    sample = tree.sample()
    assert "pid_reused_not_adopted:11" in sample["issues"]
    tree.terminate_tracked()
    assert 11 not in backend.killed and 13 not in backend.killed and 99 not in backend.killed


def test_unknown_liveness_or_missing_member_memory_does_not_claim_enforcement_complete():
    backend = FakeProcesses()
    backend.rows[12] = (11, 102, None, None)
    tree = module.OwnedProcessTree(SimpleNamespace(pid=10), backend=backend)
    sample = tree.sample()
    assert not sample["observed_memory_complete"] and sample["uncertain_member_count"] == 1
    assert not tree.terminate_tracked()["all_tracked_exited"]
    assert 12 not in backend.killed


@pytest.mark.skipif(
    not (sys.platform == "win32" or sys.platform.startswith("linux")), reason="Windows/Linux helper"
)
def test_real_owned_launcher_child_memory_and_cleanup_leave_unrelated_process_alive(tmp_path):
    if sys.platform.startswith("linux") and (
        not hasattr(os, "pidfd_open") or not hasattr(__import__("signal"), "pidfd_send_signal")
    ):
        pytest.skip("Safe pidfd termination unavailable; helper refuses PID-only fallback.")
    marker = tmp_path / "worker.json"
    child = (
        "import os,time,pathlib,json; payload=bytearray(24*1024*1024);pathlib.Path("
        + repr(str(marker))
        + ").write_text(json.dumps({'pid':os.getpid()}));time.sleep(60)"
    )
    parent = (
        "import subprocess,sys,time; child=subprocess.Popen([sys.executable,'-c',"
        + repr(child)
        + "]);time.sleep(60)"
    )
    base = getattr(sys, "_base_executable", sys.executable)
    sentinel = subprocess.Popen(
        [base, "-c", "import time;time.sleep(60)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    process = subprocess.Popen(
        [sys.executable, "-c", parent], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    tree = module.OwnedProcessTree(process)
    sentinel_tree = module.OwnedProcessTree(sentinel)
    try:
        deadline = time.monotonic() + 10
        actual = None
        sample = None
        while time.monotonic() < deadline:
            sample = tree.sample()
            if marker.exists():
                actual = json.loads(marker.read_text())["pid"]
                if actual in tree.known:
                    break
            time.sleep(0.02)
        assert actual is not None and actual in tree.known
        sample = tree.sample()
        assert sample["maximum_process_lifetime_peak_resident_bytes"] >= 24 * 1024 * 1024
        assert sentinel.pid not in tree.known
        tree.terminate_tracked()
        deadline = time.monotonic() + 5
        while tree.sample()["live_member_count"] and time.monotonic() < deadline:
            time.sleep(0.02)
        assert tree.sample()["live_member_count"] == 0
        assert sentinel.poll() is None
        process.wait(timeout=5)
    finally:
        tree.terminate_tracked()
        sentinel_tree.sample()
        sentinel_tree.terminate_tracked()
        process.wait(timeout=5)
        sentinel.wait(timeout=5)
