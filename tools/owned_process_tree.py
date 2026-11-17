"""Identity-checked monitoring of descendants of one owned Popen.

No command-name matching, no system-wide process killing. This is observational
tree tracking, not atomic Job/cgroup containment: see the integration notes.
"""

import os
import signal
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    creation_token: int


class LinuxProcesses:
    """Linux /proc identity and resident HWM; pidfd is required for termination."""

    @staticmethod
    def _stat(pid):
        try:
            raw = Path(f"/proc/{pid}/stat").read_text()
            fields = raw[raw.rfind(")") + 2 :].split()
            return dict(pid=pid, ppid=int(fields[1]), state=fields[0], creation=int(fields[19]))
        except (FileNotFoundError, ProcessLookupError, ValueError, IndexError):
            return None

    def identity_from_popen(self, process):
        # An unreaped child PID cannot be recycled between Popen and this read.
        if process.returncode is not None:
            return None
        return self.identity(process.pid)

    def identity(self, pid):
        try:
            row = self._stat(pid)
        except OSError:
            return None
        return None if row is None else ProcessIdentity(pid, row["creation"])

    def parents(self):
        rows = {}
        for path in Path("/proc").iterdir():
            if path.name.isdigit():
                try:
                    row = self._stat(int(path.name))
                except OSError:
                    continue
                if row is not None:
                    rows[row["pid"]] = row["ppid"]
        return rows

    def alive(self, identity):
        try:
            row = self._stat(identity.pid)
        except OSError:
            return None
        return (
            row is not None
            and row["creation"] == identity.creation_token
            and row["state"] not in ("Z", "X")
        )

    def memory(self, identity):
        if self.identity(identity.pid) != identity:
            return None
        try:
            rows = {
                line.split(":", 1)[0]: line.split(":", 1)[1].strip()
                for line in Path(f"/proc/{identity.pid}/status").read_text().splitlines()
                if ":" in line
            }
            current = int(rows["VmRSS"].split()[0]) * 1024
            peak = int(rows["VmHWM"].split()[0]) * 1024
        except (OSError, KeyError, ValueError):
            return None
        return (current, peak) if self.identity(identity.pid) == identity else None

    def terminate(self, identity):
        if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
            return "safe_termination_unavailable: pidfd required; no PID-only fallback"
        try:
            descriptor = os.pidfd_open(identity.pid, 0)
        except ProcessLookupError:
            return "gone"
        except OSError as exc:
            return "open_failed: " + str(exc)
        try:
            # Verify *after* opening: the fd is the actual signal target and
            # cannot accidentally follow a subsequent PID reuse.
            if self.identity(identity.pid) != identity:
                return "identity_changed: no signal sent"
            signal.pidfd_send_signal(descriptor, signal.SIGKILL, None, 0)
            return "termination_requested"
        except ProcessLookupError:
            return "gone"
        except OSError as exc:
            return "signal_failed: " + str(exc)
        finally:
            os.close(descriptor)


class WindowsProcesses:
    """Toolhelp parent links; same-handle creation checks for memory/terminate."""

    def __init__(self):
        import ctypes
        from ctypes import wintypes

        self.ctypes = ctypes
        self.wintypes = wintypes
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.psapi = ctypes.WinDLL("psapi", use_last_error=True)

        class Entry(ctypes.Structure):
            _fields_ = [
                ("dwSize", wintypes.DWORD),
                ("cntUsage", wintypes.DWORD),
                ("th32ProcessID", wintypes.DWORD),
                ("th32DefaultHeapID", ctypes.c_size_t),
                ("th32ModuleID", wintypes.DWORD),
                ("cntThreads", wintypes.DWORD),
                ("th32ParentProcessID", wintypes.DWORD),
                ("pcPriClassBase", wintypes.LONG),
                ("dwFlags", wintypes.DWORD),
                ("szExeFile", wintypes.WCHAR * 260),
            ]

        class Memory(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("page_faults", wintypes.DWORD)] + [
                (name, ctypes.c_size_t)
                for name in (
                    "peak_working_set",
                    "working_set",
                    "peak_paged_pool",
                    "paged_pool",
                    "peak_nonpaged_pool",
                    "nonpaged_pool",
                    "pagefile",
                    "peak_pagefile",
                )
            ]

        self.Entry = Entry
        self.Memory = Memory
        kernel = self.kernel
        kernel.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
        kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        for name in ("Process32FirstW", "Process32NextW"):
            function = getattr(kernel, name)
            function.argtypes = [wintypes.HANDLE, ctypes.POINTER(Entry)]
            function.restype = wintypes.BOOL
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        kernel.GetProcessId.argtypes = [wintypes.HANDLE]
        kernel.GetProcessId.restype = wintypes.DWORD
        kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [
            ctypes.POINTER(wintypes.FILETIME)
        ] * 4
        kernel.GetProcessTimes.restype = wintypes.BOOL
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.GetExitCodeProcess.restype = wintypes.BOOL
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        kernel.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel.TerminateProcess.restype = wintypes.BOOL
        self.psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(Memory),
            wintypes.DWORD,
        ]
        self.psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

    def _creation(self, handle):
        values = [self.wintypes.FILETIME() for _ in range(4)]
        if not self.kernel.GetProcessTimes(handle, *[self.ctypes.byref(value) for value in values]):
            return None
        return (int(values[0].dwHighDateTime) << 32) | int(values[0].dwLowDateTime)

    def _open(self, pid, access=0x1000):
        return self.kernel.OpenProcess(access, False, pid)

    def _identity_handle(self, handle, pid):
        creation = self._creation(handle)
        return (
            None
            if creation is None or self.kernel.GetProcessId(handle) != pid
            else ProcessIdentity(pid, creation)
        )

    def identity_from_popen(self, process):
        # Popen owns this kernel process handle even if its PID later vanishes.
        handle = getattr(process, "_handle", None)
        return None if handle is None else self._identity_handle(int(handle), process.pid)

    def identity(self, pid):
        handle = self._open(pid)
        if not handle:
            return None
        try:
            return self._identity_handle(handle, pid)
        finally:
            self.kernel.CloseHandle(handle)

    def parents(self):
        snapshot = self.kernel.CreateToolhelp32Snapshot(0x00000002, 0)
        if snapshot in (None, self.ctypes.c_void_p(-1).value):
            raise OSError(self.ctypes.get_last_error(), "Cannot snapshot process parent links.")
        try:
            entry = self.Entry()
            entry.dwSize = self.ctypes.sizeof(entry)
            rows = {}
            present = self.kernel.Process32FirstW(snapshot, self.ctypes.byref(entry))
            while present:
                rows[int(entry.th32ProcessID)] = int(entry.th32ParentProcessID)
                present = self.kernel.Process32NextW(snapshot, self.ctypes.byref(entry))
            if self.ctypes.get_last_error() not in (0, 18):
                raise OSError(
                    self.ctypes.get_last_error(), "Process parent snapshot was incomplete."
                )
            return rows
        finally:
            self.kernel.CloseHandle(snapshot)

    def alive(self, identity):
        handle = self._open(identity.pid, 0x1000 | 0x00100000)
        if not handle:
            return False if self.ctypes.get_last_error() == 87 else None
        try:
            current = self._identity_handle(handle, identity.pid)
            if current != identity:
                return False if current is not None else None
            state = self.kernel.WaitForSingleObject(handle, 0)
            return True if state == 258 else False if state == 0 else None
        finally:
            self.kernel.CloseHandle(handle)

    def memory(self, identity):
        handle = self._open(identity.pid, 0x0400 | 0x0010)
        if not handle:
            return None
        try:
            if self._identity_handle(handle, identity.pid) != identity:
                return None
            counter = self.Memory()
            counter.cb = self.ctypes.sizeof(counter)
            if not self.psapi.GetProcessMemoryInfo(handle, self.ctypes.byref(counter), counter.cb):
                return None
            return int(counter.working_set), int(counter.peak_working_set)
        finally:
            self.kernel.CloseHandle(handle)

    def terminate(self, identity):
        handle = self._open(identity.pid, 0x1000 | 0x0001)
        if not handle:
            return "gone_or_access_denied: no termination attempted"
        try:
            if self._identity_handle(handle, identity.pid) != identity:
                return "identity_changed: no termination attempted"
            # The checked handle is the termination target; never re-open by PID.
            if self.kernel.TerminateProcess(handle, 1):
                return "termination_requested"
            return "termination_failed: " + str(self.ctypes.get_last_error())
        finally:
            self.kernel.CloseHandle(handle)


class OwnedProcessTree:
    """Track only creation-verified descendants of one caller-owned Popen.

    A poll is not atomic containment. Descendants that spawn/exit entirely
    between observations may be unobserved, which must remain a stated limit.
    """

    def __init__(self, process, *, backend=None):
        if backend is None and not isinstance(process, subprocess.Popen):
            raise TypeError(
                "The monitoring root must be this caller-owned Popen, not an arbitrary PID/name."
            )
        self.process = process
        self.backend = backend or (
            WindowsProcesses()
            if sys.platform == "win32"
            else LinuxProcesses()
            if sys.platform.startswith("linux")
            else None
        )
        if self.backend is None:
            raise NotImplementedError(
                "Identity-checked tree monitoring is supported on Windows/Linux only."
            )
        self.root = self.backend.identity_from_popen(process)
        if self.root is None:
            raise RuntimeError(
                "Cannot establish the owned root creation identity; do not claim tree enforcement."
            )
        self.known = {self.root.pid: (self.root, None)}
        self._peak = 0

    def sample(self):
        issues = []
        try:
            parents = self.backend.parents()
        except OSError as exc:
            parents = {}
            issues.append("parent_snapshot_failed: " + str(exc))
        changed = True
        while changed:
            changed = False
            for pid, parent_pid in parents.items():
                if pid in self.known:
                    current = self.backend.identity(pid)
                    if current is not None and current != self.known[pid][0]:
                        issues.append(f"pid_reused_not_adopted:{pid}")
                    continue
                if parent_pid not in self.known:
                    continue
                parent = self.known[parent_pid][0]
                if (
                    self.backend.identity(parent.pid) != parent
                    or self.backend.alive(parent) is not True
                ):
                    continue
                identity = self.backend.identity(pid)
                if identity is None:
                    issues.append(f"descendant_identity_unavailable:{pid}")
                    continue
                if identity.creation_token < parent.creation_token:
                    # Stale parent PID from a prior incarnation is not lineage.
                    continue
                if (
                    self.backend.identity(parent.pid) != parent
                    or self.backend.alive(parent) is not True
                ):
                    continue
                self.known[pid] = (identity, parent)
                changed = True
        members = []
        currents = []
        for identity, parent in self.known.values():
            alive = self.backend.alive(identity)
            memory = self.backend.memory(identity) if alive is True else None
            if memory is not None:
                currents.append(memory[0])
                self._peak = max(self._peak, *memory)
            elif alive is not False:
                issues.append(f"live_member_memory_unavailable:{identity.pid}")
            members.append(
                dict(
                    identity=asdict(identity),
                    proven_parent=None if parent is None else asdict(parent),
                    alive=alive,
                    current_resident_bytes=None if memory is None else memory[0],
                    peak_resident_bytes=None if memory is None else memory[1],
                )
            )
        return dict(
            root=asdict(self.root),
            members=members,
            live_member_count=sum(row["alive"] is True for row in members),
            uncertain_member_count=sum(row["alive"] is None for row in members),
            maximum_process_current_resident_bytes=max(currents) if currents else None,
            maximum_process_lifetime_peak_resident_bytes=self._peak or None,
            aggregate_tree_current_resident_bytes_diagnostic=sum(currents) if currents else None,
            observed_memory_complete=not issues,
            issues=issues,
            atomic_containment=False,
            metric=(
                "Maximum of identity-verified per-process resident HWM; tree sum is separate"
                " diagnostic, not replacement for worker peak."
            ),
        )

    def terminate_tracked(self):
        """Leaf-first requests only for reverified observed owned identities."""
        self.sample()
        actions = []

        def depth(record):
            identity, parent = record
            count = 0
            while parent is not None:
                count += 1
                parent = self.known.get(parent.pid, (None, None))[1]
            return count

        for identity, parent in sorted(self.known.values(), key=depth, reverse=True):
            if self.backend.identity(identity.pid) != identity:
                status = "gone_or_identity_changed: untouched"
            elif self.backend.alive(identity) is False:
                status = "gone"
            elif self.backend.alive(identity) is None:
                status = "liveness_unverified: untouched"
            else:
                status = self.backend.terminate(identity)
            actions.append(dict(identity=asdict(identity), status=status))
        remaining = [
            asdict(identity)
            for identity, parent in self.known.values()
            if self.backend.alive(identity) is not False
        ]
        return dict(
            actions=actions,
            remaining_tracked=remaining,
            all_tracked_exited=not remaining,
            whole_tree_containment_proven=False,
            scope=(
                "Only previously proven owned identities; re-poll to observe asynchronous"
                " termination. Unobserved spawn/exit races are not claimed contained."
            ),
        )
