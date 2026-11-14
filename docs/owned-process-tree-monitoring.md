# Owned process-tree monitor: future runner integration

This helper is a targeted safety/observability improvement. It is not connected
to any previously measured or currently running benchmark and does not change
the fixed snapshots, data, thresholds or numerical results.

## Why the existing parent sampling was incomplete

On Windows a venv `python.exe` can be a launcher that starts a base-Python child.
The original benchmark parent polled `Popen.pid`, so its periodic RSS sample and
PID-only timeout/budget termination could cover the launcher rather than the
actual numerical worker. A child could survive killing only that launcher.

The recorded **worker post-transport footer** uses the actual worker's own PID
and kernel resident high-water mark. Those peaks remain valid evidence for the
completed runs, whose observed worker peaks stayed well below2GiB. Existing raw
records and statistical results are retained unchanged. They do **not** prove
that the parent could immediately stop the numerical worker if a budget were
crossed, or completely clean up descendants after a timeout. This limitation
must accompany the original Radiomics/other affected report, rather than
claiming that an untested guard succeeded because no over-budget run occurred.

Suggested precise original-report qualification:

> The actual-worker post-transport kernel peak remains the memory observation.
> On Windows venv launchers, parent periodic sampling was limited to the launcher
> PID, so immediate budget enforcement and descendant timeout cleanup were not
> demonstrated. All recorded worker peaks remained below the fixed budget; no
> original sample, input, threshold or result was changed after this discovery.

## Helper contract

`tools/owned_process_tree.py` takes one caller-owned `Popen`, not a process name
or arbitrary PID. Windows anchors its identity using the Popen kernel handle,
discovers parent links with Toolhelp and checks process creation FILETIME.
Memory/termination uses an opened handle with the same creation identity; the
checked handle itself is the termination target, preventing PID-reuse redirection.
Linux uses `/proc` parent/starttime identity and pidfd-based signal delivery;
without safe pidfd support it refuses a PID-only termination fallback.

Only descendants whose live parent identity and creation ordering were verified
are adopted. Previously tracked identity mismatch/PID reuse is untouched and
reported. Termination requests are leaf-first and only for reverified tracked
identities; unknown liveness/access does not become a false cleanup success.
No command-line matching, wildcard Python killing, executable-name filtering or
system-wide termination is performed.

The primary memory quantity stays **maximum identity-verified per-process
resident HWM**, combined with the actual-worker footer by max. An aggregate
tree current-RSS sum is exposed only as a separately named diagnostic. A caller
must not silently replace the old per-process peak by that different definition.

## Integration sketch, not applied to the live runners

```python
process = subprocess.Popen(argv, ...)
tree = OwnedProcessTree(process)
while True:
    sample = tree.sample()
    # Keep actual worker footer and per-process tree HWM as separate originals.
    # Use max for the original peak metric; do not relabel the tree sum.
    if budget_or_timeout_crossed:
        cleanup = tree.terminate_tracked()
        # Re-poll asynchronous termination; record remaining/unknown identities.
        break
    if process.poll() is not None and sample['live_member_count'] == 0:
        break
```

Polling ancestry is **observational**, not atomic Job/cgroup containment. A
descendant that starts/exits entirely between samples, becomes orphaned before
its chain is observed, or is inaccessible may remain unproven. The helper
therefore never claims whole-tree containment. Future strict isolation would
need a separately reviewed spawn-time Job/cgroup design; it is outside this
minimal patch. Missing identities/memory must be visible in evidence, not zeroed
or counted as complete resource matching.

Primary API definitions: [Microsoft process handles/identifiers](https://learn.microsoft.com/en-us/windows/desktop/procthread/process-handles-and-identifiers),
[Toolhelp parent links](https://learn.microsoft.com/en-us/windows/win32/api/tlhelp32/ns-tlhelp32-processentry32),
[GetProcessTimes](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-getprocesstimes),
[TerminateProcess](https://learn.microsoft.com/windows/win32/api/processthreadsapi/nf-processthreadsapi-terminateprocess),
[Linux pidfd_open](https://www.man7.org/linux/man-pages/man2/pidfd_open.2.html),
[pidfd_send_signal](https://www.man7.org/linux/man-pages/man2/pidfd_send_signal.2.html).
