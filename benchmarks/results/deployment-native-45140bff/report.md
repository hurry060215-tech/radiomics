# Independent L5 deployment evidence

Plan SHA256: `d807983caaee9bb95f6046327c014bb9a2b86c6e421e77e96b66e938c0c6bfa6`.

| Endpoint | n | Median ms | IQR ms | Empirical p95 ms | Output validated |
|---|---:|---:|---:|---:|---|
| cli_version_fresh_process | 7 | 3.611 | 0.567 | 48.817 | True |
| python_import_backend_info_fresh_process | 7 | 45.052 | 3.447 | 51.142 | True |

Single offline wheel-payload install: 0.762398s, n=1. This excludes dependencies, downloads, resolver and environment/base-Python creation; it is not complete installation cost.

Wheel: 3723631 compressed bytes, 31 ZIP files, 9880697 uncompressed bytes. CLI: 9058304 bytes.
Installed project payload after startup: 59 regular files, 10074196 logical bytes.

The two endpoints perform different work; no CLI/Python speed ratio is computed. No language-wide speedup or combined Radiomics/ML/Pathology score is inferred.

- One offline --no-deps wheel-payload install; no dependency resolution/download/install cost
- Fresh interpreter/process, not flushed OS disk/bytecode cache
- Logical bytes/file counts, not filesystem allocation/shared base Python size
- Only Windows/native snapshot and these two endpoints measured; no algorithm or full-workflow timing
- Seven repeats give empirical p95 only; unrelated OS/user background processes not controlled
