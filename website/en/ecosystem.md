# Ecosystem and responsibilities

[简体中文](../guide/ecosystem.md) · [Installation](installation.md)

The independent package owns feature extraction and modeling through its Rust
core. An optional OmicVerse adapter provides a convenient namespace while
retaining separate installation and licensing boundaries.

| Entry point | Responsibility | Dependency boundary |
|---|---|---|
| `ov.radiology` | DICOM geometry, measurements and image review | Imaging tools maintained by OmicVerse |
| `ov.radiomics` | Optional lazy access to the independent radiomics package | The adapter is maintained in the OmicVerse repository |
| `ov_radiomics` | Declared native feature subset, identified tables, modeling and evaluation | Independently distributed MIT package |
| `radiomics` CLI | File-driven extraction, modeling and prediction | The shared Rust core without Python |

```{mermaid}
flowchart TD
    R[ov.radiology: geometry and review] -->|explicit image and mask inputs| P[Independent ov_radiomics package]
    A[ov.radiomics: optional lazy adapter] -->|optional package installation| P
    P --> C[Shared Rust core]
    L[Independent radiomics CLI] --> C
```

The diagram assigns responsibilities; it does not make DICOM conversion a new
radiomics feature or transform CT voxels into an AnnData expression matrix.
Existing arrays and external embeddings enter only through their documented,
identified contracts. Optional interoperation does not add an OmicVerse runtime
dependency to the independent package.

The adapter becomes an installation promise only when a compatible independent
wheel and the corresponding OmicVerse extra are actually released. Importing
OmicVerse must not eagerly import that optional package or new imaging backends.

The independent package is MIT licensed; OmicVerse is GPL-3.0. The GPL project
may depend on MIT code. GPL implementations must not be copied back and
redistributed as MIT code. See the [contribution policy](../contributing.md).

Read the [native boundaries](../reference/native-boundaries.md) for optional
Python paths, unsupported adapters and explicitly deferred capabilities.
Read the [English installation guide](installation.md) for candidate wheels and
the separate CLI.
