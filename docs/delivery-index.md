# Development delivery index

The package remains version 0.1.0. These are development artifacts and evidence,
not a PyPI/crates.io publication. The agreed development scope is complete with
the numerical, installation and artifact evidence below. Performance target
misses and measurement limits remain disclosed. The versioned completion audit
and latest PR CI must be consulted together.

## Read and reproduce

- [Chinese technical report](technical-report.md): architecture, algorithms,
  correctness, measured performance, deployment and limitations.
- [Support matrix](support-matrix.md) and [requirement audit](completion-audit.md).
- [Six native tutorials](python-native.md), with fresh output directories.
- [Native CLI](cli-workflows-native.md) and [pathology CLI](pathology-cli-native.md).
- [Portable source assets and build instructions](native-package-assets.md).

## Versioned evidence

- [Radiomics results](radiomics-benchmark-v2-results.md): immutable source48ecb310,
  854 retained observations, real CT speed misses and per-case uncertainty.
- [L5 deployment](deployment-benchmark-v2-results.md): immutable source45140bff,
  14 startup observations and one offline payload install, with limited scope.
- [ML/Statistics and Pathology](learning-pathology-benchmark-v2-results.md):
  224 complete observations, separate suites and whole-scene stages.
- [Public CT acceptance](public-ct-native-acceptance.md): independent scalar,
  matrix, buffer and physical-coordinate checks.
- [Actual optional containers](optional-interop-acceptance.md): pinned dense
  AnnData/WSIData metadata paths, preserving exact IDs/history and unknown MPP.
- [Independent API/IBSI audit](acceptance-v2-report.md).

## Local build artifacts

`E:/radiomics-build/snapshot-native-v3/build-manifest.json` binds the Windows
wheel and CLI to source45140bff, compiler/target/release flags, implementation
identity and every wheel member. The wheel is under `wheels/` and the executable
under `cli/`. It was installed into the isolated native environment and passed
71 native-only tests. Later CLI extensions require a newer final artifact.

`E:/radiomics-build/source-delivery-v1/` contains the inspected source archive,
unpacked tree, six successful tutorial outputs and `tutorial-evidence.json`.
That receipt verifies tutorial/assets execution with the installed v3 wheel;
it does not substitute for the separate locked source-build validation.

The later `snapshot-native-v4` directory contains the final CLI center extension
and a wheel verified by71 installed-native-only tests. CI run43615315617 passed
all9 jobs at76808bb, including the center extension; later numerical source code
is unchanged. Core archive standalone150tests and derived-sdist206Python tests
are documented in the package-assets evidence. Raw references,
measurements and failures remain available; no unfavorable observation is removed
to produce a completion claim.
