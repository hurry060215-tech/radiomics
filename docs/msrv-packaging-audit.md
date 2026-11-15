# Rust 1.85 and crate packaging audit

Audit date: 2026-11-14. Source baseline:
`cbdbdebbf418681f3e1d215a754e3251ef0083f6`, the independent `research-m3`
worktree. The main worker owns subsequent shared-manifest and packaging changes;
this report does not assume those later changes have been tested.

The audit did not change dependency versions, Cargo.lock, workspace metadata,
default toolchain, package version or publication settings. No publishing command
was run. Research/habitat modules were temporarily registered in this worktree's
lib.rs for the workspace MSRV check and second development archive, then restored
without committing that registration.

## Results

| Check | Observed result | Meaning |
|---|---|---|
| Rust 1.85.0 workspace/all-targets/locked check, including research/habitat | **Passed**, exit 0 | The declared toolchain supports this Windows/MSVC locked graph and source/test compilation |
| Three `cargo package --list` commands | **Passed**, exit 0 | Cargo enumerated files; this alone does not prove an installable/testable artifact |
| Core development `.crate`, actual `cargo +1.85.0 package` verification | **Passed**, exit 0 | Packaged library and newly registered M3 modules build on 1.85.0 |
| Core unpacked archive `check --lib --locked` | **Passed**, exit 0 | The original source archive builds independently of the repository as a library |
| Core archive `check --all-targets --locked` | **Failed**, exit 101 | Included tests refer to Golden files outside the archive |
| CLI actual `cargo package --locked --no-verify` | **Failed**, exit 101 | Path dependency on core has no version requirement |
| Python crate actual `cargo package --locked --no-verify` | **Failed**, exit 101 | Same missing path-dependency version; `publish=false` remains intentional |
| Linux/MSRV, linking and executing all three crates with 1.85 | **Not checked here** | `cargo check` is compilation/type checking, not a cross-platform runtime result |

No dependency MSRV failure occurred in the successful locked Windows workspace
check. This supports the declared Rust 1.85 floor for that scope; it does not
claim that older Rust versions work or that a future unlocked graph will work.

## Toolchain and commands

The isolated repository toolchain directory was
`D:\OnekeyPlatfrom50\radiomics\.build-tools\rustup`; cache and build directories
were task-owned locations on E:. Installed minimal toolchain:

```text
rustc 1.85.0 (4d91de4e4 2025-02-17)
cargo 1.85.0 (d73d2caf9 2024-12-31)
host: x86_64-pc-windows-msvc
LLVM: 19.1.7
```

`rustup toolchain install 1.85.0 --profile minimal` downloaded and installed the
components, but its final self-update check exited 1 because CARGO_HOME had been
redirected to the cache directory rather than the existing rustup executable
directory. Actual subsequent `rustup toolchain list`, `cargo +1.85.0 -V`,
`rustc +1.85.0 -Vv` and the full check succeeded. The active/default toolchain
remained `stable-x86_64-pc-windows-msvc` (Rust 1.100.0).

```powershell
. D:/OnekeyPlatfrom50/radiomics/tools/windows-rust-env.ps1
$env:CARGO_HOME='E:\radiomics-build\cargo'
$env:CARGO_TARGET_DIR='E:\radiomics-build\target-msrv'
$env:TEMP='E:\radiomics-build\tmp'
$env:TMP=$env:TEMP
cargo +1.85.0 check --workspace --all-targets --locked
cargo package --list -p ov-radiomics-core --locked
cargo package --list -p ov-radiomics-cli --locked
cargo package --list -p ov-radiomics-python --locked
cargo +1.85.0 package -p ov-radiomics-core --locked --allow-dirty `
    --target-dir E:/radiomics-build/target-package-verify
```

`--allow-dirty` on the final development package acknowledges only temporary M3
module registration. The archive's `.cargo_vcs_info.json` explicitly records the
baseline SHA and `dirty:true`. It is a development audit artifact, not a release
artifact or permission to publish.

## Packaging findings requiring main-worker integration

### Core-dependent crates lack a packaging version requirement

Both CLI and Python manifests contain:

```toml
ov-radiomics-core = { path = "../radiomics-core" }
```

Cargo rejects actual packaging before verification:

```text
all dependencies must have a version requirement specified when packaging.
dependency `ov-radiomics-core` does not specify a version
```

The main worker should add the intended compatible core version alongside `path`
and choose an explicit development-artifact/registry-verification strategy. A
packaged path dependency becomes a registry dependency; adding a version does not
prove registry availability or authorize publication. This audit did not query
or claim ownership/availability of a crates.io version and did not change
`publish=false` for the Python binding crate.

### Golden paths escape the package root

Core's real package contains its source and integration tests, including
`src/*_tests.rs` and `tests/golden_*.rs`, but contains no `golden/` files. Many
compile-time includes and runtime paths refer to `../../golden` or
`../../../golden`. Cargo cannot supply these repository-external paths from the
standalone `.crate`.

The verified second development archive includes the M3 module registration,
eliminating an unrelated missing-export failure from the first audit archive.
Its real `cargo +1.85.0 check --manifest-path <archive>/Cargo.toml --all-targets
--locked` still fails with missing statistics, imaging, pathology, WSI, research
and other Golden includes; the library test target alone reports 30 missing-file
errors. Integration tests using runtime file reads would also need their data.
Default `cargo package` library verification passing does not contradict this
test-target failure.

The main worker should make the intended distributed test scope explicit and
make every enabled test's data path package-local, through a reviewed fixture
layout or reproducible package-staging process. Simply listing the archive,
skipping verification, or excluding `tests/` does not repair compile-time Golden
includes inside the library's `cfg(test)` modules. This audit did not suppress
tests, change Golden data or introduce a hidden skip mechanism.

### README/license material and IBSI attribution

None of the three actual package lists contains a README or LICENSE file from
the repository root. The normalized core manifest reports `readme=false` while
retaining the valid SPDX declaration `license="MIT"`. Missing a license file is
not the Cargo manifest failure reported above, but the independent artifact's
documentation/attribution contents still need an explicit delivery decision.

No IBSI image, mask, license or source README is currently shipped in these
archives. Therefore the current archive cannot claim to distribute a standalone
IBSI test dataset. If Golden fixtures are staged into a package, preserve the
digital phantom's separate CC-BY-4.0 license, Alex Zwanenburg attribution,
SOURCE_README and input hashes. Do not relabel that data MIT. The explicitly
restricted/noncommercial CT reference data must remain separate from an MIT-only
artifact unless its own allowed-distribution terms are satisfied. This audit did
not add any CT data or data licensed with unclear terms.

### Build identity changes under packaging

Core's build.rs hashes normalized `.rs` source text, its Cargo.toml/build.rs and
the first Cargo.lock found at or above the crate root. The archive contains a
Cargo-normalized manifest and a package-local reduced lockfile, rather than the
workspace manifest/lock. A source digest therefore changes under packaging even
when algorithm source bytes are the same.

For the first clean committed archive, actual build-script digest was
`d2810c3df1b2d88477272671ec0317349592406083fce958756cc676b90c47d7`.
Recomputing the same build-script input order for the matching committed
workspace source gives
`5b3f7b2ec87df4cf694ea70bb0103a2d9de7395d87312845d5f18075f8d63d58`.
The corresponding lockfile SHA256 values are
`cb265e7fdb02dfae32fd9a315e3495c431a2ea694649a7fb43560a9e46b8b272`
and `ff310d5320801bf8d4a02308e560aa0876f7786521b8101f7334ce099dabc77d`.

The independently compiled workspace check with temporary M3 exports emitted
`9675ac74a6b5bbe79f5cddb40f2d2bf93166ea07362ca17c33215b772d04176b`;
that source intentionally differs from the first archive's unregistered lib.rs.
Actual build-script output order was used and reproduced exactly when checking
these hashes; filesystem-language path sorting must not be assumed equivalent.

This is a cache/provenance distinction, not an observed numerical mismatch. The
main worker should document artifact-specific identities or deliberately separate
algorithm-source and dependency/build identities. The script's nearest lockfile
hash should not be presented as proof of a consuming project's actual resolved
dependency graph. No hashing policy was changed during this audit.

## Local development artifacts and raw evidence

Verified development core archive:

```text
E:\radiomics-build\target-package-verify\package\ov-radiomics-core-0.1.0.crate
67 files; 235027 bytes
SHA256 c2de4d6983323f71393e05f3b07568703e9ce676227dd1ab0b0302fdc11195da
```

The first clean, unverified-package-command archive was
`E:\radiomics-build\target-msrv\package\ov-radiomics-core-0.1.0.crate`
(235347 bytes, SHA256
`a9e122138f11f1868f87dc44b52dcb271e357c46dd458157a3c81d30cb468f99`).
It was separately unpacked and its library checked successfully with Rust 1.85;
the second archive additionally passed Cargo's actual package verification.

Raw command logs and actual package lists are retained under
`E:\radiomics-build\msrv-packaging-audit\`, including:

- `msrv-check-1.85.0.txt`
- `ov-radiomics-{core,cli,python}-package-list.txt`
- `ov-radiomics-{core,cli,python}-package-build.txt`
- `core-package-verified.txt`
- `core-archive-lib-check.txt`
- `core-archive-all-targets-check.txt`
- `core-verified-archive-all-targets-check.txt`
- `source-digest-comparison.json`

Passing the MSRV compilation/library-package checks does not complete standalone
crate testability, CLI/binding packaging, Linux validation, full runtime tests or
M5 release acceptance. Those findings remain actionable integration work.
