# Source tags, candidates and registry releases

The immutable `v0.1.0` Git tag is a **source release**. It does not promise a
PyPI upload, and it is not moved to match later repairs on `main`. GitHub source
archives and prepared installation sdists are different artifacts; see the
[frozen-tag build prerequisites](../website/en/installation.md#frozen-source-tag-builds).

Registry publication is a separate approval action. No token is stored in this
repository. The manual `release.yml` workflow defaults to `destination=artifacts`:
it checks the requested version against Python, Cargo and CITATION metadata,
runs the acceptance workflow, and builds Linux and Windows x86-64 abi3 wheels
plus a prepared sdist from the selected, immutable Actions commit SHA.

Two `cp39-abi3` wheels cover the eight Python 3.9–3.12/platform combinations.
The same two candidate wheels are smoke-tested across all eight combinations;
eight duplicate filenames must not be uploaded to the registry. The acceptance
workflow also tests installation from the prepared sdist without a checkout.
macOS, ARM and Python versions beyond the tested matrix are not release claims.

## Maintainer setup and release sequence

1. Configure GitHub environments named `testpypi` and `pypi` with required
   reviewers and deployment restricted to `main`. Configure matching Trusted
   Publishers separately on each registry: owner `hurry060215-tech`, repository
   `radiomics`, workflow `release.yml`, and the corresponding environment name.
   A pending publisher can create a new registry project. This account setup
   cannot be replaced by adding a workflow file.
2. Dispatch the workflow on `main` with the exact version and `artifacts` first.
   Review the commit SHA, acceptance checks, wheel installation matrix and sdist
   installation results. Source changes still require normal PR review.
3. After explicit approval, dispatch to `testpypi`. In a new environment, install
   base dependencies from PyPI and the candidate from TestPyPI only:

   ```bash
   python -m pip install numpy pandas matplotlib
   python -m pip install --index-url https://test.pypi.org/simple/ --no-deps ov-radiomics==0.1.0
   python -c "import ov_radiomics as r; print(r.__version__, r.backend_info())"
   ```

4. Review that real TestPyPI installation before approving a `pypi` dispatch.
   A successful artifact build is not a TestPyPI or PyPI publication. Registry
   versions cannot be overwritten; decide a new version explicitly if needed.
5. Verify `python -m pip index versions ov-radiomics` and installation in a new
   environment after publication. Enable the commented OmicVerse optional extra
   only after the public-registry install works. Until then use an accepted local
   wheel; the OmicVerse adapter already carries the inactive placeholder.

Trusted Publisher setup follows the
[PyPI publishing guide](https://docs.pypi.org/trusted-publishers/using-a-publisher/).
