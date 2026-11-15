# IBSI-1 digital phantom

Original phantom design: **Alex Zwanenburg**. This copy uses the unmodified
NIfTI image and mask from the official
[IBSI data repository](https://github.com/theibsi/data_sets/tree/6da96021bc91faf4c0cb7fd7fa56a4225d2064a8/ibsi_1_digital_phantom),
revision `6da96021bc91faf4c0cb7fd7fa56a4225d2064a8`.

The data is licensed under **Creative Commons Attribution 4.0 International**.
The original license and source README are retained beside the files. The
project's MIT code license does not replace this data license.

Independent benchmark facts are extracted from the official
[IBSI manual source](https://github.com/theibsi/ibsi-doc/tree/b952e71e9d40569e8a9ff3c740bc389c04dd6c28/docs),
revision `b952e71e9d40569e8a9ff3c740bc389c04dd6c28`. Source sections, lines,
printed numeric strings, published tolerances and reported precision are
recorded in `published_reference.json`.

This is a source freeze, before the native independent acceptance. It does not
claim full IBSI conformance. Raw kurtosis in `python_golden_v1` differs from IBSI
excess kurtosis by three. The corresponding mesh-based shape features are
outside the supported `shape_basic` subset.
