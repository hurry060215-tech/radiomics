# Public CT source and derived reference notice

This directory records provenance, an explicit numerical validation protocol and
derived numeric references for the IBSI-1 PAT1 CT/GTV-1 NIfTI dataset. These
records identify their CC-BY-NC-3.0 source and are not asserted to be covered by the
software repository's MIT license. Original CT/ROI images are not stored here.

The pinned [dataset README](https://raw.githubusercontent.com/theibsi/data_sets/6da96021bc91faf4c0cb7fd7fa56a4225d2064a8/ibsi_1_ct_radiomics_phantom/README.md)
and [original license text](https://raw.githubusercontent.com/theibsi/data_sets/6da96021bc91faf4c0cb7fd7fa56a4225d2064a8/ibsi_1_ct_radiomics_phantom/LICENSE.md)
identify the Creative Commons Attribution-NonCommercial 3.0 Unported license.
See [CC-BY-NC-3.0](https://creativecommons.org/licenses/by-nc/3.0/) for its terms.
The numerical records retain this source-specific noncommercial provenance;
they are not a general permission for commercial redistribution of the data.

Philippe Lambin contributed the phantom through CancerData,
[DOI 10.17195/candat.2016.08.1](https://doi.org/10.17195/candat.2016.08.1).
Alex Zwanenburg converted the DICOM/RTSTRUCT source to NIfTI. Dataset citations
include Lambin P, *Radiomics Digital Phantom*, CancerData (2016), and Lambin et al.,
*Radiomics: the bridge between medical imaging and personalized medicine*,
Nature Reviews Clinical Oncology 14(12):749 (2017).

Reference generation converts already-integral image values losslessly from
float32 to float64, computes two explicitly declared ROI feature sets and a
fixed subgrid's texture counts. It applies no rounding, registration, resampling
or implicit CT preset. These derived numerical transformations are documented
in `protocol.json`; raw source bytes remain unchanged in external artifacts.

`tools/package_native.py` does not include this profile in its explicit asset
allowlist. Do not add it or its raw inputs to an MIT-only crate, wheel or source
artifact without a separately reviewed licensing/distribution decision.
