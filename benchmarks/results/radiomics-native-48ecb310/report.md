# Radiomics Benchmark v2 measured report

This report covers only the predeclared Radiomics suite. Correctness, speed targets, memory regression checks and deployment evidence remain distinct.

Plan SHA256: `a37b033956721ce3ae6ef54fa4cc4a9e62bf001b50611b4886a873e77786058c`.
L1 is a selected-family public array call, including conversion/binning and its actual matrix/scalar work. L2–L4 compute the exact 55-feature list plus the separately described project shape_basic descriptors on both backends. No mesh/filter feature is timed.

| Case | Python median s | Rust median s | Python/Rust speedup | Rust/Python peak RSS | Correct matching output |
|---|---:|---:|---:|---:|---|
| L1-synthetic_2d_g8-firstorder-n16-j1-warm_prepared_arrays | 0.0196872 | 0.0190754 | 1.032 | 0.5493 | passed applicable comparison |
| L1-synthetic_2d_g8-glcm-n16-j1-warm_prepared_arrays | 0.0132062 | 0.0187266 | 0.7052 | 0.5498 | passed applicable comparison |
| L1-synthetic_2d_g8-glrlm-n16-j1-warm_prepared_arrays | 0.0106262 | 0.0194259 | 0.547 | 0.5536 | passed applicable comparison |
| L1-synthetic_2d_g8-glszm-n16-j1-warm_prepared_arrays | 0.0085224 | 0.019488 | 0.4373 | 0.5513 | passed applicable comparison |
| L1-synthetic_2d_g8-gldm-n16-j1-warm_prepared_arrays | 0.0085264 | 0.0189996 | 0.4488 | 0.5498 | passed applicable comparison |
| L1-synthetic_2d_g8-ngtdm-n16-j1-warm_prepared_arrays | 0.0080883 | 0.0182493 | 0.4432 | 0.5508 | passed applicable comparison |
| L1-synthetic_2d_g8-shape_basic-n16-j1-warm_prepared_arrays | 0.0044557 | 0.0182444 | 0.2442 | 0.5508 | passed applicable comparison |
| L2-synthetic_2d_g8-full-n16-j1-warm_prepared_arrays | 0.0477536 | 0.0248769 | 1.92 | 0.5542 | passed applicable comparison |
| L3-synthetic_2d_g8-full-n8-j1-warm_uncached | 0.0642 | 0.0083145 | 7.721 | 0.5319 | passed applicable comparison |
| L3-synthetic_2d_g8-full-n8-j1-cold_process_total | 0.716527 | 0.419002 | 1.71 | 0.5304 | passed applicable comparison |
| L1-synthetic_2d_g32-firstorder-n16-j1-warm_prepared_arrays | 0.0209482 | 0.0366836 | 0.5711 | 0.5601 | passed applicable comparison |
| L1-synthetic_2d_g32-glcm-n16-j1-warm_prepared_arrays | 0.0261716 | 0.0377471 | 0.6933 | 0.5603 | passed applicable comparison |
| L1-synthetic_2d_g32-glrlm-n16-j1-warm_prepared_arrays | 0.0125993 | 0.0387992 | 0.3247 | 0.5604 | passed applicable comparison |
| L1-synthetic_2d_g32-glszm-n16-j1-warm_prepared_arrays | 0.0100523 | 0.03809 | 0.2639 | 0.5612 | passed applicable comparison |
| L1-synthetic_2d_g32-gldm-n16-j1-warm_prepared_arrays | 0.0101273 | 0.0380921 | 0.2659 | 0.5613 | passed applicable comparison |
| L1-synthetic_2d_g32-ngtdm-n16-j1-warm_prepared_arrays | 0.0102182 | 0.0372852 | 0.2741 | 0.5601 | passed applicable comparison |
| L1-synthetic_2d_g32-shape_basic-n16-j1-warm_prepared_arrays | 0.0045653 | 0.0356597 | 0.128 | 0.5615 | passed applicable comparison |
| L2-synthetic_2d_g32-full-n16-j1-warm_prepared_arrays | 0.0677136 | 0.0470218 | 1.44 | 0.5604 | passed applicable comparison |
| L1-synthetic_3d_g8-firstorder-n16-j1-warm_prepared_arrays | 0.0185708 | 0.0429446 | 0.4324 | 0.5618 | passed applicable comparison |
| L1-synthetic_3d_g8-glcm-n16-j1-warm_prepared_arrays | 0.0148387 | 0.04343 | 0.3417 | 0.5614 | passed applicable comparison |
| L1-synthetic_3d_g8-glrlm-n16-j1-warm_prepared_arrays | 0.0165601 | 0.0519993 | 0.3185 | 0.5633 | passed applicable comparison |
| L1-synthetic_3d_g8-glszm-n16-j1-warm_prepared_arrays | 0.0105762 | 0.0453423 | 0.2333 | 0.5635 | passed applicable comparison |
| L1-synthetic_3d_g8-gldm-n16-j1-warm_prepared_arrays | 0.0102466 | 0.044298 | 0.2313 | 0.5632 | passed applicable comparison |
| L1-synthetic_3d_g8-ngtdm-n16-j1-warm_prepared_arrays | 0.0095742 | 0.0433628 | 0.2208 | 0.5627 | passed applicable comparison |
| L1-synthetic_3d_g8-shape_basic-n16-j1-warm_prepared_arrays | 0.0042761 | 0.0423659 | 0.1009 | 0.5622 | passed applicable comparison |
| L2-synthetic_3d_g8-full-n16-j1-warm_prepared_arrays | 0.058811 | 0.0597179 | 0.9848 | 0.5636 | passed applicable comparison |
| L1-synthetic_3d_g32-firstorder-n16-j1-warm_prepared_arrays | 0.0248508 | 0.117225 | 0.212 | 0.5738 | passed applicable comparison |
| L1-synthetic_3d_g32-glcm-n16-j1-warm_prepared_arrays | 0.0318742 | 0.121782 | 0.2617 | 0.576 | passed applicable comparison |
| L1-synthetic_3d_g32-glrlm-n16-j1-warm_prepared_arrays | 0.0289832 | 0.142597 | 0.2033 | 0.5772 | passed applicable comparison |
| L1-synthetic_3d_g32-glszm-n16-j1-warm_prepared_arrays | 0.0160256 | 0.127124 | 0.1261 | 0.5762 | passed applicable comparison |
| L1-synthetic_3d_g32-gldm-n16-j1-warm_prepared_arrays | 0.0155331 | 0.122881 | 0.1264 | 0.5773 | passed applicable comparison |
| L1-synthetic_3d_g32-ngtdm-n16-j1-warm_prepared_arrays | 0.0147046 | 0.119335 | 0.1232 | 0.5734 | passed applicable comparison |
| L1-synthetic_3d_g32-shape_basic-n16-j1-warm_prepared_arrays | 0.0044105 | 0.114603 | 0.03849 | 0.5765 | passed applicable comparison |
| L2-synthetic_3d_g32-full-n16-j1-warm_prepared_arrays | 0.107711 | 0.166168 | 0.6482 | 0.576 | passed applicable comparison |
| L3-synthetic_3d_g32-full-n8-j1-warm_uncached | 0.0948684 | 0.0296165 | 3.203 | 0.5325 | passed applicable comparison |
| L3-synthetic_3d_g32-full-n8-j1-cold_process_total | 0.721011 | 0.435187 | 1.657 | 0.5313 | passed applicable comparison |
| L1-ibsi_digital_3d-firstorder-n16-j1-warm_prepared_arrays | 0.0183627 | 0.0092438 | 1.986 | 0.5439 | passed applicable comparison |
| L1-ibsi_digital_3d-glcm-n16-j1-warm_prepared_arrays | 0.0125092 | 0.0094429 | 1.325 | 0.5449 | passed applicable comparison |
| L1-ibsi_digital_3d-glrlm-n16-j1-warm_prepared_arrays | 0.01024 | 0.0097667 | 1.048 | 0.5448 | passed applicable comparison |
| L1-ibsi_digital_3d-glszm-n16-j1-warm_prepared_arrays | 0.0080358 | 0.0091866 | 0.8747 | 0.5447 | passed applicable comparison |
| L1-ibsi_digital_3d-gldm-n16-j1-warm_prepared_arrays | 0.0081453 | 0.0092838 | 0.8774 | 0.5453 | passed applicable comparison |
| L1-ibsi_digital_3d-ngtdm-n16-j1-warm_prepared_arrays | 0.0077529 | 0.0090305 | 0.8585 | 0.545 | passed applicable comparison |
| L1-ibsi_digital_3d-shape_basic-n16-j1-warm_prepared_arrays | 0.0040489 | 0.0088964 | 0.4551 | 0.5451 | passed applicable comparison |
| L2-ibsi_digital_3d-full-n16-j1-warm_prepared_arrays | 0.0423657 | 0.0134437 | 3.151 | 0.5487 | passed applicable comparison |
| L3-ibsi_digital_3d-full-n8-j1-warm_uncached | 0.0594852 | 0.007893 | 7.536 | 0.5319 | passed applicable comparison |
| L3-ibsi_digital_3d-full-n8-j1-cold_process_total | 0.68981 | 0.407455 | 1.693 | 0.5307 | passed applicable comparison |
| L4-throughput_3d_g8-full-n100-j1-warm_uncached | 0.663711 | 0.0586062 | 11.32 | 0.5516 | passed applicable comparison |
| L4-throughput_3d_g8-full-n100-j4-warm_uncached | 0.483568 | 0.0436474 | 11.08 | 0.559 | passed applicable comparison |
| L4-throughput_3d_g8-full-n500-j1-warm_uncached | 3.32524 | 0.25959 | 12.81 | 0.5976 | passed applicable comparison |
| L4-throughput_3d_g8-full-n500-j4-warm_uncached | 2.37456 | 0.193538 | 12.27 | 0.6097 | passed applicable comparison |
| L4-throughput_3d_g8-full-n1000-j1-warm_uncached | 6.67001 | 0.517182 | 12.9 | 0.6317 | passed applicable comparison |
| L4-throughput_3d_g8-full-n1000-j4-warm_uncached | 4.68314 | 0.376744 | 12.43 | 0.6403 | passed applicable comparison |
| L4-throughput_3d_g32-full-n100-j1-warm_uncached | 0.817535 | 0.117975 | 6.93 | 0.5516 | passed applicable comparison |
| L4-throughput_3d_g32-full-n100-j4-warm_uncached | 0.752028 | 0.0568079 | 13.24 | 0.5638 | passed applicable comparison |
| L4-throughput_3d_g32-full-n500-j1-warm_uncached | 4.02717 | 0.573824 | 7.018 | 0.5987 | passed applicable comparison |
| L4-throughput_3d_g32-full-n500-j4-warm_uncached | 3.71145 | 0.263242 | 14.1 | 0.61 | passed applicable comparison |
| L4-throughput_3d_g32-full-n1000-j1-warm_uncached | 8.10913 | 1.14279 | 7.096 | 0.6277 | passed applicable comparison |
| L4-throughput_3d_g32-full-n1000-j4-warm_uncached | 7.43196 | 0.51699 | 14.38 | 0.6447 | passed applicable comparison |
| L2-public_ct_pat1-full-n1-j1-warm_prepared_arrays | 0.157666 | 0.394163 | 0.4 | 0.7617 | passed applicable comparison |
| L3-public_ct_pat1-full-n1-j1-warm_uncached | 0.225082 | 0.183807 | 1.225 | 0.7331 | passed applicable comparison |

Each JSON summary retains n, q25/q75, IQR, empirical p95, min/max, paired ratios, ROI/s, process-resident peaks, source/toolchain provenance and all raw observations. Seven repetitions do not provide a precise tail estimate.

## Missed engineering thresholds and targets

- `L2-synthetic_2d_g32-full-n16-j1-warm_prepared_arrays`: runtime regression=False, memory regression=False, >=1.5× speed target met=False. Retain and investigate; do not change correctness tolerances or claim an algorithm failure merely because it is slower.
- `L2-synthetic_3d_g8-full-n16-j1-warm_prepared_arrays`: runtime regression=False, memory regression=False, >=1.5× speed target met=False. Retain and investigate; do not change correctness tolerances or claim an algorithm failure merely because it is slower.
- `L2-synthetic_3d_g32-full-n16-j1-warm_prepared_arrays`: runtime regression=True, memory regression=False, >=1.5× speed target met=False. Retain and investigate; do not change correctness tolerances or claim an algorithm failure merely because it is slower.
- `L2-public_ct_pat1-full-n1-j1-warm_prepared_arrays`: runtime regression=True, memory regression=False, >=1.5× speed target met=False. Retain and investigate; do not change correctness tolerances or claim an algorithm failure merely because it is slower.
- `L3-public_ct_pat1-full-n1-j1-warm_uncached`: runtime regression=False, memory regression=False, >=1.5× speed target met=False. Retain and investigate; do not change correctness tolerances or claim an algorithm failure merely because it is slower.

## Scope and limitations

- Raw records and all failed observations remain in `raw-observations.jsonl`; no cross-suite combined acceleration score is produced.
- Batch sources intentionally repeat a disclosed synthetic image/ROI with unique identifiers. Content cache is disabled for uncached throughput. Native cache population/hits are a separate noncomparative series.
- Peak RSS covers one complete worker process lifetime, including imports, fixed fixture buffers, warmup, timed calls and result-transport bookkeeping. Both backends use outer threads in that same process. These numbers are not Python traced allocations or per-call allocation counts.
- Cold means fresh interpreter process, with startup/import/API/export/numerical-report transport retained. External package/input audits and artifact-file hashing occur outside the cold clock. It is not a flushed-disk-cache measurement or Python-free CLI startup result.
- Observed scaling is limited to the frozen workers=1/4 and ROI-count grid under a 2GiB budget. It does not establish a theoretical maximum throughput.
- Median/IQR/p95 and alternating raw pairs describe uncertainty. Gains within observed noise should not be advertised as acceleration.
- This is not a new IBSI conformance claim, clinical validation or blanket project Release Gate pass. The independent IBSI matrix and Golden rules are unchanged.
- L5 CLI install/startup performance measured separately; allocator count/bytes unmeasured; per-stage extraction instrumentation unmeasured; OS disk cache uncontrolled; power mode/background load not controlled by this runner.
