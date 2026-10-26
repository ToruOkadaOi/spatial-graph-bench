# Source data rights inventory

Reviewed 2026-10-26 for preparing the corrected benchmark data deposit. These notes describe source record metadata, not a legal determination about redistribution of derived artifacts.

| Benchmark data | Source record | License / rights found | Deposit implication |
| --- | --- | --- | --- |
| MERFISH spinal cord | [Zenodo 18039571](https://zenodo.org/records/18039571) | The record is open access. Its public metadata API did not expose a `license` or `rights` field when checked. | Do not assume the source record grants CC BY reuse. Ask the source depositor or clarify terms before licensing derived data. |
| Open-ST lymph node | [GEO GSE251926](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE251926) | GEO says it places no restrictions on use or distribution, while also noting that submitters may retain intellectual property rights and GEO cannot grant unrestricted permission on their behalf. See the [GEO disclaimer](https://www.ncbi.nlm.nih.gov/geo/info/disclaimer.html). | Public access does not itself establish a blanket redistribution license for a Zenodo bundle. |
| Xenium mouse kidney | [Figshare 28761695 v2](https://doi.org/10.6084/m9.figshare.28761695.v2) | Figshare metadata identifies CC BY 4.0. | Preserve attribution and the source license when redistributing this portion. |
| Stereo-seq axolotl | [STOmics ARTISTA STDS0000056](https://db.cngb.org/stomics/datasets/STDS0000056/summary) | The data are publicly downloadable; no explicit reuse license was located on the dataset pages reviewed. | Do not treat “public access” as an explicit license grant. Clarify rights before licensing derived files. |

The repository's MIT license covers the code only. It does not grant rights to the upstream datasets or all derived feature, graph, and prediction files. Because the sources do not share one clearly documented license, do not leave Zenodo's default CC BY 4.0 on the mixed data record. Keep the record unpublished until a defensible rights statement and redistribution basis are established.
