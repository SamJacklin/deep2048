Search band: the closest 50% of late decisions (greedy relative margin < 0.00189); recovered by band: {'10': 0.27433628318584075, '25': 0.3539823008849557, '50': 0.6814159292035398}; ranking threshold 0.000268 (873 dev disagreements)

| Arm | Greedy conversion | Δ | Greedy score added | Δ | Depth-3 conversion | Δ | Depth-3 score added | Δ | Full greedy | Δ | Full depth 3 | Δ | 32768, full depth 3 | Late moves changed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 11.7% | — | 162,726 | — | 18.0% | — | 200,167 | — | 316,674 | — | 393,606 | — | 16.6% | 0.0% |
| A | 9.4% | -2.4% ± 1.2% | 151,378 | -11,347 ± 2,907 | 18.2% | +0.3% ± 2.5% | 195,592 | -4,576 ± 5,860 | 290,868 | -25,807 ± 10,291 | 386,724 | -6,882 ± 9,316 | 17.5% | 18.4% |
| B | 9.8% | -1.9% ± 1.1% | 147,357 | -15,368 ± 2,914 | 17.5% | -0.5% ± 2.7% | 197,532 | -2,635 ± 6,058 | 279,077 | -37,598 ± 10,601 | 387,416 | -6,191 ± 9,038 | 15.6% | 21.2% |

| Arm | Training moves | Decisions searched | Search disagreed | Ranking updates | Chance nodes | CPU hours | Wall hours |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 486,551,719 | 0 | 0 | 0 | 0 | 0.45 | 0.07 |
| B | 508,556,360 | 121,250,960 | 56,004,840 | 0 | 7,761,286,686 | 0.00 | 0.00 |
