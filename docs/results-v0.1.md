# LightJev-0.6B v0.1 results

[Weights, complete predictions and reports](https://huggingface.co/rongxinzy/LightJev-0.6B-v0.1). Numbers below are transcribed programmatically from the release `evaluation/ce/report.json` and `evaluation/brier/report.json`, after reloading each selected checkpoint.

CE step 250 was selected before held-out reports by training-evaluated development CE (0.53279645 versus Brier step 320 at 0.54574977). Both runs completed 20 head-only plus 300 full-model steps (5,120 sampled questions), seed 17. The selected CE step-250 checkpoint contains 20 head-only plus 230 full-model steps (4,000 sampled questions); it is earlier than the completed training budget. Brier performs better on some final test metrics, but those outcomes do not change the precommitted development-based selection. Training-time BF16-autocast development numbers need not equal final FP32 evaluation numbers.

An independent review recomputed the reported metrics from saved predictions and found them consistent. Test has 848 questions (656 hard / 192 soft); OOD has 448 (352 hard / 96 soft).

## Overall held-out comparison

| Arm | Split | Scores | All n | Hard n | Soft n | All CE | Hard accuracy | Hard Brier | Hard ECE | Soft squared L2 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ce | test | raw | 848 | 656 | 192 | 0.564371 | 79.57% | 0.266239 | 0.026456 | 0.00313175 |
| ce | test | calibrated | 848 | 656 | 192 | 0.568544 | 79.57% | 0.266815 | 0.042842 | 0.00498104 |
| ce | ood | raw | 448 | 352 | 96 | 0.659047 | 72.44% | 0.303744 | 0.064119 | 0.00396426 |
| ce | ood | calibrated | 448 | 352 | 96 | 0.688877 | 72.44% | 0.311637 | 0.093040 | 0.00567818 |
| brier | test | raw | 848 | 656 | 192 | 0.563263 | 79.73% | 0.264457 | 0.069143 | 0.00575920 |
| brier | test | calibrated | 848 | 656 | 192 | 0.561635 | 79.73% | 0.264489 | 0.047142 | 0.00507942 |
| brier | ood | raw | 448 | 352 | 96 | 0.609766 | 73.01% | 0.299688 | 0.095532 | 0.00373097 |
| brier | ood | calibrated | 448 | 352 | 96 | 0.626286 | 73.01% | 0.309694 | 0.094655 | 0.00407519 |

Hard means one-hot target; soft means exact programmatic conditional distribution. They can overlap if an exact conditional distribution is degenerate. Soft L2 is distribution error, not observed classification error. ECE uses ten equal-width top-label bins.

## Family results, raw probabilities

| Arm | Split | Family | n | Accuracy (hard families) | CE | Squared L2 |
|---|---|---|---:|---:|---:|---:|
| ce | test | catalog_lookup_v2 | 144 | 100.00% | 0.000271 | 0.00000090 |
| ce | test | grid_navigation_bfs_v1 | 160 | 61.25% | 0.896193 | 0.52835766 |
| ce | test | known_chance_v2 | 192 | — | 1.011278 | 0.00313175 |
| ce | test | smart_home_v2 | 192 | 98.96% | 0.032929 | 0.01709978 |
| ce | test | tic_tac_toe_minimax_v1 | 160 | 56.25% | 0.841681 | 0.54270191 |
| ce | ood | catalog_lookup_v2 | 96 | 100.00% | 0.000247 | 0.00000051 |
| ce | ood | grid_navigation_bfs_v1 | 80 | 51.25% | 1.347604 | 0.59160929 |
| ce | ood | known_chance_v2 | 96 | — | 1.017028 | 0.00396426 |
| ce | ood | smart_home_v2 | 96 | 95.83% | 0.087833 | 0.05488974 |
| ce | ood | tic_tac_toe_minimax_v1 | 80 | 32.50% | 1.016931 | 0.67899706 |
| brier | test | catalog_lookup_v2 | 144 | 100.00% | 0.002008 | 0.00002876 |
| brier | test | grid_navigation_bfs_v1 | 160 | 61.25% | 0.835475 | 0.50069854 |
| brier | test | known_chance_v2 | 192 | — | 1.016076 | 0.00575920 |
| brier | test | smart_home_v2 | 192 | 99.48% | 0.085741 | 0.04176224 |
| brier | test | tic_tac_toe_minimax_v1 | 160 | 56.25% | 0.825833 | 0.53343313 |
| brier | ood | catalog_lookup_v2 | 96 | 100.00% | 0.001634 | 0.00002008 |
| brier | ood | grid_navigation_bfs_v1 | 80 | 51.25% | 1.076316 | 0.57570307 |
| brier | ood | known_chance_v2 | 96 | — | 1.018003 | 0.00373097 |
| brier | ood | smart_home_v2 | 96 | 97.92% | 0.075510 | 0.03832434 |
| brier | ood | tic_tac_toe_minimax_v1 | 80 | 32.50% | 1.024196 | 0.69691227 |

## Interpretation and limitations

- Catalog lookup and smart-home rules perform better than grid-navigation and tic-tac-toe state judgments. The latter remain substantial weaknesses; action-policy questions were excluded from training.
- The selected CE model's fitted global temperature 0.8187307530779818 worsens test/OOD CE and ECE. Raw predictions remain the API default; calibration is an optional artifact, not a guarantee.
- Data is synthetic and the OOD split is defined by the upstream generator. Templates and families overlap. These results do not establish broad business, multilingual or real-world generalization.
- One seed and a short training budget do not establish superiority over NanoJev, Jev, or other baselines. No matched competitor run or RLCD implementation is claimed.
- Two preliminary CE/Brier trials used inherited BF16 master weights. They were stopped to correct master-parameter precision, excluded from selection, and restarted with FP32 masters/Adam moments and BF16 autocast. Their manifests and step logs are retained under `training/*-aborted-nativebf16`. Final weights and evaluation are FP32.
- Both successful-arm reports, complete predictions, failed-trial histories, frozen data hashes, selection receipt and package versions remain in the model repository. Source attribution and conversion are described in [data.md](data.md); exact commands are in [training-release.md](training-release.md).

## Additional manual probe

A post-benchmark CPU inference check used this out-of-template, hand-written rule:

```text
State: The room temperature is 30 degrees C. Cooling is required above 26 degrees C.
Question: Should cooling be enabled?
Candidates: [false, true]
```

The selected CE checkpoint returned `[0.5236047506332397, 0.4763951897621155]`, selecting **false**, although the stated rule requires **true**. This is a concrete failure to apply a simple rule outside the training template. It is an extra manual probe, not part of the frozen benchmark, and is not included in its aggregate scores. The README example now reproduces an original-format frozen test question instead; that example is not evidence of new-task generalization. This failed probe reinforces the synthetic-domain scope and the need for independent validation on new instructions.
