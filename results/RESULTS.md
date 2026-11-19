## Results

Holdout half of the MVTec AD test set, 15 categories, RTX 5050 laptop GPU.

| | PatchCore (no defects seen) | ResNet-18 + Grad-CAM (trained on defects) |
|---|---:|---:|
| **Mean image AUROC** | **0.988** | 0.966 |
| **Mean pixel AUROC** (defect localisation) | **0.979** | 0.893 |
| Mean F1 at the chosen threshold | 0.956 | 0.865 |
| Defects caught (recall) | 94% | 80% |
| Good parts wrongly rejected | 6% | 10% |

<details><summary>Per category</summary>

| Category | PatchCore image | PatchCore pixel | PatchCore F1 | CNN image | CNN pixel | CNN F1 |
|---|---:|---:|---:|---:|---:|---:|
| bottle | 1.000 | 0.985 | 0.968 | 1.000 | 0.907 | 0.933 |
| cable | 0.997 | 0.985 | 0.969 | 0.951 | 0.830 | 0.825 |
| capsule | 0.984 | 0.987 | 0.973 | 0.973 | 0.896 | 0.857 |
| carpet | 0.979 | 0.988 | 0.957 | 0.822 | 0.935 | 0.849 |
| grid | 0.985 | 0.978 | 0.984 | 0.952 | 0.809 | 0.889 |
| hazelnut | 1.000 | 0.984 | 1.000 | 1.000 | 0.965 | 0.957 |
| leather | 1.000 | 0.990 | 1.000 | 1.000 | 0.960 | 0.932 |
| metal nut | 1.000 | 0.985 | 1.000 | 0.987 | 0.881 | 0.843 |
| pill | 0.956 | 0.977 | 0.931 | 0.956 | 0.904 | 0.844 |
| screw | 0.982 | 0.972 | 0.940 | 1.000 | 0.799 | 0.881 |
| tile | 0.973 | 0.955 | 0.976 | 0.990 | 0.947 | 0.925 |
| toothbrush | 0.978 | 0.986 | 0.933 | 0.933 | 0.869 | 0.571 |
| transistor | 0.997 | 0.976 | 0.750 | 0.932 | 0.864 | 0.750 |
| wood | 0.994 | 0.961 | 0.984 | 1.000 | 0.878 | 0.967 |
| zipper | 0.998 | 0.983 | 0.983 | 1.000 | 0.956 | 0.957 |

</details>

**Where PatchCore is weakest:** pill (0.956), tile (0.973), toothbrush (0.978).
The classifier beats it on image AUROC in: screw, tile, wood, all categories where it trained on examples of the same defect types.
