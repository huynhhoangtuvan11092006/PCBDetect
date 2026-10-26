# DeepPCB Exploratory Data Analysis (EDA) Report

## 1. High-Level Dataset Summary

- **Total Image Pairs**: 1500 (1,000 trainval + 500 test)
- **Total Ground Truth Defect Annotations**: 10013
- **Average Defects Per Image**: 6.68 (Min: 1, Median: 7.0, Max: 15)
- **Image Resolution**: 640 x 640 pixels (~48 pixels/mm CCD scan)
- **Average Defect Area**: 1425.64 px² (0.3481% of total image area)
- **Small Object Regime (< 32x32 px)**: **32.0%** of all defects!

---

## 2. Class Distribution & Imbalance Analysis

| Defect Class | Trainval Count | Test Count | Total Count | % of All Defects |
| :--- | :--- | :--- | :--- | :--- |
| **mousebite** | 1379 | 586 | 1965 | 19.6% |
| **open** | 1283 | 659 | 1942 | 19.4% |
| **spur** | 1142 | 483 | 1625 | 16.2% |
| **short** | 1028 | 478 | 1506 | 15.0% |
| **pin_hole** | 1031 | 470 | 1501 | 15.0% |
| **copper** | 1010 | 464 | 1474 | 14.7% |

### Key Findings on Class Distribution:
1. **Relatively Balanced Defect Frequency**: All 6 classes have roughly 1,400 to 1,600 annotations each across the 1,500 images. Unlike typical industrial anomaly datasets with 99% normal images and 1% rare defects, DeepPCB includes artificially seeded defects on sampled boards, yielding an average of ~6 defects per tested image.
2. **Short and Open Circuits are Primary Reliability Killers**: While all classes are equally represented, in operational manufacturing, `open` and `short` account for fatal assembly failures and must be penalized higher during loss computation (cost-sensitive learning).

---

## 3. Geometric & Scale Analysis (Small Object Challenge)

| Metric | Width (px) | Height (px) | Area (px²) | Aspect Ratio (W/H) |
| :--- | :--- | :--- | :--- | :--- |
| **Mean** | 38.93 | 35.67 | 1425.64 | 1.12 |
| **Median** | 35.0 | 33.0 | 1184.0 | 1.00 |

- **COCO Object Scale Classification**:
  - Small (< 1,024 px² / 32x32): **32.0%**
  - Medium (1,024 – 9,216 px²): **67.9%**
  - Large (> 9,216 px²): **0.1%**

> **Factory Takeaway**: Because almost 70% of defects are in the COCO Small range, standard pre-trained architectures with aggressive downsampling (e.g., standard ResNet-50 with 32x stride down to 20x20 feature maps) will struggle without High-Resolution feature pyramids or P2 stride integration.

---

## 4. Board Layout & Data Leakage Review

DeepPCB consists of 11 distinct board groups (`group00041`, `group12000`, `group12100`, etc.).
The default `trainval.txt` and `test.txt` splits divide pairs across these groups.

```
split       test  trainval
group_id                  
group00041    21       200
group12000    14         0
group12100   146         0
group12300    98         0
group13000    18       198
group20085    34       291
group44000    50        50
group50600    29        50
group77000     7       100
group90100    74         0
group92000     9       111
```

> **Engineering Recommendation**: To demonstrate production-grade ML rigor, in Milestone 1 we will provide a **Strict Group-Disjoint Split** option where entire PCB board types are reserved exclusively for testing, guaranteeing that the model learns generalizable defect patterns rather than memorizing board trace topologies!

---

## 5. Generated Figures

- `class_distribution.png`: Bar chart of defect frequencies.
- `defect_heatmap.png`: 2D spatial density of defect locations on the 640x640 canvas.
- `defect_size_distribution.png`: Bounding box width vs height and area distributions.
- `defects_per_image.png`: Histogram of defect counts per image pair.
- `sample_defect_visualizations.png`: Side-by-side template, tested PCB, ground truth overlays, and difference image.
