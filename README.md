3D Object Detection and Multi-Object Tracking on KITTI


---

## Overview

This repository contains the Google Colab notebooks and results spreadsheet comparing 3D object detection and multi-object tracking approaches on the KITTI detection and  tracking dataset. Experiments are organised into three studies:

- **3D Object Detection Comparison** — Experiments 1–4
- **Multi-Object Tracking Comparison** — Experiments 5–8
- **4D Perception Pipeline Comparison** — Experiments 9–12


---

## 3D Object Detection Comparison (Experiments 1–4)

This study evaluates four approaches to 3D object detection from LiDAR point clouds, progressing from a pipeline validation baseline through classical geometric clustering to two state-of-the-art deep learning detectors. The goal is to establish per-class detection benchmarks for Car, Pedestrian, and Cyclist before introducing temporal tracking.

| Exp | Experiment | Method | Colab |
|---|---|---|---|
| 1 | Ground Truth Oracle Baseline | Ground truth labels as predictions | [Open](https://colab.research.google.com/drive/1TfKNxPZC0oTuwyKVPusIk_jBTnSwhDhd) |
| 2 | LiDAR Clustering Baseline | DBSCAN on LiDAR point cloud | [Open](https://colab.research.google.com/drive/16L4yUJxOIKzQANcRQDWFV5pivhAeMxyo) |
| 3 | PointPillars Detection | Pretrained PointPillars (thr=optimal) | [Open](https://colab.research.google.com/drive/1xt-J1x1aEfvZ_riM8KJ8LfW7UZAxqqT-) |
| 4 | PointRCNN Detection | Pretrained PointRCNN (thr=optimal) | [Open](https://colab.research.google.com/drive/1BEEYqHc3DBabiNsId0HgIifRtNv8qSf4) |

> **Exp 1:** Ground truth labels are treated as predictions to validate the full evaluation pipeline. A custom IoU function and greedy matching logic are implemented and reused across Experiments 2–4. Precision, Recall, IoU and AP all reach 1.0 / 100% confirming that dataset loading, bounding box parsing and metric computation are all correct.

> **Exp 2:** DBSCAN clustering (eps=0.5, min_samples=10) is applied to ground-removed, ROI-filtered LiDAR point clouds to detect objects without any neural network. The method achieves only marginal Car detection (Precision 0.011, Recall 0.006) and completely fails on Pedestrians and Cyclists, demonstrating that classical geometric clustering cannot semantically distinguish object classes.

> **Exp 3:** A pretrained PointPillars model (zhulf0804/PointPillars, checkpoint epoch_160.pth) encodes point clouds into vertical pillars processed by a single-stage 2D backbone detector. The model achieves Car AP 78.39%, Pedestrian AP 51.46% and Cyclist AP 63.00% at 3.63 FPS on a Colab T4 GPU, establishing the neural detection baseline.

> **Exp 4:** A pretrained PointRCNN model (OpenPCDet, checkpoint pointrcnn_7870.pth) processes raw point clouds through a two-stage pipeline — PointNet++ for per-point features and region proposals, followed by second-stage box refinement. It outperforms PointPillars on all three classes, most notably Cyclist (+7.61% AP) and Pedestrian (+3.49% AP), at the cost of slower inference (1.50 FPS). The saved detection file pointrcnn_result.pkl is used as the fixed input for all four trackers in Study 2.

**Results — Precision, Recall, IoU, 3D AP_R40 (moderate), FPS:**

| Exp | Method | Car Prec | Ped Prec | Cyc Prec | Car Rec | Ped Rec | Cyc Rec | Car IoU | Ped IoU | Cyc IoU | Car AP | Ped AP | Cyc AP | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | GT Oracle | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 100 | 100 | 100 | — |
| 2 | DBSCAN | 0.011 | 0.000 | 0.000 | 0.006 | 0.000 | 0.000 | 0.620 | 0.000 | 0.000 | 0.0001 | 0 | 0 | — |
| 3 | PointPillars | 0.876 | 0.625 | 0.781 | 0.848 | 0.585 | 0.550 | 0.870 | 0.684 | 0.799 | 78.39 | 51.46 | 63.00 | 3.63 |
| 4 | PointRCNN | 0.868 | 0.703 | 0.825 | 0.855 | 0.625 | 0.607 | 0.876 | 0.690 | 0.818 | 80.61 | 54.95 | 70.61 | 1.50 |

**Research papers (Exps 3–4):**

| Exp | Paper | Comment | Paper AP R11 (moderate) Car / Ped / Cyc |
|---|---|---|---|
| 3 | [PointPillars — Lang et al., CVPR 2019](https://arxiv.org/pdf/1812.05784) | Paper uses R11 not R40; evaluated on test split | 74.99 / 43.53 / 59.07 |
| 4 | [PointRCNN — Shi et al., CVPR 2019](https://arxiv.org/pdf/1812.04244) | Paper uses R11 not R40; evaluated on test split | 75.76 / 41.78 / 59.60 |

> **Section summary:** PointRCNN is the stronger detector across all three classes with its clearest margins on Cyclist AP (70.61% vs 63.00%) and Pedestrian AP (54.95% vs 51.46%), at the cost of 2.4× slower inference than PointPillars. Both deep learning models represent a step-change over DBSCAN, which produces near-zero results on Pedestrians and Cyclists. MOTP is consistently lower for Pedestrians than Cars across both models, reflecting the difficulty of precisely localising sparse small-object point clusters in 3D space. PointRCNN detections are carried forward as the fixed detection input for all tracking experiments.

---

## Multi-Object Tracking Comparison (Experiments 5–8)

This study evaluates four multi-object trackers on the same pre-saved PointRCNN detections, isolating each tracker's contribution independently of the detector. All trackers are evaluated on KITTI tracking sequences 0017–0020 using MOTA, MOTP and IDSW computed in BEV LiDAR space with the motmetrics library.

All trackers use the same pre-saved PointRCNN detections as input.

| Exp | Experiment | Method | Colab |
|---|---|---|---|
| 5 | Kalman Filter Tracking | Kalman Filter + Hungarian Matching | [Open](https://colab.research.google.com/drive/1wMHedOFLWQJVgfbJMjurpqWhO9a2rNm4) |
| 6 | SORT Tracker | Simple Online Realtime Tracking | [Open](https://colab.research.google.com/drive/1vNQhPNvX_3q93CH3A__tIsw_LzsyypvO) |
| 7 | DeepSORT Tracker | Motion + Appearance-based tracking | [Open](https://colab.research.google.com/drive/1iwNtfPfrXKAA-s8G8OMQAjXlWh-Q5Xgt) |
| 8 | ByteTrack | Confidence-based tracking | [Open](https://colab.research.google.com/drive/1h4NJVcZDIIvjw4WGftzKz1vhrWGCnA4w) |

> **Exp 5:** A 3D Kalman filter with a 10-dimensional state vector and constant-velocity motion model is combined with Hungarian assignment on a BEV IoU cost matrix using shapely oriented polygon intersection. Tracking is run per-class separately (Car, Pedestrian, Cyclist) with a confidence threshold of 0.5, achieving Overall MOTA 0.5951 with 85 total identity switches at 109.2 FPS — this serves as the motion-only baseline for the series.

> **Exp 6:** The official SORT library (abewley/sort, max_age=1, iou_threshold=0.1) is applied with all classes combined in a single tracker instance, using axis-aligned BEV box approximations that ignore object heading angle. Despite running at 601 FPS, SORT produces the most identity switches (255) of all four trackers, with Cyclist MOTA dropping to 0.1491 due to the heading-agnostic approximation and cross-class confusion from its single-tracker design.

> **Exp 7:** DeepSORT (nwojke/deep_sort) is extended with a custom ResNet50 appearance extractor (ImageNet weights, 2048-d L2-normalised embeddings) using per-class tracker instances (max_cosine_distance=0.4, max_age=30) and real image crops projected from 3D LiDAR boxes. Appearance features deliver the best Pedestrian MOTA (0.6160) and Cyclist MOTA (0.5257) in the series at 12.16 FPS — the lowest speed due to per-frame ResNet50 inference overhead.

> **Exp 8:** ByteTrack (ifzhang/ByteTrack, track_thresh=0.75, track_buffer=50, match_thresh=0.8) runs a dual-stage association: high-confidence detections matched first, then unmatched tracks are offered remaining low-confidence detections to recover partially occluded objects. It achieves the best Overall MOTA (0.6484) and the lowest total IDSW (50) in the entire series without any appearance model, running at 881 FPS.

**Results — MOTA, MOTP, IDSW (Car / Pedestrian / Cyclist / Overall), FPS:**

| Exp | Tracker | Car MOTA | Ped MOTA | Cyc MOTA | Overall MOTA | Car MOTP | Ped MOTP | Cyc MOTP | Overall MOTP | Car IDSW | Ped IDSW | Cyc IDSW | Total IDSW | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | Kalman | 0.6225 | 0.5785 | 0.3521 | 0.5951 | 0.7177 | 0.4569 | 0.7164 | 0.6069 | 31 | 54 | 0 | 85 | 109.2 |
| 6 | SORT | 0.6263 | 0.5141 | 0.1491 | 0.5622 | 0.7305 | 0.4306 | 0.7819 | 0.6065 | 91 | 159 | 5 | 255 | 601 |
| 7 | DeepSORT | 0.6600 | 0.6160 | 0.5257 | 0.6406 | 0.7300 | 0.4380 | 0.7849 | 0.6019 | 46 | 67 | 3 | 116 | 12.16 |
| 8 | ByteTrack | 0.6900 | 0.6180 | 0.3667 | 0.6484 | 0.7303 | 0.4311 | 0.7822 | 0.6017 | 26 | 24 | 0 | 50 | 881.1 |

> **Section summary:** ByteTrack achieves the best Overall MOTA (0.6484) and fewest identity switches (50) at 881 FPS without any appearance model, making it the dominant result of the study. DeepSORT delivers the strongest Pedestrian and Cyclist MOTA through appearance-guided re-identification but at only 12 FPS. SORT, despite running second-fastest, produces the worst MOTA (0.5622) and most identity switches (255) due to its single class-agnostic design — the custom Kalman with per-class matching substantially outperforms SORT despite using the same algorithm. MOTP is consistent across all four trackers for the same class, confirming that localisation precision is bottlenecked by detector quality rather than tracker choice.

---

## 4D Perception Pipeline Comparison (Experiments 9–12)

This study integrates PointRCNN detection and tracking in a single live loop on every frame, rather than using pre-saved detections. A confidence threshold of 0.5 is applied per class. The purpose is to measure how temporal tracking affects detection stability and identity consistency in a realistic end-to-end pipeline.

PointRCNN detection and tracking run as a single integrated pipeline (confidence threshold = 0.5).

| Exp | Experiment | Method | Colab |
|---|---|---|---|
| 9 | Detection Only Pipeline | Frame-wise PointRCNN detection | [Open](https://colab.research.google.com/drive/1upMul-CBntwqV18Gr7lrfiJKwVR1aAJW) |
| 10 | Detection + Kalman Pipeline | Detection with Kalman tracking | [Open](https://colab.research.google.com/drive/1ESeumu8CXs-DZb83GnjFBueBPA8wAy1g) |
| 11 | Detection + SORT Pipeline | Detection with SORT tracking | [Open](https://colab.research.google.com/drive/1ndUu_Gv3KO3VphEkOLN1bQY3QcEMAtc9) |
| 12 | Detection + DeepSORT Pipeline | Detection with DeepSORT tracking | [Open](https://colab.research.google.com/drive/1KBX1E4SAW-Q_pnEP9HvcfMUGczkWU5Lz) |

> **Exp 9:** PointRCNN is run frame-by-frame without any tracking at confidence threshold 0.5. Detection precision is strong across all classes (Car 0.8747, Pedestrian 0.8038, Cyclist 0.8975), establishing the live pipeline detection baseline. No MOTA or MOTP figures are recorded as there is no temporal identity assignment.

> **Exp 10:** Live PointRCNN detections are passed to the Kalman + Hungarian tracker on every frame in a single continuous loop. Overall MOTA reaches 0.6027 with MOTP 0.6057, demonstrating that adding motion tracking improves temporal consistency. Cyclist tracking remains limited (MOTA 0.3423) due to PointRCNN's 60.7% Cyclist recall causing frequent missed detections entering the tracker.

> **Exp 11:** Live PointRCNN detections are fed to the SORT tracker in the integrated pipeline. This pipeline produces the weakest tracking of the three with Overall MOTA 0.5662, and Cyclist MOTA collapses to just 0.0954, consistent with SORT's heading-agnostic BEV approximation and class-agnostic design performing poorly on non-car classes.

> **Exp 12:** Live PointRCNN detections combined with DeepSORT appearance tracking form the most complete integrated pipeline. It achieves the best Overall MOTA (0.6402) and strongest Cyclist MOTA (0.5012) among the three integrated trackers, with 101 total identity switches — the only pipeline experiment with IDSW recorded.

**Detection Results — Precision, Recall (Car / Pedestrian / Cyclist):**

| Exp | Pipeline | Car Prec | Ped Prec | Cyc Prec | Car Rec | Ped Rec | Cyc Rec |
|---|---|---|---|---|---|---|---|
| 9 | Detection Only | 0.8747 | 0.8038 | 0.8975 | 0.8024 | 0.6948 | 0.6210 |
| 10 | Det + Kalman | 0.8270 | 0.7863 | 0.5611 | 0.8398 | 0.7544 | 0.7188 |
| 11 | Det + SORT | 0.8248 | 0.7881 | 0.5475 | 0.8383 | 0.7540 | 0.7188 |
| 12 | Det + DeepSORT | 0.8242 | 0.7858 | 0.5646 | 0.8395 | 0.7492 | 0.7262 |

**Tracking Results — MOTA, MOTP (Car / Pedestrian / Cyclist / Overall):**

| Exp | Pipeline | Car MOTA | Ped MOTA | Cyc MOTA | Overall MOTA | Car MOTP | Ped MOTP | Cyc MOTP | Overall MOTP |
|---|---|---|---|---|---|---|---|---|---|
| 10 | Det + Kalman | 0.6281 | 0.5895 | 0.3423 | 0.6027 | 0.7172 | 0.4563 | 0.6949 | 0.6057 |
| 11 | Det + SORT | 0.6310 | 0.5208 | 0.0954 | 0.5662 | 0.7303 | 0.4301 | 0.7837 | 0.6049 |
| 12 | Det + DeepSORT | 0.6692 | 0.6157 | 0.5012 | 0.6402 | 0.7305 | 0.4388 | 0.7860 | 0.6029 |

**IDSW recorded for Exp 12 only — Car: 46 / Pedestrian: 54 / Cyclist: 1 / Total: 101**

> **Section summary:** All three integrated pipelines replicate the MOTA ordering from Study 2 — Det+DeepSORT > Det+Kalman > Det+SORT. Detection precision remains high across all pipelines (Car >0.82), showing tracker feedback does not degrade detection quality at threshold 0.5. The Cyclist class remains the hardest to track in the live pipeline setting — SORT produces near-zero Cyclist MOTA (0.0954) reflecting the combined effect of low PointRCNN Cyclist recall (60.7%) and SORT's heading-agnostic design. DeepSORT's Cyclist MOTA of 0.5012 is the strongest pipeline result, highlighting that appearance-based re-identification provides its largest relative benefit on the rarest and most geometrically distinctive class.

---
## Disclaimer
The content of this repository is provided for academic and research purposes only. The results and conclusions presented are based on specific models and techniques as detailed in the thesis. While every effort has been made to ensure the accuracy of the data and findings, variations may occur depending on the context and application of these methods. Users are advised to apply the information contained in this repository at their own discretion and risk.


---

## References

| Paper | Citation |
|---|---|
| KITTI | Geiger et al., *Are We Ready for Autonomous Driving? The KITTI Vision Benchmark Suite*, CVPR 2012 |
| PointPillars | Lang et al., *PointPillars: Fast Encoders for Object Detection from Point Clouds*, CVPR 2019 |
| PointRCNN | Shi et al., *PointRCNN: 3D Object Proposal Generation and Detection from Point Cloud*, CVPR 2019 |
| CLEAR MOT | Bernardin & Stiefelhagen, *Evaluating Multiple Object Tracking Performance: The CLEAR MOT Metrics*, EURASIP 2008 |
| AB3DMOT | Weng et al., *AB3DMOT: A Baseline for 3D Multi-Object Tracking and New Evaluation Metrics*, IROS/ECCVW 2020 |
| SORT | Bewley et al., *Simple Online and Realtime Tracking*, ICIP 2016 |
| DeepSORT | Wojke et al., *Simple Online and Realtime Tracking with a Deep Association Metric*, ICIP 2017 |
| ByteTrack | Zhang et al., *ByteTrack: Multi-Object Tracking by Associating Every Detection Box*, ECCV 2022 |

---
