# Temporal action spotting and localisation for fast, dense, frame-precise events (state of the art, Nov 2026)

Conventions in these notes:
- **Tolerance:** δ is in frames of a 25-30 fps video unless stated. The current system's ±4 frames at 60 fps is ±67 ms, which is about δ=2 at 30 fps (δ=2 is ±67-80 ms at 30-25 fps). δ=2 columns are the closest published match; δ=0 and δ=1 are stricter than the current metric.
- **Data:** every number below is on real sports or real-world video unless stated. None is on fighting-game video.
- **[OLDER]** marks work from before 2023.
- **How the numbers were read:** paper tables were read through a fetch tool that summarises pages, not checked by eye against the PDFs. Check any figure against the paper before relying on it for a decision.

## What do the best recent methods do for precise spotting, and how much does end-to-end gain over frozen features, at what cost?

### Takeaway
All the leading precise-spotting methods share one recipe: a tiny 2D CNN (RegNet-Y 200-800MF) with temporal shift modules, run on every frame at 224-448 px, trained end-to-end, with a light sequence model and a per-frame softmax. End-to-end beat frozen Kinetics features by about 6-17 mAP at δ=2 in the 2022 study, but only by about -1 to +6 mAP over features fine-tuned on the same dataset (+6 on FineGym, under 2 elsewhere), and the differences between recent end-to-end models at δ=1-2 are about as small as the disagreement between papers re-running the same model.

### Cited Findings

**Method table: precise spotting (end-to-end, per-frame)**

| Method (year) | Backbone / temporal | Input | Key measured result, mAP δ=1 / δ=2 unless stated | Cost | Code |
|---|---|---|---|---|---|
| E2E-Spot (2022) [OLDER] | RegNet-Y 200/800MF + GSM, 1-layer BiGRU, per-frame K+1 softmax | 224 px height, 25-30 fps, 100-frame clips, batch 8 | 200MF: Tennis 96.1/97.7, FS-Comp 81.0/93.5, FS-Perf 85.1/95.7, FineDiving 68.4/85.3, FineGym-Full 47.9/65.2 | About 8.5 min per cycle (200MF) and 14 min (800MF) on one A5000, for 50 cycles (100-150 on FineGym); 0.3-0.6 ms/frame inference | [Project page](https://jhong93.github.io/projects/spot.html) |
| T-DEED (CVPRW 2024) | RegNet-Y 200/800MF + GSF on the latter half, encoder-decoder with SGP layers and skip connections, displacement head | 224×224 crops, L=100, batch 8, 50 epochs of 5,000 clips | 800MF: FS-Comp 84.77/92.86, FS-Perf 88.17/95.87, FineDiving 73.23/88.88 | 16.42M params, 21.97 GFLOPs (200MF); 64.26M, 86.34 GFLOPs (800MF), [per AdaSpot](https://arxiv.org/html/2602.22073) | [Code](https://github.com/arturxe2/T-DEED) |
| ASTRM (Santra et al., arXiv Feb 2025) | RegNet-Y 200MF + spatial/temporal refinement, SoftIC contrastive loss | 128-frame clips; spatial size not stated | Tennis 96.21/97.75, FS-Comp 86.67/95.53, FS-Perf 89.57/97.38, FineGym 52.31/66.57; δ=0 Tennis 61.01 | 6.46M params, 57.84 GFLOPs | None listed |
| MFS (arXiv Jul 2025) | GSM + multi-scale shifts (δ 1,2,3) + grouped focus; drop-in for GSM, adds about 0.06M params | L=100, batch 8, 50 epochs on one L40S | E2E-Spot+MFS: Tennis 91.34/96.51, FS-Comp 77.72/90.04, FS-Perf 84.12/96.90, FineDiving 71.03/87.26, Table Tennis Australia 69.50/78.12 | 4.51M params, 45.06 GFLOPs | [Anonymous repo](https://anonymous.4open.science/r/MSAGSM-B0A4) |
| F3ED (ICLR 2025) | TSM on RegNet-Y 200MF, BiGRU, per-frame binary localiser, multi-label element classifier, BiGRU context module | 224×224, 96 frames at stride 2, batch 4 | F3Set high granularity (1,108 classes), ±1 frame: F1evt 40.3, F1elm 75.2, Edit 74.0; best baseline 31.4/71.4/68.7 | RTX 4090, about 10 min/epoch, 50 epochs; 5.6M params | [Code](https://github.com/F3Set/F3Set) |
| AdaSpot-s / -b (CVPR 2026) | Two RegNet-Y 200/400MF + GSF branches (low-res full frame, 112×112 RoI picked by training-free saliency), max-pool fusion, BiGRU, per-frame softmax, Soft-NMS | Video 796×448; high-res input is a 448×448 centre crop on Tennis/FineGym/F3Set, low-res is half that; L=100, batch 4 | -b: Tennis 97.36/97.79, FineDiving 72.00/87.45, FineGym 54.66/65.24, F3Set (365 classes) 69.37/69.94; δ=0: Tennis 74.02, F3Set 55.38 | RTX 6000 Ada; 7.58M params, 29.78 GFLOPs (-s); 10.63M params (-b; GFLOPs reads 56.78 in the main text and 36.78 in the appendix as extracted); 1.97 GB for single-clip inference | [Code, MIT, checkpoints](https://github.com/arturxe2/AdaSpot) |
| Dynamic label assignment (arXiv Apr 2025) | RegNet-Y + GSM, DETR-style queries, Hungarian matching on class and time | RegNet-Y 800MF results shown | Tennis 96.61/98.07, FineDiving 72.42/90.14, FG-Full 56.55/71.05; on noisy labels (σ=2.0, Tennis) gains of 19.79/9.00 over existing methods | Not stated | None listed |
| TFD + TGS (ACM MM 2026) | ViT-T/S with a multi-scale gated shift in each encoder block, EMA teacher feature distillation | Resolution not stated | δ=1 at 10% labels: Tennis 80.19, FS-Perf 50.29, FineDiving 52.87; at 80% labels Tennis 93.04 | Training details limited | [Anonymous repo](https://anonymous.4open.science/r/TFD-8535) |

Sources for the table: [E2E-Spot full text](https://ar5iv.labs.arxiv.org/html/2207.10213); [T-DEED](https://arxiv.org/html/2404.05392v2); [ASTRM](https://arxiv.org/html/2503.00147); [MFS](https://arxiv.org/html/2507.07381); [F3Set/F3ED](https://arxiv.org/html/2504.08222v2); [AdaSpot](https://arxiv.org/html/2602.22073); [Dynamic label assignment](https://arxiv.org/html/2504.00149); [TFD](https://arxiv.org/html/2607.10998v1). Venue of TFD per a [search listing](https://arxiv.org/abs/2607.10998).

**Same model, different papers: the comparison noise is large**
- E2E-Spot 200MF on Tennis at δ=0 is reported as 71.6 by its [authors](https://ar5iv.labs.arxiv.org/html/2207.10213), 69.78 in [AdaSpot](https://arxiv.org/html/2602.22073), 51.42 in [MFS](https://arxiv.org/html/2507.07381) and 45.34 in [ASTRM](https://arxiv.org/html/2503.00147).
- T-DEED 200MF on FineDiving at δ=1 is 71.48 in its [own paper](https://arxiv.org/html/2404.05392v2) and 71.07 in [AdaSpot](https://arxiv.org/html/2602.22073), while the T-DEED + GSM row in [MFS](https://arxiv.org/html/2507.07381) reads 66.18. That MFS row is not strictly the same model, because T-DEED ships with GSF, though the MFS T-DEED + GSF row is also only 67.04.
- T-DEED 200MF on Tennis at δ=0: 56.00 in [AdaSpot](https://arxiv.org/html/2602.22073), 48.82 (GSM) and 50.08 (GSF) in [MFS](https://arxiv.org/html/2507.07381), 43.49 in [ASTRM](https://arxiv.org/html/2503.00147).

**End-to-end vs frozen, measured on the same data ([E2E-Spot, 2022](https://ar5iv.labs.arxiv.org/html/2207.10213)) [OLDER], E2E-Spot 200MF against the best row of each feature family, mAP δ=1 / δ=2**

| Dataset | E2E-Spot 200MF | Best frozen Kinetics (MViT-B or I3D, no fine-tuning) | Best fine-tuned features (TSP or VC, then head) |
|---|---|---|---|
| Tennis | 96.1 / 97.7 | 67.0 / 80.8 | 92.1 / 96.2 |
| FS-Comp | 81.0 / 93.5 | 60.8 / 81.8 | 77.7 / 94.1 |
| FS-Perf | 85.1 / 95.7 | 69.0 / 89.3 | 80.2 / 94.5 |
| FineDiving | 68.4 / 85.3 | 59.3 / 78.3 | 63.2 / 83.5 |
| FineGym-Full | 47.9 / 65.2 | 31.0 / 48.6 | 41.1 / 58.8 |

The best δ=1 and δ=2 values within a column can come from different rows.

**End-to-end vs frozen, other measurements**
- AdaTAD (CVPR 2024): 75.4% average mAP on THUMOS14 with VideoMAEv2-giant end-to-end against a previous best feature-based result of 71.5%. Only the temporal-informative adapters inside the backbone are trained; an adapter outside the backbone falls from 71.5 to 70.2 in one VideoMAE-Base setting. Inputs up to 1,536 frames. [Code](https://github.com/sming256/AdaTAD), [paper](https://arxiv.org/pdf/2311.17241v2.pdf). These come from search excerpts, and total GPU memory was not retrieved. THUMOS14 is interval localisation of long actions, not spotting.
- TimeLoc (arXiv Mar 2025; a figure filename suggests ICCV 2025): frozen vs end-to-end VideoMAE gives Kinetics-GEBD F1 89.7 vs 91.5 and Charades-STA R1@0.5 64.6 vs 67.28. Its THUMOS14 average mAP is 75.0 with VideoMAE-L at 224 against 74.4 for AdaTAD with VideoMAE-H. Memory on an H100-80G at 224: a 9,216-frame window took 66.5 GB without temporal checkpointing and 21.1 GB with it, at 6.5 s/iter. Default resolution is 160×160. [Paper](https://ar5iv.labs.arxiv.org/html/2503.06526), [code (release stated as planned)](https://github.com/sming256/TimeLoc).
- V-JEPA 2 facts: tubelets are 2×16×16, ViT-L has about 300M parameters, evaluation at 256×256 uses a frozen encoder with a 4-layer attentive probe. [Paper](https://ar5iv.labs.arxiv.org/html/2506.09985), [code](https://github.com/facebookresearch/vjepa2). The 7.5 Hz token rate from 15 fps input follows from that tubelet size.
- ViT cost context: the MFS paper lists a ViT at 3,522 GFLOPs against 42-46 GFLOPs for the RegNet-Y models ([MFS](https://arxiv.org/html/2507.07381)).

**Challenge results**
- SoccerNet 2025 Team Ball Action Spotting (tolerance 1 s, not frame precision): winner 60.03 Team-mAP@1 against a T-DEED baseline at 51.72. The winner kept T-DEED, merged the heads into one joint team-action head, pretrained on 500+ broadcast matches then fine-tuned, added horizontal mirroring with label swap, camera-pan and brightness augmentation, and tuned learning rate, focal-loss weights and Soft-NMS with Bayesian optimisation. Second place merged the heads too and scored 56.78. [Report](https://arxiv.org/html/2508.19182v1). The report calls T-DEED "last year's winning method".
- SoccerNet 2026 replaced that task with Player-Centric Ball Action Spotting (±12 frames, Macro F1, FOOTPASS data): winner PAVE 58.94 against a baseline of 46.41, using ROI-aligned X3D player features per tracklet, a temporal transformer and a 4-model ensemble with agreement filtering. Ball Action Anticipation winners used RegNetY-GSF at 448p; one 21.14 entry pretrained at 224p and fine-tuned at 720p. [Report](https://arxiv.org/html/2607.07320v1).
- AdaSpot on SoccerNet Ball (seven matches) at 0.5 s / 1 s: AdaSpot-b 56.24/59.82, E2E-Spot 800MF 54.49/58.65, T-DEED 800MF 49.39/53.11 ([AdaSpot](https://arxiv.org/html/2602.22073)).

**Interval localisation family (ActionFormer, TriDet, TemporalMaxer, DyFADet, frozen features)**
- THUMOS14 average mAP on I3D: ActionFormer 66.8, TriDet 69.3, DyFADet 69.2. On VideoMAEv2-g: 69.6, 70.1, 70.5. FineAction on VideoMAEv2-g: ActionFormer 18.2, DyFADet 23.8. Encoders are small (ActionFormer 0.57M, DyFADet 0.49M params; 29.2 and 18.8 ms on an RTX 4090). [DyFADet](https://arxiv.org/html/2407.03197v1), [code](https://github.com/yangle15/DyFADet-pytorch).
- TemporalMaxer reports 67.7 average mAP on THUMOS14 ([paper, search excerpt only](https://arxiv.org/pdf/2303.09055)).
- Temporal Zoom Networks (Nov 2025) reports 58.2 average mAP on THUMOS14 over tIoU 0.3-0.7 and treats ActionFormer++ as baseline. That is below the numbers above, so it may not be comparable ([paper, search excerpt only](https://arxiv.org/pdf/2511.03943)).
- Direct test on spotting data: on F3Set high granularity, ±1 frame, with TSM features, ActionFormer scored F1evt 22.4 against 31.4 for the E2E-Spot head and 40.3 for F3ED. With SlowFast features ActionFormer (28.7) beat the E2E-Spot head (25.9) ([F3Set](https://arxiv.org/html/2504.08222v2)).
- T-DEED's ablation found a TriDet-style feature pyramid at 68.31 (FS-Comp) and 64.28 (FineDiving), against 74.29 / 63.01 for no skip connections and 77.96 / 63.67 for its SGP-Mixer skips, δ=1 ([T-DEED](https://arxiv.org/html/2404.05392v2)).

### Inferences
- At the current ±67 ms metric, swapping one recent end-to-end architecture for another is unlikely to help much, because published δ=2 differences are about 1-2 mAP and smaller than between-paper disagreement. AdaSpot's real headline gains are at δ=0.
- The 2022 frozen-vs-end-to-end gap may overstate what a V-JEPA 2 system would gain. Its frozen features were Kinetics models on sports. No study measures frozen V-JEPA 2 against end-to-end on spotting.
- The better-supported comparison is that a head trained on features fine-tuned for the domain is close to end-to-end at δ=2 (FS-Comp 94.1 vs 93.5, FineDiving 83.5 vs 85.3), so a RegNet-Y branch fine-tuned on Strive and then frozen is a cheap intermediate experiment.
- A 24 GB RTX 3090 should fit the RegNet-Y class. The published runs used batch 4-8 of about 100 frames on an A5000, RTX 4090, L40S and RTX 6000 Ada. Fine-tuning a ViT-L end-to-end at comparable temporal density looks much harder on this card, judging by the TimeLoc memory figures on an 80 GB card; adapters (AdaTAD) or checkpointing would be needed. I found no measured RTX 3090 figures.
- About 1,700 replays should be enough to train these from ImageNet initialisation. F3Set has 114 matches and 11,584 clips, Tennis 28 matches, so the replay count is not the limit, but rare classes among the 1,215 may be.

### Gaps
- No published measurement of V-JEPA 2 (or 2.1) features on any spotting or THUMOS-style benchmark was found. V-JEPA 2.1 is used frozen for [anticipation in 2026](https://arxiv.org/pdf/2606.00694), which I saw only as a search excerpt.
- Exact AdaTAD memory use and the leaderboard method details of many SoccerNet teams (fps, resolution) are not stated or were not retrieved.
- SoccerNet 2024 report ([arXiv 2409.10587](https://arxiv.org/pdf/2409.10587)) could not be parsed; the 2024 win by T-DEED rests on the 2025 report's wording.

## How is ±1-2 frame precision obtained, and how are events a few frames apart handled?

### Takeaway
Precision comes from having a feature and an output for every frame, plus a sequence model. In-backbone temporal shifts, flow and exotic modules add much less, and higher spatial resolution helps. Published methods separate close events with per-frame outputs, a small label radius, a displacement head and Soft-NMS, but no paper I found reports recall as a function of the gap between events.

### Cited Findings

**Per-frame features and sequence model ([E2E-Spot ablations](https://ar5iv.labs.arxiv.org/html/2207.10213)) [OLDER], mAP δ=1, E2E-Spot 200MF**

| Variant | Tennis | FS-Comp | FS-Perf | FineDiving | FineGym-Full |
|---|---|---|---|---|---|
| Full | 96.1 | 81.0 | 85.1 | 68.4 | 47.4 (47.9 in the main table) |
| No GRU | 95.7 | 74.3 | 77.9 | 64.1 | 32.9 |
| No GSM | 94.1 | 75.5 | 85.6 | 68.9 | 44.2 |
| No GSM, no GRU (single frame) | 60.1 | 26.9 | 41.1 | 47.0 | 22.1 |
| TSM instead of GSM | 96.1 | 78.6 | 83.3 | 65.3 | 48.1 |
| 112 px instead of 224 | 88.5 | 75.4 | 80.9 | 64.9 | 45.3 |
| 2-stream RGB + flow | 96.3 | 82.2 | 85.1 | 70.1 | 49.0 |

- The same paper says sparse sampling is ineffective for precise spotting and uses dense per-frame sampling. It has no frame-rate ablation ([E2E-Spot](https://ar5iv.labs.arxiv.org/html/2207.10213)).
- F3Set states that clip-wise extraction with 6-frame segments gives temporally coarse features and weaker precise detection, while frame-wise extraction allows end-to-end training over hundreds of frames ([F3Set](https://arxiv.org/html/2504.08222v2)). That is the closest published evidence on a coarse temporal stride. The table with its values was not in the extract.
- All the benchmarks above are 25-30 fps. FineGym is resampled to that range ([E2E-Spot](https://ar5iv.labs.arxiv.org/html/2207.10213)).

**Multi-scale temporal fusion**
- T-DEED, δ=1 (FS-Comp / FineDiving), ablations at an intermediate configuration: Transformer 69.68/59.84, GRU 72.04/59.09, SGP 73.14/60.07. Skip connections: none 74.29/63.01, SGP-Mixer 77.96/63.67 ([T-DEED](https://arxiv.org/html/2404.05392v2)).
- MFS over GSM on E2E-Spot at δ=1: Tennis 90.03 → 91.34, FS-Perf 81.27 → 84.12, FineDiving 67.89 → 71.03, TTA 66.42 → 69.50, at +1.3% params ([MFS](https://arxiv.org/html/2507.07381)).
- Clip length: T-DEED peaks at 100 frames (25: 71.02/66.61; 100: 81.25/68.40; 200: 79.19/65.29; [T-DEED](https://arxiv.org/html/2404.05392v2)). E2E-Spot keeps rising slightly to 250 ([E2E-Spot](https://ar5iv.labs.arxiv.org/html/2207.10213)).

**Spatial resolution**
- F3ED, F1evt at high granularity: 224 px 40.3, 336 px 43.2, 448 px 44.4. The table lists identical FLOPs at all three, so its FLOPs column looks wrong ([F3Set](https://arxiv.org/html/2504.08222v2)).
- AdaSpot's two-branch full model on Tennis δ=0/1/2 and SN-BAS 0.5 s/1 s: 73.30/96.90/97.47 and 53.02/56.43, against 71.18/96.73/97.42 and 47.98/51.69 for low-res only, and 71.91/96.62/97.26 and 52.13/55.95 for high-res only. RoI crop size varies little between 56 and 224 px ([AdaSpot](https://arxiv.org/html/2602.22073)). AdaSpot adds about 6 GFLOPs for the 112×112 RoI branch.

**Close events and post-processing**
- T-DEED labels frames within a radius r_E of an event (1 frame on FigureSkating, 2 on FineDiving). Displacement ablation, δ=1 FS-Comp/FineDiving: no displacement, dilation 0: 74.74/63.67; dilation 1: 77.96/61.97; displacement with r_E=1: 78.20/67.49; r_E=2: 77.12/68.31 ([T-DEED](https://arxiv.org/html/2404.05392v2)).
- Soft-NMS over NMS in T-DEED: 81.27 → 85.15 (FS-Comp) and 68.40 → 71.48 (FineDiving) ([T-DEED](https://arxiv.org/html/2404.05392v2)). AdaSpot uses Soft-NMS windows of 2 frames for precise spotting and 12 for coarse ([AdaSpot](https://arxiv.org/html/2602.22073)).
- Uncertainty-aware displacement in ASTRA (frozen-feature, 1-s scale soccer): tight 66.09 → 66.63, a small gain ([ASTRA](https://arxiv.org/html/2404.01891v1)).
- DETR-style query matching on clean labels stays within about 3 mAP of T-DEED, in either direction, but gains 19.79 (δ=1) and 9.00 (δ=2) on Tennis with σ=2.0 label noise ([Dynamic label assignment](https://arxiv.org/html/2504.00149)).
- Density: F3Set has about one event per second, under 3% of frames contain events, and usually fewer than 20 events per 96-frame clip ([F3Set](https://arxiv.org/html/2504.08222v2)). The MFS authors introduce table tennis as a denser PES benchmark (39 matches, 4,878 events), and for E2E-Spot it has the lowest δ=2 scores in the MFS table, 76.90 (GSM) to 78.12 (MFS) ([MFS](https://arxiv.org/html/2507.07381)). The density claim itself is from a [search excerpt](https://arxiv.org/html/2507.07381v3).
- F3ED's event sequence context (BiGRU over predicted events) took high-granularity scores from 37.9/74.3/71.7 to 40.3/75.2/74.0; a Transformer context gave 39.0/74.3/72.8 ([F3Set](https://arxiv.org/html/2504.08222v2)).

### Inferences
- **The system's 47% recall on starts under 8 frames apart lines up with its token spacing.** At 7.5 Hz one token spans 133 ms, which is 8 frames at 60 fps. Two starts under 8 frames apart can land in one token. A single class plus one 8-way offset per side and token can only output one of them, so part of that loss may be structural. Check first whether it is: count how many ground-truth starts share a token per side, which gives an upper bound on recall independent of any model. Also check whether NMS removes close pairs. Cross-side pairs should not collide, since each side has its own head.
- The literature suggests fixing the output and feature rate before anything else. Removing per-frame temporal reasoning loses a great deal, while choice of shift module moves mAP 1-3. Possible orders of work: (a) re-extract V-JEPA 2 at 30 fps to get 15 Hz tokens, at a cost of double the feature storage and with V-JEPA 2 now seeing different motion speeds than it was trained on; (b) add a per-frame RegNet-Y 200/400MF + GSM/GSF branch at 30 or 60 fps and fuse it with interpolated V-JEPA tokens; (c) keep the rate and predict up to two events per token. These are my suggestions, not tested results.
- 60 fps may be worth testing for this data specifically: the game animates at 60 Hz, there are no published results above 30 fps, and game video has less motion blur. A 100-frame clip at 60 fps covers only 1.67 s, while published windows of 100 frames at 25-30 fps cover 3.3-4 s and the T-DEED/E2E-Spot clip-length results show that context matters, so the clip would need to grow or a stride be used.
- Soft-NMS with a small window is a cheap test against the current post-processing.
- Label noise methods probably add little if labels come from the game memory logger, since the DLA gain concentrates on noisy labels.

### Gaps
- I found no study varying input fps for spotting, no 60 fps benchmark, and no paper with recall by inter-event gap. The user's own harness will have to measure this.
- No published results on two events in one output bin, or on multi-instance offset heads for spotting.

## What works for fine-grained action classes that look alike?

### Takeaway
The strongest evidence is F3Set (1,108 classes on real tennis): factorising a flat class into separate element predictions gave +6.5 F1evt at high granularity on its own, and context added +2.4 more. Input resolution, balanced mixup and mirror-with-label-swap also have measured gains, each smaller. Whether the same transfers to whiffs and throws is untested.

### Cited Findings
- F3ED, high granularity (F1evt/F1elm/Edit): TSM + E2E-Spot 31.4/71.4/68.7; plus multi-label 37.9/74.3/71.7; plus BiGRU context 40.3/75.2/74.0. Scores rise as granularity falls: F1evt 40.3 (1,108 classes), 48.0 (365), 68.4 (38). The authors say flat classification over 1,000+ combinations reduces training efficiency and worsens long-tail bias, using the example of two events that differ only in shot technique ([F3Set](https://arxiv.org/html/2504.08222v2)).
- Localisation vs classification: on F3Set high granularity the localiser reaches F1 86.7 while the full event F1 is 40.3 ([F3Set](https://arxiv.org/html/2504.08222v2)).
- AdaSpot on F3Set (365 classes) uses per-category heads and multiplies category probabilities: 53.55-55.38 / 67.76-69.37 / 68.41-69.94 at δ=0/1/2, against 24.79/60.71/64.79 for F3ED, at 448 px base vs F3ED's 224 ([AdaSpot](https://arxiv.org/html/2602.22073)). That F3ED figure comes from the AdaSpot paper, and the input differs.
- Skeleton vs RGB on F3Set high: ST-GCN++ F1evt 25.4, PoseConv3D 20.1, RGB TSM + F3ED 40.3 ([F3Set](https://arxiv.org/html/2504.08222v2)). 2D-VPD pose + MS-TCN on figure skating matched RGB: 83.5/96.2 and 85.2/96.4 ([E2E-Spot](https://ar5iv.labs.arxiv.org/html/2207.10213)) [OLDER]. Skeleton-to-RGB few-shot distillation exists but its abstract reports no numbers ([arXiv 2604.22839](https://arxiv.org/abs/2604.22839)).
- Long tail: ASTRA's balanced mixup moved tight A-mAP from 63.97 (none) to 64.97 (standard) to 65.82 (balanced) on SoccerNet ([ASTRA](https://arxiv.org/html/2404.01891v1)). SoftIC contrastive + mixup is ASTRM's imbalance answer, and it reports no separate loss-only figure in the extract ([ASTRM](https://arxiv.org/html/2503.00147)).
- FineGym is the hardest public benchmark, with 32 classes, 80,166 events, about 52-57 at δ=1 and 64-71 at δ=2 across recent methods ([AdaSpot](https://arxiv.org/html/2602.22073), [DLA](https://arxiv.org/html/2504.00149)).
- Actor-centric: SoccerNet 2026's winner classifies per-player tracklets with ROI-aligned features ([report](https://arxiv.org/html/2607.07320v1)). Soccer, ±12 frames.
- Fine-grained THUMOS-style: FineAction average mAP rose from 18.2 (ActionFormer) to 23.8 (DyFADet) on the same features ([DyFADet](https://arxiv.org/html/2407.03197v1)).

### Inferences
- **A factorised head fits this problem well.** 1,215 classes likely decompose into character, then input (button, direction, stance or air, special or normal). Since characters are known for a replay, conditioning or masking classes by character may be a free gain. It is the single best-evidenced classification change, although the 6.5-point gain comes from a very different scene.
- F3Set's localiser-vs-full gap suggests measuring the system's "found a start, wrong class" share separately from "missed the start". Whiff recall of 57% is a localisation-side issue if whiffs are missed, and throws may be a class-side issue if labelled as something else.
- Whiffs may be weak because hit effects, HP drops and the HUD all accompany hits but not whiffs, and frozen V-JEPA tokens at 256 px may not resolve small sprite differences. That points to higher-res crops of the fighters, for which position could come from the existing memory logger when training. AdaSpot's RoI branch is a tested form of this, at low cost. Untested on game video.
- Mirror augmentation with side-swap labels is natural here, as long as the HUD is handled.
- Pose-based routes look poor value: skeleton methods lost to RGB on F3Set, and I assume no ready pose model exists for anime-styled characters (not checked).

### Gaps
- No paper specifically tackles detection of non-contact or failed actions (the whiff analogue), or very short, near-identical start-ups (the throw analogue).
- I found no per-class breakdown for AdaSpot; Appendix G was cut off in the extract.
- I did not retrieve the FineTennis dataset or a benchmark under that name. The nearest are Tennis and F3Set.

## Is there any work directly on esports, fighting games or game video event detection?

### Takeaway
I found no peer-reviewed work that spots moves from fighting-game video with frame precision, and no public benchmark for it. What exists is student projects, a patent without published measurements, and first-person-shooter highlight work, so this system appears to have no direct published comparison.

### Cited Findings
- Electronic Arts patent US11625880B2 (granted 2023): "Machine-learning models for tagging video frames". Fighting games are its main example, with contact tags and body-part sub-types. It uses a frame-level encoder (LSTM or conv) over keypoints, then an animation-level LSTM with per-frame softmax, or an alternative hierarchy of blank/non-blank, then tag type, then sub-type. It works from animation data with skeletal joints, not gameplay video. No quantitative result appears in the portion read ([patent](https://patents.google.com/patent/US11625880B2/en)); the final third of the page was not read.
- Stanford CS231n project (undated): CNNs and deep spatial autoencoders turn Super Smash Bros. Melee video re-emulated from Slippi replays into state representations of observations and player actions ([report, search excerpt only](https://cs231n.stanford.edu/papers/text_file_840589945-Parsing_Super_Smash_Bros__Melee_Frames.pdf)).
- Cal Poly thesis, 2018 [OLDER]: real-time tracking of four Melee characters, trained on 2,296 annotated frames ([thesis, search excerpt only](https://digitalcommons.calpoly.edu/theses/1845)).
- 2017 arXiv [OLDER]: pixel-based imitation on N64 Super Smash Bros. from about 128×128 frames as a 30-class classification ([paper, search excerpt only](https://arxiv.org/pdf/1702.05663.pdf)).
- Melee agents read memory or replay state rather than pixels ([Firoiu et al. 2017, search excerpt only](https://arxiv.org/pdf/1702.06230) [OLDER]; [MIMIC, search excerpt only](https://huggingface.co/erickfm/MIMIC/commit/a30bee705203a00ae103c62a3f11cf3bdedeacc5)).
- A 2025 paper classifies one-second clips of first-person-shooter gameplay as interesting or background with CLIP-style models on an in-house dataset ([arXiv 2505.07721, search excerpt only](https://arxiv.org/pdf/2505.07721)). That is coarse highlight detection, far from frame precision. A 2018 paper does unsupervised multi-view stream highlights ([arXiv 1807.09715](https://arxiv.org/pdf/1807.09715)) [OLDER].
- A May 2025 survey of sports event detection lists TAL, spotting and precise spotting datasets and methods ([survey](https://arxiv.org/html/2505.03991v3)). I did not read the whole survey for esports content, and my searches turned up no esports benchmark in it.

### Inferences
- The closest benchmark in structure is F3Set: over 1,000 classes, ±1 frame, two-sided play. Its 40.3 F1evt at about ±33-40 ms is not comparable with the system's 0.735 at ±67 ms, since data, tolerance and class distribution all differ. The system's score should not be read as beating or trailing that benchmark.
- Game video differs from sport in ways that favour this system and that sports results cannot capture: fixed camera framing, repeatable rendering, identical animations for a move, a 60 Hz animation clock, and labels that can come from game memory. So gains from augmentation and from robustness modules may not transfer, while rate and class-structure changes are more likely to.
- The EA hierarchy (blank/non-blank, type, sub-type) is the same idea as F3ED's localiser plus element classifier, reached independently for fighting-game animation.

### Gaps
- Searches for Guilty Gear, Street Fighter and Tekken move recognition returned no academic paper. Absence from about six searches is not proof that none exists, and non-English or hobbyist work (for example, GitHub projects) was not checked systematically.
- No esports or MOBA/Rocket League video event-detection benchmark turned up in my searches.
- The specific hit-confirm, whiff and throw distinction has no published evaluation.
