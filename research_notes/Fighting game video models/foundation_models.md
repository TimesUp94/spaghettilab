# Foundation video/image models, small-domain fine-tuning and CPU distillation (as of November 2026)

Scope note. Project numbers quoted (opener move classification, whole chain, old / new test replays): ResNet18+ResNet50 ensemble 0.887 / 0.855; V-JEPA 2 ViT-L last 8 blocks fine-tuned 0.869 / 0.835 (4 epochs, NaN in epoch 2); frozen V-JEPA 2 0.844 / 0.792; frozen DINOv2-B 0.811 / 0.764; frozen SigLIP 2-B 0.805 / 0.760. These come from the assignment brief, not from web sources. Differences computed from them are arithmetic on those values.

Source quality. Several numbers come from search snippets because PDFs could not be parsed; those are marked "(snippet)". Everything else was read from the page itself. Numbers across different papers use different protocols and are not directly comparable unless stated.

## Which backbones lead now for video understanding, and which do well when fine-tuned on small datasets?

### Takeaway
V-JEPA 2 / 2.1 lead the published frozen-probe results, and the distilled V-JEPA 2.1 ViT-B (80M parameters) scores above the undistilled ViT-L in Meta's tables, which makes it the most interesting new candidate. No source compares these backbones after fine-tuning on a small dataset, and none uses game footage, so the project's own measurements carry more weight than any published ranking. Licences rule out the Hiera video weights for SaaS, and they leave VideoMAE V2 unconfirmed.

### Cited Findings

**Frozen probe results, V-JEPA family**
- V-JEPA 2 attentive probe, frozen: ViT-L/16 73.7% SSv2, 89.0% Diving48, 32.7 R@5 EK100; ViT-g/16 at 384: 77.3% SSv2, 90.2% Diving48, 39.7 R@5 EK100 — [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2)
- Same page gives prior bests of 69.7% SSv2 and 86.4% Diving48 for InternVideo2-1B (probe) — [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2)
- V-JEPA 2.1 (released 2026-03-16) checkpoints: ViT-B/16 80M, ViT-L/16 300M, ViT-g/16 1B, ViT-G/16 2B, all at 384 px — [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2)
- V-JEPA 2.1 Table 11, ViT-G (2B): 77.7 SSv2, 87.7 K400, 85.5 IN1K. ViT-L undistilled: 74.2 / 84.3 / 81.8. ViT-L distilled: 76.5 / 86.9 / 84.9. ViT-B distilled: 74.5 / 85.1 / 84.0 — [V-JEPA 2.1 paper](https://arxiv.org/html/2603.14482)

**Image backbone plus temporal module, all frozen (May 2026, authors call it work in progress)**
- SSv2 under one protocol (Table 3): DINOv3-L + GMMix temporal module 66.4; V-JEPA-L 66.0; RVM-L 66.7; VideoMAE-L 62.7; 4DS-L 57.6; DINOv3-Ldist alone 55.9; DINOv2-Ldist alone 52.2. Base size: DINOv3-B + GMMix 60.7; RVM-B 61.4; 4DS-Bdist 60.3; VideoMAE-B 52.3; DINOv3-Bdist alone 50.6 — [Towards Data-Efficient Video Pre-training](https://arxiv.org/html/2605.19137v1)
- A trained temporal module added 10.5 points to DINOv3-L (55.9 to 66.4) and 10.1 points to DINOv3-B (50.6 to 60.7) on SSv2 (arithmetic on the same table) — [same](https://arxiv.org/html/2605.19137v1)
- At 25% of SSv2 training data, DINOv3 + GMMix reached 56.5 — [same](https://arxiv.org/html/2605.19137v1)
- The "V-JEPA-L" 66.0 there and the V-JEPA 2 ViT-L 73.7 in Meta's repo are different protocols (and the paper's label does not say V-JEPA 2), so they should not be compared.
- Temporal module sizes (encoder included) are 347M to 405M for DINOv3-L, which is not a light serving model — [same](https://arxiv.org/html/2605.19137v1)

**Controlled SSL objectives at a fixed 80k-update budget (ViT-B/16, K700, frozen heads)**
- DINOv2 objective led K700 (59.8%); V-MAE led SSv2 (51.3 vs 49.5). Joint DINOv2 + V-MAE got 60.8 K700 and 52.8 SSv2. Budget: 8 H100 GPUs, 144 to 224 GPU-hours per run — [Controlled Study of SSL Pretraining](https://arxiv.org/html/2608.13183)

**Other video models**
- Hiera K400 top-1 (3x5 views) and A100 fp16 speed: B 84.0, 133.6 clip/s; B+ 85.0, 84.1 clip/s; L 87.3, 40.8 clip/s; H 87.8, 20.9 clip/s — [facebookresearch/hiera](https://github.com/facebookresearch/hiera)
- VideoMAE V2 distilled, K400 top-1 at 16x5x3: ViT-S 83.7, ViT-B 86.6; fine-tuned after distillation at 16x7x3: ViT-S 84.0, ViT-B 87.1 — [VideoMAE V2 model zoo](https://github.com/OpenGVLab/VideoMAEv2/blob/master/docs/MODEL_ZOO.md)
- VideoMAE V2 authors note that fine-tuning very large video models on small datasets may overfit (snippet) — [VideoMAE V2 paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_VideoMAE_V2_Scaling_Video_Masked_Autoencoders_With_Dual_Masking_CVPR_2023_paper.pdf)
- Small comparison: on 140 UCF Sports sequences DINOv3 clustered classes better than V-JEPA 2 (silhouette 0.31 vs 0.21). Very small sample, UCF Sports is appearance-heavy (snippet) — [Temporal vs. Spatial](https://arxiv.org/pdf/2509.21595)

**CNN versus ViT on a small synthetic domain: the closest public analogue found**
- DINOv2-B on rendered fonts (226K images, 394 classes, about 575 per class), on an RTX 3090: linear probe 36.6%; full fine-tune 93.4% at 50 epochs (95.9% converged at about 80); LoRA r=8 98.7% at 50 epochs and 99.0% at 100; ImageNet ResNet-50 fine-tuned 98.8% at 50 epochs. About 20 min per epoch — [Parameter-Efficient Fine-Tuning of DINOv2](https://arxiv.org/html/2602.13889)

**Licences (commercial/SaaS)**

| Model | Status found | Source |
|---|---|---|
| V-JEPA 2 | Repo mostly MIT, three files Apache-2.0; HF ViT-L weights tagged MIT | [repo](https://github.com/facebookresearch/vjepa2), [HF card](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256) |
| V-JEPA 2.1 weights | Not stated on the repo page. The arXiv paper itself is CC BY-NC-ND, which concerns the paper rather than the weights | [repo](https://github.com/facebookresearch/vjepa2), [paper](https://arxiv.org/html/2603.14482) |
| DINOv3 | Custom licence. No ban on commercial use, royalty-free; bars military end uses under trade controls; distribution requires including the licence and showing "Built with DINOv3"; derivatives you distribute stay under it; IP suit against Meta ends the licence. SaaS not mentioned; a GitHub issue recommends legal review | [licence](https://ai.meta.com/resources/models-and-libraries/dinov3-license), [issue](https://github.com/facebookresearch/dinov3/issues/31) |
| Hiera (video/image weights) | Code Apache-2.0, weights **CC BY-NC 4.0: not usable for SaaS** | [repo](https://github.com/facebookresearch/hiera) |
| SAM 2 Hiera checkpoints | HF cards tagged Apache-2.0 | [HF card](https://huggingface.co/facebook/sam2-hiera-large-hf/raw/main/README.md) |
| VideoMAE V2 | Repo MIT; weights have no stated licence; request form for the large models, distilled ones direct from HF | [repo](https://github.com/OpenGVLab/VideoMAEv2), [zoo](https://github.com/OpenGVLab/VideoMAEv2/blob/master/docs/MODEL_ZOO.md) |
| InternVideo2 | HF cards tagged Apache-2.0 with a gated prompt about not harming human subjects (snippet). A third-party summary says the repo README states no licence, so these conflict | [HF card](https://huggingface.co/OpenGVLab/InternVideo2-Stage1-1B-224p-f8-MiT/blob/main/README.md), [summary](https://www.sourcepulse.org/projects/1845322) |

### Inferences
- The project's data agree with the literature's direction. Frozen video beats frozen image by about 3 points (V-JEPA 2 versus DINOv2: +0.033 / +0.028), and motion-heavy tasks favour video pretraining. The DINOv3 + temporal module result suggests an image backbone can close much of that gap, but only by training a temporal module.
- The V-JEPA 2 fine-tune has not reached its ceiling: 4 epochs with a NaN still gained +0.025 / +0.043 over frozen, and it sits only 0.018 / 0.020 below the CNN ensemble. So "CNNs are better here" is not established. A clean run is the highest-value next experiment.
- Fine-tuning also narrows the old-to-new drop: ensemble 0.032, fine-tuned V-JEPA 2 0.034, against frozen 0.052 (V-JEPA 2), 0.047 (DINOv2), 0.045 (SigLIP 2). Domain adaptation seems to be buying generalisation to new replays, not only fit.
- The font result matches this project's shape (synthetic look, small per-class data): a fine-tuned ResNet-50 matched the best ViT setup, and full ViT fine-tuning was worse than both LoRA and ResNet-50. It is a single paper on still images with an easier task.
- Candidates worth testing, in order: V-JEPA 2 ViT-L clean fine-tune (already wired up, MIT); V-JEPA 2.1 ViT-B distilled as a smaller teacher or even a server candidate, once its weight licence is confirmed; an image backbone (DINOv3 ViT-S/B or ConvNeXt-T) with a temporal module, subject to the licence terms.
- Hiera's video weights are out for SaaS. SAM 2's Hiera is Apache but trained for segmentation, so it would be an image backbone with no video pretraining advantage.
- 384 px for V-JEPA 2.1 costs serving time. Its token count is about 2.25x that of 256 px per frame ((384/16)^2 = 576 versus 256).

### Gaps
- No source found comparing V-JEPA 2, VideoMAE V2, DINOv3, Hiera or InternVideo2 after fine-tuning on a small dataset. Published numbers are mostly frozen probes or full-Kinetics fine-tunes.
- No measured numbers obtained for TSM, X3D, TimeSformer or ConvNeXt/RegNet with temporal shift on small data.
- Licences not checked in this session: DINOv2, SigLIP 2, X3D, MoViNet, TimeSformer, RVM, 4DS, and timm CNN weights. They must be read from the repos before any SaaS use. DINOv2 and SigLIP 2 matter least, since both lost on the project's data.
- No later V-JEPA successor beyond 2.1 surfaced, but the search was not exhaustive.

## How much do domain-specific self-supervised pretraining and LoRA-style adapters add over plain fine-tuning? Any evidence for animation or game footage?

### Takeaway
Target-domain SSL shows large gains over from-scratch training and modest ones over a good transfer start. LoRA has one strong small-data result and ExPLoRA offers a way to combine both. No study found measures any of this on animation, HUD or game footage, so the project would be producing the evidence itself.

### Cited Findings

**Domain-specific SSL**
- VideoMAE ViT-B pretrained only on the target dataset's own unlabeled clips, then fine-tuned: UCF101 (9.5k clips) 91.3 vs 51.4 from scratch vs 81.7 MoCo v3; HMDB51 (3.5k clips) 62.6 vs 18.0 vs 39.2 — [VideoMAE](https://arxiv.org/html/2203.12602)
- VideoMAE K400-pretrained, then transferred: SSv2 68.5, UCF101 96.1, HMDB51 73.3. So on UCF101 and HMDB51, a larger out-of-domain pretrain beat pretraining on the small target set (96.1 vs 91.3; 73.3 vs 62.6) — [VideoMAE](https://arxiv.org/html/2203.12602)
- The same authors report domain shift matters: transfer from Kinetics to Something-Something was slightly worse than pretraining on the target directly (snippet) — [VideoMAE PDF](https://arxiv.org/pdf/2203.12602). A summary says about 42k in-domain SSv2 videos beat 240k K400 videos (snippet, secondary) — [alphaXiv](https://alphaxiv.org/overview/2203.12602v3)
- MotionMAE, predicting motion as well as frames, reports beating VideoMAE by up to 3.2% in the domain-specific setting (snippet) — [MotionMAE](https://arxiv.org/pdf/2210.04154)
- At a fixed 80k updates, the SSL objective matters a lot. V-MAE beat DINOv2 at the temporal SSv2 task and DINOv2 beat V-MAE at K700 and image tasks. Joint training improved both on SSv2 — [Controlled Study](https://arxiv.org/html/2608.13183)
- The temporal module of DINOv3 gained only +1.3 on SSv2 when initialised from RVM pretraining, whereas RVM-L's own gained +9.5, so a strong frozen image encoder leaves less for video SSL to add — [Towards Data-Efficient](https://arxiv.org/html/2605.19137v1)

**LoRA and adapters**
- LoRA r=8 on DINOv2-B fonts: 98.7% at 50 epochs with about 900K trainable parameters, versus full fine-tune 93.4% with 87.2M. Reaches 95% by epoch 5, whereas ResNet-50 needs about 10. LoRA ranks 8 and 16 are not significantly different; r=4 97.9% — [Font paper](https://arxiv.org/html/2602.13889)
- ExPLoRA: continue unsupervised pretraining on the new domain with 1 to 2 ViT blocks unfrozen and LoRA on the rest, then supervised LoRA fine-tuning. Up to 8% better linear probing with under 10% of the parameters of prior fully tuned methods. Tested on satellite imagery, and self-reported — [ExPLoRA abstract](https://arxiv.org/abs/2406.10973)

**Animation or game evidence**
- DACoN uses DINOv2 on anime line drawings and reports the features understand stylised art, a qualitative claim on colourisation, not a pretraining comparison (snippet) — [search result, alphaXiv](https://www.alphaxiv.org/overview/2509.14685)
- Searches for MAE/JEPA pretraining on gameplay, HUD or esports footage returned only general video MAE work.

### Inferences
- The right baseline is the existing K400/V-JEPA start, not from-scratch training. Against that baseline, expect small gains, a single-digit points effect at most by analogy, not the 40-point numbers above. This is an estimate from the analogy, not a measured prediction.
- The argument for in-domain SSL here is stronger than usual, because fighting game footage is far from Kinetics, and thousands of hours exceeds the datasets above by a large margin. Fixed camera, repeated HUD and backgrounds also mean a lot of redundancy, so a raw hour count overstates the useful data. Sampling across characters and stages matters more than hours.
- A cheap test before a full SSL run: ExPLoRA-style continued pretraining of V-JEPA 2 ViT-L with last blocks unfrozen, judged on the same old/new replay split as the frozen probe. Frozen accuracy above 0.844 / 0.792 is the pass mark.
- HUD masking deserves attention in any SSL run. HUD pixels are easy to reconstruct and carry health and timer state, which could absorb model capacity (reasoning, no source).
- LoRA is mainly useful when overfitting is the problem. The project's new-replay gap (about 0.03) suggests testing it against the last-8-blocks setup, as it would also make it safe to adapt every layer rather than the last few.

### Gaps
- No source on LoRA or adapters for video encoders with measured small-data results.
- No published cost/benefit for JEPA pretraining on a single GPU. The Controlled Study's 144 to 224 GPU-hours per 80k-update ViT-B run was on 8 H100s and is the only cost anchor found. A single RTX 3090 would take longer, by an amount not measured.
- No published evidence on animation, game or HUD footage at all.

## Knowledge distillation for video: student choices, accuracy loss, CPU speed, quantisation, real CPU throughput

### Takeaway
Measured ViT-to-smaller-ViT distillation costs about 1 to 3 points for a one-to-two-size drop and about 6 for giant-to-small. Published video student-on-CPU numbers are almost absent. INT8 speedups depend heavily on hardware: about 2x with a dot-product instruction, while it was **slower** than FP32 on a CPU without one, so check the VPS's flags before planning around quantisation.

### Cited Findings

**Accuracy loss when distilling**
- V-JEPA 2.1 teacher ViT-G (2B) to distilled ViT-L (300M): -1.2 SSv2, -0.8 K400. To distilled ViT-B (80M): -3.2 SSv2, -2.6 K400 (arithmetic on Table 11). Distilled ViT-L beats undistilled ViT-L by +2.3 SSv2 and +2.6 K400 — [V-JEPA 2.1](https://arxiv.org/html/2603.14482)
- That distillation is self-supervised: frozen teacher replaces the EMA target; loss on the teacher's last layer only; a 12-block predictor; same masking and schedule as pretraining. Compute cost not stated — [V-JEPA 2.1](https://arxiv.org/html/2603.14482)
- VideoMAE V2 teacher ViT-g (83.8 K710) to distilled ViT-B 81.5 (-2.3), ViT-S 77.6 (-6.2) on K710. Further fine-tuning on the target adds +0.5 for B and +0.3 for S on K400 at 16x7x3 — [model zoo](https://github.com/OpenGVLab/VideoMAEv2/blob/master/docs/MODEL_ZOO.md)
- 4DS-B distilled gets 60.3 on SSv2 versus 49.6 undistilled: +10.7, in one table — [Towards Data-Efficient](https://arxiv.org/html/2605.19137v1)
- ActionCLIP with a distilled CLIP backbone: best student 66.5 vs teacher 68.8 on HMDB51; inference time fell steeply. Master's thesis (snippet) — [Utrecht thesis](https://studenttheses.uu.nl/handle/20.500.12932/46534)
- DINOv3 distilled students published: ViT-S 21M, S+ 29M, B 86M, L 300M, H+ 840M. ConvNeXt T 29M, S 50M, B 89M, L 198M are listed but not labelled as distilled; ConvNeXt distillation code was released November 2025 — [dinov3 repo](https://github.com/facebookresearch/dinov3)
- MobiAct: MobileNetV4 student scored best with a ConvNeXt teacher; no CPU figures in the excerpt (snippet) — [MobiAct](https://arxiv.org/pdf/2510.19273)
- I3D-to-3D-MobileNet with quantisation on driver activity reports a considerable accuracy drop and a 2273.312 ms CPU inference figure; the snippet does not make clear which model the figure belongs to (snippet) — [Quantized Distillation](https://ar5iv.labs.arxiv.org/html/2311.05970)

**Lightweight video models on CPU**
- MoViNet-A0 reaches similar K600 accuracy to MobileNetV3-large+TSM with 75% fewer FLOPs, and streaming MoViNets are reported faster on Xeon W-2135 CPU than MobileNetV3 with TSM (snippet; exact ms not obtained) — [MoViNets](https://arxiv.org/pdf/2103.11511)

**Quantisation, measured**

| Model | CPU and runtime | FP32 | INT8 | Effect | Source |
|---|---|---|---|---|---|
| ResNet-50 | Core i9-10900K (no AVX-512 VNNI), ORT 1.23.2 | 9.28 ms | 16.34 ms | **1.76x slower** | [INT8 portability study](https://arxiv.org/html/2609.16085) |
| ResNet-50 | Cortex-A78AE (dotprod), ORT 1.23.2 | 38.47 ms | 18.22 ms | 2.11x faster | same |
| ResNet-50 | Cortex-A76 (dotprod), ORT 1.28.0 | 144.95 ms | 79.08 ms | 1.83x faster | same |
| ResNet-50 | Cortex-A53 (no dotprod), ORT 1.17.1 | 680.20 ms | 1123.02 ms | 1.65x slower | same |
| ResNet-50 v1-12 | Unspecified Xeon, ORT 1.8.0 | 520.8 samples/s | 1008.72 samples/s | 1.94x (snippet, Intel blog) | [Intel](https://community.intel.com/t5/Blogs/Tech-Innovation/Artificial-Intelligence-AI/Quantizing-ONNX-Models-using-Intel-Neural-Compressor/post/1355237) |
| ResNet-50 | Xeon Platinum 8270, OpenVINO 2022.1 | n/a | n/a | 3.5x to 4.0x (snippet; old) | [OpenVINO](https://docs.openvino.ai/2022.1/openvino_docs_performance_int8_vs_fp32.html) |
| ViT-B/16 224 | Apple arm64, ORT, dynamic INT8 | 92.51 ms | 37.05 ms | 2.50x; size 346 to 89 MB | [HF model card](https://huggingface.co/nielsr/vit-base-patch16-224-r3al-int8) |

- Study caveat: single-run p50 values, one device per CPU class, thread counts not given — [INT8 portability study](https://arxiv.org/html/2609.16085)
- ResNet-50 accuracy loss under INT8 is about 0.1 to 0.2 top-1 points in OpenVINO's tables (snippet, old) — [OpenVINO](https://docs.openvino.ai/2022.1/openvino_docs_performance_int8_vs_fp32.html); about -0.12 pp in the portability study — [same study](https://arxiv.org/html/2609.16085)
- ViT-B dynamic INT8 lost 83% to 81% top-1 on just 100 ImageNet images, which the page calls a regression check, not an evaluation. LayerNorm and Softmax stay unquantised — [HF model card](https://huggingface.co/nielsr/vit-base-patch16-224-r3al-int8)
- One benchmark found ORT dynamic quantisation covered only MatMul/Gemm because that build's CPU registry lacked ConvInteger, so CNNs may benefit less from *dynamic* INT8 (snippet, low-authority page) — [edge quantisation benchmark](https://darkweb007-ml-portfolio-suite.static.hf.space/05-edge-quantization-benchmark/web/index.html)
- OpenVINO's FAQ says INT8 can help without VNNI — [OpenVINO FAQ](https://docs.openvino.ai/2024/about-openvino/performance-benchmarks/performance-benchmarks-faq.html). This is in tension with the i9-10900K ORT result; the two used different runtimes, and the sign may depend on the runtime as well as the CPU.

### Inferences
- **Budget.** 240 s over 2 threads for 180 s of 30 fps video is 22.5 frames/s if every frame runs, or about 44 ms per frame for the whole chain. The real allowance depends on how many frames or clips the opener classifier actually sees, which the brief does not state. Opener classification only needs clips near round starts, so the budget per clip may be far larger than this worst case.
- **ViT-L on the VPS is implausible for full-rate use.** The nearest measurement is 92 ms for one 224 px ViT-B image on an Apple CPU with unknown threads. A 16-frame V-JEPA 2 ViT-L clip at 256 px has about 2,048 tokens (8 x 16 x 16, if tubelet size is 2) against 196, with a larger model. That suggests several seconds per clip on two shared threads. This is an estimate that needs measuring. FP32 weights alone come to about 1.2 GB (300M x 4 bytes), which fits in 4.4 GB but with little room if other work shares it.
- **Student choice.** Distilling into the existing ResNet18/50 (or a ConvNeXt-T / MobileNetV4 with temporal shift) has a practical advantage: it keeps today's serving path, and only the training labels change (teacher soft labels or features on unlabeled footage). With thousands of hours available, the unlabeled pool is this project's real distillation asset, and it could help even if the teacher's accuracy is not above the ensemble's, since the student sees far more data.
- **Reasonable expectation:** a distilled student lands within 1 to 3 points of the teacher if the size step is moderate, by analogy with the ViT-to-ViT numbers above. No published number exists for ViT-to-CNN on video.
- **Licence effect.** Distilling from DINOv3 could make the student a derivative under the DINOv3 licence, with "Built with DINOv3" obligations; whether this applies to a hosted service needs legal review. V-JEPA 2 under MIT avoids this.
- **Order of checks on the VPS:** (1) `lscpu` for `avx512_vnni` or `avx_vnni`; (2) run FP32 ORT and OpenVINO on the current ResNet18/50 with two threads, as a shared VPS may differ widely from any table; (3) only then try static (QDQ) INT8. The brief's budget is a real measured constraint, and the published multipliers range from 0.57x to about 4x, so they cannot be used for planning.

### Gaps
- No measured CPU throughput found for TSM-R50, X3D-XS, MoViNet-A0, or a distilled video ViT on a Xeon/EPYC. arXiv 2111.09641 benchmarks 13 lightweight models on 10 devices including TSM-R50, X3D-XS and MoViNet-A0 and would answer this directly, but it could not be read (PDF unparseable, ar5iv 502): [paper](https://arxiv.org/pdf/2111.09641.pdf). The same goes for MoViNet's exact ms: [paper](https://arxiv.org/pdf/2103.11511).
- No current (2026) OpenVINO FPS tables for ResNet-50 or ViT retrieved. The official FAQ says they are published per release: [FAQ](https://docs.openvino.ai/2026/about-openvino/performance-benchmarks/performance-benchmarks-faq.html).
- No source for ViT static INT8 accuracy on a full validation set.
- No paper found that distills a video ViT into a CNN and reports both accuracy and CPU speed.

## Practical recipe for ViT fine-tuning on a 24 GB GPU: layer-wise LR decay, gradient checkpointing, causes of NaNs with mixed precision

### Takeaway
Use bf16 without GradScaler, gradient clipping or StableAdamW, and layer-wise decay around 0.75 with a lower base LR. Float16 overflow is a well documented NaN cause in ViT training, but this was not shown to be the project's cause, so reproduce under fp32 first.

### Cited Findings

**NaN causes**
- Float16 represents about 5.96e-8 to 65504; bf16 about 1e-38 to 3e38. ViT-Huge CLIP training slowly diverged under float16 and was fixed by bf16 — [Stable low-precision training](https://arxiv.org/html/2304.13013)
- Loss spikes follow an RMS gradient spike by 1 to 8 iterations in 28 of 30 detected cases, against a 1% chance rate, and are driven mainly by out-of-date AdamW second moments in the patch embedding layer. These coincide with activation and gradient spikes — [same](https://arxiv.org/html/2304.13013)
- PyTorch's loss scaler starts at 65536, and repeated Inf/NaN events drive it down — [same](https://arxiv.org/html/2304.13013)
- A ViT-L/16 under `16-true` went from 2.4105 at batch 0 to NaN by batch 300; `bf16-true` fixed this with no accuracy loss — [Raschka](https://sebastianraschka.com/blog/2023/pytorch-memory-optimization.html)
- bf16 is not a guaranteed fix: one unresolved report of NaN under bf16 in a different model (snippet) — [svd-mv issue](https://github.com/king159/svd-mv/issues/2). Another forum thread suspects softmax overflow in the attention path under mixed precision (snippet) — [PyTorch forum](https://discuss.pytorch.org/t/training-with-mixed-precision-loss-is-nan-despite-finite-output-in-forward-pass/162937)

**NaN countermeasures with evidence**
- Use bf16 instead of fp16; StableAdamW (per-tensor update clipping) beat gradient clipping at norm 1 on zero-shot accuracy; β2 of 0.99 worked better with clipping, as lowering too far slows learning; skip Inf/NaN per tensor with a fixed loss scale; freeze the patch embedding if Inf/NaN events are frequent — [Stable low-precision training](https://arxiv.org/html/2304.13013)

**bf16 on the RTX 3090**
- `torch.cuda.is_bf16_supported()` is true for Ampere or newer on a supporting build (snippet) — [PyTorch docs mirror](https://www.paddlepaddle.org.cn/documentation/docs/en/guides/model_convert/convert_from_pytorch/api_difference/cuda/torch.cuda.is_bf16_supported.html), [PyTorch forum](https://discuss.pytorch.org/t/bfloat16-on-nvidia-v100-gpu/201629/8). No direct RTX 3090 confirmation found; a one-line check on the machine settles it.

**Hyperparameters (VideoMAE fine-tuning)**
- AdamW; weight decay 0.05; betas 0.9, 0.999; cosine schedule; 5 warmup epochs; layer decay 0.7 for ViT-S and 0.75 for B/L/H; base LR 1e-3 (S), 5e-4 (B, L) on SSv2, batch 512; drop path 0.1 (S, B), 0.2 (L, H); dropout 0.5 for L; label smoothing 0.1; RandAug (9, 0.5); mixup 0.8, CutMix 1.0; repeated augmentation 2; no flip on SSv2; 40 epochs (S, B) or 30 (L) on SSv2 — [VideoMAE](https://arxiv.org/html/2203.12602)

**Memory**
- ViT-L/16 peak on one GPU: baseline 26.79 GB; `16-mixed` 18.21 GB; `bf16-true` 13.82 GB; batch 16 5.69 GB. Gradient accumulation at micro-batch 4 kept the result, but runtime went from 3.96 to 12.91 min. This is image ViT at 224 px, not a video ViT — [Raschka](https://sebastianraschka.com/blog/2023/pytorch-memory-optimization.html)
- Fonts on an RTX 3090 with fp16: about 20 min per epoch for 226K images, no NaN reported — [Font paper](https://arxiv.org/html/2602.13889)

### Inferences
Suggested recipe (synthesis, not a tested setup; adjust from the project's own runs):

1. **Find the NaN first.** Run 200 to 500 steps in fp32 with anomaly detection, logging the per-layer gradient norm and the AMP scale. Compare with fp16 and bf16 under the same seed. If only fp16 fails, the cause is the range issue above. If all fail, suspect LR, a bad clip or data, or a division by zero in preprocessing or normalisation. A NaN in epoch 2 fits an LR or gradient-spike cause as well as an overflow one, and the evidence cannot separate them yet.
2. **Precision.** `torch.autocast(dtype=torch.bfloat16)` with no GradScaler. bf16 uses low-precision rounding, so keep the final loss and probe computed in fp32.
3. **Optimiser.** AdamW with betas (0.9, 0.99) or StableAdamW, gradient clipping at norm 1, layer decay 0.75, warmup 5 epochs or more. 5e-4 is calibrated to batch 512, so a much smaller batch needs a much smaller absolute rate (the common linear rule is LR x batch/256, not confirmed in the sources read here). Layer decay of 0.75 means the patch embed of a 24-block ViT-L trains at about 0.1% of the head's rate (0.75^24, arithmetic), which is close to freezing it, in line with the freeze advice.
4. **Unfreezing.** Layer decay across all layers may replace "last 8 blocks": deep layers still learn while the early ones barely move. Test it against last-8 plus LoRA, with the full-chain old/new split as the judge.
5. **Regularisation for small data.** Drop path 0.2 for ViT-L. Be careful with VideoMAE's augmentation recipe: flips are invalid here (they change side, and SSv2 itself disables flip). Mixup and CutMix on move classes with little visual difference need testing rather than copying.
6. **Memory.** Gradient checkpointing on the trainable blocks, micro-batch small, accumulation to the target effective batch. Expect time, not memory, to be the limit: accumulation more than tripled runtime in the one measurement above. Do not accumulate under fp16 until the NaN is explained: one low-authority guide warns this can magnify numerical errors (snippet) — [guide](https://claudskills.com/skills/mixed-precision-deployment/SKILL.md).
7. **Commit limit.** With the 94 GB commit limit noted in the brief, use fewer dataloader workers or decode ahead to disk. Video decode workers are the usual heavy consumer (reasoning, no source).
8. **Validation.** Fine-tune for at least 20 to 30 epochs, judged on the new-replay split, not a loss curve, since the current 4-epoch number does not tell where this model tops out.

### Backbone strategy implied by the above (inference)
- **Serving stays CNN for now.** Nothing measured here shows a ViT meeting the VPS budget, and the fine-tuned CNN ensemble is still the most accurate model on the project's data.
- **Training gets a stronger teacher.** Fix the NaN, run V-JEPA 2 ViT-L for a full schedule, and only then decide. If it passes the ensemble, distil it into the CNN via soft labels on unlabeled footage. If it does not, an ensemble of teacher and CNNs may still give better soft labels than either alone, and distillation remains worthwhile because of the unlabeled pool.
- **Domain SSL is the later, bigger bet.** Run it only after a clean fine-tune sets the baseline, as otherwise a gain cannot be attributed.
- **Decision rule worth pre-registering.** A new backbone must beat 0.887 / 0.855 on the same chain and split at a serving cost within budget; a gain on old replays alone does not count, since new-replay generalisation is the product requirement.

### Gaps
- No V-JEPA 2-specific NaN report found. The official repo's issues were not searched directly.
- No measured gradient-checkpointing memory or time numbers for a video ViT on a 24 GB card. The memory figures above are image ViT at 224 px and do not include checkpointing.
- No source for layer decay values specific to V-JEPA 2 fine-tuning; 0.75 is carried over from VideoMAE.
