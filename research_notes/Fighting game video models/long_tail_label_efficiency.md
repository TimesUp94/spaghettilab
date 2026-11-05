# Long-tailed, fine-grained classification and label-efficient learning for move recognition

Reading notes. Unless marked "search summary only", numbers were read from the paper's own PDF text. Some PDF tables extracted with scrambled row labels; where I realigned them I checked at least one anchor against the paper's prose or another table. No source studies fighting-game video. Every number below comes from a different kind of data, and each finding says which. Pre-2021 papers are marked, and are included because they are the primary source of the method or number.

Project numbers used in the Inferences are from the brief: 34 characters, 904 opener and 1,215 transcript classes, about 27 and 36 classes per character after masking.

## Which long-tail methods give real gains, and how large?

### Takeaway
On the benchmarks, re-balancing raises tail accuracy by 10-20+ points and balanced accuracy by 4-6 points, but it pays for much of this with head accuracy, and those scores use a balanced test set. Our natural-frequency metric may gain far less or even fall. Cheap first tries are weight-decay tuning, post-hoc logit adjustment with a tuned temperature, and classifier retraining. On the one large long-tailed video benchmark, gains were only about 1-2 points.

### Cited Findings
**Survey with one common protocol** [different data: natural images, balanced test set; ImageNet-LT with ResNeXt-50, iNaturalist 2018 with ResNet-50]
- ImageNet-LT, 200 epochs, head/middle/tail and overall top-1 (rows realigned, anchored on Table 7 and the prose): Softmax 66.9/40.4/12.6, 46.8; Balanced Softmax 62.4/47.7/32.1, 51.2; LDAM 60.0/49.2/31.9, 51.1; IB-cRT 64.2/46.1/26.0, 50.3; PaCo 63.2/51.6/39.2, 54.4; RIDE 67.9/52.3/36.0, 56.1; SADE 67.2/55.3/40.0, 58.8 — [Zhang et al., Deep Long-Tailed Learning survey](https://arxiv.org/abs/2110.04596)
- iNaturalist 2018 (fine-grained, naturally long-tailed), 200 epochs, head/middle/tail/all: Softmax 75.3/66.4/60.4/64.9; Balanced Softmax 70.9/70.7/70.4/70.6; LDAM 57.4/62.7/63.8/62.8; IB-cRT 73.2/68.8/65.1/67.8; RIDE 71.5/70.0/71.6/71.8; SADE 74.4/72.5/73.1/72.9 — [same survey](https://arxiv.org/abs/2110.04596)
- Two methods fell below Softmax on ImageNet-LT: BBN (41.2 / 44.7 at 90 / 200 epochs) and Decouple-CB-CRT (44.9 / 43.0). The authors blame over-focus on the tail and poor representation learning under class-balanced sampling. Their stated best practice is ensembles plus class re-balancing used together — [same survey](https://arxiv.org/abs/2110.04596)
- Stacking re-balancers overshoots: from Balanced Softmax alone (51.2 overall), adding logit adjustment and/or re-sampling gave 46.6, 49.1 and 44.6 (which row is which combination was lost in text extraction). The survey warns against simply combining them — [same survey, Table 7](https://arxiv.org/abs/2110.04596)

**Logit adjustment** [different data: images, balanced error; ICLR 2021]
- Balanced error, ERM to post-hoc adjustment (tau=1) to tuned-tau to the loss form: ImageNet-LT 53.11 / 49.66 / 49.56 / 48.89; iNaturalist 38.66 / 33.98 / 33.80 / 33.64. LDAM ("adaptive", numbers taken from Cao et al.): 52.15 and 35.42 — [Menon et al.](https://arxiv.org/abs/2007.07314)

**Decoupled training and prototype heads** [different data: images; ICLR 2020]
- iNaturalist ResNet-50 at the two training lengths reported: Joint 61.7/65.8; NCM (prototype head) 58.2/63.1; cRT 65.2/67.6; tau-norm 65.6/69.3; LWS 65.9/69.5. Here prototypes were the worst option — [Kang et al.](https://arxiv.org/abs/1910.09217)
- ImageNet-LT ResNeXt-50, Joint against NCM and cRT, many/medium/few/all, first of the two training-length columns: Joint 64.0/33.8/5.8/41.6; NCM 53.1/42.3/26.5/44.3; cRT 58.8/44.0/26.1/47.3. NCM matches cRT on few-shot but loses more on many-shot. Few-shot in this benchmark means under 20 images — [Kang et al.](https://arxiv.org/abs/1910.09217)

**Regularisation alone** [different data: images; CVPR 2022]
- Tuning weight decay only, naive to WD to WD + two-stage classifier (CB, WD, MaxNorm): ImageNet-LT 38.0 / 48.6 / 53.9; iNaturalist 46.1 / 65.4 / 70.2; CIFAR100-LT IF100 38.38 / 46.08 / 53.35. Tail on iNaturalist 43.9 / 61.5 / 69.7. Note that "naive" is a weak baseline, so these jumps are not comparable with Softmax rows above — [Alshammari et al., Weight Balancing](https://arxiv.org/abs/2203.14197)

**Long-tailed video** [different data: VideoLT, 1,004 classes, untrimmed weakly labelled video, multi-label mAP on frozen features, tail = under 100 videos; ICCV 2021]
- TSM ResNet-50 features, baseline to FrameStack (overall mAP, tail mAP): nonlinear model 0.565 to 0.580, tail 0.436 to 0.459; NetVLAD 0.660 to 0.667-0.669, tail 0.554 to 0.566-0.568. FrameStack beats image-based long-tail methods by 1.5 and 2.3 points (overall, tail) — [Zhang et al., VideoLT](https://arxiv.org/abs/2105.02668)
- Resampling frames by inverse class frequency cost about 0.5% on NetVLAD. The authors attribute weakness partly to background frames in weakly labelled clips and report that long-tail image methods do not transfer well to video — [VideoLT](https://arxiv.org/abs/2105.02668)

**Other figures**
- Retraining the classifier after Balanced Softmax: ImageNet-LT few-shot 38.21 to 45.13, Top-1 51.01 to 49.37; Places-LT few-shot +9.78 — [arXiv 2607.09832, search summary only](https://arxiv.org/abs/2607.09832) [different data: images]
- Survey claim that loss reweighting gives 2-4% balanced-accuracy improvement over CE, and is harder to improve with many classes — [arXiv 2404.15593, search summary only](https://arxiv.org/abs/2404.15593) [different data: images]

**Hierarchical classification** [different data: images; CVPR 2020]
- Hierarchical cross-entropy is a trade-off: it reduces mistake severity, and heavier hierarchy weighting raises top-1 error. In one results block (read as iNaturalist19-H, table layout scrambled), HXE alpha=0.1 has top-1 error 60.11, alpha=0.5 45.00 and a YOLO-v2 hierarchical baseline 44.04. The paper describes top-1 and hierarchical distance as closely linked at k=1 — [Bertinetto et al.](https://arxiv.org/abs/1912.09393)
- A later paper also describes this line as lowering severity at the cost of top-1 error — [HAF, ECCV 2022, search summary only](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840249.pdf)

### Inferences
- **The metric decides the answer.** All benchmark scores weight classes equally. Ours weights moves by how often they occur, so a head loss like iNaturalist's 75.3 to 70.9 costs more than the tail gain returns. Keep tau as a post-hoc knob (Menon's method needs no retraining), and report natural-frequency whole-chain accuracy and macro per-move accuracy side by side.
- **Our problem differs in scale.** Our 5-19 bin scores 0.84-0.92, far above ImageNet-LT few-shot (under 20 images, 12.6-40), though masking leaves us only about 27-36 candidates per character against 1,000. So most benchmark gains apply to a regime we have already left. Our real tail is under 5 examples, currently not predictable at all as separate classes. That makes "can we un-merge the rare class?" a few-shot question, not a re-weighting one. No benchmark above tests under-5 classes on their own.
- Our dense frame-exact labels remove VideoLT's stated background-frame problem, so its small gains are a pessimistic bound, not a forecast, though its frozen-feature setup also limits what the methods could do.
- Suggested order (cheapest first): weight-decay sweep; post-hoc logit adjustment; cRT or tau-norm with frozen backbone; only then few-shot heads for under-5 moves. Avoid stacking several re-balancers. Avoid class-balanced sampling during representation training (CB-CRT result). Avoid LDAM as a first try: on iNaturalist it fell below Softmax in the survey (62.8 against 64.9), and in Menon it beat ERM but trailed logit adjustment (35.42 against 33.64-33.98 balanced error).
- Hierarchy (category > button > move) is more likely to improve the usefulness of errors than top-1, and the per-character masks already provide its top level. A practical use is splitting the catch-all into per-character rare-by-button classes, so a rare move at least reports its button. This is untested.

### Gaps
- I found no measured results for few-shot heads on classes with under 5 examples inside an otherwise well-sampled supervised classifier, only NCM on the under-20 bin.
- I found no measured top-1 gain from hierarchical losses on rare classes, and did not read Valmadre et al. directly.
- I found no study of long-tail methods on natural-frequency test sets, or with per-group class masks like ours.
- The share of moves or frames in the under-5 catch-all was not in the brief. It bounds the possible whole-chain gain and should be measured first.

## Does targeted data collection beat algorithm changes, and what should the learning curve predict at 1,700 replays?

### Takeaway
Fitting our own curve gives roughly 0.89-0.91 whole-chain at 1,700 replays, with 0.90 the central figure, and about 0.90-0.93 at 3,400. Targeted recording of under-5 moves probably beats algorithm work for those moves, judging by our own bins and the rare-class evidence, though no study compares the two directly. Pool-based active-learning results transfer only partly, because we choose what to record instead of paying per label.

### Cited Findings
**Learning curves** [different data; mostly pre-2021]
- Power-law error scaling holds across factors; model improvements shift the error but do not appear to change the exponent. Empirical exponents run -0.35 to -0.07 against about -0.5 in theory, with saturation at irreducible error for large sets — [Hestness et al. 2017, search summary only](https://arxiv.org/abs/1712.00409) [different data: translation, language models, image, speech]
- The extended power law fits well, but its individual parameters are unstable under measurement noise, so they summarise curves by error at a reference size plus data-reliance. The authors treat a 4x extrapolation (n=400 to n=1600) as the limit of accurate extrapolation — [Hoiem et al., ICML 2021](https://arxiv.org/abs/2010.11029) [different data: CIFAR, Places365]
- An error of only 1-6% in the fitted curve mis-estimated data needs by 120,000-310,000 images against a true 900,000. The fitted functions are often optimistic or pessimistic depending on the setting, with no single best one, and they recommend a correction factor and collecting over several rounds. ImageNet extrapolation RMSE fell from 3.19-60.1 fitting on 10% of data to 0.74-5.06 fitting on 50% — [Mahmood et al., CVPR 2022](https://arxiv.org/abs/2207.01725) [different data: image classification and others]
- At very large scale, performance grows logarithmically with data (JFT-300M, with about 20% label error in precision) — [Sun et al. 2017](https://arxiv.org/abs/1707.02968) [different data: images, pre-2021]
- In the low-data regime, learning curves do not show the asymptotic power law — [arXiv 2012.04115, search summary only](https://arxiv.org/abs/2012.04115)

**Active learning** [different data: image classification with a per-label cost]
- With proper classifier configuration, AL gains over random are often small: "the benefits of AL being generally smaller compared to" the right training strategy and hyperparameter selection. But AL "generally provides substantial gains in class-imbalanced settings", BADGE was best with no sub-random failures, and FixMatch "may struggle" on class-imbalanced data. AL was tested on CIFAR-10/100 (LT), MIO-TCD and ISIC-2019 — [Lüth et al., NeurIPS 2023](https://arxiv.org/abs/2301.10625)
- With strong regularisation, AL methods show marginal or no advantage over random sampling, and differences between studies are often within variance — [Munjal et al., CVPR 2022](https://arxiv.org/abs/2002.09564)
- Traditional AL was often less effective than random when data come from multiple domains — [arXiv 2312.00364, search summary only](https://arxiv.org/abs/2312.00364)
- Common AL methods amplify imbalance in the selected set, and adding an explicit class-balance term generally gave a gain, including on balanced data — [Bengar et al., WACV 2022](https://arxiv.org/abs/2110.04543)

### Inferences
- **Our curve (computed here).** 722 to 1,090 replays gives error 0.135 to 0.113, exponent -0.43, within theory but steeper than Hestness's range. Projected whole-chain:

| Assumption | 1,700 replays | 3,400 replays |
|---|---|---|
| Fit, no error floor (exponent -0.43) | 0.907 | 0.931 |
| Fit, floor 0.03 | 0.906 | 0.927 |
| Fit, floor 0.05 | 0.904 | 0.923 |
| Fit, floor 0.08 | 0.901 | 0.912 |
| Literature exponent -0.35 | 0.903 | 0.924 |
| Literature exponent -0.07 | 0.890 | 0.896 |

- Going 1,090 to 1,700 is a 1.56x step, inside Hoiem's 4x limit, but it is a two-point fit and assumes the two scores used comparable test sets and models. If the model also changed between 722 and 1,090, the exponent is inflated (model changes should move error, not the exponent). A measured result well below about 0.89 would point to something other than data volume. A result above about 0.91 would suggest the earlier curve was pessimistic.
- The first dataset's curve (error 0.321 / 0.281 / 0.205) has exponents -0.19 then -0.45. Steepening is not a clean power law, consistent with the low-data caveat, so I would not fit through those points together with the later ones.
- Doubling from 1,700 to 3,400 replays is worth only about 1-2.5 points overall by the fits (0.5 at the pessimistic literature exponent), so whole-dataset recording is close to diminishing returns relative to its cost. A retrain at 25/50/75/100% of 1,700 on a frozen test set would replace all this with a real curve.
- **Targeted collection.** Our bins show about 0.57 under 5, 0.84-0.92 at 5-19, 0.96-0.97 at 50+. Taken at face value, getting a move from under 5 to 5-19 is worth about +0.3 on it, larger than the 10-20 point tail gains above. This comparison is confounded: the catch-all measures something different from per-move scores, and rare moves may be inherently harder. Still, recording to reach about 20 examples for each move looks like the best-supported single action for rare moves.
- AL's "class-imbalance gains" finding supports targeting, but we are not selecting from an unlabelled pool, so query strategies such as BADGE do not map directly. The nearest usable procedure is picking replays (character, rank, opponent) by predicted per-class payoff, in several rounds as Mahmood suggests. Multiple-round collection also guards against mis-estimation.
- **Suggested ranking of work by evidence for rare moves:** (1) measure the attribution of whole-chain errors per move; (2) targeted recording or bot generation for under-5 moves; (3) weight decay, logit adjustment and cRT; (4) ensemble and distillation; (5) SSL and domain adaptation only for stream VOD deployment.

### Gaps
- I found no direct head-to-head study of targeted class-level recording against long-tail algorithms, on any data.
- The AL literature assumes annotation is the cost. For us, the cost is recording time and rare-move availability, and I found no benchmark for that.
- The current per-move learning curve at 1,700 replays was not in the brief.

## Is semi-supervised learning or domain adaptation useful for unlabeled footage and stream VODs?

### Takeaway
Published SSL gains are large at label scarcity (1-20% of labels) and shrink to none in realistic settings: out-of-class unlabeled data can push results below the supervised baseline. With 8.2M labelled frames, SSL has low priority for clean-footage accuracy. The real value of the unlabeled footage is measuring and reducing the gap to stream VODs, where compression costs 5-9 points in a benchmark, and evidence that specific augmentation helps is missing.

### Cited Findings
**Fine-grained SSL** [different data: Semi-Aves, 200 long-tailed bird classes, 6K labelled, 27K in-class and 122K out-of-class unlabeled]
- Top-1 from ImageNet pretraining: supervised 52.7, oracle with all labels 68.5, FixMatch in-class 57.4, FixMatch with out-of-class unlabeled 49.7 (below baseline). Self-Training with both 55.5. From iNaturalist pretraining: supervised 65.4; FixMatch in-class 70.2, with out-of-class 64.2. Their finding is that out-of-class data "often hurts" when using a good pretrained model — [Su et al., CVPR 2021](https://arxiv.org/abs/2104.00679)

**Video SSL** [different data: UCF101, Kinetics-400 action videos at low label fractions]
- UCF101 at 10% labels, R-50: Pseudo-Label 30.4 to 70.0, UDA 47.0 to 77.8, FixMatch 48.5 to 79.1 when run with MvPL (multi-view: RGB, flow, temporal gradients) — [Xiong et al., MvPL, ICCV 2021](https://arxiv.org/abs/2104.00682)
- 3D-ResNet-50, UCF101 at 1/5/10/20/50% labels: TimeBalance 30.1/53.5/81.1/83.3/85.0; Kinetics-400 at 1/10%: 19.6/61.2. FixMatch SlowFast-R50: UCF101 16.1 at 1% and 55.1 at 10%; Kinetics 10.1/49.4 — [Dave et al., CVPR 2023](https://arxiv.org/abs/2303.16268)
- FixMatch transfers poorly to video because it lacks temporal augmentation, and directly applying it to ViTs performs worse — [SVFormer, search summary only](https://arxiv.org/abs/2211.13222)

**SSL at large scale** [different data: ImageNet, 300M unlabeled, pre-2021]
- EfficientNet-L2 85.5% to 88.4% with Noisy Student — [Xie et al.](https://arxiv.org/abs/1911.04252)

**Robustness and adaptation** [different data: Kinetics, Something-Something; H.265, not H.264]
- Mini Kinetics-C, averaged over severities, clean / ABR / CRF: I3D 70.5 / 64.3 / 60.8; TimeSformer 82.2 / 76.5 / 73.0. Mini SSV2-C, TimeSformer: 60.5 / 54.9 / 53.8. Models with more capacity are more robust. Efficient models lose more on temporal corruptions — [Yi et al., NeurIPS 2021](https://arxiv.org/abs/2110.06513)
- Gaussian-noise augmentation lowered clean accuracy and degraded robustness on 10 of 12 corruption types, unlike image models. Whether corruption augmentation in general helps video remains open by their account — [Yi et al.](https://arxiv.org/abs/2110.06513)
- Test-time adaptation added 6.5% on Mini Kinetics-C and 4.1% on Mini SSV2-C on average — [TeCo, ICLR 2023, search summary only](https://arxiv.org/pdf/2302.14309.pdf)
- Video UDA with self-training: 92.5 HMDB to UCF, 95 UCF to HMDB — [UNITE, CVPR 2024; aggregator figures, unverified against the paper](https://arxiv.org/abs/2312.02914). CleanAdapt treats source-free video adaptation as learning from noisy labels, with 88.6 on U to H — [CleanAdapt, search summary only](https://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/images/ConferencePapers/2025/Source-free-video.pdf)

### Inferences
- We are outside the regime where video SSL shines. The large gains above appear at 1-20% label fractions, and shrink as labels rise (TimeBalance 83.3 to 85.0 from 20% to 50%). The class-imbalance finding (Lüth) and out-of-class finding (Su) both apply to our footage, which contains non-match segments, other versions and possibly unknown moves. Pseudo-label only from high-confidence segments in verified gameplay, and test that on rare moves separately, because a rare class is the one a confident teacher gets wrong.
- Self-supervised pretraining (MoCo + Self-Training) helped most in Su "from scratch" (20.6 to 41.2 with out-of-class data), and far less from ImageNet pretraining (52.7 to 55.9 in-class, 53.9 with out-of-class). Since our model is already trained on millions of frames, I would not expect much here.
- For stream VODs the mechanism is known: re-encode clean recordings using the real stream pipeline settings (bitrate, 720p, H.264) and paste plausible overlay graphics, then validate on real VOD clips. This differs from Gaussian noise, which is what Yi et al. tested, and matches the known corruption rather than adding unrelated noise, but its benefit is not established by these sources. The overlay is manually set and may be absent per earlier project lessons, so training must include examples both with and without it.
- A labelled stream test set could be built from our own logged replays re-streamed through a real streaming path, giving exact labels on realistic degradation. This looks more valuable than any adaptation algorithm because it makes every other choice measurable.

### Gaps
- I found no measured study of H.264-at-stream-bitrate augmentation for video classifiers, or of overlay robustness.
- I found no adaptation benchmark for the same content with different compression; UCF/HMDB adaptation is a different type of shift.
- I found no semi-supervised results on long-tailed fine-grained video with 90%+ labelled data.

## How should label noise be handled, and are ensembles, mixed precision and distillation worth it?

### Takeaway
Our noise is structured and locatable, so fix or mask it and clean the test set before reaching for noise-robust losses, which also fail on tail classes. A 5-model ensemble cuts error by about 11-14% in the benchmarks, and distillation keeps roughly two-thirds of the gain in one model. Mixed precision saves memory and time but does not itself raise accuracy.

### Cited Findings
**Label noise**
- An estimated minimum of 3.3% label errors on average across 10 benchmark test sets; about 54% of algorithmically flagged candidates were confirmed by people. Rankings on ImageNet and CIFAR-10 survived correction but can flip with small changes in the error mix — higher-capacity models fit systematic errors more closely — [Northcutt et al., NeurIPS 2021, search summary only](https://arxiv.org/abs/2103.14749) [different data: human-labelled benchmarks]
- The small-loss trick fails under long-tailed labels because networks cannot separate correct from mislabeled examples in tail classes; a prototype-distance detector with soft pseudo-labels beat DivideMix by 3% — [Wei et al., RoLT](https://arxiv.org/abs/2108.11569) [different data: images with injected noise]

**Ensembles** [different data: images]
- ImageNet ResNet-50 top-1 error: single 23.86; deep ensemble 5 / 10 / 50 members 21.19 / 20.90 / 20.63; snapshot 5 / 10 / 50 members 22.21 / 21.75 / 21.48. WRN-28-10 CIFAR-100: single 19.31; deep ensemble 5 / 10 / 100: 16.55 / 16.17 / 15.77. On ImageNet, 5 and 10 snapshots equal about 2.2 and 3.2 independent models (deep ensemble equivalent). Snapshots scale nearly linearly on CIFAR but saturate on ImageNet, perhaps from a suboptimal schedule. On ImageNet, FGE stays close to a single model (23.61 at 5 members) — [Ashukha et al., ICLR 2020](https://arxiv.org/abs/2002.06470)
- An ensemble of six ResNet-18s improves on a single one, but has higher error and data-reliance than single larger networks such as ResNet-101, Wide-ResNet-50 or Wide-ResNet-101 — [Hoiem et al.](https://arxiv.org/abs/2010.11029)
- Ensemble-based long-tail methods lead the survey's results (RIDE 56.1 and SADE 58.8 against Softmax 46.8 on ImageNet-LT at 200 epochs), though part of that gain comes from higher upper-reference accuracy, not only imbalance handling — [Zhang et al.](https://arxiv.org/abs/2110.04596)

**Distillation** [different data: CIFAR]
- WRN-28-10 CIFAR-100: single 81.51; 10-model ensemble 84.69; distilled from ensemble 83.81; self-distilled 83.56. CIFAR-10: 96.70; 97.20; 97.22 / 97.13 — [Allen-Zhu & Li, ICLR 2023](https://arxiv.org/abs/2012.09816)

**Mixed precision** [different data: general DNN workloads; pre-2021]
- FP16 nearly halves memory with no loss of accuracy against FP32 when using a master FP32 weight copy and loss scaling; DeepBench operations on a Volta GPU sped up 2-6x, though full-training speedup depends on the network and libraries — [Micikevicius et al., ICLR 2018](https://arxiv.org/abs/1710.03740)

### Inferences
- **Treat "older logs miss object credits" as missing labels, not noise.** You know which logs and which labels are affected. The cleanest options are to re-derive them by re-running the logger where possible, or to mask the projectile-vs-special decision on frames from old logs, so those frames still train other outputs. Generic noise-robust methods would likely remove rare moves too (RoLT result).
- **Clean the test first.** Before optimising projectile vs special, score only against logs with full object credit. Otherwise the benchmark may reward a model that reproduces the missing-credit pattern, the Northcutt effect. Then a confusion that remains on clean logs is model error, and the rest is label artefact.
- A simple check of the split: compare projectile/special confusion on old against new logs with one model. A large difference estimates how much of it is noise.
- **Ensemble payoff for us (extrapolated, not measured).** A 11-14% relative error reduction from 5 models (23.86 to 21.19; 19.31 to 16.55) on 0.10 error is about +1.1-1.4 points. Distillation retained about 72% (ensemble) or 64% (self) of the ensemble gain on CIFAR-100 (2.30 and 2.05 of 3.18 points), and adds no inference cost, which matters if inference runs on users' machines.
- Training 5 models serially on a single RTX 3090 costs 5x time. Snapshot ensembling is the budget variant, at about 2-3 models' worth. A practical sequence: first evaluate 3 seeds to measure ensemble gain on our data, including on rare moves; distil only if it is worthwhile. Larger single-model width may be competitive (Hoiem), so test that as well.
- Mixed precision is worthwhile as an enabler (more clips, longer temporal windows or larger models within 24 GB), not an accuracy method. If already enabled, no accuracy gain should be expected from it.

### Gaps
- No source measured ensembling on tail classes specifically under natural-frequency evaluation, apart from the ensemble-based long-tail methods above.
- I found no direct evidence on training with partly missing labels from an instrumented source such as a game log.
- I did not find a 2021-2026 replication of snapshot ensembles on video, and the 3090 speedup from AMP on our model is unmeasured. It would be a one-epoch test.

## Should opener and transcript share multi-task training, and can the game itself generate rare moves?

### Takeaway
Multi-task sharing is worth one simple trial with static loss weights, as specialised optimisers show no measured benefit. The evidence on synthetic rare-class data, even across a real render gap, is strongly positive at low volumes, and our same-engine case should be better on appearance. The main risks are prior bias from too much synthetic data and behavioural differences between bots and human matches.

### Cited Findings
**Multi-task** [different data: language and vision tasks]
- Specialised multi-task optimisers gave no improvement beyond a static weighted sum of losses; multi-task models are sensitive to learning rate and weight decay, and insufficient baseline tuning can create false gains; some reported results could not be reproduced — [Xin et al., NeurIPS 2022](https://arxiv.org/abs/2209.11379)
- Unitary scalarisation with standard regularisation matches or beats those optimisers — [Kurin et al., NeurIPS 2022](https://arxiv.org/abs/2201.04122)

**Synthetic data for rare classes** [different data: wildlife camera traps, Unity-rendered animals against real photos, so a true sim-to-real gap exists; WACV 2020]
- Rare class error on new locations with synthetic added: 120 images, 28% lower error, with under 0.5% rise in average per-class error on other classes; 100K images, 39% lower, again under 0.5%; 1.4M images, 88% lower, but other classes' error rose 13% as the classifier became biased. The threshold for noticeable bias was above 325K images, where the next-largest real class had 2,514 — [Beery et al.](https://arxiv.org/abs/1904.05916)
- Wider simulation variation gave the best results, with some axes more valuable than others (night images most for their data) — [Beery et al.](https://arxiv.org/abs/1904.05916)

### Inferences
- **Multi-task.** Since opener and transcript moves likely share low-level appearance, a shared backbone with separate heads and a fixed weighted loss is the only version the evidence supports trying. Compare against separately trained models tuned equally well, because Xin warns that undertuned baselines create fake wins. The opener task may see fewer frames, so it could gain more. This is untested.
- **Game-generated data: same engine removes the render gap, not the distribution gap.** Beery's gains came despite a real render gap, so ours should not be lower on appearance grounds alone. What remains: bot-driven moves lack human context (spacing, cancels, combos and hit-confirm states, and matchups), and training-mode screens may differ in HUD or stage from real matches. Generate in the most match-like mode the game allows, with varied opponents, stages, positions, hit/block and combo contexts, following Beery's "wider variation" result. Check the HUD and lighting against replays first.
- **Volume.** Beery's rare-class gain arrived with tiny amounts (120 images gave 28%), and damage began only with an overwhelming prior. For us, a useful start is raising each under-5 move to about 20-50 instances, keeping each below a modest share of its character's data, plus logit adjustment with the natural prior so the extra samples do not bias predictions.
- **Validation.** Evaluate only on real human replays. Keep generated data out of test and report rare-move accuracy with and without it. If the memory logger runs in that mode, it should give the same exact labels, which is a major advantage over the literature: the generated samples need no separate labelling.
- Generation may be the cheapest route to reaching the 5-19 bin, since waiting for ranked replays cannot easily yield a move humans rarely use. For moves that are that rare in real play, the gain to whole-chain accuracy will also be small, so prioritise by error attribution as noted above.

### Gaps
- I found no study of supplementing real data with data from the same engine for the same task. The closest evidence is cross-domain (Beery), so the size of gains for our case is unknown.
- I did not find measured per-task gains from sharing a backbone for related fine-grained video tasks; Xin and Kurin address optimisers, not whether to share.
- Whether Strive's training mode, CPU matches or replay playback expose match-like HUD and full move availability was not researched here, and should be verified in-game.
