# Recognition methods specific to fighting games and game footage, including palette robustness

Evidence tags used below: **[full text]** = numbers read from the paper's own text or tables. **[abstract]** = abstract or landing page only. **[search summary]** = seen only through a search-result summary, not checked against the paper, so verify before quoting. **[project]** = this repo's own notes, not external evidence. **Analogy** means the evidence is not from game footage or does not test palette swaps.

Overall: I found **no published study of move classification from fighting-game video that reports per-move accuracy on modern 3D or cel-shaded games, and none that tests palette swaps**. Nearly all the "colour" and "pose" evidence is analogy.

## Existing fighting-game move recognition, frame data, replay analysis and combo work

### Takeaway
The public record is thin: student projects on Melee from pixels with weak results, a lot of RL and imitation work that does not measure recognition, and tools that avoid vision by reading memory or replay files. This project's own results (move 0.944 given a found opener, chain 0.887 on a replay-split test fold) are better documented and stronger than anything comparable I found, so external baselines are of little use as targets. The lessons worth taking are about leakage and class imbalance.

### Cited Findings

**Move or state recognition from pixels (the closest work)**
- **Slippify (Stanford CS231n, Melee).** CNNs predict state from frames: positions, percent, facing and action, where "action" means controller input, not animation. Frames came from only "a couple" of replays (over 10k frames each), with test windows blocked out of the same replays. It reports normalised MSE only, with no per-action accuracy. Observation test loss: small CNN 0.11592, large CNN 0.08699; with colour jitter 0.11747 and 0.10390. So jitter did not help and hurt the large model, but the authors say the same-replay test set may be unable to show any benefit from jitter. Action loss by window length: 1 frame 0.777-0.797, 3 frames 0.569-0.600, 5 frames 0.901-0.957, 7 frames 0.664-0.747, 9 frames 0.731-0.784. The authors do not explain the bad 5-frame result. The unsupervised spatial autoencoder locked onto the clock and percentages rather than the characters. [full text] — [Slippify](https://cs231n.stanford.edu/papers/text_file_840589945-Parsing_Super_Smash_Bros__Melee_Frames.pdf)
- **Yale poster (Melee, AlexNet and time-distributed AlexNet, frames to inputs, one matchup, one stage, 15 fps).** The five configurations scored 98.8%, 47%, 45%, 55% and 52%. The 98.8% run included a HUD/graphic in the image; the poster's Figure 5 shows the on-screen input graphic, which was also the label source, drawing the activation. Removing it gave 47%. Fixed camera gave 55% against 47%. Image augmentation lowered accuracy (47 to 45, 55 to 52). `null`, `left` and `right a` dominated predictions, so rare inputs were missed. The authors cite the character's small share of a very large image as a difficulty. The test split is not described. [full text] — [Rodriguez-Vars, Lachance, Radev](https://yale-lily.github.io/public/s2018/BenVars_AustinLachance_presentation.pdf)
- **Chen and Yi 2017 (N64 Smash, vision-only imitation).** 30-class input classification, 80% top-1 and 95% top-3 validation, competitive with the highest-level built-in AI after slight test-time fine-tuning. This predicts player input, not moves. [abstract] — [arXiv 1702.05663](https://arxiv.org/abs/1702.05663)
- **Cal Poly thesis 2018.** 2,296 annotated Melee frames to track the coordinates of 4 characters in real time. I could not retrieve the accuracy figures. [search summary] — [Cal Poly](https://digitalcommons.calpoly.edu/theses/1845)
- **Roboflow "StreetFighter" dataset.** Ken and Ryu states such as idle, hit and specific normals, with a pretrained model. The page returned 403, so I have no metrics. [search summary] — [Roboflow](https://universe.roboflow.com/dojocapstoneproject-xnq0n/streetfighter)

**Pixel-input game AI (recognition not measured)**
- **DIAMBRA Arena.** Fighting-game RL platform with pixel and RAM observations. The authors state the RAM values are redundant with what the screen shows. PPO agents show human-like behaviour. [search summary] — [arXiv 2210.10595](https://arxiv.org/abs/2210.10595)
- **FightLadder (ICML 2024).** Covers SF2, SF3, Fatal Fury 2, Mortal Kombat and KOF97. A general agent consistently beats 12 built-in characters, and a non-exploitable two-player agent is hard to train without human knowledge. [search summary] — [arXiv 2406.02081](https://arxiv.org/html/2406.02081v2)

**State-based work and tools (no vision, using structured ground truth)**
- **Hamelink 2025 (Leiden MSc).** Melee imitation agents use a 91-feature state per frame, not pixels, built on Eric Gu's HAL, which trained on 3 billion frames. In the Fox-ditto validation, the character being imitated mattered far more to loss than the opponent's character. Loss curves only. [full text] — [Hamelink thesis](https://theses.liacs.nl/pdf/2025-2026-HamelinkJJosef.pdf)
- **Slippi.** Melee replays expose per-frame state, and the combo tooling parses those files. [search summary] — [slippi-combo-detector](https://gitlab.com/slippi-development/slippi-combo-detector), [Project Clippi](https://github.com/vinceau/project-clippi)
- **TekkenBot.** A GUI for studying frame data in game, which records its own frame-data database. Its readme says it can no longer parse the move list from game memory. Not vision. [search summary] — [TekkenBot](https://github.com/dcep93/TekkenBot)
- **SF6 community work.** A mod converts SF6 matches to a text script for AI analysis. A Japanese write-up found a video AI needed the command list and move names supplied to interpret a combo. Both anecdotal, no accuracy. [search summary] — [OpenAI forum](https://community.openai.com/t/ai-focused-on-street-fighter-6/699412), [note.com](https://note.com/clear_gecko3242/n/n24f3e0f05f6a?hl=en)

**This project, for comparison**
- ResNet18 + ResNet50 ensemble with frame-shift TTA and a whole-frame stream. 2026-10-30 v2 on 1,090 replays: move 0.944 given found, chain 0.887 on the old 144 test replays, 0.855 on the new 76. A learning curve at 25/50/100% of data gave 0.68/0.72/0.80 at the earlier v1 stage. Mirror TTA, a per-character prior and 224 px / 12 frames were dead ends. [project] — `C:\Users\feder\.claude\projects\C--Projects-replanal\memory\ggst-search-batch.md`

### Inferences
- The Slippi pattern (structured state as labels for video) is what this project's UE4SS logger already does, and none of the public work has an equivalent at this scale for a modern game.
- The Yale 98.8% to 47% drop is the clearest measured warning in game footage: any on-screen element correlated with the label will dominate a CNN. For this project that argues for checking that the whole-frame stream isn't learning palette, stage or player-specific correlations rather than the move.
- Yale and Slippify both saw augmentation fail to help, but neither had a test set with colour shift, so they do not show that jitter is useless. They do suggest jitter costs a little when the test set has no shift. That cost must be measured against the gain on alternate colours.

### Gaps
- No per-move accuracy on Guilty Gear, SF6 or Tekken from video in any paper I found. My searches returned nothing, which is not proof of absence: forum, Discord, Japanese and Chinese sources were barely covered.
- The Hamelink thesis has no vision component; it is included only for the character-specificity finding.
- I did not retrieve FightLadder's exact tables, or Cal Poly's accuracy.

## Colour robustness: augmentation, normalisation, randomisation, and palette-swap augmentation

### Takeaway
Strong analogy evidence says plain hue jitter closes most of a global hue-shift gap at little or no cost. That makes it the first thing to try, and it is cheap. Global hue shift is not a palette swap, however, and nobody has measured palette swaps. Colour-equivariant networks and grayscale add little beyond jitter, while random convolutions and IBN-style normalisation are the best-evidenced next steps for appearance changes beyond hue.

### Cited Findings

**Colour Equivariant Networks (NeurIPS 2023)** give the most directly useful numbers, all analogy (real photos, whole-image hue rotation). [full text]. Accuracy on the unshifted test set, and averaged over test hue shifts — [Lengyel et al.](https://arxiv.org/html/2310.19368v1)

| Dataset | Baseline clean | Baseline, hue-shift avg | Jitter clean | Jitter, hue-shift avg | Best CEConv + jitter, hue-shift avg |
|---|---|---|---|---|---|
| Flowers-102 | 66.79 | 13.41 | 68.75 | 68.44 | 73.86 |
| Caltech-101 | 71.61 | 51.14 | 73.93 | 73.61 | 73.57 |
| Stanford Cars | 76.54 | 55.59 | 80.59 | 80.65 | 80.28 |
| Oxford Pets | 69.87 | 37.56 | 72.71 | 72.31 | 72.94 |
| STL-10 | 83.80 | 67.60 | 83.91 | 83.71 | 84.31 |
| CIFAR-100 | 71.28 | 47.01 | 69.23 | 69.12 | 71.33 |
| CIFAR-10 | 93.69 | 85.26 | 93.03 | 92.91 | 93.86 |
| ImageNet | 69.71 | 54.72 | 69.37 | 67.10 | 68.85 |

- Same paper: when colour is needed, removing it is disastrous. Long-tailed ColorMNIST: colour-equivariant CNN 91.35%, CNN 71.59%, colour-invariant variants 24.19% and 29.43%. [full text] — [Lengyel et al.](https://arxiv.org/html/2310.19368v1)

**Random Convolutions (RandConv).** Accuracy on domains the model never saw, compared directly with grayscale and jitter. [full text] — [Xu et al., arXiv 2007.13003](https://ar5iv.labs.arxiv.org/html/2007.13003)

| Target domain | Baseline | Grayscale | Colour jitter | RandConv (mix) |
|---|---|---|---|---|
| MNIST to MNIST-M | 58.87 | 58.41 | 62.72 | 87.76 |
| MNIST to SVHN | 33.41 | 36.06 | 39.61 | 57.52 |
| PACS Cartoon target | 64.12 | 62.12 | 61.43 | 64.05 |
| PACS Sketch target | 55.28 | 60.07 | 62.44 | 69.74 |
| PACS Photo target | 86.77 | 83.93 | 84.61 | 86.60 |

**Normalisation: IBN-Net.** Mixing instance and batch norm, ImageNet top-1 error increase under appearance changes. These are pixel intensity transforms, and the fetched page does not define the "RGB+50" label. Cross-domain mIoU, including a game domain: Cityscapes to GTA5 ResNet50 22.2, IBN-a 26.0, IBN-b 29.6; GTA5 to Cityscapes 29.4, 32.5, 37.9. [full text] — [Pan et al., arXiv 1807.09441](https://ar5iv.labs.arxiv.org/html/1807.09441)

| Transform | ResNet50 | IBN-Net50-a | IBN-Net50-b |
|---|---|---|---|
| RGB+50 | +3.94 | +3.00 | +0.18 |
| R+50 | +3.26 | +2.66 | +1.46 |
| contrast (std x1.5) | +15.74 | +13.43 | +0.00 |
| Monet style | +30.24 | +29.03 | +26.81 |

**Video specifically**
- Motion Coherent Augmentation permutes RGB channels (SwapMix) and aligns predictions between original and recoloured clips. TSM Top-1: SSv1 45.63 to 47.57, SSv2 59.29 to 60.71, UCF101 79.57 to 81.87, HMDB51 48.63 to 51.63, Kinetics400 70.28 to 71.08. Plain hue jitter on SSv1 reaches 46.32. The paper reports that hue jitter causes a drop on CIFAR10 image classification but helps video. The fetched page does not name a venue; a search result points to ICLR 2024. [full text] — [arXiv 2403.09506](https://arxiv.org/html/2403.09506v2)
- A CVPR 2022 supplement on semi-supervised action recognition: converting inputs to grayscale cost the baseline 28.0 points and the proposed method 14.6. [search summary] — [Xiao et al. supplement](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Xiao_Learning_From_Temporal_CVPR_2022_supplemental.pdf)

**Shape bias and caveats on jitter**
- A study of 39 augmentation types finds shape bias correlates with out-of-distribution robustness in some cases but is not causal. [search summary] — [arXiv 2406.05006](https://arxiv.org/html/2406.05006v1)
- Standard colour jitter damages colour features in self-supervised learning. Planckian Jitter (ICLR 2023) offers a physics-based alternative, and combining both representations gives gains. No figures on the page I read. [abstract] — [Zini et al.](https://arxiv.org/abs/2202.07993)
- In a 2026 robotics VLA study, a naively augmented model scored 90.5% on grayscale and 89.8% on RGB, which the authors read as the model having gone colour-blind. [search summary] — [arXiv 2607.14698](https://arxiv.org/pdf/2607.14698)

**Engine-rendered variation and palette swaps themselves**
- GameCLR uses a game engine to render controlled scene variations for contrastive learning. It reports beating SimCLR on state inference, but only on the CARLA driving simulator, and no numbers appear on the abstract page. [abstract] — [Trivedi et al.](https://arxiv.org/abs/2207.01289)
- I found no paper evaluating palette-swap augmentation on game sprites or characters. — [search returning only adjacent work, e.g. arXiv 2505.17328](https://arxiv.org/abs/2505.17328)

### Inferences
- **This project is in the favourable case for jitter.** Jitter recovered nearly all the hue-shift loss even on Flowers-102, where colour is informative, and it cost clean accuracy only on some datasets (CIFAR-100 lost about 2 points). Move class is mostly shape-defined, so full-range hue augmentation looks justified. One exception needs care: colour can be informative in Strive (for example Roman Cancel flash colours and some effects). Check whether any class depends on colour, from domain knowledge not verified here. Hue jitter could blur these, and the ColorMNIST result shows how badly invariance hurts when colour is the signal.
- **Test first, cheaply, before changing training.** Copy the CEConv protocol: run the current model on the existing test fold at fixed hue rotations (say 12 steps) plus channel permutations, and report accuracy per shift. If accuracy collapses like Flowers-102 does, the model is colour-dependent, and jitter will likely recover most of it. If it holds up, global colour is not the main problem. This costs one evaluation pass and no training.
- **Global hue rotation is the wrong physical model.** A palette swap maps each material separately (hair, coat, skin) and leaves stage and effects alone, whereas jitter changes everything together. A closer simulation: cluster colours in the character region of a crop and shift each cluster independently, keeping lightness, optionally with a character mask. This is a hypothesis, since nothing measures it. The cel-shaded flat colour regions should make clustering easier than on photos.
- **Whole-image jitter also changes the scene stream and the hit effects.** If hue jitter hurts, apply it per stream or per character region rather than dropping it.
- **Real alternate-colour data beats synthetic.** The 1,090+ replays presumably contain many player-chosen colours. If the colour ID can be logged, the model's accuracy can be stratified by colour, and rare colours held out as a real test of generalisation. This would directly measure the problem rather than an analogy for it.
- **Order to try:** (1) hue-shift evaluation, (2) hue and channel-permutation augmentation, with MCA-style consistency if plain jitter is short, (3) per-material palette simulation, (4) RandConv or IBN-b as a backbone change. CEConv looks low priority: its gain over jitter is about 2 points or less outside Flowers-102 (5.4), and the fully equivariant CEConv-4 lost accuracy on ImageNet (64.52 hue-shift average against 67.10 for jitter alone). Grayscale alone looks poor: it was weaker than jitter on most targets above and throws away colour-defined classes.

### Gaps
- No measurement of palette swaps or material-wise recolouring in any setting I found.
- No direct comparison of batch and instance norm under palette changes. IBN evidence is for intensity and style shifts on real images and GTA5.
- RandConv and IBN have not been tested on fine-grained action classes, and RandConv may damage fine shape detail at 192 px (unmeasured).
- Whether this project's data actually has a colour-related accuracy gap is unmeasured. The Slayer error was at character identification, not the move model.

## Pose, silhouette and segmentation approaches for stylised characters

### Takeaway
On real humans, pose can match or beat RGB for fine body-motion classes (FineGYM) and fusing the two is often best, but pose collapses when context and objects matter (Kinetics). Off-the-shelf pose models lose much of their accuracy on illustrated characters, and nothing I found tests fighting-game characters or SAM 2 on anime or cartoons. For this project, pose makes sense only as a possible auxiliary stream, not a replacement, and it would need a custom-trained estimator.

### Cited Findings

**PoseConv3D: pose against RGB.** Real humans, so analogy. Top-1, except FineGYM mean top-1. [full text] — [Duan et al., arXiv 2104.13586](https://ar5iv.labs.arxiv.org/html/2104.13586)

| Dataset | Pose only | RGB only | RGB + pose |
|---|---|---|---|
| FineGYM (fine-grained gym moves) | 93.2 (94.3 joint+limb) | 87.2-88.5 | 93.6-94.1 (95.6 in another table) |
| NTU60 X-Sub | 93.7 (94.1) | 94.1-94.9 | 96.2-96.5 (97.0 in another table) |
| Kinetics400 | 46.0 (47.7) | 84.9 (prior VideoSwin, not their own baseline) | 85.5 |

- The paper does not explain the differences between its tables.
- Estimator quality matters moderately. NTU-60: HRNet top-down (COCO AP 0.746) 93.6, HRNet bottom-up (0.654) 93.0, MobileNet top-down (0.646) 92.0. On FineGYM, training on high-quality and testing on low-quality poses gives 86.5, and low-to-low 90.7. Keypoint dropping costs PoseConv3D little: 92.4 to 91.5. [full text] — [same paper](https://ar5iv.labs.arxiv.org/html/2104.13586)
- Hand-object verbs (Assembly101): RGB-only about 55.6-56.6%, RGB plus pose about 69.2-69.3%. PoseConv3D is reported as weak on hand actions since it assumes full-body poses. [search summary] — [HandFormer supplement](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01025-supp.pdf)

**Pose estimation on stylised characters**
- Chen and Zwicker (WACV 2022) on 4,000 illustrated characters (3,200 training images, single full-body characters) measured OKS@50 / OKS@75. Their best model 0.8982 / 0.7930 (feature concatenation, new data). Feature matching 0.8953 / 0.7907 at about 1/9 the parameters. COCO-pretrained Mask R-CNN 0.7584 / 0.6724. OpenPose 0.4922 / 0.4222. Column mapping came from flattened PDF text, so check it against Table 1 before quoting. [full text] — [arXiv 2108.01819](https://arxiv.org/abs/2108.01819)
- Pose augmentation with varied head-body ratios reportedly helps on images rendered from 3D models. [search summary] — [IEICE](https://ken.ieice.org/ken/paper/20200227L1Vz/)

**Segmentation**
- My search found no evaluation of SAM 2 on anime, cartoons or game characters. The SAM 2 paper's evaluation covers 37 zero-shot datasets, none identified as cartoon in what I saw. [search summary] — [SAM 2](https://arxiv.org/pdf/2408.00714)
- A SAM 2 study on camouflaged objects offers a usable evaluation protocol. [search summary] — [arXiv 2409.18653](https://arxiv.org/pdf/2409.18653)

### Inferences
- FineGYM's RGB gap probably comes partly from RGB models latching onto scene and clothing. Where pose won, the classes were body-configuration-defined. Strive's moves share that property, but many are also defined by weapons, projectiles, summons and effects, which is closer to the Kinetics and object cases where pose fails badly. Non-humanoid or weapon-defined characters would be a poor fit for a 17-keypoint skeleton.
- This project's RGB result (0.944 move given found) is already in the same range as the best pose-only numbers above, though these are different tasks and not comparable. Pose replacing RGB seems unlikely to pay. The realistic value is a colour-invariant auxiliary stream or attention signal, and the RGB + pose improvements (about 1-2 points on NTU) suggest a small gain at high cost.
- Any pose route needs labels on Strive characters. Bones are in the game engine, so a logger or mod could perhaps supply 3D keypoints as ground truth, which would remove the annotation problem. Feasibility is unverified.
- A silhouette or mask stream directly removes palette, but also internal shape (arm in front of body), which fine moves may need. SAM 2 with detector-box prompts is cheap to trial on sampled frames. Masks would also support the per-material palette augmentation, and that may be the better use: mask for augmentation at training time only, avoiding inference cost and failure modes during hit effects.

### Gaps
- No skeleton-based action recognition on anime, cel-shaded or game characters found. No direct RGB-versus-silhouette comparison on a stylised set found.
- No SAM 2 cartoon or anime IoU, no measurement with heavy VFX, and no throughput figures gathered for this hardware.
- The anime datasets are mostly still illustrations, not rendered 3D with motion blur, compression or overlapping characters, so transfer to Strive is unknown.

## Exploiting deterministic animation: nearest neighbours, metric learning, frame identity

### Takeaway
I found no measured work exploiting deterministic game animation for recognition. This is the biggest open opportunity and the least evidenced: everything here is analogy or proposal. The project's logger may give an unusually strong basis for it, because sprite or animation identity could serve as a metric-learning label that crosses colours.

### Cited Findings
- **Finite Automata Extraction (2025).** Learns a neural sprite dictionary from gameplay video and then a program per sprite. It is about world modelling, not recognition from noisy broadcast video. Numbers not read. [search summary] — [arXiv 2508.11836](https://arxiv.org/html/2508.11836v1)
- **Game engine learning from video (Georgia Tech).** Pairs each sprite with its nearest neighbour in the next frame to derive facts such as velocity. Secondary source only. [search summary] — [Hackernoon](https://hackernoon.com/how-ai-can-be-used-to-replicate-a-game-engine-6160d000b1e3)
- **GameCLR.** Engine-rendered variations as contrastive pairs: same state, varied rendering. Only CARLA, no numbers seen. This is the conceptual pattern closest to cross-colour positives. [abstract] — [Trivedi et al.](https://arxiv.org/abs/2207.01289)
- **VideoNet (2026).** A kNN classifier on video features with 3 support examples per action, no two clips from the same source video. A real-world, domain-specific action benchmark. I saw the setup but no figures. [search summary] — [arXiv 2605.02834](https://arxiv.org/pdf/2605.02834)
- **Few-shot action recognition with CLIP and metric matching (CLIP-FSAR).** Classifies queries by metric matching with no test-time fine-tuning. Real video, numbers not read. [search summary] — [arXiv 2303.02982](https://arxiv.org/pdf/2303.02982)
- **Partial video copy detection (WACV 2022).** A kNN frame-feature database plus a temporal network to localise the copied segment. This is structurally the same as "which reference move is this clip", and it comes closest to the deterministic-replay assumption. Numbers not read. [search summary] — [Tan et al.](https://arxiv.org/abs/2105.01713v2)
- **DINOv2 near-duplicate filtering.** Cosine similarity with a threshold around 0.9, reported as better than average-pooled ResNet features for dropping near-identical frames. An engineering note, not a benchmark. [search summary] — [arXiv 2412.07584](https://arxiv.org/pdf/2412.07584)
- **Project assets.** The logger records the current action name per player, and characters are identified from sprite codes. [project] — `C:\Users\feder\.claude\projects\C--Projects-replanal\memory\ggst-memory-map.md`, `ggst-search-batch.md`

### Inferences
These are all proposals, untested:
- **A move is a known sequence of poses, and the label is available per frame.** Most action recognition cannot assume this. If sprite codes change per animation frame, then (sprite code, character) is a much finer label than the move: thousands of pose classes with the same pose seen in many colours, stages and scales. Supervised contrastive training on those labels would treat "same sprite, different colour" as positives, teaching colour invariance from real data rather than from augmentation. Verify sprite-code granularity first.
- **Reference-frame kNN helps with the learning-curve limit.** The curve (0.68/0.72/0.80 at 25/50/100%) shows the move model is data-limited. Rare moves could get a kNN or prototype head over a learned embedding, since one clean example of an animation should carry far more information here than in real video. Compare per-class accuracy on rare moves against the softmax head.
- **DTW or sequence alignment to reference.** Per-frame embeddings aligned to each move's reference sequence would handle hitstop and cancelled moves (which are prefixes of the reference). The cancel case is specific to this domain and standard classifiers have no way to express it.
- **Limits on pixel matching.** "Same animation" does not mean same pixels: camera zoom, compression, 720p, side flipping, effects and opposing-character occlusion all vary. Raw template or perceptual-hash matching is likely fragile, so matching belongs in a learned or DINOv2-like space. The project's dead end on mirror TTA is a reminder that flips may not be symmetric (Strive characters are often not mirror-identical; unverified).
- **Cheap test.** Extract DINOv2 or current-backbone embeddings for test openers and run kNN against training openers. Compare with the softmax head overall, on rare moves, and across colours. A few hours of work would say whether identity matching has merit.

### Gaps
- No paper or tool found that matches video frames to game animation data for recognition, and no measurement of how much determinism helps.
- Whether a frame data resource such as hitbox or animation dumps could provide reference frames for Strive was not investigated.
- Whether sprite codes are per animation frame or per pose group is unverified from the notes I read.

## Tracking two players who overlap and cross sides, including mirror matches

### Takeaway
No fighting-game-specific tracking evidence exists that I could find. The closest benchmarks agree that when players look alike, motion beats appearance and appearance can actively hurt. With fast motion, appearance adds a little on top of motion. A fighting game's constraints (exactly two objects, HUD side, logger ground truth) make the problem much smaller than these benchmarks.

### Cited Findings
- **DanceTrack (CVPR 2022, uniform appearance with crossovers).** Oracle analysis with ground-truth boxes: IoU only HOTA 72.8, adding appearance 69.4, all cues 59.7. With YOLOX detections: IoU HOTA 44.7, SORT 47.8, DeepSORT (adds appearance) 45.8, LSTM motion 51.6, OC-SORT 52.1. Re-ID features are "highly entangled" on DanceTrack, and the authors say appearance can have negative effects. OC-SORT test: HOTA 63.2 on MOT17 and 55.1 on DanceTrack; AssA 63.2 and 38.3. DetA is higher on DanceTrack, so association, not detection, is the bottleneck. [full text] — [Sun et al.](https://ar5iv.labs.arxiv.org/html/2111.14690)
- **SportsMOT (ICCV 2023).** Fusion weight sweep, OC-SORT with MixSort, HOTA on test. MixSort's overall gains: OC-SORT 73.7 to 74.1 and ByteTrack 64.1 to 65.7. The authors conclude that faster motion depends more on appearance. [full text] — [Cui et al.](https://arxiv.org/pdf/2304.05170)

| Sport | Pure motion | Best fusion | Pure appearance |
|---|---|---|---|
| Basketball | 65.9 | 66.2 | 26.9 |
| Volleyball | 76.0 | 76.9 | 40.9 |
| Football | 71.9 | 73.2 | 65.7 |

- Later appearance-plus-motion methods self-report HOTA 61.3 on DanceTrack (Deep OC-SORT) and 80.1 on SportsMOT (Deep HM-SORT). [search summary] — [Deep OC-SORT](https://arxiv.org/abs/2302.11813v1), [Deep HM-SORT](https://arxiv.org/html/2406.12081v1)
- Melee object detection tracked 4 specific characters from 2,296 annotated frames. Accuracy not retrieved. [search summary] — [Cal Poly](https://digitalcommons.calpoly.edu/theses/1845)

### Inferences
- **Mirror matches correspond to DanceTrack:** do not rely on appearance embeddings there. Non-mirror matches correspond to SportsMOT at best: appearance is helpful but secondary to motion, and a character classifier is a much stronger cue than generic re-ID since the two characters differ in class.
- **N is always 2**, so association reduces to one binary decision per frame: same or swapped. This removes birth, death and many-to-many assignment, which are much of what the benchmarks test. A small HMM or Viterbi pass over a two-state "swap" variable, with per-frame evidence, fits this directly and is simple to build.
- **Useful evidence specific to this game:** character class per box; facing direction (characters normally face each other, so crossing shows as a facing flip); the attacker's combo counter side and which health bar drops, which ties an event to a player; and colour within a mirror match, if the two differ. This is one case where colour is a helpful signal, so a colour-invariant move model should not share its embedding with the identity tracker.
- **Hitstop and knockdown overlap** are probably where swaps happen, and the existing logger gives true player positions, so swaps can be measured directly rather than estimated. Any tracker change should report swap rate per replay on the replay-split test fold, split by mirror and non-mirror.

### Gaps
- No published fighting-game two-player tracking results found, and no study isolating two-object overlap swaps. The benchmarks have many objects, so their numbers don't transfer directly.
- Whether Strive forces different colours in a mirror match, and how the existing `video_players` step handles swaps, was not checked here. Its swap rate is not in the notes I read.
