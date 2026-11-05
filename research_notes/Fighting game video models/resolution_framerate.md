# Input resolution and frame rate for Strive video models

Bottom line for the writer: **the case for higher resolution is weak.** It is not disproved, because our data has barely tested a model that actually consumes more pixels. Raising the source or upload resolution has no support. The case for 30 fps / 15 Hz tokens is moderate, and only after a no-new-pixels fix of the output head. Frame rate is the better bet of the two, and the literature agrees.

Source quality: "own data" means files in this repo or `G:/Projects/replanal-data/ggst_dataset/model/`, read directly. X3D and the F3Set stride table and resolution prose were read from the papers' own text. Other paper numbers came through a page-summarising fetch tool, so re-check them before quoting them as exact.

## What do resolution studies show, and when is the gain real? Is resolution or frame rate the limit for tiny, fast motion?

### Takeaway
Gains from resolution are real but small and diminishing, and they are largest when the action occupies a small part of a squashed whole frame. Our crops already remove that condition. Studies that vary both axes get more from temporal density than from pixels on fine-grained, fast events.

### Cited Findings
**Own data (old 144 test replays unless stated)**
- The only run that fed the model more pixels: ResNet18, 224 px, 12 frames against 192 px, 8 frames, on the same 578 training replays. It came out slightly **worse**, not just level: chain 0.8431 vs 0.8511, move 0.9064 vs 0.9127, category 0.9116 vs 0.9192, `normal_acc` 0.9265 vs 0.9332. — `results_exp_moves_r224f12.json`, `results_exp_moves_ep20.json`; settings from the `.pt` checkpoints; "did not help" in commit 40a071d. Resolution and frame count changed together (offsets -15..+2), so the run does not separate the two axes. It is also only a 17% linear increase. Epochs are not recorded in either results file.
- The opener chain's inputs are small compared with the source: a 640 px crop at 1080p goes to 192 px, and the whole frame goes to 384×216. — `model/video_crops.py` (`S, T = 640, 224`), `model/video_moves.py` (`RES = 192`, `SCENE`)
- The transcript and the 60 fps DINOv2 features decode at **960×540**, so a player square is 320 px before it is resized to 256 or 224. — `model/transcript_feats.py`, `model/inputs_frame_feats.py` (`DEC_W, DEC_H`)
- 720p30 at 3 Mbit vs 1080p60 for the chain: whole chain 0.888 vs 0.887, starts 0.947 vs 0.955 at 0.943 precision, exact move 0.945 vs 0.94, category 0.93 for both. — `tools/ggst_logger/README.md`
- The transcript and inputs models were never run on 720p30 copies. — README ("Not tried on the CPU or on 720p30 copies"; "not tried on 720p30 copies")
- The largest measured spatial improvement on our data needed no new pixels. Keeping 2×2 quadrant tokens instead of mean-pooling raised transcript move F1@4 from 0.697 to 0.735 (old) and 0.678 to 0.716 (new); bootstrap +0.038 [0.033, 0.042] and +0.039 [0.033, 0.045]. — README
- Other levers measured on the same chain: frame-shift averaging plus a ResNet50 ensemble took the chain from 0.838 to 0.865 (commit 40a071d). More data took it from 0.865 to 0.887. — README
- Fine-tuning V-JEPA 2 instead of freezing it: +0.025 / +0.043 chain. — README

**Literature**
- X3D, Kinetics-400 classification. The expansion order is bottleneck width, temporal resolution (one to two frames), spatial resolution, depth, duration, then more frame-rate/duration steps, with spatial resolution again only at steps 8 and 9. Any axis helps. — [X3D](https://arxiv.org/abs/2004.04730)
- X3D sizes: X3D-S 13×160² 72.9% at 1.96 GFLOPs; X3D-M 16×224² 74.6% at 4.73; X3D-XL 16×312² 78.4% at 35.84 (depth and width also grow). — [X3D](https://arxiv.org/abs/2004.04730)
- X3D's comparison with a 3D EfficientNet-style scaling: EfficientNet3D-B4 74.5% at 23.80 GFLOPs against X3D-M 74.6% at 4.73. — [X3D](https://arxiv.org/abs/2004.04730)
- Tiny Video Networks, Moments in Time: the search keeps 128–224 px and varies frames. TVN-1 224² × 2 frames 23.1%; TVN-5 160² × 16 frames 29.8%; TVN-6 160² × 32 frames 30.7%, all at stride 4. — [TVN](https://ar5iv.arxiv.org/html/1910.06961)
- F3Set is the closest task: tennis at 25–30 fps, events under 0.1 s, about one per second, ±1 frame tolerance, 1,108 classes. Input resolution, TSM + F3ED, F1_evt / F1_elm / Edit: 224² 40.3 / 75.2 / 74.0; 336² 43.2 / 77.1 / 74.8; 448² 44.4 / 78.1 / 74.5. The authors report diminishing gains and keep 224². Its table lists the same 77.3 FLOPs for all three sizes, which looks like an error, so no cost can be read from it. — [F3Set appendix H.3](https://arxiv.org/html/2504.08222v2)
- F3Set temporal side, same model: stride 2 (default) 40.3 / 75.2 / 74.0; stride 4 25.9 / 69.2 / 62.7; stride 8 14.0 / 56.7 / 44.3. SlowFast 4×16 at 336² (37.4) falls below TSM at 224² (40.3), which the authors put down to lower temporal resolution. — [F3Set](https://arxiv.org/html/2504.08222v2)
- GolfDB/SwingNet, percentage of correct events: 224 px 66.8 (10.32 GFLOPs), 160 px 62.4 (5.33), 128 px 57.7 (3.45). 192 px gave 45.7, which the authors flag as not behaving as expected. At 160 px, a sequence of 64 instead of 32 gave 71.1 vs 62.4 at 10.65 GFLOPs. So the same doubling of cost buys +4.4 from resolution and +8.7 from context. — [GolfDB](https://ar5iv.labs.arxiv.org/html/1903.06528)
- FineGym Gym99, TSN mean accuracy by frames: 1 frame 35.46, 3 61.4, 5 70.8, 7 74.4, 12 78.82, saturating around 12 frames. Flow beats RGB (75.6 vs 61.4). Skeleton (ST-GCN) gives 25.2, blamed on pose estimation. No resolution experiment was found in the text read. — [FineGym](https://ar5iv.labs.arxiv.org/html/2004.06704)
- Skeletons also lose on F3Set: Edit on G_high is 74.0 for TSM vs 56.1 for ST-GCN++ and 53.2 for PoseConv3D. — [F3Set](https://arxiv.org/html/2504.08222v2)
- Small-object sport is the exception. SoccerNet Ball Action Spotting solutions run high resolution: 2024 winner T-DEED at 796×448, mAP@1 73.39; second place at 720p grayscale, 71.35. — [SoccerNet 2024](https://arxiv.org/html/2409.10587)
- 2023 SoccerNet winner: 1280×736 grayscale, every second frame of 25 fps, mAP@1 86.47 on the challenge set. Its largest stated gain was architecture: slow fusion about 65% to 80%. — [lRomul repo](https://github.com/lRomul/ball-action-spotting). Scores across years should not be compared; I did not verify that the task definitions match.
- Precise-spotting work on human actions stays at 224 px: E2E-Spot on Tennis, Figure Skating, FineGym and FineDiving ([E2E-Spot](https://ar5iv.labs.arxiv.org/html/2207.10213)); T-DEED on FigureSkating and FineDiving, 71–88 mAP at δ=1 ([T-DEED](https://arxiv.org/html/2404.05392)).
- A larger backbone at 224 is not a reliable gain either. E2E-Spot RegNet-Y 200MF vs 800MF at δ=1: Tennis 96.1 vs 96.8, FS-Comp 81.0 vs 84.0, FS-Perf 85.1 vs 83.6, FineDiving 68.4 vs 64.6, FineGym-Full 47.9 vs 50.1. — [E2E-Spot](https://ar5iv.labs.arxiv.org/html/2207.10213)

### Inferences
- **The headline 720p = 1080p result cannot show that resolution doesn't matter.** At 720p the 640 px crop is 427 source pixels, still 2.2× the chain's 192 px input and above the transcript's 320 px decode. The extra 1080p pixels were discarded before any model saw them. The result means "source resolution above 720p is wasted with the current inputs", which is useful and firm for the opener chain only. It says nothing on whether a 384 px input would help.
- The true test of input resolution on our data is one confounded run with a 17% linear bump. So the evidence is thin rather than strongly negative.
- Our crops already provide the zoom that high-resolution inputs buy elsewhere. A 640 px square at 256 px is 0.4× of 1080p. F3Set at 448² on a whole frame would be about 0.23× horizontally if its sources are 1080p (the paper gives no pixel dimensions). So we sit past F3Set's top setting in pixels per source pixel, and its gain of 2.9 F1_elm from 224 to 448 mostly belongs to a condition we don't have. Tennis players are also far smaller on screen than Strive characters.
- If higher resolution helps on the transcript, the post-encoder pooling is the likelier bottleneck. A ViT patch at 256 px covers 40 px of 1080p, yet the tokens are then averaged to 2×2. That grid gave +0.038, and a 4×4 grid costs no new pixels.
- Ranking by the literature and our data for tiny fast motion: temporal density and context first, then fine-tuning or the data, then pixels.

### Gaps
- AR-Net (resolution-adaptive): only the abstract was retrieved, with no numbers. — [AR-Net](https://ecva.net/papers/eccv_2020/papers_ECCV/html/152_ECCV_2020_paper.php)
- I found no study of input resolution on stylised or animated fighting game footage.
- The EfficientNet paper itself was not read; only X3D's comparison was.
- I did not find a saturation study specific to 112/224/448 on Kinetics.

## For frame-exact events, what does temporal sampling do? What fps is needed, and when does it stop helping?

### Takeaway
Dense sampling matters a lot once sampling gets coarser than the event, and the steepest measured loss falls at about the regime of our 7.5 Hz tokens. But 30 fps video is the norm for frame-precise work, and our own timing gains are capped by label alignment, by a one-start-per-tubelet head and by how few events timing can rescue. The requested 25 vs 50 fps comparison could not be found.

### Cited Findings
**Own data**
- Transcript: 256 px at 15 fps, one token per 8 video frames (7.5 Hz); each tubelet predicts **one** start with an 8-way offset; merge distance 8 frames. Starts under 8 frames after the previous one are found 0.47 of the time. — README
- Merge distance 8 was the edge of its grid (4, 6, 8). — README
- Timing share of the transcript's error, category micro F1 old: k=2 0.749, k=4 0.815, k=8 0.840. Exact move: 0.685, 0.735, 0.750. — README
- 60 fps DINOv2-small on player squares: the dense model did not improve (F1 ±1 0.122 vs 0.125, ±2 0.181 vs 0.183, direction 0.554 vs 0.553). — README
- Refiner with those features: start exactly right 0.336 → 0.376 / 0.346 → 0.378; within ±1 0.792 → 0.828 / 0.776 → 0.806; within ±2 slightly **worse**, 0.933 → 0.925 / 0.919 → 0.912. The press is not improved. — README
- Labels are right to about ±1 frame (pixel check on 6 test replays), which bounds exact-frame numbers. — README
- Whiff and interrupted moves are found 0.57 and 0.64 of the time, against 0.90+ for contact moves. — README

**Literature**
- F3Set: stride 2 to 4 costs 14.4 F1_evt and 11.3 Edit at ±1 frame; stride 8 costs 26.3 F1_evt. Frame-wise TSM + E2E-Spot beats clip-wise I3D + E2E-Spot (31.4 vs 22.7 F1_evt, high granularity). — [F3Set](https://arxiv.org/html/2504.08222v2)
- E2E-Spot runs all frames at 25–30 fps (FineGym resampled down from up to 60) and gets, at δ=1, Tennis 96.1, FS-Comp 81.0, FS-Perf 85.1, FineDiving 68.4, FineGym-Full 47.9; at δ=0, 71.6, 36.7, 40.5, 30.1, 22.4. — [E2E-Spot](https://ar5iv.labs.arxiv.org/html/2207.10213)
- End-to-end vs frozen features: pretrained Kinetics features are 9.1–29.1 mAP below E2E-Spot at δ=1; fine-tuned two-phase features 3.9–25.1 below. — [E2E-Spot](https://ar5iv.labs.arxiv.org/html/2207.10213)
- Adding flow to RGB gains only 0.0–1.7 mAP at δ=1. — [E2E-Spot](https://ar5iv.labs.arxiv.org/html/2207.10213)
- T-DEED: output temporal resolution matters; predicting at coarser pyramid levels reduces precision, so the decoder restores full resolution. — [T-DEED](https://arxiv.org/html/2404.05392)
- SoccerNet Ball practice is 25 fps or lower: the 2024 second place and the 2023 winner use every second frame of 25 fps (12.5 fps), and 2025 participants describe 25 fps clips. — [SoccerNet 2024](https://arxiv.org/html/2409.10587), [lRomul repo](https://github.com/lRomul/ball-action-spotting), [int8 write-up](https://int8.io/team-ball-action-spotting-challenge-2025)
- GolfDB adds a counterpoint on sampling: real-time video outscored slow motion, 79.2 vs 72.5 PCE. The authors blame slower phases producing more frames that look like the event. — [GolfDB](https://ar5iv.labs.arxiv.org/html/1903.06528)

### Inferences
- **Most of the 0.47 close-start figure is structural.** Under the current head and an 8-frame merge, two starts under 8 frames apart in one tubelet cannot both be output. A two-slot head on existing tokens should be tried first. Only what stays missing can be attributed to the 7.5 Hz features.
- **Better timing has a low ceiling.** F1@8 minus F1@2 is 0.091 on category and 0.065 on exact move. A perfect localiser that finds no new events can gain about that much at k=2 and very little at k=4. The rest of the loss (F1@8 0.84) is events never found or wrongly named. Whiffs are in that group and are not a timing problem.
- Our 7.5 Hz tokens are about the sampling rate of F3Set's stride 4 at 30 fps, where it loses most, if stride means source frames. That supports 15 Hz tokens, but the tasks differ: F3Set's score needs ±1 frame at 25–30 fps, while our k=4 is ±67 ms. Our k=2 is ±33 ms, about equal to theirs.
- Evidence supports about 30 fps being enough. Nothing I found shows a gain past it, with the caveat that nothing tested it directly. Our one 60 fps test is mixed: +0.036 within ±1 and −0.008 within ±2, close to the size of the label error. It cannot show a true 60 fps need until labels are aligned per replay.
- V-JEPA 2's tubelet covers 2 input frames, so 15 Hz tokens mean 30 fps through the encoder. That is the most a 720p30 upload can supply anyway, so 30 fps is also the product's natural limit.

### Gaps
- **No source found comparing 25 vs 50 fps in SoccerNet Ball Action Spotting.** Searches, the 2024 report's section and the winners' material did not mention 50 fps. Don't state such a result without another source. T-DEED's fps for SoccerNet Ball was not in the text read.
- E2E-Spot's text as read has no stride ablation; F3Set's is the measured one. The last 9% of the E2E-Spot page was not read.
- F3Set has no stride 1 row, so it doesn't show whether 25–30 fps beats 12.5–15 fps.
- I found no study with sources above 30 fps on frame-exact game footage.

## Compression: effect of low-bitrate video, and of training with compression augmentation

### Takeaway
Accuracy holds until heavy compression, then falls fast. Training on compressed video mostly doesn't help except at extreme levels. Our 3 Mbit result is good for the chain but tests a single clean encode, not a real tournament VOD with overlays and repeated encodes.

### Cited Findings
- Own data: 720p30 3 Mbit (`libx264 veryfast`, maxrate 3.5M) chain 0.888 and `normal_acc` 0.960. — `results_degraded.json`, `model/degrade_eval.py`. `--light` 0.878 and 0.947 (`results_degraded_light.json`).
- Own data: the transcript and inputs models have no compressed test. Their numbers are 1080p60 only. — README
- Pretrained models on Kinetics, H.264, top-1 change: at CRF 30 −0.50 to −0.89; at CRF 40 −4.46 to −5.78; at CRF 50 −22.0 to −31.87 (X3D-M, R50, SlowFast, TimeSformer, Swin-B). CRF 20 and below shows no meaningful loss. — [Otani et al.](https://ar5iv.labs.arxiv.org/html/2204.09166)
- Trained on transcoded Kinetics, original → CRF 20 / 30 / 40: X3D-M 59.53 → 58.99 / 58.49 / 56.28; TimeSformer 76.90 → 77.06 / 76.58 / 74.80; R50 51.29 → 52.63 / 50.86 / 47.05. Gains appear in some cases at mild compression only. The fetch noted small table/text inconsistencies in the paper. — [Otani et al.](https://ar5iv.labs.arxiv.org/html/2204.09166)
- Two-stream on UCF101, not retrained → retrained: CRF 25 73.60 → 71.76; CRF 30 72.23 → 70.58; CRF 40 70.88 → 67.12; CRF 50 41.48 → 56.23. The motion stream is the weak part (CRF 40 from 38.96 to 9.26 after retraining). — [arXiv 2007.14314](https://ar5iv.labs.arxiv.org/html/2007.14314)
- Enhancement with pixel losses can over-smooth features useful to recognition (search-result summary only, full paper not read). — [Pourreza et al.](https://openaccess.thecvf.com/content_ICCVW_2019/html/RLQ/Pourreza_Recognizing_Compressed_Videos_Challenges_and_Promises_ICCVW_2019_paper.html)

### Inferences
- The chain tolerates 3 Mbit partly because of downsampling: a 427 px crop to 192 px averages out much of the codec noise. The same reasoning means **raising input resolution would also expose more compression artefacts**. High-resolution gains at 1080p60 clean may not carry to VODs at 3–8 Mbit. This is an argument against, and it's unmeasured here.
- Motion is the thing compression hurts first in the literature. Frame-level timing (the refiner) is therefore more at risk than the opener chain, and it has not been tested at all at low bitrate.
- Compression augmentation has no measured need on the chain today. Since it can hurt at moderate levels, run it only if a bitrate ladder shows a real drop.

### Gaps
- CRF-to-bitrate mapping for our content is unmeasured, so the literature's CRF steps cannot be placed against 3 Mbit.
- No real tournament VOD with logged labels exists, so our "typical VOD" is a simulation: a single encode of a clean recording, no overlay, no platform re-encode. Double-encode and overlay effects are unmeasured.
- I found no study of compression augmentation for action spotting; the one supporting source is on image detection and was not read in full ([arXiv 2211.05805](https://arxiv.org/pdf/2211.05805.pdf)).

## Super-resolution and high-resolution crops: do attention or crop-based approaches give the benefit at lower cost?

### Takeaway
Yes, crops give most of the benefit, and we already use them. Learned attention to crops would add little since the game positions are known. Super-resolution solves a problem we don't have.

### Cited Findings
- AdaFocus on Something-Something, patches against a full-frame TSM (46.1 / 59.1, 32.7 GFLOPs): 144² 47.0 / 59.7 at 23.5; 160² 47.6 / 60.2 at 27.5; 176² 48.1 / 60.7 at 33.7. — [AdaFocus](https://ar5iv.labs.arxiv.org/html/2105.03245)
- AdaFocus mAP by patch size (with global features): 96² 71.9, 128² 75.0, 160² 76.0, 192² 76.7, diminishing. The table is probably ActivityNet; the page didn't confirm this. — [AdaFocus](https://ar5iv.labs.arxiv.org/html/2105.03245)
- Video super-resolution on HMDB51, low-resolution input: spatial stream 42.81 → 53.59, temporal 56.54 → 61.5. — [Zhang et al. ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Zhang_Two-Stream_Action_Recognition-Oriented_Video_Super-Resolution_ICCV_2019_paper.html)
- Tiling work is on small-object detection. Azure's documentation says a 3×2 grid costs about 7× the computation. — [Azure docs](https://docs.azure.cn/en-us/machine-learning/how-to-use-automl-small-object-detect), [Unel et al.](https://openaccess.thecvf.com/content_CVPRW_2019/papers/UAVision/Unel_The_Power_of_Tiling_for_Small_Object_Detection_CVPRW_2019_paper.pdf)
- Own data: crops plus 2×2 tokens +0.038. Scene stream: whole frame at 384×216, added for projectiles and hits away from the attacker. Whole chain 0.79 with the attacker crop alone. — README
- Own data, weak points tied to where the crop looks: projectile category 0.77 / 0.69 right. Teleports, overlaps and cinematics can make both squares show the same player. — README

### Inferences
- We've effectively built AdaFocus by hand, with a far better localiser (characters right on every old replay). A learned-patch policy has no room to pay off.
- Reject super-resolution: it helps when pixels are missing, and the 720p crop already has 1.7–2.2× more pixels per side than the 192–256 px inputs (1.3× the transcript's 320 px decode). It only becomes relevant for sub-480p sources, and the app rejects under 360p. Separate check first with a 480p copy. (`lab_remote` `too_small`, `docs/desktop-login.md`.)
- Tiling is also a poor fit: characters are not tiny, and its cost is multiplicative.
- The crop-based idea most likely to pay is a **second, tighter crop around the active limb or weapon**. It has no direct evidence on our data, so it is an option after experiment E3 below, not a recommendation.
- Projectile errors are about what's seen in the frame, not sharpness: the scene stream sees them at 0.2× and could be tested at a modestly higher size alone, with less cost than increasing crops.

### Gaps
- No study found of a limb/weapon crop or hand-detail stream in fighting games.
- Attention/crop results above are classification, not spotting.
- AdaFocus numbers compare to its own baseline; the input size of TSM wasn't stated in the text read.

## Where would higher resolution likely matter given our numbers, what cheap experiments settle it, and what does it cost?

### Takeaway
Most likely button/weapon confusions in the transcript and possibly whiffs, not close starts or the opener chain. A one-night ladder on existing assets can settle it. Don't re-extract the 1,700 replays or change the upload until it does.

### Cited Findings (own data unless linked)
- Where it could matter:
  - Button/stance: opener chain 0.96 given found, and `normal_acc` 0.960 at 720p (`results_degraded.json`); the transcript's F1@4 is 0.712 / 0.685 at a 320→256 input.
  - Whiffs: found 0.57, move right when found 0.82 vs 0.93–0.94 for contact.
  - Rare: `<5` moves score 0.57; these are a catch-all class, so a data problem.
  - Throws 0.68 F1 in the transcript but 0.985–1.00 recall on the openers' category (`results_exp_moves_r224f12.json`, `results_degraded.json`).
  - Close starts 0.47; presses without a visible move 0.09 (README: "likely not at all" visible).
- Costs:
  - Transcript features: 6.7 h for three views plus 4.7 h for the grid, 1,090 replays; 10.4 GB. GPU 99.8% busy at 98 sampled frames/s, 1.6 GB of GPU memory. Training holds all features on the GPU at about 15 GB peak. README.
  - Encoder throughput: about 290 frames/s at 256×256. ROADMAP.
  - Inference on a median match: about 62 s wall, 21 s of it GPU; heads 3 s. README.
  - 60 fps DINOv2 on 563 replays: 2.4 h, 7.1 GB. README.
  - ResNet18 move net: 146 min for 20 epochs from ImageNet on the old split; ResNet18 feature extraction 10 min. README.
  - CPU server: 4 min 15 s and 4.4 GB per 3-minute match on 4 threads. ROADMAP.
- Upload: limit 500 MiB (`docs/desktop-login.md`) and 15 min. The app stream-copies when small enough, otherwise encodes to 720p30, 3 Mbit. — `app/src-tauri/src/lab_remote.rs`
- The chain's decoder upscales any input to 1920×1080 and resamples it to 60 fps before cropping. — `model/video_crops.py` `read_frames`

### Inferences
**Costs at 1,700 replays (arithmetic from the numbers above; none measured)**

The transcript pass is GPU-bound (99.8% busy), so it scales with the number of tokens at least. ViT attention can be worse, up to quadratic.

| Setting (three views unless noted) | GPU time | Disk | Notes |
|---|---|---|---|
| Current, 256 px, 15 fps | 10.4 h, plus 7.3 h grid (or one combined pass) | 16 GB | Baseline |
| 30 fps, 256 px | 21 h, plus 15 h grid | 32 GB | Roadmap says double |
| 384 px, 15 fps | 23–52 h (×2.25 to ×5) | 16 GB | With the same grid; position embedding must interpolate; memory 1.6 GB grows |
| 384 px, 30 fps | 47–104 h | 32 GB | Not worth considering before E3 |
| 512 px, 15 fps | 42–166 h | 16 GB | Rejected on cost alone |
| Grid 4×4, 256 px | Same encoder pass | About 21 GB for grid part (vs 5.3) | No new pixels |

- Training memory is a hidden limit: 15 GB × 1.56 is about 23.4 GB at 1,700 replays with current features, close to the 3090's 24 GB. Doubling the rate or grid means streaming features from RAM or disk, so training would no longer be 12.5 min per net.
- Source data: 849 GB / 1,700 is about 500 MB per replay, or about 123 s at the 32 Mbit/s implied by 730 MB per 3 min. 720p30 3 Mbit copies of all would take about 78 GB, if we keep them as an augmentation or test source.
- Upload: 500 MiB allows 23 Mbit/s for 3 minutes but only 4.7 Mbit/s for the 15-minute maximum. So a 1080p60 8 Mbit upload fits one match (180 MB) and not a full set (900 MB). A 1080p tier is feasible only for single matches, and the evidence does not ask for it.
- Decoding a 720p30 upload to 1080p60 buys the chain nothing in pixels. Using a direct 720p decode could save CPU on the server, but must be checked for identical crops. An aside, not measured.

**Experiments, cheapest first. All on the frozen fold; report old / new / 720p30 with paired bootstrap over replays.**

| # | Experiment | Assets | Estimated cost | What it settles |
|---|---|---|---|---|
| E0 | Look at existing button errors: how many are S vs H in saved test probabilities; view 50 errors at model input and at native pixels | `video_moves_v2_test.pt`, tiles | Under 1 h, no GPU | If a person can tell them apart at 192 px, pixels are not the limit |
| E1 | Two-slot head or wider merge on existing tokens; per-replay ±1 label alignment | Existing features | 1–2 h GPU (net 12.5 min) | How much of 0.47 is structural; sets the true baseline for any fps test |
| E2 | Transcript, refiner and state stream on the existing 720p30 copies | 144 old copies | About 1–3 h for the features; the refiner test needs 30 fps frame features | Owed anyway by the roadmap; whether the +0.036 refiner gain survives 30 fps |
| E3 | Resolution sensitivity with one axis at a time: ResNet18 at **128 vs 192 vs 288 px**, same 8 frames, split and epochs. Score button, normals, move and chain | Tiles are 224 px, so 288 needs a re-cut from 1080p | 128: about 1 h; 288: about 5.5 h (146 min × 2.25) plus re-cut | The only clean measurement of the slope. **If 128 loses little, stop.** |
| E4 | Grid 4×4 from V-JEPA at 256 px, on the 563-replay subset vs its 2×2 control | Encoder pass on subset | About 3–6 h at batch rates; 1 h heads | Whether spatial information was lost after the encoder rather than in pixels |
| E5 | Bitrate ladder for chain and transcript: 3, 1.5, 0.8 Mbit at 720p30; 480p; double encode; overlay | `degrade_eval.py` | Encode plus 2–3 h each | Whether augmentation is needed at all, and the real floor |
| E6 | 30 fps tokens (15 Hz) on the 563 subset vs control, only if E1 still shows missing close starts | Subset | About 7 h, 11 GB | The real benefit of more frames for close starts and whiffs |
| E7 | V-JEPA 384 px player views on the same subset, only if E0, E3 or E4 are positive | Subset | About 8–16 h | Last resort, the expensive test |

Suggested decision rule (mine, not measured): adopt a higher-cost setting only if the paired-bootstrap lower bound is above zero **and** the gain reaches about 0.01 chain or 0.02 on button/normals, **on the 720p30 copies**. A gain only on clean 1080p60 does not count, since the server and VODs don't see that.

**Plain conclusion**
- Higher source or upload resolution: **no case**, with firm evidence for the opener chain. Unmeasured for the transcript, though its 320 px decode makes a gain unlikely.
- Higher model input resolution: **weak case.** One confounded test on our data was slightly negative. Literature gains are about 2–4 points for a 2× linear increase, mostly where subjects are tiny. Costs are at least 2.25×, which directly harms the priority of ROADMAP point 6 (a CPU student).
- Close starts and whiffs are unlikely to be resolution problems. Close starts are first a head-structure issue, and whiffs a detection one.
- Frame rate: moderate case for 15 Hz tokens at 30 fps, after E1. No case for 60 fps inputs in production unless the refiner gain survives alignment, since uploads are 30 fps.
- Better uses of the same budget, on measured gains: post-encoder spatial tokens (+0.038), fine-tuning V-JEPA 2 (+0.025 / +0.043 chain over frozen, still below the ResNets), and data (0.865 to 0.887).

### Gaps
- No measured transcript result at any input size other than 256 px, and none at other rates than 15 fps; the table above is arithmetic.
- 384 px through V-JEPA 2 is untested here: GPU memory and whether interpolated position embeddings hold quality are unknown.
- Stated "1.6× cost" for the 224/12 run comes from the assignment. It's not in the results files. Pixel-frame count is 2.04×.
- Whiff scores don't exist for the opener net, so E3 can't address whiffs; only E7 could.
- The README's 1,090 replays and the assignment's 1,700 differ; scaling uses ×1.56 linearly.
