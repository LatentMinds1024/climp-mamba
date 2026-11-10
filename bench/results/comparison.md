# CLIMP: paper vs. released checkpoint (our eval) vs. ours

Protocol for all evaluated columns: authors' CLIMP repo (square bicubic resize, ImageNet norm, no text truncation, fp16 autocast), CLIP-Benchmark templates/classnames/metrics. "Image retrieval" = text→image (T→I); "text retrieval" = image→text (I→T).

## 0. Reproduction check — NoCaps val @224 vs authors' README

| | T→I R@1 | T→I R@5 | T→I R@10 | I→T R@1 | I→T R@5 | I→T R@10 |
|---|---|---|---|---|---|---|
| Authors' README | 38.2 | 70.5 | 81.8 | 51.8 | 81.9 | 90.9 |
| Paper ckpt (our eval) | 38.3 | 70.6 | 81.8 | 52.6 | 82.3 | 90.6 |
| Ours | 25.5 | 53.2 | 65.8 | 35.9 | 65.7 | 77.6 |

## 1. CLIP-Benchmark summary (Table 1) @224

Acc = mean over the 25 classification datasets of paper Table 9; IR@5/TR@5 = mean over flickr30k, flickr8k, mscoco (Table 10).

| Model | Acc@1 | Acc@5 | IR@5 (T→I) | TR@5 (I→T) |
|---|---|---|---|---|
| Paper, CLIMP-Mamba1 (Table 1 as printed) | 29.6 | 58.5 | 65.5 | 77.0 |
| Paper, CLIMP-Mamba1 (recomputed from Tables 9/10, same 25+3 datasets) | 27.9 | 58.5 | 65.5 | 77.0 |
| Paper, best transformer (RoPE-ViT-B/16 + LLaMA-1B) | 27.3 | 59.0 | 63.4 | 72.9 |
| Paper ckpt (our eval) | 28.2 | 58.6 | — | — |
| Ours (partial: 0/25) | — | — | — | — |

## 2. OOD robustness (Table 2) @224, Acc@1 / Acc@5

| Model | IN-1k | IN-V2 | IN-R | IN-A | IN-O | Sketch | Avg (5 variants) |
|---|---|---|---|---|---|---|---|
| Paper CLIMP-Mamba1 (Table 9 cols) | 44.0/74.8 | 37.5/68.4 | 46.6/74.5 | 15.6/45.3 | 49.8/78.8 | 27.5/54.4 | 35.4/64.3 |
| Paper ckpt (our eval) | 43.8/74.7 | 37.5/68.2 | 46.9/74.7 | 15.2/44.6 | 49.5/78.3 | 27.6/54.5 | 35.3/64.1 |
| Ours | —/— | —/— | —/— | —/— | —/— | —/— | —/— |

Note: the paper's Table 2 and Table 9 disagree on which CLIMP variant has IN-O 48.1 vs 49.8; Table 9 values are shown.

## 3. Per-dataset zero-shot classification (Table 9) @224, Acc@1

| Dataset | Paper CLIMP-M1 | Paper best transformer | Paper ckpt (our eval) | Ours |
|---|---|---|---|---|
| cars | 9.1 | 8.9 | 8.6 | — |
| country211 | 4.3 | 4.7 | 4.3 | — |
| fer2013 | 18.8 | 35.9 | 21.3 | — |
| fgvc_aircraft | 2.5 | 2.6 | 2.3 | — |
| imagenet-a | 15.6 | 16.3 | 15.2 | — |
| imagenet-o | 49.8 | 40.1 | 49.5 | — |
| imagenet-r | 46.6 | 47.8 | 46.9 | — |
| imagenet1k | 44.0 | 40.7 | 43.8 | — |
| imagenet_sketch | 27.5 | 27.4 | 27.6 | — |
| imagenetv2 | 37.5 | 34.4 | 37.5 | — |
| objectnet | 28.1 | 33.1 | 27.6 | — |
| sun397 | 48.2 | 53.2 | 47.9 | — |
| voc2007 | 75.8 | 77.9 | 76.3 | — |
| cifar100 | 48.3 | 66.3 | 50.9 | — |
| cifar10 | 83.0 | 94.2 | 85.9 | — |
| clevr_closest_object_distance | 18.3 | 18.7 | 18.8 | — |
| clevr_count_all | 16.2 | 19.8 | 15.8 | — |
| diabetic_retinopathy | 71.3 | 30.7 | 71.7 | — |
| dmlab | 17.8 | 21.2 | 17.7 | — |
| dsprites_label_orientation | 2.8 | 2.7 | 2.7 | — |
| dsprites_label_x_position | 3.2 | 3.2 | 3.2 | — |
| dsprites_label_y_position | 3.4 | 3.2 | 3.5 | — |
| smallnorb_label_azimuth | 5.1 | 5.4 | 5.0 | — |
| smallnorb_label_elevation | 10.3 | 12.9 | 10.0 | — |
| svhn | 9.9 | 20.8 | 10.0 | — |

## 4. Zero-shot retrieval (Table 10) @224 — R@1 / R@5 / R@10

| Dataset | Direction | Paper CLIMP-M1 R@5 | Paper ckpt (our eval) | Ours |
|---|---|---|---|---|
| flickr30k | I→T | 87.5 | —/—/— | —/—/— |
| flickr30k | T→I | 76.5 | —/—/— | —/—/— |
| flickr8k | I→T | 81.7 | —/—/— | —/—/— |
| flickr8k | T→I | 70.5 | —/—/— | —/—/— |
| mscoco_captions | I→T | 61.9 | —/—/— | —/—/— |
| mscoco_captions | T→I | 49.6 | —/—/— | —/—/— |

## 5. Resolution scaling on CLIP-Benchmark (Table 3) — Acc@1/Acc@5 and IR@5/TR@5

| Model | 224 | 320 | 384 |
|---|---|---|---|
| Paper CLIMP-Mamba1 | 29.6/58.5 · 65.5/77.0 | 28.4/57.5 · 66.0/76.9 | 27.3/56.3 · 65.0/75.2 |
| Paper ckpt (our eval) | 28.2/58.6 · —/— | 24.5/52.4 · —/— | —/— · —/— |
| Ours | —/— · —/— | —/— · —/— | —/— · —/— |

## 6. High-resolution retrieval, NoCaps & Crossmodal-3600 (Tables 4, 11) — R@5

Paper Table 11 is printed with columns "I→T, T→I", but the authors' README for the same model gives NoCaps@224 image-retrieval (T→I) R@5 = 70.5 and text-retrieval (I→T) R@5 = 81.9, i.e. the printed labels are swapped. Below the paper values are placed under the corrected direction.

| Dataset | Res | Paper CLIMP-M1 T→I / I→T | Paper RoPE-ViT T→I / I→T | Paper ckpt (our eval) T→I / I→T | Ours T→I / I→T |
|---|---|---|---|---|---|
| nocaps | 224 | 70.5 / 81.9 | 67.6 / 78.9 | 70.6 / 82.3 | 53.2 / 65.7 |
| nocaps | 512 | 66.9 / 72.7 | 65.1 / 75.9 | 66.5 / 73.1 | 49.4 / 53.6 |
| nocaps | 896 | 57.1 / 53.5 | 38.6 / 34.9 | 57.1 / 53.6 | 40.6 / 38.6 |
| xm3600 | 224 | 65.1 / 69.1 | 64.4 / 68.2 | 64.4 / 68.8 | 42.0 / 48.5 |
| xm3600 | 512 | 62.6 / 60.8 | 62.3 / 64.1 | 63.3 / 61.4 | 40.3 / 43.1 |
| xm3600 | 896 | 55.9 / 45.4 | 36.8 / 26.9 | 53.8 / 43.1 | 34.8 / 33.0 |

## 7. Dense / long-caption retrieval (Tables 5, 12, 13) — R@1/R@5/R@10

| Benchmark | Res | Direction | Paper CLIMP-M1 | Paper best transformer | Paper ckpt (our eval) | Ours |
|---|---|---|---|---|---|---|
| DOCCI test (5k) | 896 | T→I | 37.3/67.1/77.8 | 29.1/55.3/67.5 (NaFlex) | 36.5/66.4/77.8 | 12.9/32.1/42.8 |
| DOCCI test (5k) | 896 | I→T | 23.8/50.4/62.7 | 26.1/53.4/65.1 (NaFlex) | 23.6/50.0/61.9 | 9.7/26.9/37.3 |
| DOCCI test (5k) | 224 | T→I | — | — | 50.2/79.7/87.9 | 12.9/31.6/43.6 |
| DOCCI test (5k) | 224 | I→T | — | — | 52.0/80.4/89.3 | 15.9/37.1/47.9 |
| Urban1k (extra) | 224 | T→I | — | — | 40.3/65.8/75.2 | —/—/— |
| Urban1k (extra) | 224 | I→T | — | — | 48.4/73.0/83.3 | —/—/— |
| Flickr8k-Rephrased | 224 | T→I / I→T | 67.0/89.4/94.7 · 81.3/95.9/98.4 | — | not released | not released |

## 8. Embedding geometry on NoCaps (Table 6) — lower is better

Alignment = E‖f(x)−g(y)‖² over matched pairs (α=2); uniformity = log E exp(−2‖u−v‖²) (t=2); hubness = skewness of the k-occurrence distribution, k=10, cross-modal. The paper does not state k/t; these are the standard defaults.

| Model | Alignment | Unif. image | Unif. text | Hub. image | Hub. text |
|---|---|---|---|---|---|
| Paper CLIMP-Mamba1 | 0.982 | -3.18 | -3.54 | 1.01 | 1.15 |
| Paper RoPE-ViT | 1.052 | -3.16 | -3.52 | 1.04 | 1.25 |
| Paper ckpt (our eval) | 0.979 | -3.199 | -3.534 | 1.720 | 3.981 |
| Ours | 0.910 | -3.405 | -3.502 | 1.764 | 3.350 |

## 9. Efficiency (Fig. 3 + latency / throughput / energy)

GPU: NVIDIA L40S. Vision towers, batch 1 (memory = resolution-specific peak above resident weights).

| Tower | Res | Params (M) | GFLOPs | Mem fp32 (MB) | Lat fp32 (ms) | Lat bf16 (ms) | Thr bf16 (img/s) | Energy bf16 (J/img) |
|---|---|---|---|---|---|---|---|---|
| VMamba-B (CLIMP) | 64 | 87.5 | 1.3 | 40.4 | 10.7 | 12.0 | 4673 | 1.2 |
| ViT-B/16 naive-attn | 64 | 85.7 | 1.5 | 3.2 | 1.9 | 2.1 | 27668 | 0.6 |
| ViT-B/16 SDPA | 64 | 85.7 | 1.5 | 3.2 | 1.4 | 1.6 | 31213 | 0.7 |
| VMamba-B (CLIMP) | 128 | 87.5 | 5.0 | 38.9 | 10.4 | 11.7 | 2253 | 1.3 |
| ViT-B/16 naive-attn | 128 | 85.7 | 5.6 | 4.4 | 1.9 | 2.2 | 8667 | 0.0 |
| ViT-B/16 SDPA | 128 | 85.7 | 5.6 | 4.4 | 1.4 | 1.6 | 9846 | 0.9 |
| VMamba-B (CLIMP) | 224 | 87.5 | 15.4 | 59.3 | 10.1 | 12.1 | 784 | 1.4 |
| ViT-B/16 naive-attn | 224 | 85.8 | 17.6 | 8.1 | 1.9 | 2.2 | 1857 | 0.0 |
| ViT-B/16 SDPA | 224 | 85.8 | 16.9 | 7.5 | 1.4 | 1.6 | 2875 | 0.0 |
| VMamba-B (CLIMP) | 256 | 87.5 | 20.1 | 71.5 | 10.0 | 12.1 | 594 | 1.9 |
| ViT-B/16 naive-attn | 256 | 85.8 | 23.2 | 13.3 | 1.8 | 2.2 | 1258 | 0.7 |
| ViT-B/16 SDPA | 256 | 85.8 | 22.0 | 11.8 | 1.7 | 1.6 | 2241 | 0.0 |
| VMamba-B (CLIMP) | 384 | 87.5 | 45.1 | 164.7 | 10.0 | 11.4 | 238 | 2.4 |
| ViT-B/16 naive-attn | 384 | 86.1 | 55.5 | 44.4 | 2.6 | 2.2 | 343 | 0.9 |
| ViT-B/16 SDPA | 384 | 86.1 | 49.4 | 22.2 | 2.6 | 1.6 | 879 | 0.0 |
| VMamba-B (CLIMP) | 512 | 87.5 | 80.2 | 287.1 | 9.8 | 12.0 | 138 | 2.3 |
| ViT-B/16 naive-attn | 512 | 86.4 | 107.1 | 121.2 | 7.7 | 5.8 | 122 | 2.1 |
| ViT-B/16 SDPA | 512 | 86.4 | 87.8 | 44.0 | 5.5 | 2.3 | 483 | 1.0 |
| VMamba-B (CLIMP) | 768 | 87.5 | 180.5 | 644.2 | 20.1 | 15.5 | 60 | 5.5 |
| ViT-B/16 naive-attn | 768 | 87.4 | 295.3 | 535.3 | 31.3 | 32.8 | 30 | 10.6 |
| ViT-B/16 SDPA | 768 | 87.4 | 197.4 | 89.8 | 16.9 | 4.8 | 187 | 2.0 |
| VMamba-B (CLIMP) | 896 | 87.5 | 245.7 | 881.6 | 27.8 | 19.0 | 45 | 6.7 |
| ViT-B/16 naive-attn | 896 | 88.1 | 450.0 | 965.7 | 53.5 | 57.9 | 17 | 17.8 |
| ViT-B/16 SDPA | 896 | 88.1 | 268.6 | 119.5 | 28.5 | 7.4 | 135 | 2.1 |
| VMamba-B (CLIMP) | 1024 | 87.5 | 320.9 | 1144.5 | 37.4 | 23.9 | 34 | 7.7 |
| ViT-B/16 naive-attn | 1024 | 88.8 | 660.2 | 1620.8 | 86.5 | 95.6 | 10 | 30.0 |
| ViT-B/16 SDPA | 1024 | 88.8 | 350.8 | 156.0 | 40.7 | 9.5 | 93 | 2.3 |

Text towers (batch 1):

| Tower | Tokens | Params (M) | ~GFLOPs | Lat bf16 (ms) | Mem bf16 (MB) | Thr bf16 (captions/s) |
|---|---|---|---|---|---|---|
| in-house Mamba-1 99M (ours) | 16 | 99 | 1.9 | 6.2 | 249 | 6749 |
| in-house Mamba-1 99M (ours) | 77 | 99 | 9.3 | 6.5 | 77 | 3998 |
| in-house Mamba-1 99M (ours) | 142 | 99 | 17.1 | 6.4 | 79 | 2404 |
| in-house Mamba-1 99M (ours) | 256 | 99 | 30.9 | 6.4 | 82 | 2313 |
| in-house Mamba-1 99M (ours) | 512 | 99 | 61.8 | 6.5 | 91 | 1129 |
| Mamba-1 1.4B (paper) | 16 | 1372 | 40.6 | 18.6 | 1556 | 774 |
| Mamba-1 1.4B (paper) | 77 | 1372 | 195.5 | 18.7 | 1560 | 382 |
| Mamba-1 1.4B (paper) | 142 | 1372 | 360.5 | 18.7 | 1566 | 225 |
| Mamba-1 1.4B (paper) | 256 | 1372 | 649.8 | 18.8 | 1573 | 184 |
| Mamba-1 1.4B (paper) | 512 | 1372 | 1299.7 | 18.9 | 1591 | 95 |

Paper (Fig. 3 text): at 896×896 CLIMP 10.0 MB vs ViT 50.4 MB resolution-specific memory; 259.7 vs 457.9 GFLOPs.

## 10. Qualitative similarity (Fig. 2, 5) — same NoCaps images and captions as the paper

| Caption | Paper CLIMP score | Paper ckpt (our eval) | Ours |
|---|---|---|---|
| A large porch with a wooden fence and no roof. | 0.545 | 0.549 | 0.670 |
| A woman with red hair hugging a man. | 0.628 | 0.633 | 0.561 |
| A hand holding a glass that has some wine in it. | 0.646 | 0.651 | 0.740 |

Maps: bench/results/qualitative.png
