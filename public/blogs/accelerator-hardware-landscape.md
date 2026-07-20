---
title: "Training Silicon Across a Decade: NVIDIA, AMD, TPU, and Ascend Compared"
date: "2026/7/20"
---

Every post in this series so far has treated "the GPU" as a stand-in for the whole category — the [first post](#/blog?id=gpu-field-guide-for-dl) built a mental model around one, mostly-NVIDIA-shaped chip. That's a reasonable simplification for learning the concepts, but it quietly erases a real and interesting fact: there are at least four seriously different lineages of training silicon in production today, from four organizations that have made four genuinely different bets about what matters. This post is a pure hardware comparison, restricted to what actually matters for training large models — memory capacity, memory bandwidth, compute throughput, interconnect, power — across NVIDIA's datacenter GPUs, AMD's Instinct MI series, Google's TPUs, and Huawei's Ascend accelerators, laid out chronologically within each lineage and then compared across all four. Two things are deliberately out of scope: software ecosystems (CUDA, ROCm, XLA, CANN, and so on), and anything that hasn't actually shipped yet — every chip below is a real, purchasable (or at least deployed) product, not a roadmap slide.

A note on confidence before the numbers: NVIDIA and AMD publish detailed datasheets, so those sections are the most solid. Google discloses generous architectural detail in blog posts but has never once published an official TDP for any TPU — those figures are third-party estimates, marked as such. Huawei is the genuinely hard case: post-2020 export controls mean there is close to no official spec-sheet culture for Ascend chips, and several of the numbers in that section are best-available third-party estimates rather than vendor-confirmed figures. Anywhere a number is an estimate rather than a vendor-published spec, it's marked.

## 1. NVIDIA: the incumbent, expanding on every axis at once

| Chip | Year | Memory | Bandwidth | Dense compute (BF16/FP16-class) | Interconnect | TDP |
|---|---|---|---|---|---|---|
| P100 | 2016 | 16 GB HBM2 | 0.72 TB/s | 21.2 TFLOPS FP16 (no Tensor Cores) | NVLink 1, 160 GB/s | 300 W |
| V100 | 2017 | 32 GB HBM2 | 0.9 TB/s | 125 TFLOPS (1st-gen Tensor Cores) | NVLink 2, 300 GB/s | 300 W |
| A100 (80GB) | 2020 | 80 GB HBM2e | 2.04 TB/s | 312 TFLOPS (TF32 introduced) | NVLink 3, 600 GB/s | 400 W |
| H100 SXM5 | 2022 | 80 GB HBM3 | 3.35 TB/s | 990 TFLOPS (**FP8 introduced**, 1979 TFLOPS) | NVLink 4, 900 GB/s | 700 W |
| H200 | 2024 | 141 GB HBM3e | 4.8 TB/s | 990 TFLOPS (same die as H100, memory-only refresh) | NVLink 4, 900 GB/s | 700 W |
| B200 | 2024 | 192 GB HBM3e | 8 TB/s | 2,250 TFLOPS (**FP4 introduced**, ~9 PFLOPS FP4 sparse) | NVLink 5, 1.8 TB/s | 1,000 W |
| B300 / GB300 | 2025 | 288 GB HBM3e | 8 TB/s | ~2,500 TFLOPS (FP64 sharply de-emphasized vs. B200) | NVLink 5, 1.8 TB/s | 1,400 W |

The clearest structural break in this table is at Blackwell (B100/B200/B300): every NVIDIA datacenter GPU up through H200 was a single monolithic die, pushed right up against the reticle size limit (H100's die is 814 mm²). Blackwell is the first generation to go **dual-die** — two reticle-sized dies joined by a 10 TB/s die-to-die link, presented to software as a single GPU — because there was no more room to grow a single die, only to add a second one next to it.

The other clear pattern is the "rack as the real product" shift. Starting with GB200 NVL72, NVIDIA's highest-end offering isn't really a chip you buy one of — it's a 72-GPU, liquid-cooled rack sharing a single NVLink domain with 130+ TB/s of aggregate bandwidth, and per-GPU numbers increasingly need that context to mean anything. TDP has grown right alongside all of this — 300 W (Pascal) to 700 W (Hopper) to 1,000–1,400 W (Blackwell) — nearly a 5× increase in under a decade, and the reason liquid cooling stopped being optional starting with Blackwell Ultra.

## 2. AMD Instinct: betting on memory capacity

| Chip | Year | Memory | Bandwidth | Dense compute (BF16/FP16-class) | Interconnect | TDP |
|---|---|---|---|---|---|---|
| MI100 | 2020 | 32 GB HBM2 | 1.2 TB/s | 184.6 TFLOPS FP16 | Infinity Fabric, ~276 GB/s | 300 W |
| MI250X | 2021 | 128 GB HBM2e | 3.2 TB/s | 383 TFLOPS | Infinity Fabric, 8×100 GB/s | 560 W |
| MI300X | 2023 | 192 GB HBM3 | 5.3 TB/s | 1,307 TFLOPS (**FP8 introduced**, 2,615 TFLOPS) | Infinity Fabric, 7×128 GB/s | 750 W |
| MI325X | 2024 | 256 GB HBM3e | 6.0 TB/s | 1,307 TFLOPS (same compute dies as MI300X) | Infinity Fabric, 7×128 GB/s | 1,000 W |
| MI355X | 2025 | 288 GB HBM3e | 8 TB/s | ~2,500 TFLOPS (**MXFP6/MXFP4 introduced**) | Infinity Fabric, 7×153.6 GB/s | 1,400 W |

AMD went multi-die earlier than anyone else at this scale — MI250X (2021) already packages two compute dies (GCDs), and MI300X (2023) pushes to eight compute chiplets plus separate I/O dies, a genuinely more aggressive chiplet strategy than NVIDIA adopted until Blackwell three years later. The strategic signature that shows up clearly in the numbers, though, is memory capacity. Checked against NVIDIA's contemporaneous flagship at each point in time, AMD's memory lead has been real and consistent, even as it's been narrowing:

| Era | AMD chip | AMD memory | NVIDIA contemporary | NVIDIA memory | AMD's edge |
|---|---|---|---|---|---|
| Late 2023 | MI300X | 192 GB | H100 SXM5 | 80 GB | 2.4× |
| Late 2024 | MI325X | 256 GB | H200 | 141 GB | 1.8× |
| 2025 | MI355X | 288 GB | B200 | 192 GB | 1.5× |

The trade-off is legible too: AMD has generally matched, rather than led, on raw bandwidth per chip in the most recent generations (MI355X's 8 TB/s ties B200 exactly), and NVIDIA's NVLink fabric remains well ahead of AMD's Infinity Fabric ecosystem in scale-up sophistication and aggregate rack-level bandwidth. The fair read: AMD trades interconnect polish for the biggest memory pool you can buy per chip — useful for fitting a bigger model or KV-cache on fewer GPUs — while NVIDIA has prioritized bandwidth and multi-GPU fabric scale.

## 3. Google TPU: winning on architecture, not raw scaling

| Generation | Year | Memory | Bandwidth | Dense compute (BF16) | Interconnect | TDP |
|---|---|---|---|---|---|---|
| v2 | 2017 | 8 GB HBM | ~0.25 TB/s (est.) | 45 TFLOPS | ICI (undisclosed) | not disclosed |
| v3 | 2018 | 32 GB HBM2 | 0.9 TB/s | 123 TFLOPS | ICI | 123–262 W (measured) |
| v4 | 2021 | 32 GB HBM2 | 1.2 TB/s | 275 TFLOPS | ICI, **optical circuit switches** | 90–192 W (measured) |
| v5e | 2023 | 16 GB HBM2 | 0.82 TB/s | 197 TFLOPS | ICI, 400 GB/s | not disclosed |
| v5p | 2023 | 95 GB HBM2e | 2.77 TB/s | 459 TFLOPS | ICI, 1.2 TB/s | not disclosed |
| v6e (Trillium) | 2024 | 32 GB HBM | 1.64 TB/s | 918 TFLOPS | ICI, 800 GB/s | not disclosed |
| v7 (Ironwood) | 2025 | 192 GB HBM (2×96 GB chiplets) | 7.38 TB/s | 2,307 TFLOPS (**FP8 introduced**, 4,614 TFLOPS) | ICI, 1.2 TB/s + fast die-to-die link | ~1,000 W (back-calculated est.) |

Two things make TPU's table read differently from the other three. First, Google has never once published TDP for any generation — every figure in that column is a third-party estimate (SemiAnalysis, ServeTheHome), except the two generations (v3, v4) where Google's own Cloud docs happen to list measured min/mean/max power draw. Second, and more interesting: the jump from v5p to Trillium (v6e) delivered roughly double the peak FLOPS in a single generational step — the gain came from quadrupling the systolic array from 128×128 to 256×256 MXU tiles, an architectural change rather than simply scaling up power or die size. That's the cleanest "architecture, not brute force" data point in this whole comparison.

Ironwood (v7) is TPU's Blackwell moment: it abandons the unified "MegaCore" die design used since v4 in favor of two chiplets per chip, and it's the first TPU with native FP8 hardware — a real jump in memory capacity (32 GB → 192 GB) and bandwidth (1.64 → 7.38 TB/s) in one step, both squarely aimed at the memory-bound parts of large model training.

## 4. Huawei Ascend: scaling out to compensate for scaling up less

| Chip | Year | Memory | Bandwidth | Dense compute (BF16/FP16-class) | Interconnect | TDP |
|---|---|---|---|---|---|---|
| 910 | 2019 | 32 GB HBM2 | 1.23 TB/s | ~320 TFLOPS FP16 | HCCS (early) | ~310 W |
| 910B | 2023 | 64 GB HBM2e | 1.6 TB/s | ~400 TFLOPS FP16 (est.) | HCCS, ~336 GB/s (est.) | ~400 W (est.) |
| 910C | 2025 | 128 GB HBM2e (reported) | ~3.2 TB/s (est.) | ~800 TFLOPS (est.) | HCCS + die-to-die (unconfirmed) | not disclosed |

Ascend is the lineage where the public numbers carry the most uncertainty of the four — Huawei doesn't publish detailed datasheets the way NVIDIA and AMD do, and most figures above are compiled from Chinese-market cloud documentation, conference disclosures, and third-party analysis rather than an official spec sheet, so treat them as directional rather than exact. What's clear from even the directional numbers: each generation roughly doubles memory capacity (32 → 64 → 128 GB) and bandwidth, which is exactly the axis that matters most for fitting bigger models and longer context windows onto a single accelerator.

The 910C (2025) is also a confirmed dual-chiplet package — two 910B-class compute dies in one package, the same structural move NVIDIA made with Blackwell and AMD made years earlier — a way of adding more memory, bandwidth, and compute per accelerator without needing a single larger die. And the training-relevant story that matters most for this lineage isn't really about any one chip at all: Huawei leans harder than any other vendor here on cluster-scale systems as the actual product. The CloudMatrix 384 system links 384 Ascend accelerators into a single tightly-coupled cluster, aiming to make up in aggregate cluster memory, bandwidth, and compute what any individual chip in the lineup doesn't match on its own.

![Four different bets on the same problem](blogs/images/accelerator-four-philosophies.svg?v=1)
^Same underlying constraint — not enough memory, bandwidth, or compute on one die — four structurally different answers to it.

## 5. Everyone eventually went multi-die

Look across all four tables and one architectural transition shows up in every single lineage, just on a different clock: the move from a single monolithic die to a package of multiple compute chiplets. AMD got there first, out of necessity, with MI250X in 2021. NVIDIA held out on a monolithic H100/H200 die through 2024, then switched decisively with Blackwell once a bigger single die became physically impossible to manufacture. TPU held out even longer, keeping its "MegaCore" unified-die design all the way through Trillium (2024), before Ironwood (2025) adopted the same dual-chiplet pattern. Huawei's 910C (2025) did the same.

![Everyone eventually went multi-die](blogs/images/accelerator-chiplet-timeline.svg?v=1)
^Four lineages, four different timelines, the same destination: once a single die can't grow any further, the only way to add more memory, bandwidth, and compute is to add another die next to it.

This isn't a coincidence — every one of these chips was chasing more memory capacity, more bandwidth, and more compute per accelerator, and every one of them eventually ran into the same wall: a single die can only be made so large. Once you see chiplets as the response to that specific wall, rather than a stylistic choice, the fact that four independent organizations converged on the same architectural move, within about four years of each other, stops looking like imitation and starts looking like the only available path to keep the training-relevant numbers growing.

## 6. The trends, quantified

Pulling the numbers out of the tables above and plotting them against release year makes a few things vivid that are easy to miss chip-by-chip.

**Memory bandwidth** has grown roughly 11× at NVIDIA's flagship tier in under a decade (0.72 TB/s → 8 TB/s), driven almost entirely by successive HBM generations (HBM2 → HBM2e → HBM3 → HBM3e) rather than clock speed — a direct, quantitative echo of the [first post's](#/blog?id=gpu-field-guide-for-dl) point that moving data, not computing on it, is what modern accelerator design is actually optimizing for.

**Memory capacity** is the axis where the four vendors diverge the most visibly — AMD's chiplet-enabled capacity lead over NVIDIA at the same point in time, TPU's unusual 2023 split into a 16 GB "efficient" v5e and a 95 GB "performance" v5p, and Huawei's steady doubling from 32 to 128 GB across three generations.

**TDP** has climbed at every vendor that discloses it, and climbed hardest at the leading edge: NVIDIA's 300 W → 1,400 W (nearly 5×, under a decade), AMD's 300 W → 1,400 W over five years. Liquid cooling has gone from optional to structurally required at the top of every lineage that's disclosed enough to tell.

![interactive:accel-trend](#)

For a closer look at any individual chip — or to compare two you have in mind directly — the panel below is a full lookup across everything in the tables above.

![interactive:accel-lookup](#)

## 7. How to actually read one of these spec sheets

**"TFLOPS" numbers are not one number.** Every modern vendor now publishes both a "dense" figure and a "with structured sparsity" figure that's roughly 2× higher, and increasingly a whole family of numbers across FP16/BF16/FP8/FP6/FP4. A headline PFLOPS figure without a precision and a sparsity qualifier attached is close to meaningless for comparison purposes — always check which one you're looking at, on both sides of a comparison.

**Bandwidth-to-compute ratio matters more than either number alone.** This is the [roofline model](#/blog?id=gpu-field-guide-for-dl) from the first post in this series, applied across vendors instead of within one chip: a chip with enormous peak FLOPS and comparatively modest bandwidth will look fantastic on a compute-bound workload and disappointing on a memory-bound one (which describes a large share of real deep learning workloads), and vice versa. Comparing two accelerators on FLOPS alone, without checking where their respective ridge points fall, is a very easy way to draw the wrong conclusion.

**Confidence isn't uniform across vendors, and pretending otherwise is itself a mistake.** NVIDIA and AMD's numbers in this post come from official datasheets and should be treated as reliable. Google's architectural claims are credible but its power figures are estimates. Huawei's numbers are compiled from a thinner and less official evidence base than the other three lineages — treat the Ascend row in any comparison as directional rather than exact.

*The tables in this post reflect the best publicly available information as of mid-2026 for shipped, purchasable hardware only, compiled from vendor datasheets, official cloud documentation, and (where noted) third-party industry analysis. The durable part of this post isn't any individual number — it's the shape of each vendor's bet: NVIDIA's rack-scale interconnect, AMD's memory-capacity maximalism, TPU's architecture-over-brute-force philosophy, and Ascend's cluster-scale compensation.*
