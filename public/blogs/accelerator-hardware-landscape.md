---
title: "Training Silicon Across a Decade: NVIDIA, AMD, TPU, and Ascend Compared"
date: "2026/7/20"
---

Every post in this series so far has treated "the GPU" as a stand-in for the whole category — the [first post](#/blog?id=gpu-field-guide-for-dl) built a mental model around one, mostly-NVIDIA-shaped chip. That's a reasonable simplification for learning the concepts, but it quietly erases a real and interesting fact: there are at least four seriously different lineages of training silicon in production today, from four organizations that have made four genuinely different bets about what matters. This post is a pure hardware comparison — memory capacity, memory bandwidth, compute throughput, interconnect, power — across NVIDIA's datacenter GPUs, AMD's Instinct MI series, Google's TPUs, and Huawei's Ascend accelerators, laid out chronologically within each lineage and then compared across all four. Deliberately out of scope: CUDA, ROCm, XLA, CANN, or any other software layer. This is about the silicon alone.

A note on confidence before the numbers: NVIDIA and AMD publish detailed datasheets, so those sections are the most solid. Google discloses generous architectural detail in blog posts but has never once published an official TDP or process node for any TPU — those figures are third-party estimates, marked as such. Huawei is the genuinely hard case: post-2020 export controls mean there is no official spec sheet culture at all for Ascend chips, and some of the most interesting facts about them (like what process node they're really built on) are actively disputed in the trade press. Anywhere a number is an estimate rather than a vendor-published spec, it's marked.

## 1. NVIDIA: the incumbent, expanding on every axis at once

| Chip | Year | Memory | Bandwidth | Dense compute (BF16/FP16-class) | Interconnect | TDP | Process |
|---|---|---|---|---|---|---|---|
| P100 | 2016 | 16 GB HBM2 | 0.72 TB/s | 21.2 TFLOPS FP16 (no Tensor Cores) | NVLink 1, 160 GB/s | 300 W | TSMC 16nm |
| V100 | 2017 | 32 GB HBM2 | 0.9 TB/s | 125 TFLOPS (1st-gen Tensor Cores) | NVLink 2, 300 GB/s | 300 W | TSMC 12nm |
| A100 (80GB) | 2020 | 80 GB HBM2e | 2.04 TB/s | 312 TFLOPS (TF32 introduced) | NVLink 3, 600 GB/s | 400 W | TSMC 7nm |
| H100 SXM5 | 2022 | 80 GB HBM3 | 3.35 TB/s | 990 TFLOPS (**FP8 introduced**, 1979 TFLOPS) | NVLink 4, 900 GB/s | 700 W | TSMC 4N |
| H200 | 2024 | 141 GB HBM3e | 4.8 TB/s | 990 TFLOPS (same die as H100, memory-only refresh) | NVLink 4, 900 GB/s | 700 W | TSMC 4N |
| B200 | 2024 | 192 GB HBM3e | 8 TB/s | 2,250 TFLOPS (**FP4 introduced**, ~9 PFLOPS FP4 sparse) | NVLink 5, 1.8 TB/s | 1,000 W | TSMC 4NP |
| B300 / GB300 | 2025 | 288 GB HBM3e | 8 TB/s | ~2,500 TFLOPS (FP64 sharply de-emphasized vs. B200) | NVLink 5, 1.8 TB/s | 1,400 W | TSMC 4NP |
| Rubin *(announced, not shipping)* | 2026 | 288 GB HBM4 | 22 TB/s | not fully disclosed | NVLink 6, 3.6 TB/s | ~1,800–2,300 W | TSMC N3 |

The clearest structural break in this table is at Blackwell (B100/B200/B300): every NVIDIA datacenter GPU up through H200 was a single monolithic die, pushed right up against the reticle size limit (H100's die is 814 mm²). Blackwell is the first generation to go **dual-die** — two reticle-sized dies joined by a 10 TB/s die-to-die link, presented to software as a single GPU — because there was no more room to grow a single die, only to add a second one next to it. Rubin, per NVIDIA's own preview material, pushes this further to four dies per package (two compute, two I/O).

The other clear pattern is the "rack as the real product" shift. Starting with GB200 NVL72, NVIDIA's highest-end offering isn't really a chip you buy one of — it's a 72-GPU, liquid-cooled rack sharing a single NVLink domain with 130+ TB/s of aggregate bandwidth, and per-GPU numbers increasingly need that context to mean anything. TDP has grown right alongside all of this — 300 W (Pascal) to 700 W (Hopper) to 1,000–1,400 W (Blackwell) to a roadmap 1,800–2,300 W for Rubin — an 8× increase over a decade, and the reason liquid cooling stopped being optional starting with Blackwell Ultra.

## 2. AMD Instinct: betting on memory capacity

| Chip | Year | Memory | Bandwidth | Dense compute (BF16/FP16-class) | Interconnect | TDP | Process |
|---|---|---|---|---|---|---|---|
| MI100 | 2020 | 32 GB HBM2 | 1.2 TB/s | 184.6 TFLOPS FP16 | Infinity Fabric, ~276 GB/s | 300 W | TSMC 7nm |
| MI250X | 2021 | 128 GB HBM2e | 3.2 TB/s | 383 TFLOPS | Infinity Fabric, 8×100 GB/s | 560 W | TSMC 6nm (2 GCDs) |
| MI300X | 2023 | 192 GB HBM3 | 5.3 TB/s | 1,307 TFLOPS (**FP8 introduced**, 2,615 TFLOPS) | Infinity Fabric, 7×128 GB/s | 750 W | TSMC 5nm/6nm (8 XCDs + 4 IODs) |
| MI325X | 2024 | 256 GB HBM3e | 6.0 TB/s | 1,307 TFLOPS (same compute dies as MI300X) | Infinity Fabric, 7×128 GB/s | 1,000 W | TSMC 5nm/6nm |
| MI355X | 2025 | 288 GB HBM3e | 8 TB/s | ~2,500 TFLOPS (**MXFP6/MXFP4 introduced**) | Infinity Fabric, 7×153.6 GB/s | 1,400 W | TSMC N3P (8 XCDs + 2 IODs) |
| MI400 (MI455X) *(announced, not shipping)* | 2026 | 432 GB HBM4 | 19.6 TB/s | ~10,000 TFLOPS (AMD's own FP16 figure) | UALink / Infinity Fabric | ~2,000 W (est.) | TSMC N2 |

AMD went multi-die earlier than anyone else at this scale — MI250X (2021) already packages two compute dies (GCDs), and MI300X (2023) pushes to eight compute chiplets plus separate I/O dies, a genuinely more aggressive chiplet strategy than NVIDIA adopted until Blackwell three years later. The strategic signature that shows up clearly in the numbers, though, is memory capacity. Checked against NVIDIA's contemporaneous flagship at each point in time, AMD's memory lead has been real and consistent, even as it's been narrowing:

| Era | AMD chip | AMD memory | NVIDIA contemporary | NVIDIA memory | AMD's edge |
|---|---|---|---|---|---|
| Late 2023 | MI300X | 192 GB | H100 SXM5 | 80 GB | 2.4× |
| Late 2024 | MI325X | 256 GB | H200 | 141 GB | 1.8× |
| 2025 | MI355X | 288 GB | B200 | 192 GB | 1.5× |
| 2026 roadmap | MI455X | 432 GB | Rubin | 288 GB | 1.5× |

The trade-off is legible too: AMD has generally matched, rather than led, on raw bandwidth per chip in the most recent generations (MI355X's 8 TB/s ties B200 exactly), and NVIDIA's NVLink fabric remains well ahead of AMD's UALink/Infinity Fabric ecosystem in scale-up sophistication and aggregate rack-level bandwidth. The fair read: AMD trades interconnect polish for the biggest memory pool you can buy per chip — useful for fitting a bigger model or KV-cache on fewer GPUs — while NVIDIA has prioritized bandwidth and multi-GPU fabric scale.

## 3. Google TPU: betting on architecture over the process node

| Generation | Year | Memory | Bandwidth | Dense compute (BF16) | Interconnect | TDP | Process |
|---|---|---|---|---|---|---|---|
| v2 | 2017 | 8 GB HBM | ~0.25 TB/s (est.) | 45 TFLOPS | ICI (undisclosed) | not disclosed | ~16nm (est.) |
| v3 | 2018 | 32 GB HBM2 | 0.9 TB/s | 123 TFLOPS | ICI | 123–262 W (measured) | ~12nm (est.) |
| v4 | 2021 | 32 GB HBM2 | 1.2 TB/s | 275 TFLOPS | ICI, **optical circuit switches** | 90–192 W (measured) | ~7nm (est.) |
| v5e | 2023 | 16 GB HBM2 | 0.82 TB/s | 197 TFLOPS | ICI, 400 GB/s | not disclosed | ~5nm (est.) |
| v5p | 2023 | 95 GB HBM2e | 2.77 TB/s | 459 TFLOPS | ICI, 1.2 TB/s | not disclosed | ~5nm (est.) |
| v6e (Trillium) | 2024 | 32 GB HBM | 1.64 TB/s | 918 TFLOPS | ICI, 800 GB/s | not disclosed | TSMC N5 |
| v7 (Ironwood) | 2025 | 192 GB HBM (2×96 GB chiplets) | 7.38 TB/s | 2,307 TFLOPS (**FP8 introduced**, 4,614 TFLOPS) | ICI, 1.2 TB/s + fast die-to-die link | ~1,000 W (back-calculated est.) | ~3nm (est.) |
| v8t / v8i *(announced, targeting late 2027)* | 2027 | not disclosed | not disclosed | 121 EFLOPS FP4 per 9,600-chip pod (per-chip not disclosed) | new "Virgo" fabric, >134,000 chips | not disclosed | TSMC N2 |

Two things make TPU's table read differently from the other three. First, Google has never once published TDP or process node for any generation — everything in those two columns is a third-party estimate (SemiAnalysis, ServeTheHome), except the two generations (v3, v4) where Google's own Cloud docs happen to list measured min/mean/max power draw. Second, and more interesting: the jump from v5p to Trillium (v6e) happened on the **same TSMC N5 node**, yet delivered roughly double the peak FLOPS — the gain came from quadrupling the systolic array from 128×128 to 256×256 MXU tiles, an architectural change, not a smaller transistor. That's about as clean a "architecture beat process node" data point as this whole comparison produces.

Ironwood (v7) is TPU's Blackwell moment: it abandons the unified "MegaCore" die design used since v4 in favor of two chiplets per chip, and it's the first TPU with native FP8 hardware. And the 2027 roadmap makes a move neither NVIDIA nor AMD has made yet: splitting entirely into two different chips with two different silicon partners — "v8t" (training, built with Broadcom) and "v8i" (inference, built with MediaTek) — rather than one chip with an "e" and a "p" variant the way v5 did it.

## 4. Huawei Ascend: compensating for a process-node disadvantage

| Chip | Year | Memory | Bandwidth | Dense compute (BF16/FP16-class) | Interconnect | TDP | Process | Confidence |
|---|---|---|---|---|---|---|---|---|
| 910 | 2019 | 32 GB HBM2 | 1.23 TB/s | ~320 TFLOPS FP16 | HCCS (early) | ~310 W | **TSMC 7nm — confirmed** (pre-Entity-List) | High |
| 910B | 2023 | 64 GB HBM2e | 1.6 TB/s | ~400 TFLOPS FP16 (real generational gain closer to +20% per normalized analysis) | HCCS, ~336 GB/s (est.) | ~400 W (est.) | SMIC "N+2," 7nm-class — **disputed exact node** | Medium |
| 910C | 2025 | 128 GB HBM2e (reported) | ~3.2 TB/s | ~800 TFLOPS (est.) | HCCS + die-to-die (unconfirmed) | not disclosed | **Contested**: officially domestic, but teardowns found 2020-vintage **TSMC 7nm dies** inside shipped units | Low–Medium |
| 950 *(announced, targeting 2026)* | 2026 | first in-house HBM (capacity undisclosed) | not disclosed | roadmap target: 1 PFLOPS FP8 | ~2 TB/s (2.5× 910C, claimed) | not disclosed | not disclosed | Low (roadmap slide) |
| 960 / 970 *(roadmap, 2027–2028)* | 2027–2028 | 2×/further increase over 950 (claimed) | 2× over 950 (claimed) | 2/4 PFLOPS FP8/FP4 (claimed) | 2× ports vs. 950 (claimed) | not disclosed | not disclosed | Very low (slide bullet points) |

This is the lineage where the hardware numbers are inseparable from the export-control story, and it's worth being explicit about the confidence level dropping row by row. The Ascend 910 (2019) is solidly documented — TSMC 7nm, confirmed in a peer-reviewed architecture paper, fabbed before the 2020 Entity List cutoff. The 910B (2023) is the chip most used for domestic LLM training after A100/H100 access was restricted, and CSET's rigorous reverse-engineering of Huawei's own internal server documentation is the best public source on it — their normalized analysis suggests the real generation-over-generation compute gain was closer to 20% than the ~25% implied by Huawei's own headline TFLOPS figure, alongside a genuinely uncertain process node (SMIC doesn't publicly disclose true node geometry the way TSMC does).

The 910C (2025) is where the story gets stranger: it's a confirmed dual-chiplet package — two 910B-class compute dies in one package, structurally similar to NVIDIA's own B200 — which is a real architectural response to a process-node ceiling. But independent teardowns (TechInsights, reported via SemiWiki and others) found that the compute dies inside shipped 910C units are old-stock **TSMC 7nm silicon dating to 2020**, not domestically fabricated as officially positioned, reportedly sourced through shell companies before export controls fully closed that door. Whatever Ascend chip ships next on a genuinely domestic node will be a more meaningful data point than anything in this table so far.

Going forward, Huawei's own disclosed roadmap (950 → 960 → 970, through 2028) leans hard into exactly the pattern the 910C already showed: since single-die process node is the constraint that's hardest to fix, the roadmap emphasizes **in-house HBM** (breaking dependence on Samsung/SK hynix/Micron), **larger interconnect bandwidth jumps** (2.5× per generation, claimed), and **enormous cluster scale** (the CloudMatrix 384 system links 384 chips, and later SuperPods are claimed to reach far higher counts) — compensating at the package and cluster level for a gap that's hard to close chip-for-chip.

![Four different bets on the same problem](blogs/images/accelerator-four-philosophies.svg?v=1)
^Same underlying constraint — not enough memory, bandwidth, or compute on one die — four structurally different answers to it.

## 5. Everyone eventually went multi-die

Look across all four tables and one architectural transition shows up in every single lineage, just on a different clock: the move from a single monolithic die to a package of multiple compute chiplets. AMD got there first, out of necessity, with MI250X in 2021. NVIDIA held out on a monolithic H100/H200 die through 2024, then switched decisively with Blackwell once TSMC's reticle limit made a bigger single die physically impossible. TPU held out even longer, keeping its "MegaCore" unified-die design all the way through Trillium (2024), before Ironwood (2025) adopted the same dual-chiplet pattern. Huawei's 910C (2025) did the same, for the added reason of needing to double throughput without a smaller node available to it at all.

![Everyone eventually went multi-die](blogs/images/accelerator-chiplet-timeline.svg?v=1)
^Four lineages, four different timelines, the same destination: once a single die can't grow any further, the only way to add more transistors is to add another die next to it.

This isn't a coincidence — it's the same physical constraint (the reticle limit on how large a single die can be manufactured) arriving at each vendor's roadmap at a different point, depending on how much headroom their specific node and design had left. Once you understand the reticle limit as a hard physical ceiling rather than an engineering choice, the fact that four independent organizations converged on the same architectural response, within about four years of each other, stops looking like imitation and starts looking like the only available move.

## 6. The trends, quantified

Pulling the numbers out of the tables above and plotting them against release year makes a few things vivid that are easy to miss chip-by-chip.

**Memory bandwidth** has grown roughly 30× at NVIDIA's flagship tier over ten years (0.72 TB/s → 22 TB/s roadmap), driven almost entirely by successive HBM generations (HBM2 → HBM2e → HBM3 → HBM3e → HBM4) rather than clock speed — a direct, quantitative echo of the [first post's](#/blog?id=gpu-field-guide-for-dl) point that moving data, not computing on it, is what modern accelerator design is actually optimizing for.

**Memory capacity** is the axis where the four vendors diverge the most visibly — AMD's chiplet-enabled capacity lead over NVIDIA at the same point in time, TPU's unusual 2023 split into a 16 GB "efficient" v5e and a 95 GB "performance" v5p, and Huawei's roadmap explicitly betting on in-house HBM as the next lever to pull.

**TDP** has climbed at every vendor that discloses it, and climbed hardest at the leading edge: NVIDIA's 300 W → 1,400 W (8×, ten years), AMD's 300 W → 1,400 W-shipping / ~2,000 W-roadmap over a shorter six years. Liquid cooling has gone from optional to structurally required at the top of every lineage that's disclosed enough to tell.

![interactive:accel-trend](#)

For a closer look at any individual chip — or to compare two you have in mind directly — the panel below is a full lookup across everything in the tables above.

![interactive:accel-lookup](#)

## 7. How to actually read one of these spec sheets

**"TFLOPS" numbers are not one number.** Every modern vendor now publishes both a "dense" figure and a "with structured sparsity" figure that's roughly 2× higher, and increasingly a whole family of numbers across FP16/BF16/FP8/FP6/FP4. A headline PFLOPS figure without a precision and a sparsity qualifier attached is close to meaningless for comparison purposes — always check which one you're looking at, on both sides of a comparison.

**Bandwidth-to-compute ratio matters more than either number alone.** This is the [roofline model](#/blog?id=gpu-field-guide-for-dl) from the first post in this series, applied across vendors instead of within one chip: a chip with enormous peak FLOPS and comparatively modest bandwidth will look fantastic on a compute-bound workload and disappointing on a memory-bound one (which describes a large share of real deep learning workloads), and vice versa. Comparing two accelerators on FLOPS alone, without checking where their respective ridge points fall, is a very easy way to draw the wrong conclusion.

**"Announced" and "shipping" are different categories**, and the gap between them has been getting more aggressively marketed across every vendor in this post — Rubin, MI400, TPU v8, and Ascend 950 are all, as of this writing, roadmap disclosures rather than benchmarkable silicon. Treat every number in a "not yet shipping" row as preliminary, subject to change, and — especially for the chips furthest from launch — closer to a target than a spec.

**Confidence isn't uniform across vendors, and pretending otherwise is itself a mistake.** NVIDIA and AMD's numbers in this post come from official datasheets and should be treated as reliable. Google's architectural claims are credible but its power/process figures are estimates. Huawei's numbers — especially anything about process node or the 910C's actual composition — are genuinely disputed in ways that reflect real, unresolved reporting, not just a gap in this post's research; treat the lower-confidence rows in Section 4 accordingly.

*The tables in this post reflect the best publicly available information as of mid-2026, compiled from vendor datasheets, official cloud documentation, and (where noted) third-party teardown and industry analysis. Datacenter accelerator specs update on a roughly annual cadence per vendor and roadmap items in particular should be expected to shift before shipping. The durable part of this post isn't any individual number — it's the shape of each vendor's bet: NVIDIA's rack-scale interconnect, AMD's memory-capacity maximalism, TPU's architecture-over-node philosophy, and Ascend's cluster-scale compensation for a contested process-node gap.*
