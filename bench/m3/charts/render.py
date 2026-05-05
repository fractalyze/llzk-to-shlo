#!/usr/bin/env python3
"""Render M3 report charts from §4.1 / §4.2 measurement tables.

Outputs:
  docs/charts/throughput_vs_N.png         — log-log line plot (§4.1)
  docs/charts/stage_breakdown_at_N65536.png — stacked bar (§4.2)

Data is hardcoded from docs/M3_REPORT.md to avoid coupling to incomplete
bench/m3/results/*.csv. Re-run when those tables change.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

OUT_DIR = Path(__file__).resolve().parents[3] / "docs" / "charts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# §4.1 throughput in witnesses/second. None marks OOM / TBD / N/A.
N_GRID = [1, 64, 4096, 65536, 262144]
THROUGHPUT = {
    "aes_256_encrypt": {
        "gpu_zkx": [1.3, 84.8, 3132.6, None, None],
        "cpu_circom": [23.8, 207.5, 236.4, None, None],
    },
    "aes_256_ctr": {
        "gpu_zkx": [41.3, 2130.5, 39124.1, 31500.4, 28177.2],
        "cpu_circom": [7.8, 82.1, 98.3, 225.9, 226.4],
    },
    "aes_256_key_expansion": {
        "gpu_zkx": [0.9, 42.0, 910.8, None, None],
        "cpu_circom": [24.9, 425.8, 754.0, None, None],
    },
    "keccak_chi": {
        "gpu_zkx": [235.0, 13519.0, 45004.9, 49928.6, None],
        "cpu_circom": [24.9, 362.6, 452.6, 463.6, None],
    },
    "keccak_iota3": {
        "gpu_zkx": [1036.7, 43074.6, 51396.1, 41956.8, None],
        "cpu_circom": [25.2, 552.8, 814.1, 854.5, None],
    },
    "keccak_iota10": {
        "gpu_zkx": [988.5, 26882.7, 34376.0, 27584.6, None],
        "cpu_circom": [10.4, 219.2, 220.3, 778.3, None],
    },
    "keccak_round0": {
        "gpu_zkx": [37.7, 2480.5, 24617.2, 49681.1, None],
        "cpu_circom": [5.7, 83.3, 95.6, 303.7, None],
    },
    "keccak_round20": {
        "gpu_zkx": [43.9, 2641.2, 38464.3, 48748.5, None],
        "cpu_circom": [10.1, 109.1, 275.5, 304.5, None],
    },
    "keccak_pad": {
        "gpu_zkx": [142.8, 8717.6, 46244.3, 40865.4, None],
        "cpu_circom": [26.6, 741.9, 1286.7, 730.1, None],
    },
    "keccak_rhopi": {
        "gpu_zkx": [237.7, 13695.0, 48994.4, 51266.0, None],
        "cpu_circom": [24.7, 446.8, 633.4, 647.1, None],
    },
    "keccak_squeeze": {
        "gpu_zkx": [10306.7, 186566.6, 153537.0, 109285.0, 125186.7],
        "cpu_circom": [25.7, 562.9, 863.4, 880.8, 886.7],
    },
    "keccak_theta": {
        "gpu_zkx": [1013.6, 38934.2, 55511.5, 39252.9, None],
        "cpu_circom": [24.9, 394.1, 544.1, 533.7, None],
    },
}

# §4.2 stage time in ms at the listed N (AES encrypt/ke at N=4096 due to OOM
# at N=65 536 — see notes ³ / ²¹). harness_overhead = total - kernel - d2h.
STAGE_BREAKDOWN = [
    # (label, N, compile, jit, kernel, harness_overhead)
    ("aes_256_encrypt (N=4 096)", 4096, 86.7, 3214.5, 800.3, 507.2),
    ("aes_256_ctr (N=65 536)", 65536, 293.4, 1984.6, 191.6, 1888.9),
    ("aes_256_key_expansion (N=4 096)", 4096, 289.6, 14530.2, 3993.2, 504.2),
    ("keccak_chi (N=65 536)", 65536, 12.4, 365.2, 17.2, 1295.4),
    ("keccak_iota3 (N=65 536)", 65536, 3.6, 339.3, 10.9, 1551.1),
    ("keccak_iota10 (N=65 536)", 65536, 5.0, 997.4, 11.0, 2364.8),
    ("keccak_round0 (N=65 536)", 65536, 17.8, 350.1, 41.6, 1277.5),
    ("keccak_round20 (N=65 536)", 65536, 20.6, 1166.8, 41.3, 1303.1),
    ("keccak_pad (N=65 536)", 65536, 3.0, 200.5, 26.4, 1577.3),
    ("keccak_rhopi (N=65 536)", 65536, 14.8, 367.8, 17.3, 1261.1),
    ("keccak_squeeze (N=65 536)", 65536, 3.3, 380.9, 0.8, 598.9),
    ("keccak_theta (N=65 536)", 65536, 11.6, 371.5, 11.0, 1658.6),
]

# Stable color map: AES family in warm hues, keccak in cool, sorted by chip.
CIRCUIT_COLORS = {
    "aes_256_encrypt": "#d62728",
    "aes_256_ctr": "#ff7f0e",
    "aes_256_key_expansion": "#bcbd22",
    "keccak_chi": "#1f77b4",
    "keccak_iota3": "#17becf",
    "keccak_iota10": "#2ca02c",
    "keccak_round0": "#9467bd",
    "keccak_round20": "#8c564b",
    "keccak_pad": "#e377c2",
    "keccak_rhopi": "#7f7f7f",
    "keccak_squeeze": "#393b79",
    "keccak_theta": "#637939",
}


def render_throughput_vs_n() -> Path:
    fig, ax = plt.subplots(figsize=(11, 7))
    for circuit, backends in THROUGHPUT.items():
        color = CIRCUIT_COLORS[circuit]
        for backend, ys in backends.items():
            xs = [N_GRID[i] for i, y in enumerate(ys) if y is not None]
            ys_clean = [y for y in ys if y is not None]
            if len(xs) < 2:
                continue
            style = "-" if backend == "gpu_zkx" else "--"
            marker = "o" if backend == "gpu_zkx" else "s"
            label = f"{circuit} ({backend})"
            ax.plot(
                xs,
                ys_clean,
                style,
                marker=marker,
                color=color,
                linewidth=1.6,
                markersize=5,
                alpha=0.85,
                label=label,
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Batch size N (log)")
    ax.set_ylabel("Throughput (witnesses/sec, log)")
    ax.set_title(
        "§4.1 Per-circuit throughput vs N — gpu_zkx (solid) vs cpu_circom (dashed)"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, ncol=1)
    out = OUT_DIR / "throughput_vs_N.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def render_stage_breakdown() -> Path:
    fig, ax = plt.subplots(figsize=(12, 7))
    labels = [row[0] for row in STAGE_BREAKDOWN]
    compile_ = [row[2] for row in STAGE_BREAKDOWN]
    jit = [row[3] for row in STAGE_BREAKDOWN]
    kernel = [row[4] for row in STAGE_BREAKDOWN]
    overhead = [row[5] for row in STAGE_BREAKDOWN]

    x = list(range(len(labels)))
    bars_compile = ax.bar(x, compile_, label="compile (one-time)", color="#bdbdbd")
    bars_jit = ax.bar(x, jit, bottom=compile_, label="jit (one-time)", color="#9ecae1")
    bottoms_kernel = [c + j for c, j in zip(compile_, jit)]
    bars_kernel = ax.bar(
        x, kernel, bottom=bottoms_kernel, label="kernel (on-device)", color="#3182bd"
    )
    bottoms_oh = [b + k for b, k in zip(bottoms_kernel, kernel)]
    bars_oh = ax.bar(
        x,
        overhead,
        bottom=bottoms_oh,
        label="harness overhead (host stitch)",
        color="#e6550d",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Wallclock (ms)")
    ax.set_title(
        "§4.2 Stage breakdown at the largest measured N "
        "(AES encrypt/ke capped at N=4 096 by gpu_zkx OOM)"
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left")
    out = OUT_DIR / "stage_breakdown_at_N65536.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    p1 = render_throughput_vs_n()
    p2 = render_stage_breakdown()
    print(f"wrote {p1}")
    print(f"wrote {p2}")
