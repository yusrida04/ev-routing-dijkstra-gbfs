"""
Membuat grafik dan tabel ringkasan dari CSV hasil eksperimen
(experiment_regional_runs.csv). Tidak mengubah file apa pun selain
menulis ke folder keluaran.

Jalankan dari folder utama proyek:
    python src/make_report.py
    python src/make_report.py --runs outputs/results_uji/experiment_regional_runs.csv --out outputs/figures_uji

Hasil:
    fig1_success_rate.png    tingkat keberhasilan per level baterai
    fig2_selisih_jarak.png   selisih panjang rute terhadap Dijkstra + SPKLU
    fig3_simpul.png          jumlah simpul yang dikembangkan (skala log)
    fig4_lambda.png          pengaruh lambda pada usaha pencarian
    ringkasan_hasil.csv      tabel ringkasan (bisa dibuka di Excel)
    alasan_gagal.csv         rekap alasan kegagalan
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS = PROJECT_ROOT / "outputs/results/experiment_regional_runs.csv"
DEFAULT_OUT = PROJECT_ROOT / "outputs/figures"

NO_CHARGE = "Dijkstra (tanpa pengisian)"
DIJ = "Dijkstra + SPKLU"
GBFS_D = "GBFS Distance"
GBFS_B = "GBFS Battery-Aware"

ALGO_ORDER = [NO_CHARGE, DIJ, GBFS_D, GBFS_B]

COLORS = {
    NO_CHARGE: "#9aa0a6",
    DIJ: "#1f77b4",
    GBFS_D: "#ff7f0e",
    GBFS_B: "#2ca02c",
}

STRATA_PREFERRED = ["near", "medium", "far"]

PAIR_KEYS = [
    "pair_index",
    "battery_percentage",
    "energy_consumption_kwh_per_km",
]

REQUIRED_COLUMNS = PAIR_KEYS + [
    "lambda", "algorithm", "success", "distance_km", "expanded",
]


def normalize_algorithm(name):
    name = str(name).strip()

    if name.startswith("Dijkstra ("):
        return NO_CHARGE

    return name


def load_runs(path):
    if not Path(path).exists():
        raise FileNotFoundError(
            f"File tidak ditemukan: {path}\n"
            "Jalankan eksperimen dulu, atau beri jalur dengan --runs."
        )

    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]

    if missing:
        raise ValueError(
            f"Kolom berikut tidak ada di CSV: {missing}\n"
            f"Kolom yang ada: {list(df.columns)}"
        )

    df["algorithm"] = df["algorithm"].map(normalize_algorithm)
    df["success"] = df["success"].astype(str).str.lower().eq("true")
    df["lambda"] = pd.to_numeric(df["lambda"], errors="coerce")

    for column in ["distance_km", "expanded"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if "distance_stratum" not in df.columns:
        df["distance_stratum"] = "semua"

    # Rute bersifat deterministik; pengulangan hanya untuk waktu.
    df = df.drop_duplicates(subset=PAIR_KEYS + ["lambda", "algorithm"])

    return df


def order_strata(df):
    present = list(df["distance_stratum"].dropna().unique())
    first = [s for s in STRATA_PREFERRED if s in present]
    rest = sorted(s for s in present if s not in first)

    return first + rest


def reference_rows(df, lambda_ref):
    """Battery-Aware hanya diwakili satu nilai lambda."""
    is_battery = df["algorithm"] == GBFS_B
    matches = np.isclose(df["lambda"].fillna(-1), lambda_ref)

    return df[~is_battery | matches].copy()


def compute_deltas(df_ref):
    base = df_ref[(df_ref["algorithm"] == DIJ) & df_ref["success"]][
        PAIR_KEYS + ["distance_km"]
    ].rename(columns={"distance_km": "distance_dijkstra"})

    cmp = df_ref[
        df_ref["algorithm"].isin([GBFS_D, GBFS_B]) & df_ref["success"]
    ]

    merged = cmp.merge(base, on=PAIR_KEYS, how="inner")
    merged = merged[merged["distance_dijkstra"] > 0].copy()
    merged["delta_pct"] = (
        (merged["distance_km"] - merged["distance_dijkstra"])
        / merged["distance_dijkstra"] * 100
    )

    return merged


def grouped_bars(ax, categories, series, width=0.8):
    count = len(series)
    bar_width = width / count
    x = np.arange(len(categories))

    for i, (name, values) in enumerate(series.items()):
        ax.bar(
            x - width / 2 + bar_width * (i + 0.5),
            values,
            bar_width,
            label=name,
            color=COLORS[name]
        )

    ax.set_xticks(x)
    ax.set_xticklabels(categories)


def finish(fig, ax, title, xlabel, ylabel, out_path):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  tersimpan: {out_path.name}")


def fig_success(df_ref, out_dir):
    levels = sorted(df_ref["battery_percentage"].unique())
    series = {}

    for algo in ALGO_ORDER:
        values = []

        for level in levels:
            group = df_ref[
                (df_ref["algorithm"] == algo)
                & (df_ref["battery_percentage"] == level)
            ]
            values.append(
                group["success"].mean() * 100 if len(group) else np.nan
            )

        series[algo] = values

    fig, ax = plt.subplots(figsize=(8, 4.8))
    grouped_bars(ax, [f"{int(l)}%" for l in levels], series)
    ax.set_ylim(0, 105)

    finish(
        fig, ax,
        "Tingkat keberhasilan per level baterai awal",
        "Baterai awal",
        "Perjalanan berhasil (%)",
        out_dir / "fig1_success_rate.png"
    )


def fig_delta(delta_df, strata, out_dir):
    if delta_df.empty:
        print("  fig2 dilewati: tidak ada pasangan yang sama-sama berhasil.")
        return

    algos = [GBFS_D, GBFS_B]
    width = 0.35
    rng = np.random.default_rng(0)
    labeled = set()

    fig, ax = plt.subplots(figsize=(8, 4.8))

    for i, algo in enumerate(algos):
        for j, stratum in enumerate(strata):
            values = delta_df[
                (delta_df["algorithm"] == algo)
                & (delta_df["distance_stratum"] == stratum)
            ]["delta_pct"].to_numpy()

            if len(values) == 0:
                continue

            position = j + (i - 0.5) * width

            ax.bar(
                position, values.mean(), width * 0.9,
                color=COLORS[algo], alpha=0.55,
                label=algo if algo not in labeled else None
            )
            labeled.add(algo)

            ax.scatter(
                position + rng.uniform(-width * 0.3, width * 0.3, len(values)),
                values,
                s=14, color=COLORS[algo],
                edgecolor="black", linewidth=0.3, zorder=3
            )

    ax.set_xticks(np.arange(len(strata)))
    ax.set_xticklabels(strata)
    ax.axhline(0, color="black", linewidth=0.8)

    finish(
        fig, ax,
        "Selisih panjang rute terhadap Dijkstra + SPKLU\n"
        "(batang = rata-rata, titik = tiap skenario)",
        "Strata jarak",
        "Rute lebih panjang (%)",
        out_dir / "fig2_selisih_jarak.png"
    )


def fig_expanded(df_ref, strata, out_dir):
    algos = [DIJ, GBFS_D, GBFS_B]
    ok = df_ref[df_ref["success"]]
    series = {}

    for algo in algos:
        values = []

        for stratum in strata:
            group = ok[
                (ok["algorithm"] == algo)
                & (ok["distance_stratum"] == stratum)
            ]["expanded"]
            values.append(
                max(group.median(), 1) if len(group) else np.nan
            )

        series[algo] = values

    fig, ax = plt.subplots(figsize=(8, 4.8))
    grouped_bars(ax, strata, series)
    ax.set_yscale("log")

    finish(
        fig, ax,
        "Simpul yang dikembangkan (median, hanya yang berhasil)",
        "Strata jarak",
        "Jumlah simpul (skala log)",
        out_dir / "fig3_simpul.png"
    )


def fig_lambda(df, strata, out_dir):
    ok = df[(df["algorithm"] == GBFS_B) & df["success"]]

    if ok.empty or ok["lambda"].nunique() < 2:
        print("  fig4 dilewati: kurang dari 2 nilai lambda.")
        return

    fig, ax = plt.subplots(figsize=(8, 4.8))

    for stratum in strata:
        group = ok[ok["distance_stratum"] == stratum]

        if group.empty:
            continue

        line = (
            group.groupby("lambda")["expanded"].median().clip(lower=1)
        )
        ax.plot(line.index, line.values, marker="o", label=stratum)

    ax.set_yscale("log")
    ax.set_xscale("log")

    finish(
        fig, ax,
        "Pengaruh lambda pada simpul yang dikembangkan\n"
        "(GBFS Battery-Aware, median)",
        "Lambda",
        "Jumlah simpul (skala log)",
        out_dir / "fig4_lambda.png"
    )


def build_summary(df_ref, delta_df, strata):
    rows = []

    for algo in ALGO_ORDER:
        for stratum in strata:
            group = df_ref[
                (df_ref["algorithm"] == algo)
                & (df_ref["distance_stratum"] == stratum)
            ]

            if group.empty:
                continue

            ok = group[group["success"]]
            delta = delta_df[
                (delta_df["algorithm"] == algo)
                & (delta_df["distance_stratum"] == stratum)
            ]["delta_pct"]

            rows.append({
                "algoritma": algo,
                "strata": stratum,
                "jumlah_skenario": len(group),
                "berhasil": len(ok),
                "success_rate_persen": round(len(ok) / len(group) * 100, 1),
                "median_simpul": (
                    float(ok["expanded"].median()) if len(ok) else np.nan
                ),
                "rata2_selisih_jarak_persen": (
                    round(float(delta.mean()), 2) if len(delta) else np.nan
                ),
                "n_perbandingan_jarak": len(delta),
            })

    return pd.DataFrame(rows)


def failure_table(df):
    if "failure_reason" not in df.columns:
        return None

    failed = df[~df["success"]].copy()

    if failed.empty:
        return None

    failed["failure_reason"] = failed["failure_reason"].fillna("(kosong)")

    return pd.crosstab(failed["algorithm"], failed["failure_reason"])


def identical_route_report(df):
    ok = df[df["success"]]

    battery = ok[ok["algorithm"] == GBFS_B]
    distance = ok[ok["algorithm"] == GBFS_D]

    merged = battery.merge(
        distance, on=PAIR_KEYS, suffixes=("_b", "_d")
    )

    print("\nPEMERIKSAAN KESAMAAN HASIL")

    if merged.empty:
        print("  Tidak ada kasus yang berhasil di kedua heuristik.")
    else:
        same = (
            merged["distance_km_b"] - merged["distance_km_d"]
        ).abs() < 1e-6
        print(
            f"  Jarak rute GBFS Battery-Aware = GBFS Distance pada "
            f"{int(same.sum())} dari {len(same)} kombinasi "
            "(pasangan x skenario x lambda)."
        )

    spread = battery.groupby(PAIR_KEYS)["distance_km"].agg(
        lambda s: s.round(6).nunique()
    )

    if len(spread):
        print(
            f"  Jarak rute berbeda antar nilai lambda pada "
            f"{int((spread > 1).sum())} dari {len(spread)} skenario."
        )


def main():
    parser = argparse.ArgumentParser(
        description="Grafik dan tabel ringkasan dari CSV eksperimen."
    )
    parser.add_argument("--runs", default=str(DEFAULT_RUNS))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--lambda-ref", type=float, default=10.0,
        help="Nilai lambda yang mewakili GBFS Battery-Aware di grafik 1-3."
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_runs(args.runs)
    strata = order_strata(df)

    print(f"Membaca {args.runs}")
    print(f"  baris: {len(df)} | strata: {strata}")
    print(f"  jumlah pasangan: {df['pair_index'].nunique()}")

    df_ref = reference_rows(df, args.lambda_ref)

    if not (df["algorithm"] == GBFS_B).any():
        print("  peringatan: tidak ada baris GBFS Battery-Aware.")
    elif df_ref[df_ref["algorithm"] == GBFS_B].empty:
        print(
            f"  peringatan: lambda {args.lambda_ref} tidak ada di data; "
            "Battery-Aware tidak tampil di grafik 1-3. "
            "Nilai lambda yang ada: "
            f"{sorted(df['lambda'].dropna().unique())}"
        )

    delta_df = compute_deltas(df_ref)

    print("\nMEMBUAT GRAFIK")
    fig_success(df_ref, out_dir)
    fig_delta(delta_df, strata, out_dir)
    fig_expanded(df_ref, strata, out_dir)
    fig_lambda(df, strata, out_dir)

    summary = build_summary(df_ref, delta_df, strata)
    summary_path = out_dir / "ringkasan_hasil.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(f"\nTABEL RINGKASAN ({summary_path.name})")
    print(summary.to_string(index=False))

    failures = failure_table(df)

    if failures is not None:
        failures_path = out_dir / "alasan_gagal.csv"
        failures.to_csv(failures_path, encoding="utf-8-sig")
        print(f"\nALASAN KEGAGALAN ({failures_path.name})")
        print(failures.to_string())

    identical_route_report(df)

    print(f"\nSelesai. Semua file ada di: {out_dir}")


if __name__ == "__main__":
    main()