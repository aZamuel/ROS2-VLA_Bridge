# eval.py
# Minimal, fast: scan record*.csv, take first 50 rows, compute KPIs, save per-file & overall summaries + simple plots.
# Usage:
#   python eval.py                  # process all record*.csv next to this file
#   python eval.py --one record01.csv  # process a single file
#   python eval.py --out eval_out   # change output dir

import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)

REQUIRED_COLS = [
    "t_client_start","t_publish","t_vla_in","t_vla_out",
    "success","error",
    "x","y","z","roll","pitch","yaw",
    "gripper_width","prompt"
]

def ns_to_ms(s): return s.astype("float64") / 1e6

def load_first50(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns: {missing}")
    df = df.head(50).copy()
    # force dtypes
    num_cols = ["t_client_start_ns","t_publish_ns","t_vla_in_ns","t_vla_out_ns","x","y","z","roll","pitch","yaw","gripper_width","success"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["error"] = df["error"].astype(str).fillna("")
    df["prompt"] = df["prompt"].astype(str).fillna("")
    # derived latencies (ms)
    df["e2e_ms"]      = ns_to_ms(df["t_publish_ns"] - df["t_client_start_ns"])
    df["backend_ms"]  = ns_to_ms(df["t_vla_out_ns"] - df["t_vla_in_ns"])
    df["outbound_ms"] = ns_to_ms(df["t_vla_in_ns"] - df["t_client_start_ns"])
    # basic sanity mask (keep but mark invalid)
    df["_valid"] = np.isfinite(df["e2e_ms"]) & np.isfinite(df["backend_ms"]) & np.isfinite(df["outbound_ms"])
    return df

def kpi(series: pd.Series, prefix: str):
    s = series.dropna()
    if len(s) == 0:
        return {f"{prefix}_count": 0}
    q = s.quantile([0.5, 0.95, 0.99])
    return {
        f"{prefix}_count": int(s.size),
        f"{prefix}_mean_ms": float(s.mean()),
        f"{prefix}_std_ms": float(s.std(ddof=1)) if s.size > 1 else 0.0,
        f"{prefix}_median_ms": float(q.loc[0.5]),
        f"{prefix}_p95_ms": float(q.loc[0.95]),
        f"{prefix}_p99_ms": float(q.loc[0.99]),
        f"{prefix}_min_ms": float(s.min()),
        f"{prefix}_max_ms": float(s.max()),
    }

def summarize(df: pd.DataFrame, record_id: str):
    out = {"record_id": record_id, "n_rows": int(len(df))}
    # all rows
    out.update(kpi(df["e2e_ms"], "e2e"))
    out.update(kpi(df["backend_ms"], "backend"))
    out.update(kpi(df["outbound_ms"], "outbound"))
    # success-only
    ok = df[df["success"] == 1]
    out.update({k.replace("e2e", "e2e_ok"): v for k, v in kpi(ok["e2e_ms"], "e2e").items()})
    out["success_rate"] = float((df["success"] == 1).mean()) if len(df) else 0.0
    # error breakdown (top 3)
    err = df.loc[df["success"] != 1, "error"].value_counts().head(3)
    for i, (name, cnt) in enumerate(err.items(), start=1):
        out[f"err{i}_name"] = str(name)
        out[f"err{i}_count"] = int(cnt)
    return out

def save_ecdf(series: pd.Series, title: str, outpath: Path):
    s = series.dropna().values
    if s.size == 0:
        return
    x = np.sort(s)
    y = np.arange(1, x.size + 1) / x.size
    plt.figure()
    plt.step(x, y, where="post")
    plt.xlabel("Latency (ms)")
    plt.ylabel("ECDF")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()

def save_traj3d(df: pd.DataFrame, title: str, outpath: Path):
    xs, ys, zs = df["x"].values, df["y"].values, df["z"].values
    if not (np.isfinite(xs).any() and np.isfinite(ys).any() and np.isfinite(zs).any()):
        return
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(xs, ys, zs, linewidth=1.0)
    # start/end markers
    ax.scatter(xs[:1], ys[:1], zs[:1], s=30, marker="o")  # start
    ax.scatter(xs[-1:], ys[-1:], zs[-1:], s=30, marker="^")  # end
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--one", type=str, default=None, help="Process a single CSV file (path).")
    p.add_argument("--out", type=str, default="eval_out", help="Output directory.")
    args = p.parse_args()

    here = Path(__file__).resolve().parent
    out_root = (here / args.out)
    figs = out_root / "figs"
    data_dir = out_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    files = [Path(args.one)] if args.one else sorted(here.glob("record*.csv"))
    if not files:
        print("No record*.csv found. Use --one to specify a file.", file=sys.stderr)
        sys.exit(2)

    per_rows = []
    combined_50 = []

    for f in files:
        try:
            df = load_first50(f)
        except Exception as e:
            print(f"[SKIP] {f.name}: {e}", file=sys.stderr)
            continue

        record_id = f.stem
        # KPIs
        per_rows.append(summarize(df, record_id))
        combined_50.append(df)

        # Plots (PDF, LaTeX-friendly)
        save_ecdf(df["e2e_ms"], f"ECDF e2e — {record_id}", figs / f"ecdf_e2e_{record_id}.pdf")
        save_ecdf(df["backend_ms"], f"ECDF backend — {record_id}", figs / f"ecdf_backend_{record_id}.pdf")
        save_traj3d(df, f"Trajectory (first 50) — {record_id}", figs / f"traj3d_{record_id}.pdf")

    if not per_rows:
        print("No valid files processed.", file=sys.stderr)
        sys.exit(3)

    # Per-file summary
    per_df = pd.DataFrame(per_rows).sort_values("record_id")
    per_df.to_csv(data_dir / "summary_per_file.csv", index=False)

    # Overall (concatenate first-50 windows)
    all_df = pd.concat(combined_50, ignore_index=True)
    overall = summarize(all_df, record_id="ALL_FIRST50")
    pd.DataFrame([overall]).to_csv(data_dir / "summary_overall.csv", index=False)

    print(f"✔ Wrote {len(per_rows)} rows → {data_dir/'summary_per_file.csv'}")
    print(f"✔ Wrote overall → {data_dir/'summary_overall.csv'}")
    print(f"✔ Plots in → {figs}")

if __name__ == "__main__":
    main()
