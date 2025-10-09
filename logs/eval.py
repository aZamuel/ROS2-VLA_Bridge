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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import math

REQUIRED_COLS = [
    "t_client_start","t_publish","t_vla_in","t_vla_out",
    "success","error","x","y","z","qx","qy","qz","gw","prompt"
]

def ns_to_ms(s): return s.astype("float64") / 1e6

def load_first50(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns: {missing}")
    df = df.head(50).copy()

    num_cols = [
        "t_client_start","t_publish","t_vla_in","t_vla_out",
        "x","y","z","qx","qy","qz","gw","success"
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["error"] = df["error"].astype(str).fillna("")
    df["prompt"] = df["prompt"].astype(str).fillna("")

    # latencies (ms)
    df["e2e_ms"]      = ns_to_ms(df["t_publish"] - df["t_client_start"])
    df["backend_ms"]  = ns_to_ms(df["t_vla_out"] - df["t_vla_in"])
    df["outbound_ms"] = ns_to_ms(df["t_vla_in"] - df["t_client_start"])

    df["_valid"] = np.isfinite(df["e2e_ms"]) & np.isfinite(df["backend_ms"]) & np.isfinite(df["outbound_ms"])
    return df

def quantiles(series: pd.Series):
    s = series.dropna()
    if s.empty:
        return {"min": np.nan, "mean": np.nan, "p95": np.nan, "max": np.nan}
    q95 = float(s.quantile(0.95))
    return {
        "min": float(s.min()),
        "mean": float(s.mean()),
        "p95": q95,
        "max": float(s.max()),
    }

def summarize_basic(df: pd.DataFrame, record_id: str):
    out = {"record_id": record_id, "n_rows": int(len(df))}
    # keep existing per-file summary for convenience
    for name, col in [("e2e","e2e_ms"), ("backend","backend_ms"), ("outbound","outbound_ms")]:
        s = df[col].dropna()
        if s.empty:
            stats = {f"{name}_count": 0}
        else:
            q = s.quantile([0.5,0.95,0.99])
            stats = {
                f"{name}_count": int(s.size),
                f"{name}_mean_ms": float(s.mean()),
                f"{name}_std_ms": float(s.std(ddof=1)) if s.size > 1 else 0.0,
                f"{name}_median_ms": float(q.loc[0.5]),
                f"{name}_p95_ms": float(q.loc[0.95]),
                f"{name}_p99_ms": float(q.loc[0.99]),
                f"{name}_min_ms": float(s.min()),
                f"{name}_max_ms": float(s.max()),
            }
        out.update(stats)
    out["success_rate"] = float((df["success"] == 1).mean()) if len(df) else 0.0
    err = df.loc[df["success"] != 1, "error"].value_counts().head(3)
    for i, (name, cnt) in enumerate(err.items(), start=1):
        out[f"err{i}_name"] = str(name)
        out[f"err{i}_count"] = int(cnt)
    return out

def save_per_record_kpis(df: pd.DataFrame, outpath: Path):
    rows = {
        "outbound": quantiles(df["outbound_ms"]),
        "backend":  quantiles(df["backend_ms"]),
        "e2e":      quantiles(df["e2e_ms"]),
    }
    kdf = pd.DataFrame.from_dict(rows, orient="index")[["min","mean","p95","max"]]
    kdf = kdf.round(3)  # 3 decimals
    outpath.parent.mkdir(parents=True, exist_ok=True)
    kdf.to_csv(outpath, index=True)

def _set_axes_equal(ax):
    xlim, ylim, zlim = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
    xmid, ymid, zmid = (np.mean(xlim), np.mean(ylim), np.mean(zlim))
    r = max(xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0]) / 2.0
    ax.set_xlim3d(xmid - r, xmid + r)
    ax.set_ylim3d(ymid - r, ymid + r)
    ax.set_zlim3d(zmid - r, zmid + r)

def save_traj3d(df: pd.DataFrame, title: str, outpath: Path):
    xs, ys, zs = df["x"].to_numpy(), df["y"].to_numpy(), df["z"].to_numpy()
    if not (np.isfinite(xs).any() and np.isfinite(ys).any() and np.isfinite(zs).any()):
        return

    fig = plt.figure(figsize=(6.5, 5.5), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    # Main trajectory + endpoints
    ax.plot(xs, ys, zs, linewidth=1.2, label="trajectory")
    start_lbl = f"start ({xs[0]:.3f}, {ys[0]:.3f}, {zs[0]:.3f}) m"
    end_lbl   = f"end   ({xs[-1]:.3f}, {ys[-1]:.3f}, {zs[-1]:.3f}) m"
    ax.scatter(xs[0], ys[0], zs[0], s=36, marker="o", label=start_lbl)
    ax.scatter(xs[-1], ys[-1], zs[-1], s=48, marker="^", label=end_lbl)

    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.set_title(title)

    # Equal-aspect first so axis limits are stable:
    fig.canvas.draw()
    _set_axes_equal(ax)
    fig.canvas.draw()

    # Floor = z-axis lower bound (not data min)
    z_floor = ax.get_zlim3d()[0]

    # Floor projection (“shadow”) and vertical guides
    ax.plot(xs, ys, np.full_like(xs, z_floor), linewidth=0.8, linestyle="--", label=f"floor proj (z={z_floor:.3f} m)")
    ax.plot([xs[0], xs[0]], [ys[0], ys[0]], [zs[0], z_floor], linestyle="--", linewidth=0.8)
    ax.plot([xs[-1], xs[-1]], [ys[-1], ys[-1]], [zs[-1], z_floor], linestyle="--", linewidth=0.8)
    ax.scatter(xs[0], ys[0], z_floor, s=20, marker="o")
    ax.scatter(xs[-1], ys[-1], z_floor, s=24, marker="^")

    ax.legend(loc="best", fontsize=8, frameon=True)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath)  # no bbox_inches="tight" to avoid CL warnings
    plt.close()

def _recover_w_from_xyz(qx, qy, qz):
    # assumes near-unit quaternion; clamp to avoid tiny negatives
    ww = max(0.0, 1.0 - (qx*qx + qy*qy + qz*qz))
    return math.sqrt(ww)

def _normalize_quat(q):
    n = math.sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3])
    return (q[0]/n, q[1]/n, q[2]/n, q[3]/n) if n > 0 else (1.0, 0.0, 0.0, 0.0)

def _ensure_continuity(prev, cur):
    # flip sign if dot<0 to avoid 180° jumps
    dot = sum(p*c for p, c in zip(prev, cur))
    return tuple(-c for c in cur) if dot < 0 else cur

def _quat_to_euler_zyx(w, x, y, z):
    # returns roll(x), pitch(y), yaw(z) in radians
    # ZYX (intrinsic) — yaw(Z), pitch(Y), roll(X)
    t0 = 2.0 * (w*x + y*z)
    t1 = 1.0 - 2.0 * (x*x + y*y)
    roll = math.atan2(t0, t1)

    t2 = 2.0 * (w*y - z*x)
    t2 = max(-1.0, min(1.0, t2))  # clamp for asin
    pitch = math.asin(t2)

    t3 = 2.0 * (w*z + x*y)
    t4 = 1.0 - 2.0 * (y*y + z*z)
    yaw = math.atan2(t3, t4)
    return roll, pitch, yaw

def save_angles_plot(df: pd.DataFrame, title: str, outpath: Path):
    qxs = df["qx"].to_numpy(dtype=float)
    qys = df["qy"].to_numpy(dtype=float)
    qzs = df["qz"].to_numpy(dtype=float)
    if len(qxs) == 0 or not (np.isfinite(qxs).any() and np.isfinite(qys).any() and np.isfinite(qzs).any()):
        return

    # Build normalized, sign-continuous quaternions
    quats = []
    for i in range(len(qxs)):
        w = _recover_w_from_xyz(qxs[i], qys[i], qzs[i])
        q = _normalize_quat((w, qxs[i], qys[i], qzs[i]))
        if i > 0:
            q = _ensure_continuity(quats[-1], q)
        quats.append(q)

    # Quaternion -> Euler ZYX (rad), unwrap, then wrap to [0, 2π]
    rolls, pitchs, yaws = [], [], []
    for (w, x, y, z) in quats:
        r, p, y_ = _quat_to_euler_zyx(w, x, y, z)
        rolls.append(r); pitchs.append(p); yaws.append(y_)
    rolls = np.unwrap(np.array(rolls))
    pitchs = np.unwrap(np.array(pitchs))
    yaws = np.unwrap(np.array(yaws))

    two_pi = 2.0 * np.pi
    rolls_w  = np.mod(rolls,  two_pi)
    pitchs_w = np.mod(pitchs, two_pi)
    yaws_w   = np.mod(yaws,   two_pi)

    # Legend numbers: "<start>→<end> rad" (wrapped values)
    lab_roll  = f"roll: {rolls_w[0]:.3f}→{rolls_w[-1]:.3f} rad"
    lab_pitch = f"pitch: {pitchs_w[0]:.3f}→{pitchs_w[-1]:.3f} rad"
    lab_yaw   = f"yaw: {yaws_w[0]:.3f}→{yaws_w[-1]:.3f} rad"

    fig = plt.figure(figsize=(7.0, 3.6), constrained_layout=True)
    ax = fig.add_subplot(111)
    x = np.arange(1, len(rolls_w) + 1)

    ax.plot(x, rolls_w,  linewidth=1.2, label=lab_roll)
    ax.plot(x, pitchs_w, linewidth=1.2, label=lab_pitch)
    ax.plot(x, yaws_w,   linewidth=1.2, label=lab_yaw)

    ax.set_xlabel("request idx")
    ax.set_ylabel("angle [rad]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Fix y-axis to [0, 2π]
    ax.set_ylim(0.0, two_pi)

    ax.legend(loc="best", fontsize=8, frameon=True)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath)
    plt.close()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--one", type=str, default=None, help="Process a single CSV file (path).")
    p.add_argument("--out", type=str, default="eval_out", help="Output directory.")
    args = p.parse_args()

    here = Path(__file__).resolve().parent
    out_root = here / args.out
    figs = out_root / "figs"
    data_dir = out_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    files = [Path(args.one)] if args.one else sorted(here.glob("record*.csv"))
    if not files:
        print("No record*.csv found. Use --one to specify a file.", file=sys.stderr)
        sys.exit(2)

    per_rows, combined_50 = [], []

    for f in files:
        try:
            df = load_first50(f)
        except Exception as e:
            print(f"[SKIP] {f.name}: {e}", file=sys.stderr)
            continue

        record_id = f.stem
        per_rows.append(summarize_basic(df, record_id))
        combined_50.append(df)

        # New per-record KPI CSV (rounded)
        save_per_record_kpis(df, data_dir / f"kpis_{record_id}.csv")

        # Only trajectory plot, fixed axes
        save_traj3d(df, f"Trajectory (first 50) — {record_id}", figs / f"traj3d_{record_id}.pdf")
        save_angles_plot(df, f"Angles (ZYX) — {record_id}",     figs / f"angles_{record_id}.pdf")

    if not per_rows:
        print("No valid files processed.", file=sys.stderr)
        sys.exit(3)

    # Keep the summaries (useful for LaTeX overall tables)
    pd.DataFrame(per_rows).sort_values("record_id").to_csv(data_dir / "summary_per_file.csv", index=False)
    all_df = pd.concat(combined_50, ignore_index=True)
    overall = summarize_basic(all_df, record_id="ALL_FIRST50")
    pd.DataFrame([overall]).to_csv(data_dir / "summary_overall.csv", index=False)

    print(f"✔ Per-record KPIs in {data_dir}/kpis_<record>.csv (ms, rounded to 3 decimals)")
    print(f"✔ Per-file summary → {data_dir/'summary_per_file.csv'}; overall → {data_dir/'summary_overall.csv'}")
    print(f"✔ Trajectories (fixed axes) → {figs}")

if __name__ == "__main__":
    main()