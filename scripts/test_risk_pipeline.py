import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from backend.db_utils import get_recent_readings


def _load_risk_core_namespace() -> dict:
    """
    Load only the core risk-calculation definitions (skip demo plotting block).
    """
    src = (Path(__file__).parent.parent / "backend" / "risk_calculation.py").read_text(encoding="utf-8", errors="ignore")
    split_marker = "#--------------------------------------------------------------------------------\n# simple simulation"
    core = src.split(split_marker)[0]
    ns: dict = {}
    exec(core, ns)
    return ns


def main() -> None:
    ns = _load_risk_core_namespace()
    rows = get_recent_readings(limit=500)
    print(f"rows={len(rows)}")

    # Keep only rows that can drive VTT, then sort oldest -> newest
    rows_sorted = sorted(
        [r for r in rows if r.get("temperature_c") is not None and r.get("humidity_pct") is not None],
        key=lambda r: r["timestamp"],
    )
    if not rows_sorted:
        raise RuntimeError("No valid rows with both temperature_c and humidity_pct were found.")

    temps = [float(r["temperature_c"]) for r in rows_sorted]
    rhs = [float(r["humidity_pct"]) for r in rows_sorted]

    params = ns["VTTOriginalParams"](sample_minutes=10)
    mold_results = ns["run_vtt_original_series"](
        temperatures_c=temps,
        rhs_percent=rhs,
        params=params,
        initial_m=0.0,
    )

    if not mold_results:
        raise RuntimeError("VTT simulation returned no results.")

    # Optional heat-risk sanity output from latest hour.
    heat_in = ns["build_heat_risk_inputs_from_recent_readings"](rows)
    heat = ns["compute_heat_risk"](
        **{
            k: heat_in[k]
            for k in [
                "temp_db_mean",
                "temp_db_max",
                "rh_percent_mean",
                "high_thi_streak_minutes",
                "feed_intake",
                "water_intake",
                "thi_slope_per_hour",
            ]
        }
    )
    print("HEAT:", heat)

    last = mold_results[-1]
    print(
        "MOLD_LAST:",
        {
            "m": last.m,
            "dmdt_per_24h": last.dmdt_per_24h,
            "rhcrit": last.rhcrit,
            "mmax": last.mmax,
        },
    )

    # Full-period graph: timestamp vs M
    timestamps = [r["timestamp"] for r in rows_sorted]
    m_values = [s.m for s in mold_results]

    diffs = [m_values[i] - m_values[i - 1] for i in range(1, len(m_values))]
    nonzero = [d for d in diffs if abs(d) > 1e-12]

    print(f"Total steps: {len(m_values)}")
    print(f"Steps with change: {len(nonzero)}")
    if nonzero:
        print(f"Min delta: {min(nonzero):.12f}")
        print(f"Max delta: {max(nonzero):.12f}")
        print("Last 10 M values:")
        for v in m_values[-10:]:
            print(f"{v:.12f}")
    else:
        print("No detectable change in M (within 1e-12 tolerance).")

    plt.figure(figsize=(12, 5))
    plt.plot(timestamps, m_values, linewidth=2, label="Mould index M")
    y_min = min(m_values)
    y_max = max(m_values)
    if y_max - y_min < 1e-9:
        pad = max(abs(y_max) * 0.1, 1e-6)
    else:
        pad = (y_max - y_min) * 0.1
    plt.ylim(y_min - pad, y_max + pad)
    plt.xlabel("Time")
    plt.ylabel("M (mold score)")
    plt.title("VTT Mold Score Over Entire Simulation Period")
    plt.grid(True, alpha=0.3)
    plt.legend()
    ax = plt.gca()
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-3, 3))
    ax.yaxis.set_major_formatter(formatter)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    plt.tight_layout()
    out_path = Path("test_risk_pipeline_m_plot.png")
    plt.savefig(out_path, dpi=150)
    print(f"Plot saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
