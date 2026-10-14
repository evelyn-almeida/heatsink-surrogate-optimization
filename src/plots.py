import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from config import FIG_DIR, MASS_LIMIT, MIN_GAP
from src.simulator import (FAN_FLOW_MAX, POWER, T_AMBIENT, WIDTH, fan_pressure,
                           geometry, pressure_drop, simulate)

BLUE, ORANGE, GREEN, RED, GRAY = "#1f4e79", "#e07a1f", "#2a9d8f", "#c0392b", "#8a8a8a"
SURROGATE_COLORS = {"Gaussian process": BLUE, "Quadratic response surface": ORANGE, "Random forest": GREEN}
METHOD_COLORS = {"Bayesian optimization": BLUE, "Random search": GRAY}
DESIGN_COLORS = [GRAY, BLUE, GREEN]


def to_temp(resistance):
    return T_AMBIENT + POWER * np.asarray(resistance)


def setup_style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
    })


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def _box(ax, text, x=0.04, y=0.95, ha="left"):
    ax.text(x, y, text, transform=ax.transAxes, va="top", ha=ha,
            bbox=dict(boxstyle="round", fc="white", ec=GRAY))


def fan_operating_point(x_opt):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    q = np.linspace(1e-6, FAN_FLOW_MAX, 300)
    axes[0].plot(q * 1e3, fan_pressure(q), color="black", lw=2.5, label="Fan curve")
    for n_fins, color in zip([12, round(x_opt[0]), 60], [GREEN, BLUE, RED]):
        x = np.array(x_opt, dtype=float)
        x[0] = n_fins
        n, t, gap, H, _, L = (np.full_like(q, g[0]) for g in geometry(x[None]))
        flow = simulate(x[None])["flow"][0]
        axes[0].plot(q * 1e3, pressure_drop(q, n, t, gap, H, L), color=color, lw=2, label=f"{n_fins} fins")
        axes[0].scatter(flow * 1e3, fan_pressure(flow), color=color, s=90, edgecolor="black", zorder=5)
    axes[0].set(xlabel="Air flow (L/s)", ylabel="Pressure (Pa)", ylim=(0, 100),
                title="Operating point: where the fan meets the heat sink")
    axes[0].legend(loc="upper right")

    ns = np.arange(8, 61)
    X = np.tile(np.asarray(x_opt, dtype=float), (len(ns), 1))
    X[:, 0] = ns
    r = simulate(X)
    axes[1].plot(ns, to_temp(r["resistance"]), color=BLUE, lw=2.5, label="Chip temperature")
    axes[1].set(xlabel="Number of fins", ylabel="Max. chip temperature (°C)",
                title="More fins: more surface, but less air")
    twin = axes[1].twinx()
    twin.plot(ns, r["flow"] * 1e3, color=ORANGE, lw=2, ls="--", label="Air flow")
    twin.set_ylabel("Air flow (L/s)", color=ORANGE)
    twin.grid(False)
    blocked = r["gap_mm"] < MIN_GAP
    if blocked.any():
        axes[1].axvspan(ns[blocked].min() - 0.5, 60.5, color=RED, alpha=0.1, label=f"Gap < {MIN_GAP:g} mm")
    best = ns[~blocked][np.argmin(r["resistance"][~blocked])]
    axes[1].axvline(best, color=BLUE, ls=":", lw=1.5)
    lines = axes[1].get_legend_handles_labels()
    lines2 = twin.get_legend_handles_labels()
    axes[1].legend(lines[0] + lines2[0], lines[1] + lines2[1], loc="upper center")
    fig.tight_layout()
    save(fig, "01_fan_operating_point")


def learning_curves(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    sizes = sorted(df["n"].unique())
    for name, g in df.groupby("model"):
        stats = g.groupby("n")
        for ax, metric in zip(axes, ["mape", "r2"]):
            med = stats[metric].median()
            ax.plot(med.index, med, marker="o", lw=2.5, color=SURROGATE_COLORS[name], label=name)
            ax.fill_between(med.index, stats[metric].min(), stats[metric].max(),
                            color=SURROGATE_COLORS[name], alpha=0.15)
    axes[0].set(yscale="log", ylabel="Mean absolute error (%)", title="Prediction error vs. simulation budget")
    ticks = [1, 2, 5, 10, 20, 50]
    axes[0].set_yticks(ticks, [f"{t}%" for t in ticks])
    axes[1].set(ylabel="R² on 2,000 unseen designs", ylim=(0, 1.02), title="Explained variance")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(sizes, [str(s) for s in sizes])
        ax.minorticks_off()
        ax.set_xlabel("Number of simulations used for training")
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    save(fig, "02_learning_curves")


def parity(y_true, gp_mean, gp_lo, gp_hi, rsm_pred, n_train):
    t_true = to_temp(y_true)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharey=True)
    lims = (t_true.min() - 2, np.percentile(t_true, 99) + 5)
    covered = (y_true >= gp_lo) & (y_true <= gp_hi)
    axes[0].vlines(t_true, to_temp(gp_lo), to_temp(gp_hi), color=GRAY, alpha=0.3, lw=1)
    axes[0].scatter(t_true[covered], to_temp(gp_mean)[covered], s=14, color=BLUE, label="Inside 90% interval")
    axes[0].scatter(t_true[~covered], to_temp(gp_mean)[~covered], s=18, color=RED, label="Outside 90% interval")
    axes[1].scatter(t_true, to_temp(rsm_pred), s=14, color=ORANGE)
    for ax, pred, title in [(axes[0], gp_mean, "Gaussian process"), (axes[1], rsm_pred, "Quadratic response surface")]:
        ax.plot(lims, lims, color="black", ls="--", lw=1)
        ax.set(xlim=lims, ylim=lims, xlabel="Simulated temperature (°C)",
               title=f"{title}, trained on {n_train} simulations")
        mape = 100 * np.mean(np.abs(pred - y_true) / y_true)
        text = f"Mean abs. error = {mape:.1f}%"
        if pred is gp_mean:
            text += f"\n90% interval coverage = {covered.mean():.0%}"
        _box(ax, text)
    axes[0].set_ylabel("Predicted temperature (°C)")
    axes[0].legend(loc="lower right")
    fig.tight_layout()
    save(fig, "03_parity")


def response_surface(n_vals, h_vals, t_true, t_gp, t_std, mass, gap, train, optimum):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.9))
    levels = np.linspace(np.nanmin(t_true), np.percentile(t_true, 98), 21)
    for ax, z, title in [(axes[0], t_true, "Simulator (ground truth)"),
                         (axes[1], t_gp, f"Gaussian process ({len(train)} simulations)")]:
        cs = ax.contourf(n_vals, h_vals, z, levels=levels, cmap="RdYlBu_r", extend="both")
        ax.contour(n_vals, h_vals, mass, levels=[MASS_LIMIT], colors="black", linestyles="--", linewidths=1.5)
        ax.contour(n_vals, h_vals, gap, levels=[MIN_GAP], colors="black", linestyles=":", linewidths=1.5)
        ax.scatter(*optimum, marker="*", s=300, color="white", edgecolor="black", zorder=5)
        ax.set_title(title)
    fig.colorbar(cs, ax=axes[:2], label="Max. chip temperature (°C)", pad=0.02, format="%.0f")
    axes[0].text(0.97, 0.95, f"- - mass = {MASS_LIMIT:g} g\n···· gap = {MIN_GAP:g} mm\n★ optimum",
                 transform=axes[0].transAxes, ha="right", va="top", fontsize=9,
                 bbox=dict(boxstyle="round", fc="white", ec=GRAY))
    cs = axes[2].contourf(n_vals, h_vals, t_std, levels=15, cmap="Purples")
    fig.colorbar(cs, ax=axes[2], label="Predictive std. (°C)", pad=0.02, format="%.1f")
    axes[2].set_title("Where the surrogate is unsure")
    for ax in axes[1:]:
        ax.scatter(train[:, 0], train[:, 1], s=16, color="black", alpha=0.6, label="Training designs (projected)")
    axes[2].legend(loc="upper right", fontsize=8, frameon=True)
    for ax in axes:
        ax.set(xlabel="Number of fins", ylabel="Fin height (mm)")
        ax.grid(False)
    fig.suptitle("Design space slice: fins × height (other variables at the optimum)",
                 fontsize=13, fontweight="bold", y=1.02)
    save(fig, "04_response_surface")


def sensitivity(labels, true_idx, gp_idx, n_true, n_gp):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    y = np.arange(len(labels))
    for ax, k, title in [(axes[0], 0, "First-order index (effect alone)"),
                         (axes[1], 1, "Total index (including interactions)")]:
        ax.barh(y + 0.2, true_idx[k], height=0.38, color=GRAY, label=f"Simulator ({n_true:,} runs)")
        ax.barh(y - 0.2, gp_idx[k], height=0.38, color=BLUE, label=f"GP surrogate ({n_gp} runs)")
        ax.set(title=title, xlabel="Share of output variance", xlim=(0, true_idx[1].max() * 1.2))
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[1].legend(loc="lower right")
    fig.suptitle("Global sensitivity analysis (Sobol indices): which decisions matter?",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "05_sensitivity")


def bo_in_action(grid, true_curve, states, show=(0, 2, 5)):
    fig, axes = plt.subplots(2, len(show), figsize=(15, 6.5), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1]})
    for col, k in enumerate(show):
        s, top, bottom = states[k], axes[0, col], axes[1, col]
        top.plot(grid, true_curve, color="black", lw=1.2, ls="--", label="True (unknown)")
        top.plot(grid, s["mu"], color=BLUE, lw=2, label="GP mean")
        top.fill_between(grid, s["mu"] - 2 * s["sd"], s["mu"] + 2 * s["sd"], color=BLUE, alpha=0.18,
                         label="±2σ")
        top.scatter(s["x"], s["y"], color=ORANGE, s=60, edgecolor="black", zorder=5, label="Simulations")
        top.axvline(s["next"], color=RED, lw=1.2, ls=":")
        top.set(title=f"After {len(s['x'])} simulations", ylim=(true_curve.min() - 3, true_curve.max() + 3))
        bottom.fill_between(grid, 0, s["ei"], color=GREEN, alpha=0.4)
        bottom.axvline(s["next"], color=RED, lw=1.2, ls=":", label="Next simulation")
        bottom.set(xlabel="Number of fins", yticks=[])
    axes[0, 0].set_ylabel("Max. chip temperature (°C)")
    axes[1, 0].set_ylabel("Expected\nimprovement")
    axes[0, 0].legend(loc="upper right", fontsize=8)
    axes[1, 0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Bayesian optimization in action: balancing exploration and exploitation",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "06_bo_in_action")


def bo_convergence(df, t_opt, t_base, n_init, de_evals):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.9), gridspec_kw={"width_ratios": [2.2, 1]})
    for method, g in df.groupby("method"):
        stats = g.groupby("evaluation")["best_t"]
        med = stats.median()
        axes[0].plot(med.index, med, color=METHOD_COLORS[method], lw=2.5, label=method)
        axes[0].fill_between(med.index, stats.quantile(0.25), stats.quantile(0.75),
                             color=METHOD_COLORS[method], alpha=0.2)
    axes[0].axhline(t_opt, color=GREEN, ls="--", lw=1.5, label=f"True optimum ({t_opt:.1f} °C)")
    axes[0].axhline(t_base, color=RED, ls=":", lw=1.5, label=f"Hand-made baseline ({t_base:.1f} °C)")
    axes[0].axvspan(0.5, n_init + 0.5, color=GRAY, alpha=0.1)
    axes[0].text(n_init / 2 + 0.5, 0.97, "initial\nDoE", transform=axes[0].get_xaxis_transform(),
                 ha="center", va="top", fontsize=9, color=GRAY)
    axes[0].set(xlabel="Number of simulations", ylabel="Best chip temperature found (°C)",
                title="Median and interquartile range over independent runs",
                ylim=(t_opt - 1, max(t_base + 2, t_opt + 8)))
    axes[0].legend(loc="upper right")

    final = df[df["evaluation"] == df["evaluation"].max()]
    methods = list(METHOD_COLORS)
    data = [final.loc[final["method"] == m, "best_t"] - t_opt for m in methods]
    box = axes[1].boxplot(data, patch_artist=True, widths=0.5, tick_labels=["BO", "Random"])
    for patch, m in zip(box["boxes"], methods):
        patch.set(facecolor=METHOD_COLORS[m], alpha=0.5)
    axes[1].set(ylabel="Gap to true optimum (°C)", title=f"After {df['evaluation'].max()} simulations")
    _box(axes[1], f"Genetic-style search\n(differential evolution)\nneeded {de_evals:,} simulations\n"
                  f"to get within 0.1 °C", x=0.96, ha="right")
    fig.tight_layout()
    save(fig, "07_bo_convergence")


def pareto(cloud_mass, cloud_t, front, sur_mass, sur_t, designs, n_gp):
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    idx = np.random.default_rng(0).choice(len(cloud_mass), min(15000, len(cloud_mass)), replace=False)
    ax.scatter(cloud_mass[idx], cloud_t[idx], s=3, color=GRAY, alpha=0.15, label="Possible designs")
    order = np.argsort(cloud_mass[front])
    ax.plot(cloud_mass[front][order], cloud_t[front][order], color="black", lw=2, label="True Pareto front")
    ax.scatter(sur_mass, sur_t, s=22, color=ORANGE, edgecolor="black", lw=0.4, zorder=4,
               label=f"Front predicted by GP ({n_gp} simulations), verified")
    markers = ["o", "D", "s"]
    for (name, d), color, marker in zip(designs.items(), DESIGN_COLORS, markers):
        ax.scatter(d["mass"], d["t_max"], s=170, marker=marker, color=color, edgecolor="black", zorder=6, label=name)
    ax.axvline(MASS_LIMIT, color=RED, ls="--", lw=1)
    ax.text(MASS_LIMIT + 3, 0.97, f"weight budget\n{MASS_LIMIT:g} g", color=RED, va="top",
            transform=ax.get_xaxis_transform(), fontsize=9)
    t_hi = np.percentile(cloud_t, 60)
    ax.set(xlabel="Heat sink mass (g)", ylabel="Max. chip temperature (°C)", ylim=(cloud_t.min() - 2, t_hi),
           xlim=(0, np.percentile(cloud_mass, 99)), title="Temperature vs. weight: the trade-off frontier")
    ax.legend(loc="upper right", fontsize=9, frameon=True)
    save(fig, "08_pareto_front")


def design_comparison(designs):
    fig, axes = plt.subplots(1, len(designs), figsize=(15, 4.6), sharey=True)
    top = max(d["x"][2] + d["x"][3] for d in designs.values()) + 20
    for ax, (name, d), color in zip(axes, designs.items(), DESIGN_COLORS):
        n, *dims = (g[0] for g in geometry(np.asarray(d["x"])[None]))
        t, gap, H, tb, L = (v * 1e3 for v in dims)
        n = int(n)
        ax.add_patch(Rectangle((0, 0), WIDTH * 1e3, tb, color=color))
        for i in range(n):
            ax.add_patch(Rectangle((i * (t + gap), tb), t, H, color=color))
        ax.add_patch(Rectangle((20, -2.5), 20, 2.5, color=RED))
        ax.set(xlim=(-3, 63), ylim=(-4, top), aspect="equal", xlabel="Width (mm)", title=name)
        ax.grid(False)
        _box(ax, f"{n} fins · {t:.2f} mm thick · gap {gap:.2f} mm\n"
                 f"height {H:.0f} mm · base {tb:.1f} mm · length {L:.0f} mm\n"
                 f"T max = {d['t_max']:.1f} °C · mass = {d['mass']:.0f} g", x=0.5, y=0.99, ha="center")
    axes[0].set_ylabel("Height (mm)")
    fig.suptitle("Front view of the designs (red: 20 × 20 mm chip)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "09_design_comparison")


def resistance_breakdown(designs):
    parts = [("r_spread", "Spreading in base", ORANGE), ("r_base", "Conduction through base", RED),
             ("r_conv", "Fins to air (convection)", BLUE), ("r_air", "Air heating up", GREEN)]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    names = list(designs)
    left = np.zeros(len(names))
    for key, label, color in parts:
        vals = np.array([designs[n][key] for n in names])
        ax.barh(names, vals, left=left, color=color, label=label)
        left += vals
    for i, total in enumerate(left):
        ax.text(total + 0.005, i, f"{total:.3f} K/W", va="center")
    ax.invert_yaxis()
    ax.set(xlabel="Thermal resistance (K/W)", xlim=(0, left.max() * 1.18),
           title="Where the temperature rise comes from")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4)
    save(fig, "10_resistance_breakdown")
