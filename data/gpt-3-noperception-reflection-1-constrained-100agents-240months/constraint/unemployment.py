# -*- coding: utf-8 -*-

import os
import pickle as pkl
import csv
import matplotlib.pyplot as plt

BASE = r"/workspace/ACL24-EconAgent/data/gpt-3-noperception-reflection-1-constrained-100agents-240months"
DATA = BASE  # log文件就在BASE目录下
OUT = os.path.join(DATA, "result_analysis")
os.makedirs(OUT, exist_ok=True)

P_DENSE = os.path.join(DATA, "dense_log.pkl")


def compute_monthly_unemployment():
    """Compute monthly unemployment rate: unemployed population / labor force"""
    with open(P_DENSE, "rb") as f:
        dense = pkl.load(f)

    states = dense.get("states", [])
    monthly_rates = []

    for m in range(1, len(states)):
        unemployed = 0
        employed = 0

        for agent_id, agent_state in states[m].items():
            if agent_id == "p" or not isinstance(agent_state, dict):
                continue

            job = agent_state.get("endogenous", {}).get("job")
            if job == "Unemployment":
                unemployed += 1
            else:
                employed += 1

        labor_force = employed + unemployed
        rate = unemployed / labor_force if labor_force > 0 else 0
        monthly_rates.append(rate)

    return monthly_rates


def analyze_and_save():
    """Analyze results and save output files"""
    rates = compute_monthly_unemployment()

    # Statistical summary
    mean_rate = sum(rates) / len(rates)
    violations_low = sum(1 for r in rates if r < 0.035)
    violations_high = sum(1 for r in rates if r > 0.148)

    print("=" * 60)
    print("Monthly Unemployment Statistics")
    print("=" * 60)
    print(f"Total months: {len(rates)}")
    print(f"Average unemployment rate: {mean_rate * 100:.2f}%")
    print(f"Minimum: {min(rates) * 100:.2f}%")
    print(f"Maximum: {max(rates) * 100:.2f}%")
    print(f"Violations below 3.5%: {violations_low} times")
    print(f"Violations above 14.8%: {violations_high} times")
    print(
        f"Total violation rate: {(violations_low + violations_high) / len(rates) * 100:.2f}%"
    )

    # Save results as CSV
    csv_path = os.path.join(OUT, "unemployment_monthly.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["month", "unemployment_rate", "violation"])
        for m, rate in enumerate(rates, start=1):
            violation = "low" if rate < 0.035 else ("high" if rate > 0.148 else "")
            writer.writerow([m, rate, violation])
    print(f"\nCSV saved: {csv_path}")

    # Plot the unemployment rate trend
    fig, ax = plt.subplots(figsize=(14, 6))
    months = list(range(1, len(rates) + 1))

    ax.plot(
        months,
        [r * 100 for r in rates],
        label="Unemployment Rate",
        linewidth=1.5,
        color="#2E86AB",
    )
    ax.axhline(
        y=3.5,
        color="green",
        linestyle="--",
        alpha=0.5,
        label="Lower Bound (3.5%)",
    )
    ax.axhline(
        y=14.8,
        color="red",
        linestyle="--",
        alpha=0.5,
        label="Upper Bound (14.8%)",
    )
    ax.fill_between(months, 3.5, 14.8, alpha=0.1, color="green")

    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("Unemployment Rate (%)", fontsize=12)
    ax.set_title("Monthly Unemployment Rate", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(OUT, "unemployment_monthly.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved: {fig_path}")
    plt.close()

    print("\n" + "=" * 60)
    print("Analysis completed!")
    print("=" * 60)


if __name__ == "__main__":
    analyze_and_save()
