# -*- coding: utf-8 -*-
"""
Annual Price Inflation Rate Constraint Validation (Based on Real U.S. CPI Data)
"""
import pickle as pkl
import numpy as np
import csv
import os

# ==================== Load Simulation Data ====================
class DummyUnpickler(pkl.Unpickler):
    def find_class(self, module, name):
        if 'ai_economist' in module:
            return type(name, (), {})
        return super().find_class(module, name)

BASE = r"/workspace/ACL24-EconAgent/data/gpt-3-noperception-reflection-1-100agents-240months"
env_file = os.path.join(BASE, "env_240.pkl")
OUT = os.path.join(BASE, "result_analysis")
os.makedirs(OUT, exist_ok=True)

with open(env_file, "rb") as f:
    env = DummyUnpickler(f).load()

# Extract prices (skip month 0)
prices = list(env.world.price)[1:]  # 240 months

# Calculate annual average prices (20 years)
yearly_avg_price = []
for y in range(0, 240, 12):
    yearly_avg_price.append(np.mean(prices[y:y+12]))

# Calculate annual inflation rates (19 years, 2002-2020)
yearly_inflation = []
for i in range(1, len(yearly_avg_price)):
    inflation = (yearly_avg_price[i] - yearly_avg_price[i-1]) / yearly_avg_price[i-1]
    yearly_inflation.append(inflation)

# ==================== Real U.S. CPI Data ====================
us_cpi_inflation = {
    2002: 1.586, 2003: 2.270, 2004: 2.677, 2005: 3.393,
    2006: 3.226, 2007: 2.853, 2008: 3.839, 2009: -0.356,
    2010: 1.640, 2011: 3.157, 2012: 2.069, 2013: 1.465,
    2014: 1.622, 2015: 0.119, 2016: 1.262, 2017: 2.130,
    2018: 2.443, 2019: 1.812, 2020: 1.234, 2021: 4.698
}

us_inflation = [v/100 for v in us_cpi_inflation.values()]

# ==================== Calculate Constraints ====================
us_min = -0.00356  # -0.36% (2009)
us_max = 0.04698   # 4.70% (2021)
us_mean = np.mean(us_inflation)

# Annual maximum volatility
us_volatility = [abs(us_inflation[i] - us_inflation[i-1]) for i in range(1, len(us_inflation))]
us_max_volatility = max(us_volatility)

# ==================== Validate Simulation Data ====================
violations_low = sum(1 for r in yearly_inflation if r < us_min)
violations_high = sum(1 for r in yearly_inflation if r > us_max)
violations_total = violations_low + violations_high

# ==================== Console Output ====================
print("=" * 70)
print("Real U.S. CPI Inflation (2002-2021)")
print("=" * 70)
print(f"Sample size: 20 years")
print(f"Average: {us_mean*100:6.2f}%")
print(f"Minimum: {us_min*100:6.2f}% (2009 Financial Crisis)")
print(f"Maximum: {us_max*100:6.2f}% (2021 Post-Pandemic)")
print(f"Max annual volatility: {us_max_volatility*100:6.2f}%")

print("\n" + "=" * 70)
print("Simulated Price Inflation (2002-2020)")
print("=" * 70)
print(f"Sample size: 19 years")
print(f"Average: {np.mean(yearly_inflation)*100:6.2f}%")
print(f"Minimum: {min(yearly_inflation)*100:6.2f}%")
print(f"Maximum: {max(yearly_inflation)*100:6.2f}%")

print("\n" + "=" * 70)
print("Constraint Validation Results")
print("=" * 70)
print(f"Constraint range: [{us_min*100:.2f}%, {us_max*100:.2f}%]")
print(f"Violations below lower bound: {violations_low}/19")
print(f"Violations above upper bound: {violations_high}/19")
print(f"Total violation rate: {violations_total}/19 ({violations_total/19*100:.2f}%)")

if violations_total == 0:
    print("\nAll years satisfy constraints")
else:
    print(f"\n{violations_total} years violate constraints")

print("\n" + "=" * 70)
print("Year-by-Year Comparison (2002-2020)")
print("=" * 70)
print("Year   Simulated Rate   Constraint Range      Status")
print("-" * 70)
for i in range(len(yearly_inflation)):
    year = 2002 + i
    rate = yearly_inflation[i] * 100
    if yearly_inflation[i] < us_min:
        status = "Below lower bound"
    elif yearly_inflation[i] > us_max:
        status = "Above upper bound"
    else:
        status = "Satisfied"
    print(f"{year}   {rate:6.2f}%         [{us_min*100:.2f}%, {us_max*100:.2f}%]   {status}")

# ==================== Save to CSV ====================
# Save yearly comparison
csv_path = os.path.join(OUT, "inflation_yearly_comparison.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["year", "inflation_rate", "lower_bound", "upper_bound", "violation"])
    for i in range(len(yearly_inflation)):
        year = 2002 + i
        rate = yearly_inflation[i]
        if rate < us_min:
            violation = "low"
        elif rate > us_max:
            violation = "high"
        else:
            violation = ""
        writer.writerow([year, rate, us_min, us_max, violation])
print(f"\nCSV saved: {csv_path}")

# Save summary statistics
summary_path = os.path.join(OUT, "inflation_summary.csv")
with open(summary_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["metric", "value"])
    writer.writerow(["Real US CPI mean", us_mean])
    writer.writerow(["Real US CPI min", us_min])
    writer.writerow(["Real US CPI max", us_max])
    writer.writerow(["Real US CPI max volatility", us_max_volatility])
    writer.writerow(["Simulated mean", np.mean(yearly_inflation)])
    writer.writerow(["Simulated min", min(yearly_inflation)])
    writer.writerow(["Simulated max", max(yearly_inflation)])
    writer.writerow(["Violations below bound", violations_low])
    writer.writerow(["Violations above bound", violations_high])
    writer.writerow(["Total violations", violations_total])
    writer.writerow(["Violation rate", violations_total/19])
print(f"Summary saved: {summary_path}")