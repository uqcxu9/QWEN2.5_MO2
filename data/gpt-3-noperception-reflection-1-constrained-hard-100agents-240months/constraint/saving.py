import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta

save_path = '../'

with open(f'{save_path}dense_log.pkl', 'rb') as f:
    dense_log = pkl.load(f)

# Data sources:
states = dense_log['states']          # State data for each agent in each period
periodic_tax = dense_log['PeriodicTax']  # Tax and redistribution data

# 2. Determine Start Time (extracted from simulate.py code)
world_start_time = datetime.strptime('2001.01', '%Y.%m')  # Simulation start time

# First check the structure of periodic_tax
print("=" * 60)
print("🔍 Data Structure Check")
print("=" * 60)
print(f"States length: {len(states)} (number of time steps)")
print(f"PeriodicTax type: {type(periodic_tax)}")

if isinstance(periodic_tax, list):
    print(f"PeriodicTax length: {len(periodic_tax)}")
    print(f"PeriodicTax[0] sample keys: {list(periodic_tax[0].keys())[:3]}")
else:
    print(f"PeriodicTax keys: {list(periodic_tax.keys())[:3]}")

# 3. Calculate Saving Rate for Each Month
print("\n" + "=" * 60)
print("📊 Calculating Personal Saving Rate")
print("=" * 60)

saving_rates = []
months = []

for t in range(len(states)):
    # Current time
    current_time = world_start_time + relativedelta(months=t)
    months.append(current_time.strftime('%Y.%m'))
    
    total_DPI = 0
    total_saving = 0
    
    for agent_id in states[t].keys():
        if agent_id == 'p':  # Skip government
            continue
        
        # Data source 1: income['Coin'] - wage income
        income = states[t][agent_id]['income']['Coin']
        
        # Data source 2: lump_sum - government redistribution income
        # Data source 3: tax_paid - personal income tax
        if isinstance(periodic_tax, list):
            if t < len(periodic_tax):
                lump_sum = periodic_tax[t].get(agent_id, {}).get('lump_sum', 0)
                tax_paid = periodic_tax[t].get(agent_id, {}).get('tax_paid', 0)
            else:
                lump_sum = 0
                tax_paid = 0
        else:  # periodic_tax is a dictionary, indexed by agent_id
            agent_tax_data = periodic_tax.get(agent_id, [])
            if t < len(agent_tax_data):
                lump_sum = agent_tax_data[t].get('lump_sum', 0)
                tax_paid = agent_tax_data[t].get('tax_paid', 0)
            else:
                lump_sum = 0
                tax_paid = 0
        
        # Data source 4: consumption['Coin'] - consumption expenditure (PCE)
        consumption = states[t][agent_id]['consumption']['Coin']
        
        # Calculation formula:
        # DPI = Personal Income - Personal Taxes
        #     = (income + lump_sum) - tax_paid
        DPI = income + lump_sum - tax_paid
        
        # Personal Saving = DPI - PCE
        saving = DPI - consumption
        
        total_DPI += DPI
        total_saving += saving
    
    # Personal Saving Rate = Personal Saving / DPI
    if total_DPI > 0:
        saving_rates.append(total_saving / total_DPI)
    else:
        saving_rates.append(0)

# 4. Plot
print(f"\nStart time: {months[0]}")
print(f"End time: {months[-1]}")
print(f"Total time steps: {len(months)}")

fig, ax = plt.subplots(figsize=(14, 6))

# Plot saving rate curve
ax.plot(range(len(saving_rates)), saving_rates, linewidth=2, color='steelblue')

# X-axis labels: mark every 12 months (annually)
x_ticks = range(0, len(months), 12)
x_labels = [months[i] for i in x_ticks]
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_labels, rotation=45, ha='right')

# Format Y-axis as percentage
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

ax.set_xlabel('Time (Year.Month)', fontsize=12)
ax.set_ylabel('Personal Saving Rate', fontsize=12)
ax.set_title('Personal Saving Rate Over Time (2001.01 - 2021.01)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(f'{save_path}personal_saving_rate.png', dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: {save_path}personal_saving_rate.png")
plt.show()

# 5. Statistical Information
print("\n" + "=" * 60)
print("📈 Statistical Results")
print("=" * 60)
print(f"Average saving rate: {np.mean(saving_rates):.2%}")
print(f"Saving rate range: [{np.min(saving_rates):.2%}, {np.max(saving_rates):.2%}]")
print(f"Median saving rate: {np.median(saving_rates):.2%}")
print(f"Standard deviation: {np.std(saving_rates):.2%}")
# 🆕 添加这段代码
print("\n" + "=" * 60)
print("⚠️  Constraint Violation Analysis (1.4% - 31.8%)")
print("=" * 60)
violations = [(i, rate) for i, rate in enumerate(saving_rates) if rate < 0.014 or rate > 0.318]
violation_rate = len(violations) / len(saving_rates) * 100
print(f"Total months: {len(saving_rates)}")
print(f"Violation months: {len(violations)}")
print(f"Violation rate: {violation_rate:.2f}%")
if violations:
    print(f"\nViolation details (first 10):")
    for i, (month_idx, rate) in enumerate(violations[:10]):
        print(f"  {months[month_idx]}: {rate:.2%}")
print("=" * 60)

print("=" * 60)

# 6. Save Results to CSV
import pandas as pd

# Create DataFrame
results_df = pd.DataFrame({
    'Month': months,
    'Saving_Rate': saving_rates
})

# Save to CSV
csv_path = f'{save_path}personal_saving_rate.csv'
results_df.to_csv(csv_path, index=False)
print(f"\nData saved to: {csv_path}")

# Data Source Summary
print("\n📝 Data Source Summary:")
print("=" * 60)
print("1. income['Coin']        ← states[t][agent_id]['income']['Coin']")
print("2. lump_sum              ← periodic_tax[t][agent_id]['lump_sum']")
print("3. tax_paid              ← periodic_tax[t][agent_id]['tax_paid']")
print("4. consumption['Coin']   ← states[t][agent_id]['consumption']['Coin']")
print("\nCalculation formula:")
print("DPI = (income + lump_sum) - tax_paid")
print("Personal Saving = DPI - consumption")
print("Personal Saving Rate = Personal Saving / DPI")
print("=" * 60)