import pickle as pkl
import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Load Data
save_path = '../'

with open(f'{save_path}dense_log.pkl', 'rb') as f:
    dense_log = pkl.load(f)

states = dense_log['states']
periodic_tax = dense_log['PeriodicTax']
world_start_time = datetime.strptime('2001.01', '%Y.%m')

max_t = min(len(states), len(periodic_tax))

print("=" * 60)
print("📊 Annual MPC Analysis (Non-overlapping Windows)")
print("=" * 60)

# Calculate Annual MPC (Non-overlapping 12 months)
agent_mpc_list = []

MIN_DELTA_DPI = 1000  # Annual minimum income change threshold

print(f"\nSetting: Annual ΔDPI ≥ ${MIN_DELTA_DPI} to be included")

# Calculate number of complete years
num_complete_years = max_t // 12
print(f"Number of complete years: {num_complete_years}")
print(f"Theoretical maximum sample size: {100 * (num_complete_years - 1)}")

for agent_id in range(100):
    agent_id_str = str(agent_id)
    
    # Compare consecutive complete years (non-overlapping)
    for year_idx in range(1, num_complete_years):
        year_start = year_idx * 12
        
        # Calculate Year 1 (current year) 12-month total
        total_dpi_year1 = 0
        total_consumption_year1 = 0
        
        for t in range(year_start, year_start + 12):
            income = states[t][agent_id_str]['income']['Coin']
            lump_sum = periodic_tax[t].get(agent_id_str, {}).get('lump_sum', 0)
            tax_paid = periodic_tax[t].get(agent_id_str, {}).get('tax_paid', 0)
            consumption = states[t][agent_id_str]['consumption']['Coin']
            dpi = income + lump_sum - tax_paid
            
            total_dpi_year1 += dpi
            total_consumption_year1 += consumption
        
        # Calculate Year 0 (previous year) 12-month total
        total_dpi_year0 = 0
        total_consumption_year0 = 0
        
        for t in range(year_start - 12, year_start):
            income = states[t][agent_id_str]['income']['Coin']
            lump_sum = periodic_tax[t].get(agent_id_str, {}).get('lump_sum', 0)
            tax_paid = periodic_tax[t].get(agent_id_str, {}).get('tax_paid', 0)
            consumption = states[t][agent_id_str]['consumption']['Coin']
            dpi = income + lump_sum - tax_paid
            
            total_dpi_year0 += dpi
            total_consumption_year0 += consumption
        
        # Annual change
        delta_dpi = total_dpi_year1 - total_dpi_year0
        delta_consumption = total_consumption_year1 - total_consumption_year0
        
        # Only keep cases where income increases and the increase is large enough
        if delta_dpi >= MIN_DELTA_DPI:
            mpc = delta_consumption / delta_dpi
            
            current_time = world_start_time + relativedelta(months=year_start)
            year = current_time.strftime('%Y')
            
            agent_mpc_list.append({
                'Agent': agent_id,
                'Year': year,
                'Delta_DPI': delta_dpi,
                'Delta_Consumption': delta_consumption,
                'MPC': mpc
            })

df_mpc = pd.DataFrame(agent_mpc_list)

# Statistical Analysis
print(f"\nActual sample size: {len(df_mpc)}")

if len(df_mpc) > 0:
    print(f"\nBasic statistics:")
    print(f"Average MPC: {df_mpc['MPC'].mean():.4f}")
    print(f"Median MPC: {df_mpc['MPC'].median():.4f}")
    print(f"MPC range: [{df_mpc['MPC'].min():.4f}, {df_mpc['MPC'].max():.4f}]")
    print(f"Standard deviation: {df_mpc['MPC'].std():.4f}")
    
    # Reasonable range statistics
    reasonable = df_mpc[(df_mpc['MPC'] >= 0) & (df_mpc['MPC'] <= 1)]
    print(f"\nMPC in [0,1] range: {len(reasonable)}/{len(df_mpc)} ({len(reasonable)/len(df_mpc)*100:.2f}%)")
    
    # Literature range statistics
    literature_range = df_mpc[(df_mpc['MPC'] >= 0.05) & (df_mpc['MPC'] <= 0.9)]
    print(f"MPC in literature range [0.05,0.9]: {len(literature_range)}/{len(df_mpc)} ({len(literature_range)/len(df_mpc)*100:.2f}%)")
    
    negative = df_mpc[df_mpc['MPC'] < 0]
    over_one = df_mpc[df_mpc['MPC'] > 1]
    print(f"\nMPC < 0: {len(negative)} ({len(negative)/len(df_mpc)*100:.2f}%)")
    print(f"MPC > 1: {len(over_one)} ({len(over_one)/len(df_mpc)*100:.2f}%)")
    
    # Average MPC per Agent
    print("\n" + "=" * 60)
    print("📊 Average Annual MPC per Agent")
    print("=" * 60)
    
    agent_avg_mpc = df_mpc.groupby('Agent')['MPC'].agg(['mean', 'median', 'std', 'count'])
    agent_avg_mpc.columns = ['Mean_MPC', 'Median_MPC', 'Std_MPC', 'Sample_Count']
    
    print(f"\nAgent average MPC statistics:")
    print(f"Mean: {agent_avg_mpc['Mean_MPC'].mean():.4f}")
    print(f"Median: {agent_avg_mpc['Mean_MPC'].median():.4f}")
    print(f"Range: [{agent_avg_mpc['Mean_MPC'].min():.4f}, {agent_avg_mpc['Mean_MPC'].max():.4f}]")
    
    # Proportion of reasonable agents
    reasonable_agents = agent_avg_mpc[(agent_avg_mpc['Mean_MPC'] >= 0) & (agent_avg_mpc['Mean_MPC'] <= 1)]
    print(f"\nAgents with average MPC in [0,1]: {len(reasonable_agents)}/{len(agent_avg_mpc)} ({len(reasonable_agents)/len(agent_avg_mpc)*100:.2f}%)")
    
    # Agents within literature range
    literature_agents = agent_avg_mpc[(agent_avg_mpc['Mean_MPC'] >= 0.05) & (agent_avg_mpc['Mean_MPC'] <= 0.9)]
    print(f"Agents with average MPC in literature range [0.05,0.9]: {len(literature_agents)}/{len(agent_avg_mpc)} ({len(literature_agents)/len(agent_avg_mpc)*100:.2f}%)")
    
    # Comparison with Literature
    print("\n" + "=" * 60)
    print("📚 Comparison with Carroll et al. (2017)")
    print("=" * 60)
    
    print(f"\nLiterature range: [0.05, 0.9]")
    print(f"Your median: {df_mpc['MPC'].median():.4f}")
    
    if 0.05 <= df_mpc['MPC'].median() <= 0.9:
        print(f"✅ Median is within literature range")
    elif df_mpc['MPC'].median() < 0.05:
        print(f"⚠️ Median is below literature lower bound")
    else:
        print(f"⚠️ Median is above literature upper bound")
    
    # Save Results
    df_mpc.to_csv(f'{save_path}mpc_annual_non_overlapping.csv', index=False)
    agent_avg_mpc.to_csv(f'{save_path}mpc_agent_average_annual_non_overlapping.csv')
    
    print(f"\n✅ Results saved:")
    print(f"   - {save_path}mpc_annual_non_overlapping.csv")
    print(f"   - {save_path}mpc_agent_average_annual_non_overlapping.csv")
else:
    print("\n⚠️ No samples found meeting the criteria")

print("=" * 60)