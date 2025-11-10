# -*- coding: utf-8 -*-
"""
手动计算名义GDP：S × P
使用论文公式从原始数据计算
"""

import os
import pickle as pkl
import csv

BASE = r"/workspace/ACL24-EconAgent/data/gpt-3-noperception-reflection-1-constrained-100agents-240months"
DATA = BASE  # log文件就在BASE目录下
OUT = os.path.join(DATA, "result_analysis")
os.makedirs(OUT, exist_ok=True)


# ============================================================================
# DummyUnpickler
# ============================================================================
class DummyUnpickler(pkl.Unpickler):
    def find_class(self, module, name):
        if 'ai_economist' in module:
            return type(name, (), {})
        return super().find_class(module, name)

# ============================================================================
# 核心计算函数
# ============================================================================
def get_prices():
    """从 env.world.price 获取价格历史"""
    env_file = os.path.join(DATA, "env_240.pkl")
    
    with open(env_file, "rb") as f:
        env = DummyUnpickler(f).load()
    
    prices = list(env.world.price)
    print(f"✓ 读取价格: {len(prices)} 个月")
    return prices

def compute_supply():
    """
    根据论文 Eq.(4): S = Σ(l_j × 168 × A)
    从 actions 计算每月供给
    """
    dense_file = os.path.join(DATA, "dense_log.pkl")
    
    with open(dense_file, "rb") as f:
        dense = pkl.load(f)
    
    actions = dense.get('actions', [])
    monthly_supply = []
    
    A = 1  # 生产率
    num_labor_hours = 168  # 月工作小时数
    
    for m in range(len(actions)):
        month_actions = actions[m] or {}
        total_supply = 0
        
        for agent_id, action in month_actions.items():
            if agent_id == 'p':  # 跳过 planner
                continue
            
            # 提取工作决策 l_j (0 或 1)
            if isinstance(action, (list, tuple)) and len(action) >= 1:
                labor = int(action[0])
            elif isinstance(action, dict):
                labor = int(action.get('SimpleLabor', 0))
            else:
                labor = 0
            
            # S = Σ(l_j × 168 × A)
            total_supply += labor * num_labor_hours * A
        
        monthly_supply.append(total_supply)
    
    print(f"✓ 计算供给: {len(monthly_supply)} 个月")
    return monthly_supply

def calculate_gdp():
    """计算月度和年度GDP"""
    print("\n" + "=" * 50)
    print("手动计算名义GDP (方法: S × P)")
    print("=" * 50 + "\n")
    
    # 获取价格和供给
    prices = get_prices()
    supply = compute_supply()
    
    # 对齐长度
    min_len = min(len(supply), len(prices))
    supply = supply[:min_len]
    prices = prices[:min_len]
    
    # 计算月度 GDP = S × P
    monthly_gdp = [s * p for s, p in zip(supply, prices)]
    print(f"✓ 计算月度GDP: {len(monthly_gdp)} 个月")
    
    # 汇总年度 GDP
    yearly_gdp = []
    years = []
    
    for y in range(0, len(monthly_gdp), 12):
        chunk = monthly_gdp[y:y+12]
        if len(chunk) == 12:
            years.append(y // 12 + 1)
            yearly_gdp.append(sum(chunk))
    
    print(f"✓ 汇总年度GDP: {len(yearly_gdp)} 年\n")
    
    return yearly_gdp, years

def calculate_growth(yearly_gdp):
    """计算GDP增长率"""
    if len(yearly_gdp) < 2:
        return []
    return [(yearly_gdp[i] - yearly_gdp[i-1]) / yearly_gdp[i-1] * 100 
            for i in range(1, len(yearly_gdp))]

def save_csv(yearly_gdp, years):
    """保存GDP和增长率到CSV"""
    growth_rates = calculate_growth(yearly_gdp)
    
    # CSV 1: 完整数据
    csv_path = os.path.join(OUT, "nominal_gdp_manual_yearly.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["year", "nominal_gdp", "growth_rate"])
        for i, (y, gdp) in enumerate(zip(years, yearly_gdp)):
            if i == 0:
                w.writerow([y, f"{gdp:.2f}", "N/A"])
            else:
                w.writerow([y, f"{gdp:.2f}", f"{growth_rates[i-1]:.2f}"])
    print(f"✓ 保存: {csv_path}")
    
    # CSV 2: 增长率
    if growth_rates:
        growth_csv = os.path.join(OUT, "nominal_gdp_manual_growth_yearly.csv")
        with open(growth_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["year", "growth_rate"])
            for y, rate in zip(years[1:], growth_rates):
                w.writerow([y, f"{rate:.2f}"])
        print(f"✓ 保存: {growth_csv}")
    
    # 打印统计
    print("\n" + "=" * 50)
    print("统计摘要")
    print("=" * 50)
    print(f"年份数量: {len(yearly_gdp)}")
    print(f"平均GDP: {sum(yearly_gdp)/len(yearly_gdp):,.2f}")
    print(f"最大GDP: {max(yearly_gdp):,.2f} (第{yearly_gdp.index(max(yearly_gdp))+1}年)")
    print(f"最小GDP: {min(yearly_gdp):,.2f} (第{yearly_gdp.index(min(yearly_gdp))+1}年)")
    
    if growth_rates:
        avg_growth = sum(growth_rates)/len(growth_rates)
        print(f"\n平均增长率: {avg_growth:.2f}%")
        print(f"最高增长: {max(growth_rates):.2f}% (第{growth_rates.index(max(growth_rates))+2}年)")
        print(f"最低增长: {min(growth_rates):.2f}% (第{growth_rates.index(min(growth_rates))+2}年)")
        print(f"正增长年份: {sum(1 for g in growth_rates if g > 0)}/{len(growth_rates)}")

# ============================================================================
# 主程序
# ============================================================================
def main():
    print("\n" + "=" * 50)
    print("名义GDP手动计算程序")
    print("公式: Nominal GDP = Σ(S × P)")
    print("=" * 50)
    
    try:
        yearly_gdp, years = calculate_gdp()
        save_csv(yearly_gdp, years)
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 50)
    print("完成!")
    print("=" * 50)

if __name__ == "__main__":
    main()