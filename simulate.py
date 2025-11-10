from typing import Optional
import argparse
import fire
import os
import sys

import ai_economist.foundation as foundation
import numpy as np
import matplotlib.pyplot as plt
import yaml
from time import time
from collections import defaultdict
import re
from simulate_utils import *
import pickle as pkl
from itertools import product
from dateutil.relativedelta import relativedelta

with open('config.yaml', "r") as f:
    run_configuration = yaml.safe_load(f)
env_config = run_configuration.get('env')
        
def enforce_constraints(env, obs, actions, prev_obs):
    """
    渐进式约束强制（Gradual Constraint Enforcement）
    
    时间表：
    - Month 0-12：完全不约束（alpha=0）
    - Month 12-60：逐步收紧（alpha: 0→1）
    - Month 60+：完全约束（alpha=1）
    
    关键改进：
    1. 储蓄率下限从 0.5% 渐进到 1.4%
    2. 失业率下限从 1% 渐进到 3.5%
    3. 修正幅度按 alpha 比例调整
    """
    modified_actions = {}
    timestep = env.world.timestep
    
    # ========== 计算约束强度 alpha ==========
    if timestep < 12:
        alpha = 0.0  # 第一年：完全不约束
    elif timestep < 60:
        alpha = (timestep - 12) / 48  # 渐进期：线性增长
    else:
        alpha = 1.0  # 成熟期：完全约束
    
    # 如果不约束，直接返回
    if alpha == 0.0:
        for idx in range(env.num_agents):
            modified_actions[str(idx)] = actions[str(idx)]
        modified_actions['p'] = [0]
        return modified_actions
    
    # 统计
    stats = {
        'mpc_violations': 0,
        'mpc_fixed': 0,
        'saving_violations': 0,
        'saving_fixed': 0,
        'mpc_checks': 0,
        'saving_checks': 0
    }
    
    # ========== 个体约束修正 ==========
    for idx in range(env.num_agents):
        agent_id = str(idx)
        this_agent = env.get_agent(agent_id)
        
        work_decision = actions[agent_id][0]
        consumption_prop = actions[agent_id][1] * 0.02
        
        curr_wealth = this_agent.inventory['Coin']
        max_l = env._components_dict['SimpleLabor'].num_labor_hours
        curr_potential_income = this_agent.state['skill'] * max_l * work_decision
        
        # ========== MPC约束 ==========
        if prev_obs is not None and timestep > 0:
            try:
                prev_states = env.dense_log['states'][-1] if len(env.dense_log['states']) > 0 else None
                if prev_states is not None:
                    prev_income = prev_states[agent_id]['income']['Coin']
                    prev_consumption = prev_states[agent_id]['consumption']['Coin']
                    prev_tax_data = env.dense_log['PeriodicTax'][-1] if len(env.dense_log['PeriodicTax']) > 0 else {}
                    prev_tax = prev_tax_data.get(agent_id, {}).get('tax_paid', 0)
                    
                    curr_tax = obs['p'][f'p{idx}']['PeriodicBracketTax-tax_paid']
                    lump_sum = obs['p'][f'p{idx}']['PeriodicBracketTax-lump_sum']
                    
                    prev_dpi = prev_income - prev_tax
                    curr_dpi = curr_potential_income + lump_sum - curr_tax
                    delta_dpi = curr_dpi - prev_dpi
                    
                    planned_consumption = consumption_prop * (curr_wealth + curr_potential_income)
                    delta_consumption = planned_consumption - prev_consumption
                    
                    # 渐进式阈值：50 → 25
                    threshold = 50 * (1 - 0.5 * alpha)
                    if abs(delta_dpi) > threshold:
                        stats['mpc_checks'] += 1
                        mpc = delta_consumption / delta_dpi
                        
                        # 渐进式边界
                        lower_bound = 0.02 + 0.03 * alpha  # 0.02 → 0.05
                        upper_bound = 1.0 - 0.1 * alpha     # 1.0 → 0.9
                        
                        if mpc < lower_bound:
                            stats['mpc_violations'] += 1
                            stats['mpc_fixed'] += 1
                            # 渐进式修正：只修正 50%*alpha
                            target_mpc = lower_bound + (mpc - lower_bound) * (1 - alpha * 0.5)
                            target_consumption = prev_consumption + target_mpc * delta_dpi
                            consumption_prop = target_consumption / (curr_wealth + curr_potential_income + 1e-8)
                            consumption_prop = max(0.02, min(1.0, consumption_prop))
                            
                        elif mpc > upper_bound:
                            stats['mpc_violations'] += 1
                            stats['mpc_fixed'] += 1
                            target_mpc = upper_bound + (mpc - upper_bound) * (1 - alpha * 0.5)
                            target_consumption = prev_consumption + target_mpc * delta_dpi
                            consumption_prop = target_consumption / (curr_wealth + curr_potential_income + 1e-8)
                            consumption_prop = max(0.02, min(1.0, consumption_prop))
            except:
                pass
        
        # ========== 储蓄率约束（关键！）==========
        try:
            tax_paid = obs['p'][f'p{idx}']['PeriodicBracketTax-tax_paid']
            lump_sum = obs['p'][f'p{idx}']['PeriodicBracketTax-lump_sum']
            
            dpi = curr_potential_income + lump_sum - tax_paid
            planned_consumption = consumption_prop * (curr_wealth + curr_potential_income)
            saving = dpi - planned_consumption
            
            # 渐进式阈值：50 → 25
            threshold = 50 * (1 - 0.5 * alpha)
            if dpi > threshold:
                stats['saving_checks'] += 1
                saving_rate = saving / dpi
                
                # 🔥 关键：储蓄率下限渐进
                lower_bound = 0.005 + 0.009 * alpha  # 0.5% → 1.4%
                upper_bound = 0.318
                
                if saving_rate < lower_bound:
                    stats['saving_violations'] += 1
                    stats['saving_fixed'] += 1
                    # 渐进式修正
                    target_rate = lower_bound + (saving_rate - lower_bound) * (1 - alpha * 0.5)
                    target_consumption = dpi * (1 - target_rate)
                    consumption_prop = target_consumption / (curr_wealth + curr_potential_income + 1e-8)
                    consumption_prop = max(0.02, min(1.0, consumption_prop))
                    
                elif saving_rate > upper_bound:
                    stats['saving_violations'] += 1
                    stats['saving_fixed'] += 1
                    target_rate = upper_bound + (saving_rate - upper_bound) * (1 - alpha * 0.5)
                    target_consumption = dpi * (1 - target_rate)
                    consumption_prop = target_consumption / (curr_wealth + curr_potential_income + 1e-8)
                    consumption_prop = max(0.02, min(1.0, consumption_prop))
        except:
            pass
        
        modified_actions[agent_id] = [work_decision, int(consumption_prop / 0.02)]
    
    # ========== 失业率约束（关键！）==========
    unemployed = sum(1 for idx in range(env.num_agents) 
                     if modified_actions[str(idx)][0] == 0)
    unemployment_rate = unemployed / env.num_agents
    
    unemployment_adjusted = False
    
    # 🔥 关键：失业率下限渐进
    lower_bound = 0.01 + 0.025 * alpha   # 1% → 3.5%
    upper_bound = 0.148
    
    if unemployment_rate < lower_bound:
        unemployment_adjusted = True
        target_unemployed = int(lower_bound * env.num_agents)
        # 渐进式调整：只调整 50%*alpha
        adjust_count = max(1, int((target_unemployed - unemployed) * alpha * 0.5))
        employed_agents = [str(idx) for idx in range(env.num_agents) 
                          if modified_actions[str(idx)][0] == 1]
        if len(employed_agents) >= adjust_count:
            forced_unemployed = np.random.choice(employed_agents, 
                                                size=adjust_count, 
                                                replace=False)
            for agent_id in forced_unemployed:
                modified_actions[agent_id][0] = 0
                
    elif unemployment_rate > upper_bound:
        unemployment_adjusted = True
        target_unemployed = int(upper_bound * env.num_agents)
        adjust_count = max(1, int((unemployed - target_unemployed) * alpha * 0.5))
        unemployed_agents = [str(idx) for idx in range(env.num_agents) 
                            if modified_actions[str(idx)][0] == 0]
        if len(unemployed_agents) >= adjust_count:
            forced_employed = np.random.choice(unemployed_agents, 
                                              size=adjust_count, 
                                              replace=False)
            for agent_id in forced_employed:
                modified_actions[agent_id][0] = 1
    
    # 调试输出
    if timestep % 20 == 0 and timestep > 0:
        print(f"\n{'='*60}")
        print(f"[Gradual Constraint @ Step {timestep}]")
        print(f"{'='*60}")
        print(f"Constraint Strength (alpha): {alpha:.2%}")
        print(f"  - Saving lower: {(0.005 + 0.009 * alpha):.2%} (target: 1.4%)")
        print(f"  - Unemployment lower: {(0.01 + 0.025 * alpha):.2%} (target: 3.5%)")
        print(f"\nMPC: checks={stats['mpc_checks']}, fixed={stats['mpc_fixed']}")
        print(f"Saving: checks={stats['saving_checks']}, fixed={stats['saving_fixed']}")
        print(f"Unemployment: {unemployment_rate:.2%}, adjusted={'Yes' if unemployment_adjusted else 'No'}")
        print(f"{'='*60}\n")
    
    modified_actions['p'] = [0]
    return modified_actions

def gpt_actions(env, obs, dialog_queue, dialog4ref_queue, gpt_path, gpt_error, total_cost, model_type='gpt'):
    if not os.path.exists(gpt_path):
        os.makedirs(gpt_path)
    curr_rates = obs['p']['PeriodicBracketTax-curr_rates']
    current_time = world_start_time + relativedelta(months=env.world.timestep)
    current_time = current_time.strftime('%Y.%m')
    for idx in range(env.num_agents):
        this_agent = env.get_agent(str(idx))
        skill = this_agent.state['skill']
        wealth = this_agent.inventory['Coin']
        consumption = this_agent.consumption['Coin']
        interest_rate = env.world.interest_rate[-1]
        price = env.world.price[-1]
        tax_paid = obs['p'][f'p{idx}']['PeriodicBracketTax-tax_paid']
        lump_sum = obs['p'][f'p{idx}']['PeriodicBracketTax-lump_sum']
        max_l = env._components_dict['SimpleLabor'].num_labor_hours
        name = this_agent.endogenous['name']
        age = this_agent.endogenous['age']
        city = this_agent.endogenous['city']
        job = this_agent.endogenous['job']
        offer = this_agent.endogenous['offer']
        actions = env.dense_log['actions']
        states = env.dense_log['states']
        problem_prompt = f'''
                    You're {name}, a {age}-year-old individual living in {city}. As with all Americans, a portion of your monthly income is taxed by the federal government. This taxation system is tiered, income is taxed cumulatively within defined brackets, combined with a redistributive policy: after collection, the government evenly redistributes the tax revenue back to all citizens, irrespective of their earnings.
                    Now it's {current_time}.
                '''
        if job == 'Unemployment':
            job_prompt = f'''
                        In the previous month, you became unemployed and had no income. Now, you are invited to work as a(an) {offer} with monthly salary of ${skill*max_l:.2f}.
                    '''
        else:
            if skill >= states[-1][str(idx)]['skill']:
                job_prompt = f'''
                            In the previous month, you worked as a(an) {job}. If you continue working this month, your expected income will be ${skill*max_l:.2f}, which is increased compared to the last month due to the inflation of labor market.
                        '''
            else:
                job_prompt = f'''
                            In the previous month, you worked as a(an) {job}. If you continue working this month, your expected income will be ${skill*max_l:.2f}, which is decreased compared to the last month due to the deflation of labor market.
                        '''
        if (consumption <= 0) and (len(actions) > 0) and (actions[-1].get('SimpleConsumption', 0) > 0):
            consumption_prompt = f'''
                        Besides, you had no consumption due to shortage of goods.
                    '''
        else:
            consumption_prompt = f'''
                        Besides, your consumption was ${consumption:.2f}.
                    '''
        if env._components_dict['PeriodicBracketTax'].tax_model == 'us-federal-single-filer-2018-scaled':
            tax_prompt = f'''Your tax deduction amounted to ${tax_paid:.2f}. However, as part of the government's redistribution program, you received a credit of ${lump_sum:.2f}.
                            In this month, the government sets the brackets: {format_numbers(brackets)} and their corresponding rates: {format_numbers(curr_rates)}. Income earned within each bracket is taxed only at that bracket's rate.'''
        else:
            tax_prompt = f'''Your tax deduction amounted to ${tax_paid:.2f}. However, as part of the government's redistribution program, you received a credit of ${lump_sum:.2f}.
                            In this month, according to the optimal taxation theory, Saez Tax, the brackets are not changed: {format_numbers(brackets)} but the government has updated corresponding rates: {format_percentages(curr_rates)}. Income earned within each bracket is taxed only at that bracket's rate.'''
        if env.world.timestep == 0:
            price_prompt = f'''Meanwhile, in the consumption market, the average price of essential goods is now at ${price:.2f}.'''
        else:
            if price >= env.world.price[-2]:
                price_prompt = f'''Meanwhile, inflation has led to a price increase in the consumption market, with the average price of essential goods now at ${price:.2f}.'''
            else:
                price_prompt = f'''Meanwhile, deflation has led to a price decrease in the consumption market, with the average price of essential goods now at ${price:.2f}.'''
        job_prompt = prettify_document(job_prompt)
        obs_prompt = f'''
                        {problem_prompt} {job_prompt} {consumption_prompt} {tax_prompt} {price_prompt}
                        Your current savings account balance is ${wealth:.2f}. Interest rates, as set by your bank, stand at {interest_rate*100:.2f}%. 
                        With all these factors in play, and considering aspects like your living costs, any future aspirations, and the broader economic trends, how is your willingness to work this month? Furthermore, how would you plan your expenditures on essential goods, keeping in mind good price?
                        Please share your decisions in a JSON format. The format should have two keys: 'work' (a value between 0 and 1 with intervals of 0.02, indicating the willingness or propensity to work) and 'consumption' (a value between 0 and 1 with intervals of 0.02, indicating the proportion of all your savings and income you intend to spend on essential goods).
                    '''
        obs_prompt = prettify_document(obs_prompt)
        dialog_queue[idx].append({'role': 'user', 'content': obs_prompt})
        dialog4ref_queue[idx].append({'role': 'user', 'content': obs_prompt})
        
    def action_check(actions):
        if len(actions) != 2:
            return False
        else:
            return (actions[0] >= 0) & (actions[0] <= 1) & (actions[1] >= 0) & (actions[1] <= 1)
    if env.world.timestep%3 == 0 and env.world.timestep > 0:
        results, cost = get_multiple_completion([list(dialogs)[:2] + list(dialog4ref)[-3:-1] + list(dialogs)[-1:] for dialogs, dialog4ref in zip(dialog_queue, dialog4ref_queue)], model_type=model_type)
        total_cost += cost
    else:
        results, cost = get_multiple_completion([list(dialogs) for dialogs in dialog_queue], model_type=model_type)
        total_cost += cost
    actions = {}
    for idx in range(env.num_agents):
        content = results[idx]
        try:
            extracted_actions = list(eval(content).values())
            if not action_check(extracted_actions):
                extracted_actions = [1, 0.5]
                gpt_error += 1
        except:
            extracted_actions = [1, 0.5]
            gpt_error += 1
        extracted_actions[0] = int(np.random.uniform() <= extracted_actions[0])
        extracted_actions[1] /= 0.02
        actions[str(idx)] = extracted_actions
        dialog_queue[idx].append({'role': 'assistant', 'content': f'{content}'})
        dialog4ref_queue[idx].append({'role': 'assistant', 'content': f'{content}'})
    actions['p'] = [0]
    for idx, agent_dialog in enumerate(dialog_queue):
        with open(f'''{gpt_path}/{env.get_agent(str(idx)).endogenous['name']}''', 'a') as f:
            for dialog in list(agent_dialog)[-2:]:
                f.write(f'''>>>>>>>>>{dialog['role']}: {dialog['content']}\n''')
        
    if (env.world.timestep+1)%3 == 0:
        reflection_prompt = '''Given the previous quarter's economic environment, reflect on the labor, consumption, and financial markets, as well as their dynamics. What conclusions have you drawn?
        Your answer must be less than 200 words!'''
        reflection_prompt = prettify_document(reflection_prompt)
        for idx in range(env.num_agents):
            # dialog_queue[idx].append({'role': 'user', 'content': reflection_prompt})
            dialog4ref_queue[idx].append({'role': 'user', 'content': reflection_prompt})
        results, cost = get_multiple_completion([list(dialogs) for dialogs in dialog4ref_queue], temperature=0, max_tokens=200, model_type=model_type)
        total_cost += cost
        for idx in range(env.num_agents):
            content = results[idx]
            # dialog_queue[idx].append({'role': 'assistant', 'content': content})
            dialog4ref_queue[idx].append({'role': 'assistant', 'content': content})
        
        for idx, agent_dialog in enumerate(dialog4ref_queue):
             with open(f'''{gpt_path}/{env.get_agent(str(idx)).endogenous['name']}''', 'a') as f:
                for dialog in list(agent_dialog)[-2:]:
                    f.write(f'''>>>>>>>>>{dialog['role']}: {dialog['content']}\n''')
    return actions, gpt_error, total_cost

def complex_actions(env, obs, beta=0.1, gamma=0.1, h=1):

    def consumption_len(price, wealth, curr_income, last_income, interest_rate):
        c = (price/(1e-8+wealth+curr_income))**beta
        c = min(max(c//0.02, 0), 50)
        return c
    def consumption_cats(price, wealth, curr_income, last_income, interest_rate):
        h1 = h / (1 + interest_rate)
        g = curr_income/(last_income+1e-8) - 1
        d = wealth/(last_income+1e-8) - h1
        c = 1 + (d - h1*g)/(1 + g + 1e-8)
        c = min(max(c*curr_income/(wealth+curr_income+1e-8)//0.02, 0), 50)
        return c
    def work_income_wealth(price, wealth, curr_income, last_income, expected_income, interest_rate):
        return int(np.random.uniform() < (curr_income/(wealth*(1 + interest_rate)+1e-8))**gamma)
    
    consumption_funs = [consumption_len, consumption_cats]
    work_funs = [work_income_wealth]

    actions = {}
    for idx in range(env.num_agents):
        this_agent = env.get_agent(str(idx))
        price = env.world.price[-1]
        wealth = this_agent.inventory['Coin']
        max_l = env._components_dict['SimpleLabor'].num_labor_hours
        max_income = max_l * this_agent.state['skill']
        last_income = this_agent.income['Coin']
        expected_income = max_l * this_agent.state['expected skill']
        interest_rate = env.world.interest_rate[-1]
        if 'consumption_fun_idx' not in this_agent.endogenous:
            this_agent.endogenous['consumption_fun_idx'] = np.random.choice(range(len(consumption_funs)))
        if 'work_fun_idx' not in this_agent.endogenous:
            this_agent.endogenous['work_fun_idx'] = np.random.choice(range(len(work_funs)))
        work_fun = work_funs[this_agent.endogenous['work_fun_idx']]
        l = work_fun(price, wealth, max_income, last_income, expected_income, interest_rate)
        curr_income = l * max_income
        consumption_fun = consumption_funs[this_agent.endogenous['consumption_fun_idx']]
        c = consumption_fun(price, wealth, curr_income, last_income, interest_rate)
        actions[str(idx)] = [l, c]
    actions['p'] = [0]
    return actions
    

def main(policy_model='gpt', num_agents=100, episode_length=240, enforce_micro_constraints=False, dialog_len=3, beta=0.1, gamma=0.1, h=1, max_price_inflation=0.1, max_wage_inflation=0.05):
    env_config['n_agents'] = num_agents
    env_config['episode_length'] = episode_length
    if policy_model == 'gpt':
        total_cost = 0
        env_config['flatten_masks'] = False
        env_config['flatten_observations'] = False
        env_config['components'][0]['SimpleLabor']['scale_obs'] = False
        env_config['components'][1]['PeriodicBracketTax']['scale_obs'] = False
        env_config['components'][3]['SimpleSaving']['scale_obs'] = False
        env_config['components'][2]['SimpleConsumption']['max_price_inflation'] = max_price_inflation
        env_config['components'][2]['SimpleConsumption']['max_wage_inflation'] = max_wage_inflation
        
        gpt_error = 0
        from collections import deque
        dialog_queue = [deque(maxlen=dialog_len) for _ in range(env_config['n_agents'])]
        dialog4ref_queue = [deque(maxlen=7) for _ in range(env_config['n_agents'])]

    elif policy_model == 'complex':
        env_config['components'][2]['SimpleConsumption']['max_price_inflation'] = max_price_inflation
        env_config['components'][2]['SimpleConsumption']['max_wage_inflation'] = max_wage_inflation

    t = time()
    env = foundation.make_env_instance(**env_config)
    obs = env.reset()
    actions = {}
    if policy_model == 'complex':
        policy_model_save = f'{policy_model}-{beta}-{gamma}-{h}-{max_price_inflation}-{max_wage_inflation}'
    if policy_model == 'gpt':
        policy_model_save = f'{policy_model}-{dialog_len}-noperception-reflection-1'
        if enforce_micro_constraints:
            policy_model_save += '-constrained'
    policy_model_save = f'{policy_model_save}-{num_agents}agents-{episode_length}months'
    if not os.path.exists(f'{save_path}data/{policy_model_save}'):
        os.makedirs(f'{save_path}data/{policy_model_save}')
    if not os.path.exists(f'{save_path}figs/{policy_model_save}'):
        os.makedirs(f'{save_path}figs/{policy_model_save}')
    prev_obs = None
    for epi in range(env.episode_length):
        if policy_model == 'gpt':
            actions, gpt_error, total_cost = gpt_actions(env, obs, dialog_queue, dialog4ref_queue, f'{save_path}data/{policy_model_save}/dialogs', gpt_error, total_cost, model_type='qwen')
            if enforce_micro_constraints:
                actions = enforce_constraints(env, obs, actions, prev_obs)
            prev_obs = obs
        elif policy_model == 'complex':
            actions = complex_actions(env, obs, beta=beta, gamma=gamma, h=h)
            if enforce_micro_constraints:
                actions = enforce_constraints(env, obs, actions, prev_obs)
            prev_obs = obs

        obs, rew, done, info = env.step(actions)
        if (epi+1) % 3 == 0:
            print(f'step {epi+1} done, cost {time()-t:.1f}s')
            if policy_model == 'gpt':
                print(f'#errors: {gpt_error}, cost ${total_cost:.1f} so far')
            t = time()
        if (epi+1) % 6 == 0 or epi+1 == env.episode_length:
            with open(f'{save_path}data/{policy_model_save}/actions_{epi+1}.pkl', 'wb') as f:
                pkl.dump(actions, f)
            with open(f'{save_path}data/{policy_model_save}/obs_{epi+1}.pkl', 'wb') as f:
                pkl.dump(obs, f)
            with open(f'{save_path}data/{policy_model_save}/env_{epi+1}.pkl', 'wb') as f:
                pkl.dump(env, f)
            if policy_model == 'gpt':
                with open(f'{save_path}data/{policy_model_save}/dialog_{epi+1}.pkl', 'wb') as f:
                    pkl.dump(dialog_queue, f)
                with open(f'{save_path}data/{policy_model_save}/dialog4ref_{epi+1}.pkl', 'wb') as f:
                    pkl.dump(dialog4ref_queue, f)
            with open(f'{save_path}data/{policy_model_save}/dense_log_{epi+1}.pkl', 'wb') as f:
                pkl.dump(env.dense_log, f)
                
    with open(f'{save_path}data/{policy_model_save}/dense_log.pkl', 'wb') as f:
        pkl.dump(env.dense_log, f)
        
    if policy_model == 'gpt':
        print(f'#gpt errors: {gpt_error}')

if __name__ == "__main__":
    fire.Fire(main)
