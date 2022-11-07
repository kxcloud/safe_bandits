import importlib

import pandas as pd

import experiments.algorithms_2022_07_12 as algorithm_settings

experiment_list = [
    'dosage_bandit_zero_correlation', # Desktop
    'dosage_bandit_negative_correlation',  # Work laptop
    'dosage_bandit_positive_correlation',  # Work laptop
    'high_dim_contextual_5', # Work laptop
    'high_dim_contextual_10', # Work laptop
    'high_dim_contextual_15', # School laptop
    'power_checker_5', # School laptop
    'power_checker_10', # School laptop
    'power_checker_15', # School laptop
    'uniform_bandit', # Surface
    'all_safe', # Surface
    'polynomial_bandit', # Surface
]


def action_space_to_latex(action_space):
    if action_space[0] == 0 and action_space[-1] == 1:
        return "$[0,1]$"
    
    if len(bandit.action_space) <= 3:
        return "$\{" + ",".join([str(a) for a in bandit.action_space])+"\}$"
    
    return "$\\{"+str(bandit.action_space[0])+","+str(bandit.action_space[1])+",\dots,"+str(bandit.action_space[-1])+"\}$"

def wrap_in_table(tabular_str):
    return (
        "\\begin{table}[H]\n\centering\n"+
        tabular_str +
        "\\end{table}\n"
    )
    

# df = pd.DataFrame()
for experiment_name in experiment_list:
    experiment_module = "experiments."+experiment_name
    experiment_settings = importlib.import_module(experiment_module)
    
    tau = experiment_settings.safety_tol
    
    bandit = experiment_settings.bandit_constructor()
    bandit.reset(1,1)
    
    d = bandit.phi_XA.shape[-1]
    
    x_sample = bandit.x_dist()
    if hasattr(x_sample, "__len__"):
        p = len(x_sample)
        X = "Normal$(\\bm{0}_"+str(p)+", I_"+str(p)+")$"
    elif x_sample == 0:
        X = "-"
    else:
        X = "Uniform$([0,1])$"
    
    summary = {
        "$X$": X,
        "$\\A$": action_space_to_latex(bandit.action_space), 
        "$\\tau$" : tau, 
        "$d$" : int(d)
    }
    df = pd.DataFrame(summary, index=[0])
    # print(df)
    
    df_str = df.to_latex(escape=False, index=False, column_format="c"*len(df.columns))
    df_strs = df_str.split("\n")
    df_strs.remove("\\toprule")
    df_strs.remove("\\bottomrule")
    df_str = wrap_in_table("\n".join(df_strs))
    
    print()
    print()
    print(experiment_name)
    print()
    print(df_str)
    
# print(df)
# print()
