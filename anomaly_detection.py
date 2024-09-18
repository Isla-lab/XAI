import time
import pandas as pd
import numpy as np
import os

from tigramite import plotting as tp
from matplotlib import pyplot as plt

from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

import time

import warnings
from scipy.stats import ConstantInputWarning

warnings.filterwarnings('ignore', category=ConstantInputWarning)

# Constants
ALPHA = 0.05 # Significance level for ParCorr
TRAINING_FRAC = 0.7 # Fraction of the dataset to use for training



def read_preprocess_data(path):
    df = pd.read_csv(path, delimiter=",")
    df['Timestamp'] = df['timestamp']
    df.set_index('Timestamp', inplace=True)
    columns_to_drop = ['timestamp']
    df.drop(columns=[col for col in columns_to_drop if col in df.columns or col == df.columns[0]], inplace=True)
    return df

def run_pcmci(data, delay, link_assumptions=None):
    pcmci = PCMCI(dataframe=pp.DataFrame(data), cond_ind_test=ParCorr())
    results = pcmci.run_pcmci(tau_max=delay, pc_alpha=ALPHA, link_assumptions=link_assumptions)

    return results



end = 0
start = 0
has_ends = False
tpos = []
fpos = []
tneg = []
fneg = []

causal_model = "models/pepper_normal_07.npz"
f = np.load(causal_model, allow_pickle=True)
val_matrix = f["val_matrix"]
p_matrix = f["p_matrix"]
var = list(f["var"])
subsample = int(f['subsample'])
delay = np.shape(val_matrix)[2] - 1
nonconst = f["nonconst"]

normal_matrix = val_matrix * (p_matrix < ALPHA) * (abs(val_matrix) > np.mean(abs(val_matrix)))
normal_p_matrix = p_matrix * (p_matrix < ALPHA) * (abs(val_matrix) > np.mean(abs(val_matrix)))

#modify paths to dataset folders
normal_df = read_preprocess_data("data/pepper_csv/normal.csv")
attack_dfs = [read_preprocess_data("data/pepper_csv/WheelsControl.csv")]
attack_dfs.append(read_preprocess_data("data/pepper_csv/JointControl.csv"))
attack_dfs.append(read_preprocess_data("data/pepper_csv/LedsControl.csv"))



#PLOT CAUSAL GRAPH
# pcmci = PCMCI(dataframe=pp.DataFrame(np.nan_to_num(normal_df.values[:, nonconst])), cond_ind_test=ParCorr())
# # graph = pcmci.get_graph_from_pmatrix(p_matrix=normal_p_matrix, alpha_level=ALPHA, 
# #         tau_min=0, tau_max=delay, link_assumptions=None)
# normal_matrix[abs(normal_matrix) < 0.3] = 1 #remove weak links
# graph = pcmci.get_graph_from_pmatrix(p_matrix=normal_matrix, alpha_level=0.99, 
#         tau_min=0, tau_max=delay, link_assumptions=None)
# tp.plot_graph(
#     val_matrix=normal_matrix,
#     graph=graph,
#     var_names=f["var"][nonconst],
#     link_colorbar_label='cross-MCI',
#     node_colorbar_label='auto-MCI',
#     show_autodependency_lags=False,
#     arrow_linewidth=5,
#     tick_label_size=10,
#     link_label_fontsize=0
# )
# plt.show()
#PLOT CAUSAL GRAPH

#save normal coeffs
normal_data = normal_df.values[:int(TRAINING_FRAC*np.shape(normal_df.values)[0]), :]
normal_data = normal_data[::subsample, nonconst]
normal_data = np.nan_to_num(normal_data)
normal_data_full = normal_df.values
normal_data_full = normal_data_full[::subsample, nonconst]
normal_data_full = np.nan_to_num(normal_data_full)
indices = np.array(np.where(normal_matrix != 0))
fine_coeffs = dict()
for var in np.unique(indices[1,:]):
    var_indices = [indices[:,k] for k in range(np.shape(indices)[1]) if indices[1,k] == var]
    var_indices.sort(key= lambda a : a[-1])
    stack_list = []
    max_delay = var_indices[-1][2]
    for el in var_indices:
        stack_list.append(normal_data[max_delay-el[2] : np.shape(normal_data)[0]-el[2], el[0]])
    stack_list.append(np.ones(np.shape(normal_data)[0]-max_delay))
    coeffs = np.linalg.lstsq(np.column_stack(stack_list), normal_data[max_delay:, var])[0][:-1]
    fine_coeffs[var] = coeffs


#NORMAL OUAD
#compute online coeffs
err = dict()
max_time = np.shape(normal_data_full)[0] - np.shape(normal_matrix)[2]
norm_agg = np.zeros((max_time, len(np.unique(indices[1,:]))))
for j in range(0, max_time):
    for i in range(len(np.unique(indices[1,:]))):
        var = np.unique(indices[1,:])[i]
        var_indices = [indices[:,k] for k in range(np.shape(indices)[1]) if indices[1,k] == var]
        var_indices.sort(key= lambda a : a[-1])
        stack_list = []
        max_delay = var_indices[-1][2]
        for el in var_indices:
            stack_list.append(normal_data_full[max_delay-el[2] : j+np.shape(normal_matrix)[2]-el[2], el[0]])
        stack_list.append(np.ones(j+np.shape(normal_matrix)[2]-max_delay))
        coeffs = np.linalg.lstsq(np.column_stack(stack_list), normal_data_full[max_delay : j+np.shape(normal_matrix)[2], var])[0][:-1]
        if var not in err.keys():
            err[var] = np.zeros((max_time, len(var_indices)))
        err[var][j, :] = (coeffs - fine_coeffs[var])
        norm_agg[j,i] = np.linalg.norm(err[var][j, :])

indices_error = []
for i in range(np.shape(norm_agg)[1]):
    var = np.unique(indices[1,:])[i]
    var_indices = [indices[:,k] for k in range(np.shape(indices)[1]) if indices[1,k] == var]
    for j in range(np.shape(err[var])[1]):
        thresh = np.linalg.norm(err[var][:len(normal_data),j])
        indices_error += list(np.where(abs(err[var][:,j]) > thresh)[0])

fpos.append(len(np.unique(indices_error)))

#ATTACK OUAD
for q in range(len(attack_dfs)):
    print("ANOMALY ", q)
    dep_vars = dict()
    dep_vars_value = dict()
    #compute online coeffs
    attack_data = attack_dfs[q].values
    attack_data = attack_data[::subsample, nonconst]
    attack_data = np.nan_to_num(attack_data)
    max_time = np.shape(attack_data)[0] - np.shape(normal_matrix)[2]
    norm_agg = np.zeros((max_time, len(np.unique(indices[1,:]))))
    err_attack = dict()
    for j in range(0, max_time):
        for i in range(len(np.unique(indices[1,:]))):    
            var = np.unique(indices[1,:])[i]
            var_indices = [indices[:,k] for k in range(np.shape(indices)[1]) if indices[1,k] == var]
            var_indices.sort(key= lambda a : a[-1])
            max_delay = var_indices[-1][2]
            start_time = time.time()
            stack_list = []
            for el in var_indices:
                stack_list.append(attack_data[max_delay-el[2] : j+np.shape(normal_matrix)[2]-el[2], el[0]])
            stack_list.append(np.ones(j+np.shape(normal_matrix)[2]-max_delay))
            coeffs = np.linalg.lstsq(np.column_stack(stack_list), attack_data[max_delay : j+np.shape(normal_matrix)[2], var])[0][:-1]
            if var not in err_attack.keys():
                err_attack[var] = np.zeros((max_time, len(var_indices)))
            err_attack[var][j, :] = (coeffs - fine_coeffs[var])
            norm_agg[j,i] = np.linalg.norm(err_attack[var][j, :])

    start_index = 0
    end_index = -1
    main_vars = [] #print this to identify broken causal children
    indices_error = []
    for i in range(np.shape(norm_agg)[1]):
        var = np.unique(indices[1,:])[i]
        var_indices = [indices[:,k] for k in range(np.shape(indices)[1]) if indices[1,k] == var]
        main_vars.append(f["var"][nonconst][var])
        if f["var"][nonconst][var] not in dep_vars.keys():
            dep_vars[f["var"][nonconst][var]] = []
            dep_vars_value[f["var"][nonconst][var]] = []
        #raise alarm
        for j in range(np.shape(err_attack[var])[1]):
            thresh = np.linalg.norm(err[var][:len(normal_data),j])
            indices_error += list(np.where(abs(err_attack[var][int(start_index):int(end_index),j]) > thresh)[0])
            dep_vars[f["var"][nonconst][var]].append(f["var"][nonconst][var_indices[j][0]])
            dep_vars_value[f["var"][nonconst][var]].append(norm_agg[int(start_index):int(end_index),i])
        
    tpos.append(len(np.unique(indices_error)))
    fneg.append(np.shape(err_attack[var][int(start_index):int(end_index),j])[0] - tpos[-1])    


    # MOST ANOMALOUS VARIABLES
    dep_vars_value_aggregates = dict()
    for k in dep_vars_value.keys():
        if dep_vars_value[k] != []:
            dep_vars_value_aggregates[k] = np.linalg.norm(np.array(dep_vars_value[k]))
    # sort by aggregated error norm
    dep_vars_value_aggregates = dict(sorted(dep_vars_value_aggregates.items(), key=lambda item: item[1], reverse=True))

    dep_vars_keys = list(dep_vars_value_aggregates.keys())
    for var_idx in range(len(dep_vars_keys)):
        if var_idx > 0.1*len(dep_vars_keys):
            break
        if len(dep_vars[dep_vars_keys[var_idx]]) > 0:
            print("AGGREGATE ERROR ", dep_vars_value_aggregates[dep_vars_keys[var_idx]])
            print(dep_vars_keys[var_idx])
            # print("DEP VARS ", dep_vars[dep_vars_keys[var_idx]])
    print("===========================")

print("PRECISION")
print(np.sum(tpos) / (np.sum(tpos)+np.sum(fpos)))
print("RECALL")
print(np.sum(tpos) / (np.sum(tpos)+np.sum(fneg)))
print("F1")
print(2 * np.sum(tpos) / (2*np.sum(tpos)+np.sum(fneg)+np.sum(fpos)))

