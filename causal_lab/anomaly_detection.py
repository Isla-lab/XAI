"""
==============================================================
 Causal Discovery & Anomaly Detection (Tigramite)
==============================================================

Workflow:
1. Data Loading
2. Causal Model Learning (offline PCMCI)
3. Offline Coefficient Estimation
4. Online Monitoring (moving-window coefficient updates)
5. Anomaly Detection
6. Metrics Computation

Author: (your name)
==============================================================
"""

# ================== IMPORTS ==================
import os
import numpy as np
import pandas as pd
import warnings
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from scipy.stats import ConstantInputWarning
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=ConstantInputWarning)

# ================== GLOBAL CONFIG ==================
ALPHA = 0.05
TRAINING_FRAC = 0.7
PREFIX = "C:\\Users\\User\\tigramite\\tigramite\\tutorials\\causal_discovery\\"
TASK = "pepper"

# ================================================================
#                         DATA LOADING
# ================================================================
def read_data(path: str, task: str) -> pd.DataFrame:
    """
    Load CSV data, handle timestamp indexing.

    Visual:
        CSV -> DataFrame indexed by Timestamp
    """
    df = pd.read_csv(path, delimiter="," if task == "pepper" else ";")
    if task == "pepper":
        df["Timestamp"] = df["timestamp"]
        df.set_index("Timestamp", inplace=True)
        df.drop(columns=["timestamp"], inplace=True)
    else:
        df["Timestamp"] = pd.to_datetime(df[" Timestamp"].str.strip(),
                                         format="%d/%m/%Y %I:%M:%S %p")
        df.set_index("Timestamp", inplace=True)
        df.drop(columns=[" Timestamp"], inplace=True)
    return df


# ================================================================
#                     LEARN CAUSAL MODEL
# ================================================================
def learn_causal_model(normal_csv_path: str, save_path: str):
    """
    Learn the causal graph using PCMCI.
    Tau_max is automatically computed from the dominant frequency.

    Steps:
        1️⃣ Load normal data
        2️⃣ Filter top frequency components
        3️⃣ Remove near-constant variables
        4️⃣ Compute tau_max from max frequency
        5️⃣ Run PCMCI
        6️⃣ Save the model
    """
    print("Learning causal model...")

    df = read_data(normal_csv_path, TASK)
    n_train = int(TRAINING_FRAC * len(df))
    train_vals = np.nan_to_num(df.values[:n_train, :])

    # --- 1. Determine dominant frequencies (FFT) ---
    MAX_FREQ_COMPONENTS = 5
    freqs = []
    for col in range(train_vals.shape[1]):
        signal = train_vals[:, col]
        if np.std(signal) < 1e-12:
            continue
        w = np.fft.fft(signal)
        f = np.fft.fftfreq(len(w))
        mods = np.abs(w)
        main_freqs = [f[i] for i in np.argsort(mods)[::-1][:MAX_FREQ_COMPONENTS] if f[i] > 0]
        freqs.extend(main_freqs)

    # --- 2. Determine subsampling factor ---
    subsample = 1
    if freqs:
        sorted_freq = np.sort(np.array(freqs))[::-1]
        max_freq = sorted_freq[0]
        for freq in sorted_freq:
            if len([fr for fr in sorted_freq if fr < freq]) / len(sorted_freq) < 0.95:
                max_freq = freq
                break
        subsample = max(1, int(np.floor(0.1 / max_freq)))

    # --- 3. Remove near-constant variables ---
    train_sub = train_vals[::subsample, :]
    stds = np.std(train_sub, axis=0)
    means = np.mean(train_sub, axis=0)
    nonconst = np.where(stds > 0.01 * np.abs(means))[0]
    if len(nonconst) == 0:
        nonconst = np.arange(train_sub.shape[1])

    # --- 4. Compute tau_max from dominant frequency ---
    tau_max = max(1, int(np.round(1 / max_freq))) if freqs else 5
    print(f"Computed tau_max = {tau_max} from dominant frequency {max_freq if freqs else 'N/A'}")

    # TODO: --- 5. Run PCMCI ---

    # --- 6. Save model for reuse ---
    np.savez(save_path,
             val_matrix=results["val_matrix"],
             p_matrix=results["p_matrix"],
             var=df.columns,
             subsample=subsample,
             nonconst=nonconst)
    print(f"Saved causal model to {save_path}")
    return results, subsample, nonconst, tau_max


# ================================================================
#                  OFFLINE COEFFICIENT FITTING
# ================================================================
def fit_normal_coeffs(normal_data: np.ndarray, causal_matrix: np.ndarray):
    """
    Compute offline (baseline) coefficients for each variable.

    Visual:
        X_t = sum_j (a_j * X_parent_j_(t-delay_j)) + bias
        ------------------------------------------------
        offline coefficients: learned on full normal dataset
    """
    indices = np.array(np.where(causal_matrix != 0))
    fine_coeffs = {}
    for var in np.unique(indices[1, :]):
        # TODO: compute fine_coeffs[var]

    return fine_coeffs, indices


# ================================================================
#             ONLINE COEFFICIENTS & ERROR COMPUTATION
# ================================================================
def compute_online_errors(data: np.ndarray, fine_coeffs: dict,
                          causal_matrix: np.ndarray, indices: np.ndarray):
    """
    Recompute coefficients online over a moving window and compute deviations.

    Visual:
        Time t-3  t-2  t-1  [t]
              |---- moving window ----|
                      ↑ online regression
        norm_agg[t,i] = ||online_coeffs - offline_coeffs||
    """
    max_time = data.shape[0] - causal_matrix.shape[2]
    err = {}
    norm_agg = np.zeros((max_time, len(np.unique(indices[1, :]))))

    for t in range(max_time):
        for i, var in enumerate(np.unique(indices[1, :])):
            var_indices = [indices[:, k] for k in range(indices.shape[1]) if indices[1, k] == var]
            var_indices.sort(key=lambda x: x[2])
            max_delay = var_indices[-1][2]

            # Fit online coefficients up to time t
            stack = [data[max_delay - el[2]: t + causal_matrix.shape[2] - el[2], el[0]]
                     for el in var_indices]
            stack.append(np.ones(t + causal_matrix.shape[2] - max_delay))

            coeffs = np.linalg.lstsq(np.column_stack(stack),
                                     data[max_delay: t + causal_matrix.shape[2], var],
                                     rcond=None)[0][:-1]

            # Store deviation
            if var not in err:
                err[var] = np.zeros((max_time, len(var_indices)))
            err[var][t, :] = coeffs - fine_coeffs[var]
            norm_agg[t, i] = np.linalg.norm(err[var][t, :])

    return err, norm_agg


# ================================================================
#                        ANOMALY DETECTION
# ================================================================
def detect_anomalies(err_normal, err_attack, normal_data_len, normal):
    """
    Flag anomalies if online coefficients deviate significantly from offline baseline.

    Threshold = 0.8 * norm of offline deviations.
    """
    indices_error = []
    for var in err_attack.keys():
        for j in range(err_attack[var].shape[1]):
            thresh = 0.8 * np.linalg.norm(err_normal[var][:normal_data_len, j])
            if not normal:
                indices_error += list(np.where(abs(err_attack[var][:, j]) > thresh)[0])
            else:
                indices_error += list(np.where(abs(err_attack[var][normal_data_len:, j]) > thresh)[0])
    return len(np.unique(indices_error))



# ================================================================
#          FEATURE IMPORTANCE PLOTTING (SUBPLOTS)
# ================================================================
def plot_feature_importance_subplots(norm_agg_list, indices, nonconst, var_names, attack_names, top_frac=0.1):
    """
    Plot top anomalous variables per attack as barplots in subplots.
    
    Inputs:
        norm_agg_list : list of np.ndarray
            Aggregated online deviations for each attack (L2 norm over time)
        indices       : np.ndarray
            Indices of causal parents from offline coefficients
        nonconst      : list/np.ndarray
            Indices of non-constant variables
        var_names     : list
            Names of all variables
        attack_names  : list
            Names of attack datasets (for subplot titles)
        top_frac      : float
            Fraction of top variables to show
    """
    n_attacks = len(norm_agg_list)
    fig, axes = plt.subplots(1, n_attacks, figsize=(6*n_attacks, 5), squeeze=False)
    plt.suptitle("Top Anomalous Variables per Attack", fontsize=16)

    for i, norm_agg_attack in enumerate(norm_agg_list):
        # Compute aggregated L2 norm per variable
        dep_vals = {var_names[nonconst[var]]: np.linalg.norm(norm_agg_attack[:, j])
                    for j, var in enumerate(np.unique(indices[1, :]))}
        # Sort descending
        dep_sorted = dict(sorted(dep_vals.items(), key=lambda x: x[1], reverse=True))
        top_n = max(1, int(top_frac * len(dep_sorted)))
        top_items = list(dep_sorted.items())[:top_n]
        top_vars, top_vals = zip(*top_items)

        ax = axes[0, i]
        ax.bar(top_vars, top_vals, color='salmon')
        ax.set_xticklabels(top_vars, rotation=45, ha='right')
        ax.set_ylabel("Aggregated Error (L2 Norm)")
        ax.set_title(attack_names[i])

    plt.tight_layout()
    plt.show()


# ================================================================
#                         MAIN PIPELINE
# ================================================================
def main():
    print(f"\n========== TASK: {TASK.upper()} ==========")

    causal_path = os.path.join(PREFIX, f"{TASK}_normal_07.npz")

    # 1️⃣ Learn or load causal model
    if not os.path.exists(causal_path):
        learn_causal_model(PREFIX + "pepper_csv/normal.csv", causal_path)

    f = np.load(causal_path, allow_pickle=True)
    val_matrix, p_matrix = f["val_matrix"], f["p_matrix"]
    subsample, nonconst = int(f["subsample"]), f["nonconst"]

    normal_matrix = val_matrix * (p_matrix < ALPHA) * (abs(val_matrix) > np.mean(abs(val_matrix)))

    # 2️⃣ Load normal data
    normal_df = read_data(PREFIX + "pepper_csv/normal.csv", TASK)
    normal_data = np.nan_to_num(normal_df.values[:int(TRAINING_FRAC * len(normal_df))][::subsample, nonconst])
    normal_data_full = np.nan_to_num(normal_df.values[::subsample, nonconst])

    # 3️⃣ Offline coefficients
    fine_coeffs, indices = fit_normal_coeffs(normal_data, normal_matrix)

    # 4️⃣ Online deviations
    err_normal, norm_agg_normal = compute_online_errors(normal_data_full, fine_coeffs, normal_matrix, indices)

    # 5️⃣ Load and detect anomalies
    attack_paths = [
        PREFIX + "pepper_csv/WheelsControl.csv",
        PREFIX + "pepper_csv/JointControl.csv",
        PREFIX + "pepper_csv/LedsControl.csv"
    ]
    attack_dfs = [read_data(p, TASK) for p in attack_paths]

    tpos, fpos, fneg = [], [], []

    # False positives
    fpos.append(detect_anomalies(err_normal, err_normal, len(normal_data), normal=True))

    norm_agg_attacks = []
    attack_names = []

    for path, df_attack in zip(attack_paths, attack_dfs):
        attack_name = os.path.basename(path)
        attack_names.append(attack_name)
        print(f"\n--- Analyzing anomaly: {attack_name} ---")

        attack_data = np.nan_to_num(df_attack.values[::subsample, nonconst])
        err_attack, norm_agg_attack = compute_online_errors(attack_data, fine_coeffs, normal_matrix, indices)
        norm_agg_attacks.append(norm_agg_attack)

        tp_count = detect_anomalies(err_normal, err_attack, len(normal_data), normal=False)
        tpos.append(tp_count)
        fneg.append(attack_data.shape[0] - tp_count)

        # Top 10% variables with highest aggregated error
        dep_vals = {f["var"][nonconst][var]: np.linalg.norm(norm_agg_attack[:, i])
                    for i, var in enumerate(np.unique(indices[1, :]))}
        dep_vals_sorted = dict(sorted(dep_vals.items(), key=lambda x: x[1], reverse=True))
        top_n = max(1, int(0.1 * len(dep_vals_sorted)))
        # print("Top 10% anomalous variables:")
        # for i, (k, v) in enumerate(list(dep_vals_sorted.items())[:top_n]):
        #     print(f"{k:<20} | Aggregate Error: {v:.3f}")
        # print("========================================")

    # 6️⃣ Metrics
    precision = np.sum(tpos) / (np.sum(tpos) + np.sum(fpos))
    recall = np.sum(tpos) / (np.sum(tpos) + np.sum(fneg))
    f1 = 2 * np.sum(tpos) / (2 * np.sum(tpos) + np.sum(fpos) + np.sum(fneg))

    print("\n========== METRICS ==========")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1:.3f}")
    print("=============================\n")
    
    # ===== PLOT FEATURE IMPORTANCE FOR ALL ATTACKS =====
    plot_feature_importance_subplots(norm_agg_attacks, indices, nonconst, f["var"], attack_names, top_frac=0.1)


# ================================================================
#                          RUN SCRIPT
# ================================================================
if __name__ == "__main__":
    main()
