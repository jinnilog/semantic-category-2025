
"""
Author: Ye Jin Park <yejinn@snu.ac.kr>
Seoul National University
Human Brain Function Laboratory 

ECoG_Decoding : Semantic category [Left hemisphere]

"""

#%% Imports and Parameters
import os
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from itertools import permutations
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings("ignore")

# Paths and constants
base_path = 'G:/export'
output_dir = 'C:/Users/yejin/OneDrive/1st_result'
talairach_path = 'G:/Talairach/Talairach.xlsx'
os.makedirs(output_dir, exist_ok=True)

subjects = ['220112_NBM', '220209_CJS', '220224_PJW', '220310_JAH', '220503_HSH2',
            '220707_JHJ', '220803_SJH', '221018_JSH', '230328_KHS', '230728_LSJ_left']
subject_ba_mapping = {
    '220112_NBM': 'NBM', '220209_CJS': 'CJS', '220224_PJW': 'PJW', '220310_JAH': 'JAH',
    '220503_HSH2': 'HSH2', '220707_JHJ': 'JHJ', '220803_SJH': 'SJH', '221018_JSH': 'JSH',
    '230328_KHS': 'KHS', '230728_LSJ_left': 'LSJ_L'
}

f_low, f_high = 70, 170
win_size, step_size = 10, 5
usable_timepoints = 55
window_starts = list(range(0, usable_timepoints - win_size + 1, step_size))
window_centers = [(start + win_size // 2) * 10 for start in window_starts]
n_folds = 5
n_permutations = 1000

# Load BA mapping
ba_to_lobe = {
    'Left-AntPFC (10)': 'Frontal', 'Left-dlPFC(dorsal) (9)': 'Frontal', 'Left-dlPFC(lat) (46)': 'Frontal',
    'Left-OrbFrontal (11)': 'Frontal', 'Left-ParsOrbitalis (47)': 'Frontal', 'Left-Broca-Operc (44)': 'Frontal',
    'Left-Broca-Triang (45)': 'Frontal', 'Left-PreMot+SuppMot (6)': 'Frontal', 'Left-PrimMotor (4)': 'Frontal',
    'Left-InfTempGyrus (20)': 'Temporal', 'Left-MedTempGyrus (21)': 'Temporal', 'Left-SupTempGyrus (22)': 'Temporal',
    'Left-TemporalPole (38)': 'Temporal', 'Left-PrimAuditory (41)': 'Temporal', 'Left-Fusiform (37)': 'Occipital',
    'Left-AngGyrus (39)': 'Parietal', 'Left-SupramargGyr (40)': 'Parietal', 'Left-PrimSensory (1)': 'Parietal',
    'Left-SecVisual (18)': 'Occipital', 'Left-VisAssoc (19)': 'Occipital', 'Left-VisMotor (7)': 'Occipital'
}

#%% Step 1–3: Load data and store trials for feature selection and decoding separately
print("Step 1–3 : loading data and splitting into selection and decoding sets…")

talairach_df = pd.read_excel(talairach_path)

channel_data_train = defaultdict(lambda: defaultdict(list))  
channel_data_test  = defaultdict(lambda: defaultdict(list))  
selected_channels = defaultdict(list)                      
selected_channels_by_subj = defaultdict(list)           
channel_owner = dict()                                   

rng, train_ratio = np.random.RandomState(42), 0.7

# 1) Load and split data
for subj in subjects:
    try:
        subj_short = subject_ba_mapping[subj]

        body_data    = sio.loadmat(os.path.join(base_path, subj, f"A0_body_{subj}.mat"))['Con_TFData_cut']
        nonbody_data = sio.loadmat(os.path.join(base_path, subj, f"A1_nonbody_{subj}.mat"))['Con_TFData_cut']
        chan_list    = sio.loadmat(os.path.join(base_path, subj, "ChanList.mat"))['ChanList'][0]

        body_hg    = body_data[:,  f_low:f_high, :55, :].mean(axis=1)
        nonbody_hg = nonbody_data[:, f_low:f_high, :55, :].mean(axis=1)

        for idx in range(body_hg.shape[0]):
            real_ch = int(chan_list[idx])

            ba_row = talairach_df[(talairach_df['Subject'] == subj_short) &
                                   (talairach_df['Channel'] == real_ch)]
            if ba_row.empty:
                continue
            ba = ba_row['BA'].values[0]

            channel_owner[(ba, real_ch)] = subj

            trials_ch = []
            for tr in range(body_hg.shape[-1]):
                x = body_hg[idx, :, tr]
                if x.shape[0]==55 and not np.any(np.isnan(x)) and not np.all(x==0):
                    trials_ch.append( (x,1) )
            for tr in range(nonbody_hg.shape[-1]):
                x = nonbody_hg[idx, :, tr]
                if x.shape[0]==55 and not np.any(np.isnan(x)) and not np.all(x==0):
                    trials_ch.append( (x,0) )

            # Random split into train/test
            if len(trials_ch) < 20:
                continue
            idx_perm = rng.permutation(len(trials_ch))
            split = int(len(trials_ch)*train_ratio)
            train_idx = idx_perm[:split]
            test_idx  = idx_perm[split:]

            for i in train_idx:
                channel_data_train[ba][real_ch].append(trials_ch[i])
            for i in test_idx:
                channel_data_test[ba][real_ch].append(trials_ch[i])

    except Exception as e:
        print(f" Error loading {subj}: {e}")

# 2) Feature selection using TRAIN set
for ba, chan_dict in channel_data_train.items():
    for ch, trials in chan_dict.items():
        if len(trials) < 20:
            continue

        X_trials, y_trials = zip(*trials)
        X = np.stack(X_trials)
        y = np.array(y_trials)

        accs = []
        for start in window_starts:
            X_win = X[:, start:start+win_size]
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            fold = []
            for tr, te in cv.split(X_win, y):
                mdl = make_pipeline(StandardScaler(), SVC(kernel='linear'))
                mdl.fit(X_win[tr], y[tr])
                fold.append(mdl.score(X_win[te], y[te]))
            accs.append(np.mean(fold))

        if np.mean(accs) > 0.5:
            selected_channels[ba].append(ch)
            subj_folder = channel_owner[(ba,ch)]
            selected_channels_by_subj[ba].append((subj_folder,ch))

selected_channels_by_subject = selected_channels_by_subj

# For classifier comparison
channel_data = defaultdict(lambda: defaultdict(list))
for ba in set(channel_data_train.keys()).union(channel_data_test.keys()):
    for ch in set(channel_data_train[ba].keys()).union(channel_data_test[ba].keys()):
        channel_data[ba][ch] = channel_data_train[ba][ch] + channel_data_test[ba][ch]

print("✔ Step 1–3 complete: channels selected using 70% train split.")

#%% Step 4: Grouped Multi-channel Decoding
print("Step 4: Computing decoding accuracy")

results = defaultdict(list)
all_result_rows = []

for ba in selected_channels:
    X_all, y_all = [], []
    for ch in selected_channels[ba]:
        trials = channel_data_test[ba][ch] 
        for x, y in trials:
            X_all.append(x)
            y_all.append(y)

    if len(X_all) < 10:
        print(f"Skipping BA {ba} (not enough held-out trials).")
        continue

    X_all = np.stack(X_all)
    y_all = np.array(y_all)

    for start in window_starts:
        end = start+win_size
        X_win = X_all[:, start:end]
        X_concat = X_win.reshape(X_win.shape[0], -1)

        accs = []
        cv = StratifiedKFold(n_splits=min(n_folds, len(y_all)), shuffle=True, random_state=42)
        for train_idx, test_idx in cv.split(X_concat, y_all):
            model = make_pipeline(StandardScaler(), SVC(kernel='linear'))
            model.fit(X_concat[train_idx], y_all[train_idx])
            accs.append(model.score(X_concat[test_idx], y_all[test_idx]))

        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        results[ba].append((mean_acc, std_acc))

        ste = std_acc / np.sqrt(n_folds)
        t_center = (start + win_size //2)*10
        all_result_rows.append([ba, t_center, mean_acc, ste])

# Save results
df_all = pd.DataFrame(all_result_rows, columns=["BA","Time(ms)","Accuracy","STE"])
df_all.to_csv(os.path.join(output_dir, "AllBAs_MultiChannelGroupedDecoding.csv"), index=False)


# --- (B) Plot decoding curves per lobe ---
# Remove (number) suffix from BA names
ba_clean_map = {ba: ba.split(" (")[0] for ba in selected_channels}
ba_to_lobe_clean = {ba_clean_map[ba]: ba_to_lobe[ba] for ba in selected_channels if ba in ba_to_lobe}

# Load decoding results
df_all = pd.read_csv(os.path.join(output_dir, "AllBAs_MultiChannelGroupedDecoding.csv"))
df_all['BA_clean'] = df_all['BA'].apply(lambda x: x.split(" (")[0])
df_all['Lobe'] = df_all['BA_clean'].map(ba_to_lobe_clean)

# Plot decoding accuracy curves per lobe
for lobe in sorted(df_all['Lobe'].dropna().unique()):
    plt.figure(figsize=(10, 5))
    for ba in df_all[df_all['Lobe'] == lobe]['BA_clean'].unique():
        sub_df = df_all[(df_all['BA_clean'] == ba) & (df_all['Lobe'] == lobe)]
        plot_df = sub_df[sub_df['Time(ms)'] >= 50]
        plt.plot(plot_df['Time(ms)'], plot_df['Accuracy'], label=ba)
        plt.fill_between(plot_df['Time(ms)'],
                         plot_df['Accuracy'] - plot_df['STE'],
                         plot_df['Accuracy'] + plot_df['STE'], alpha=0.2)
    plt.title(f"{lobe} Lobe - Grouped Multi-channel Decoding")
    plt.xlabel("Time (ms)")
    plt.ylabel("Accuracy")
    plt.axhline(0.5, color='gray', linestyle='--')
    plt.ylim(0.45, 0.8)
    plt.xlim(50, 500)
    plt.xticks(np.arange(50, 501, 50))
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{lobe}_GroupedDecodingAccuracy.png"))
    plt.close()

print("✔ Step 4 complete: Grouped decoding and plots saved.")

#%% Step 5: Classifier comparison (Best SVM BA per lobe, with significance testing)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import f_oneway, ttest_ind

print("Step 5: Classifier comparison (Best SVM BA per lobe, 150–250 and averaged)...")

clf_models = {
    'SVM': SVC(kernel='linear'),
    'LDA': LDA(),
    'RF': RandomForestClassifier(n_estimators=100, random_state=42)
}
lobe_best_bas = {}
boxplot_records_150_250 = []
boxplot_records_full = []

# Find best BA per lobe based on SVM (150–250 ms)
for ba in selected_channels:
    if ba not in ba_to_lobe:
        continue
    lobe = ba_to_lobe[ba]
    chans = selected_channels[ba]
    trials = [x for ch in chans for x in channel_data[ba][ch]]
    if not trials:
        continue
    X_trials, y_trials = zip(*trials)
    X = np.stack(X_trials)
    y = np.array(y_trials)

    accs = []
    for start, center in zip(window_starts, window_centers):
        if 150 <= center <= 250:
            X_win = X[:, start:start+win_size]
            X_concat = X_win.reshape(X_win.shape[0], -1)
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            for train_idx, test_idx in cv.split(X_concat, y):
                model = make_pipeline(StandardScaler(), SVC(kernel='linear'))
                model.fit(X_concat[train_idx], y[train_idx])
                accs.append(model.score(X_concat[test_idx], y[test_idx]))
    if accs:
        avg_acc = np.mean(accs)
        if lobe not in lobe_best_bas or avg_acc > lobe_best_bas[lobe][1]:
            lobe_best_bas[lobe] = (ba, avg_acc)

# Evaluate LDA/RF on best BA per lobe
for lobe, (ba, _) in lobe_best_bas.items():
    chans = selected_channels[ba]
    trials = [x for ch in chans for x in channel_data[ba][ch]]
    X_trials, y_trials = zip(*trials)
    X = np.stack(X_trials)
    y = np.array(y_trials)

    for clf_name, clf_model in clf_models.items():
        accs_150 = []
        accs_full = []
        for start, center in zip(window_starts, window_centers):
            X_win = X[:, start:start+win_size]
            X_concat = X_win.reshape(X_win.shape[0], -1)
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            for train_idx, test_idx in cv.split(X_concat, y):
                model = make_pipeline(StandardScaler(), clf_model)
                model.fit(X_concat[train_idx], y[train_idx])
                acc = model.score(X_concat[test_idx], y[test_idx])
                if 150 <= center <= 250:
                    accs_150.append(acc)
                accs_full.append(acc)
        for acc in accs_150:
            boxplot_records_150_250.append({'Lobe': lobe, 'Classifier': clf_name, 'Accuracy': acc})
        for acc in accs_full:
            boxplot_records_full.append({'Lobe': lobe, 'Classifier': clf_name, 'Accuracy': acc})

# Save and plot
df_150 = pd.DataFrame(boxplot_records_150_250)
df_full = pd.DataFrame(boxplot_records_full)
df_150.to_csv(os.path.join(output_dir, "Classifier_Boxplot_BestBA_150to250.csv"), index=False)
df_full.to_csv(os.path.join(output_dir, "Classifier_Boxplot_BestBA_Averaged.csv"), index=False)

def plot_with_significance(df, title, fname, palette):
    plt.figure(figsize=(12, 6))
    ax = sns.boxplot(data=df, x='Lobe', y='Accuracy', hue='Classifier', palette=palette)
    plt.title(title)
    plt.ylabel("Decoding Accuracy")
    plt.tight_layout()

    # Significance stars
    lobes = df['Lobe'].unique()
    clfs = df['Classifier'].unique()
    for lobe in lobes:
        data_lobe = df[df['Lobe'] == lobe]
        accs = [data_lobe[data_lobe['Classifier'] == clf]['Accuracy'].values for clf in clfs]
        if len(accs) == 3 and all(len(a) > 1 for a in accs):
            stat, p_anova = f_oneway(*accs)
            if p_anova < 0.05:
                pairs = [(0,1), (0,2), (1,2)]
                pvals = [ttest_ind(accs[i], accs[j]).pvalue for i,j in pairs]
                _, pvals_corr, _, _ = multipletests(pvals, method='fdr_bh')
                xlocs = [0, 0.2, 0.4]
                for (i,j), p, offset in zip(pairs, pvals_corr, xlocs):
                    if p < 0.05:
                        x1 = list(clfs).index(clfs[i]) + offset
                        x2 = list(clfs).index(clfs[j]) + offset
                        y = max(np.max(accs[i]), np.max(accs[j])) + 0.02
                        plt.plot([x1, x1, x2, x2], [y, y+0.005, y+0.005, y], lw=1.5, c='k')
                        plt.text((x1+x2)*.5, y+0.007, "*", ha='center', va='bottom', fontsize=14)

    plt.savefig(os.path.join(output_dir, fname))
    plt.close()

# Plot with stars
plot_with_significance(df_150, "Classifier Comparison (Best SVM BA per Lobe) - 150–250 ms",
                       "Classifier_Boxplot_BestBA_150to250ms_withStars.png", palette="Set3")
plot_with_significance(df_full, "Classifier Comparison (Best SVM BA per Lobe) - Averaged Accuracy (50–500 ms)",
                       "Classifier_Boxplot_BestBA_Averaged_withStars.png", palette="Set3")

print("✔ Step 5 complete: Boxplots with significance testing saved.")


#%% Step 6: FDR on peak decoding accuracy
raw_pvals, ba_peak_stats = [], []
for ba in results:
    accs, stds = zip(*results[ba])
    peak_idx = int(np.argmax(accs))
    peak_acc = accs[peak_idx]
    peak_std = stds[peak_idx]
    sim_folds = np.random.normal(loc=peak_acc, scale=peak_std, size=n_folds)
    _, pval = ttest_1samp(sim_folds, 0.5, alternative='greater')
    raw_pvals.append(pval)
    ba_peak_stats.append((ba, window_centers[peak_idx], peak_acc, pval))
rejected, pvals_corr, _, _ = multipletests(raw_pvals, alpha=0.05, method='fdr_bh')
fdr_rows = [(ba, time, acc, p, pc, sig) for (ba, time, acc, p), pc, sig in zip(ba_peak_stats, pvals_corr, rejected)]
fdr_df = pd.DataFrame(fdr_rows, columns=["BA", "PeakTime(ms)", "PeakAccuracy", "p_uncorrected", "p_FDR", "Significant"])
fdr_df.to_csv(os.path.join(output_dir, "FDR_PeakAccuracy_Significance.csv"), index=False)

print("✔ Step 6 complete: FDR on peak decoding saved.")

#%% Step 7–8: Select Early/Late BAs and Prepare Trial Matrices
early_threshold, late_threshold = 250, 251
lobe_to_bas = defaultdict(lambda: {'early': [], 'late': []})
ba_to_lobe_clean = {k.split(" (")[0]: v for k, v in ba_to_lobe.items()}

for _, row in fdr_df[fdr_df['Significant']].iterrows():
    ba = row['BA']
    peak_time = row['PeakTime(ms)']
    lobe = ba_to_lobe.get(ba)
    if lobe:
        if peak_time <= early_threshold:
            lobe_to_bas[lobe]['early'].append((ba, peak_time))
        elif peak_time >= late_threshold:
            lobe_to_bas[lobe]['late'].append((ba, peak_time))

selected_ba_set = set()
ba_peak_times = {}
for lobe, group in lobe_to_bas.items():
    for tag in ['early', 'late']:
        if group[tag]:
            best = max(group[tag], key=lambda x: fdr_df[fdr_df['BA'] == x[0]]['PeakAccuracy'].values[0])
            selected_ba_set.add(best[0])
            ba_peak_times[best[0]] = best[1]

roi_data_by_category = defaultdict(lambda: defaultdict(list))
for ba in selected_ba_set:
    for ch in selected_channels[ba]:
        for x, y in channel_data[ba][ch]:
            roi_data_by_category[ba][y].append(x)

min_trials = min([len(v) for cat in [0, 1] for ba in roi_data_by_category for v in [roi_data_by_category[ba][cat]]])
roi_trimmed = {ba: {cat: np.stack(roi_data_by_category[ba][cat][:min_trials]) for cat in [0, 1]} for ba in selected_ba_set}

print("✔ Step 7–8 complete: ROI data trimmed per category.")

#%% Step 9: Plot decoding accuracy for selected BA pairs used in R²
print("Step 9: Plotting decoding accuracy curves for selected BA pairs...")

df_all = pd.read_csv(os.path.join(output_dir, "AllBAs_MultiChannelGroupedDecoding.csv"))

for src, tgt in permutations(selected_ba_set, 2):
    if src == tgt:
        continue
    src_clean = src.split(" (")[0]
    tgt_clean = tgt.split(" (")[0]

    df_src = df_all[df_all['BA'].str.startswith(src_clean)]
    df_tgt = df_all[df_all['BA'].str.startswith(tgt_clean)]

    if df_src.empty or df_tgt.empty:
        continue

    plt.figure(figsize=(10, 5))

    # Plot source BA
    sub_src = df_src[df_src['Time(ms)'] >= 50]
    plt.plot(sub_src['Time(ms)'], sub_src['Accuracy'], label=f"{src_clean}", color='tab:blue')
    plt.fill_between(sub_src['Time(ms)'],
                     sub_src['Accuracy'] - sub_src['STE'],
                     sub_src['Accuracy'] + sub_src['STE'],
                     alpha=0.2, color='tab:blue')

    # Plot target BA
    sub_tgt = df_tgt[df_tgt['Time(ms)'] >= 50]
    plt.plot(sub_tgt['Time(ms)'], sub_tgt['Accuracy'], label=f"{tgt_clean}", color='tab:orange')
    plt.fill_between(sub_tgt['Time(ms)'],
                     sub_tgt['Accuracy'] - sub_tgt['STE'],
                     sub_tgt['Accuracy'] + sub_tgt['STE'],
                     alpha=0.2, color='tab:orange')

    # Format
    plt.title(f"Decoding Accuracy: {src_clean} and {tgt_clean}")
    plt.xlabel("Time (ms)")
    plt.ylabel("Accuracy")
    plt.axhline(0.5, linestyle='--', color='gray')
    plt.xlim(50, 500)
    plt.ylim(0.45, 0.8)
    plt.xticks(np.arange(50, 501, 50))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"DecodingComparison_{src_clean}_to_{tgt_clean}.png"))
    plt.close()

#%% Step 10 : write Sem_<time>.csv  
print("Step 10 : exporting Sem_<time>.csv files (FDR-passed peaks only)…")

subject_number_map = {
    'NBM':1,'CJS':2,'PJW':3,'JAH':4,'HSH2':6,
    'JHJ':7,'SJH':8,'JSH':10,'KHS':15,'LSJ_L':19
}

# --- info from FDR step ---
fdr_passed        = fdr_df[fdr_df['Significant']]
ba_peak_time      = dict(zip(fdr_passed['BA'], fdr_passed['PeakTime(ms)']))  
ba_peak_accuracy  = dict(zip(fdr_passed['BA'], fdr_passed['PeakAccuracy']))   

# build one CSV per peak time
for t in sorted(set(ba_peak_time.values())):
    rows=[]
    for ba, peak_t in ba_peak_time.items():
        if peak_t != t:         
            continue
        acc = ba_peak_accuracy[ba]

        # channels that survived feature-selection for this BA
        for subj_folder, ch in selected_channels_by_subject.get(ba, []):
            subj_short = subject_ba_mapping[subj_folder]
            subj_num   = subject_number_map.get(subj_short)
            if subj_num is None:   
                continue

            rows.append({'Subject':subj_num,
                         'Channel':int(ch),
                         'Accuracy':acc})

    # save if anything to write
    if rows:
        df_sem = (pd.DataFrame(rows)
                    .sort_values(['Subject','Channel']))
        df_sem.to_csv(os.path.join(output_dir, f"Sem_{t}.csv"), index=False)
        print(f"  • Sem_{t}.csv  written ({len(df_sem)} rows)")

print("✔  Sem files exported (only FDR-passed peak windows).")

#%% Step 11 : PRINT SUMMARY (FDR table • best BA per lobe • planned R² pairs)
print("\n" + "="*70)
print(" STEP 11  ▸  SUMMARY OF STATISTICALLY-RELEVANT RESULTS")
print("="*70)

# 1. FDR table (only significant rows)
print("\n--- Peak Accuracy and Significance (FDR-passed BAs) ---")
cols_show = ["BA", "PeakTime(ms)", "PeakAccuracy", "p_FDR"]
print(
    fdr_df[fdr_df["Significant"]][cols_show]
    .sort_values("PeakAccuracy", ascending=False)
    .to_string(index=False)
)

# 2. Best BA per lobe (lowest p_FDR, tie-break by highest accuracy)
print("\n--- Best Performing BA per Lobe ---")
for lobe in sorted(lobe_to_bas.keys()):
    ba_list = lobe_to_bas[lobe]["early"] + lobe_to_bas[lobe]["late"]
    if not ba_list:
        continue
    best_ba = min(
        ba_list,
        key=lambda ba_tp: (
            fdr_df.loc[fdr_df["BA"] == ba_tp[0], "p_FDR"].values[0],
            -fdr_df.loc[fdr_df["BA"] == ba_tp[0], "PeakAccuracy"].values[0],
        ),
    )
    ba, t_peak = best_ba
    acc_peak = fdr_df.loc[fdr_df["BA"] == ba, "PeakAccuracy"].values[0]
    print(f"{lobe:<9}: {ba:<30}  acc={acc_peak:.3f}  @ {int(t_peak)} ms")

# 3. Build ONE definitive list of R² pairs
print("\n--- Planned Cross-Temporal R²  (Source → Target) ---")

planned_pairs = [
    (src, tgt)
    for src, tgt in permutations(sorted(selected_ba_set), 2)
    if ba_to_lobe[src] != ba_to_lobe[tgt]           # different lobe
    and ba_peak_times[src] < ba_peak_times[tgt]     # early → late
]

for src, tgt in planned_pairs:
    print(f"{src}  ({ba_to_lobe[src]})  →  {tgt}  ({ba_to_lobe[tgt]})")

print(f"Total pairs queued: {len(planned_pairs)}")
print("="*70 + "\n")

#%% Step 12 : Cross-Temporal R²
category_names = {0: "NonBody", 1: "Body"}

print("Step 12: computing category-specific cross-temporal R² …")

num_pairs     = len(planned_pairs)
num_matrices  = num_pairs * 2         
print(f"▶  Will process {num_pairs} BA pairs → {num_matrices} R² matrices")

for src, tgt in planned_pairs:            
    for cat in (0, 1):                 
        label = category_names[cat]

        # trials for this single category only
        X_src = roi_trimmed[src][cat]     
        Y_tgt = roi_trimmed[tgt][cat]  

        r2_matrix   = np.zeros((usable_timepoints, usable_timepoints))
        pval_matrix = np.ones_like(r2_matrix)

        for t1 in range(usable_timepoints):
            x = X_src[:, t1].reshape(-1, 1)
            for t2 in range(usable_timepoints):
                y = Y_tgt[:, t2]

                r2_obs = LinearRegression().fit(x, y).score(x, y)
                r2_matrix[t1, t2] = r2_obs

                null_scores = [
                    LinearRegression().fit(x, shuffle(y))
                                     .score(x, shuffle(y))
                    for _ in range(n_permutations)
                ]
                pval_matrix[t1, t2] = np.mean(np.array(null_scores) >= r2_obs)

        # tidy BA names for files
        s_name = src.split(" (")[0].replace(" ", "")
        t_name = tgt.split(" (")[0].replace(" ", "")

        # save outputs
        pd.DataFrame(r2_matrix).to_csv(
            os.path.join(output_dir,
                         f"R2_{s_name}_to_{t_name}_{label}.csv"),
            index=False,
        )
        pd.DataFrame(pval_matrix).to_csv(
            os.path.join(output_dir,
                         f"Pval_{s_name}_to_{t_name}_{label}.csv"),
            index=False,
        )

        # heat-map
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            r2_matrix, cmap="viridis",
            xticklabels=10, yticklabels=10,
            cbar_kws={"label": "R²"}
        )
        ticks = np.arange(0, usable_timepoints, 10)
        plt.xticks(ticks, ticks * 10)
        plt.yticks(ticks, ticks * 10)
        plt.title(f"R² • {label}:  {s_name} → {t_name}")
        plt.xlabel("Target time (ms)")
        plt.ylabel("Source time (ms)")
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir,
                         f"Heat_{s_name}_to_{t_name}_{label}.png"),
            dpi=300,
        )
        plt.close()

print("✔  Step 12 complete – category-specific R² matrices & plots saved.")
