# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 17:03:18 2024

@author: ruffy
"""

#%% Group
import numpy as np
import scipy.io as sio
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Helper functions
def load_mat_file(file_path):
    return sio.loadmat(file_path)['Con_TFData_cut']

def load_data(subject, base_path):
    body_file = f'{base_path}{subject}/A0_body_{subject}.mat'
    nonbody_file = f'{base_path}{subject}/A1_nonbody_{subject}.mat'
    data_body = load_mat_file(body_file)
    data_nonbody = load_mat_file(nonbody_file)
    return data_body, data_nonbody

def get_ba_for_channel(channel_real, talairach_df, subject_code, subject_ba_mapping):
    mapped_subject = subject_ba_mapping.get(subject_code, subject_code)  # Fallback to the original if no mapping exists
    filtered_df = talairach_df[(talairach_df['Subject'] == mapped_subject) & (talairach_df['Channel'] == channel_real)]
    return filtered_df['BA'].iloc[0] if not filtered_df.empty else "Unknown BA"

def map_to_fixed_range(lower_percentage, upper_percentage, fixed_ranges):
    average_percentage = (lower_percentage + upper_percentage) / 2
    for lower_bound, upper_bound in fixed_ranges:
        if lower_bound <= average_percentage < upper_bound:
            return f"{lower_bound}-{upper_bound}%"
    return "80-100%"

# Global variables
base_path = 'G:/export/'
subjects = ['220112_NBM', '220209_CJS', '220224_PJW', '220310_JAH', '220503_HSH2', '220707_JHJ', '220803_SJH', '221018_JSH', '230328_KHS', '230728_LSJ_left']
speech_onset = {
    '220112_NBM': 1000,
    '220209_CJS': 1200,
    '220224_PJW': 900,
    '220310_JAH': 800,
    '220503_HSH2': 1200,
    '220707_JHJ': 750,
    '220803_SJH': 1000,
    '221018_JSH': 1350,
    '230328_KHS': 800,
    '230728_LSJ_left': 850
}

subject_ba_mapping = {
    '220112_NBM': 'NBM',
    '220209_CJS': 'CJS',
    '220224_PJW': 'PJW',
    '220310_JAH': 'JAH',
    '220503_HSH2': 'HSH2',
    '220707_JHJ': 'JHJ',
    '220803_SJH': 'SJH',
    '221018_JSH': 'JSH',
    '230328_KHS': 'KHS',
    '230728_LSJ_left': 'LSJ_L'
}

freq_bands = {'Theta': [4, 8, 25], 'Alpha': [8, 12, 20], 'Beta': [12, 30, 20], 'Gamma': [30, 50, 15], 'HG': [70, 170, 15]}
random_states = [346, 298, 439, 245, 796, 963, 909, 24, 723, 969]
all_data = []
fixed_ranges = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]

for subject in subjects:
    print(f"Processing data for {subject}...")
    data_body, data_nonbody = load_data(subject, base_path)
    talairach_df = pd.read_excel('G:/Talairach/Talairach.xlsx')
    chan_list_data = sio.loadmat(f'{base_path}{subject}/ChanList.mat')
    chan_list = chan_list_data['ChanList'][0]
    num_channels, _, max_time, num_trials_body = data_body.shape
    _, _, _, num_trials_nonbody = data_nonbody.shape
    y = np.array([0] * num_trials_body + [1] * num_trials_nonbody)
    data_records = []

    for band, (low_freq, high_freq, window_size) in freq_bands.items():
        num_time_windows = max_time // window_size
        for time_window in range(num_time_windows):
            start_time = time_window * window_size
            end_time = start_time + window_size

            data_body_window = data_body[:, low_freq:high_freq, start_time:end_time, :]
            data_nonbody_window = data_nonbody[:, low_freq:high_freq, start_time:end_time, :]
            time_window_data = np.mean(np.concatenate((data_body_window, data_nonbody_window), axis=3), axis=(1, 2))

            for channel in range(num_channels):
                channel_real = int(chan_list[channel])
                channel_data = time_window_data[channel, :]

                # Perform Mann-Whitney U test
                group0 = channel_data[y == 0]
                group1 = channel_data[y == 1]
                if len(group0) > 0 and len(group1) > 0:
                    stat, p = mannwhitneyu(group0, group1)
                    if p < 0.05:
                        accuracies = []
                        for random_state in random_states:
                            X_train, X_test, y_train, y_test = train_test_split(channel_data.reshape(-1, 1), y, test_size=0.2, random_state=random_state, stratify=y)
                            clf = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100))
                            clf.fit(X_train, y_train)
                            accuracies.append(clf.score(X_test, y_test))
                        if accuracies:
                            mean_accuracy = np.mean(accuracies)
                            accuracy_difference = mean_accuracy - 0.5
                            ba = get_ba_for_channel(channel_real, talairach_df, subject, subject_ba_mapping)
                            if accuracy_difference > 0 and 'visual' not in ba.lower() and 'eye' not in ba.lower():
                                feature_info = f"{band} | {ba} | {map_to_fixed_range(100 * start_time / max_time, 100 * end_time / max_time, fixed_ranges)}"
                                data_records.append([subject, feature_info, channel_real, accuracy_difference])

    df = pd.DataFrame(data_records, columns=["Subject", "Feature Information", "Channel", "Accuracy Difference"])
    individual_path = f"G:/export/{subject}/A_body_nonbody/variables/accuracy_{subject}.csv"
    df.to_csv(individual_path, index=False)
    all_data.append(df)
    print(f"Data processing and saving completed for {subject}.")

combined_df = pd.concat(all_data, ignore_index=True)
combined_path = "G:/export/sorted_features/Sem_left/ver2_accuracy_aggregated.csv"
combined_df.to_csv(combined_path, index=False)
print("All data combined and saved successfully.")