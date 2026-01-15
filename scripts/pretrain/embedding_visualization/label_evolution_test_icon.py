import pandas as pd
import matplotlib.pyplot as plt
import re

# --- CONFIG ---
cropped = "_cropped" # set to "" if not cropped
CSV_PATH = f"/data1/fig/dcv2_ir108_128x128_k9_expats_70k_200-300K_CMA/test/teamx{cropped}/features_dcv2_ir108_128x128_k9_expats_70k_200-300K_CMA{cropped}.csv"  # replace with your CSV
TIME_REGEX = r"(\d{8}[_-]?\d{4})"  # matches 20250701_2300
OUTPUT_PLOT = f"/data1/fig/dcv2_ir108_128x128_k9_expats_70k_200-300K_CMA/test/teamx{cropped}/case_study_labels_over_time{cropped}.png"
# --- LOAD DATA ---
df = pd.read_csv(CSV_PATH, low_memory=False)

# --- FILTER CASE STUDY ---
df_case = df[df['case_study'] == True].copy()

# --- EXTRACT TIMESTAMP ---
def extract_timestamp(path):
    if pd.isna(path):
        return None
    m = re.search(TIME_REGEX, path)
    return m.group(1) if m else None

df_case['timestamp'] = df_case['location'].apply(extract_timestamp)

# --- CONVERT TO DATETIME ---
df_case['datetime'] = pd.to_datetime(df_case['timestamp'], format="%Y%m%d_%H%M")

#extract only row wit date on 06/30/2025
df_case = df_case[df_case['datetime'].dt.date == pd.to_datetime("2025-06-30").date()]

# --- SORT BY TIME ---
df_case = df_case.sort_values("datetime")

# --- SEPARATE MSG AND ICON ---
df_msg = df_case[df_case['vector_type'] == 'msg']
df_icon = df_case[df_case['vector_type'] == 'icon']

# --- PLOT ---
plt.figure(figsize=(8,5))

plt.plot(df_msg['datetime'], df_msg['label'], marker='o', linestyle='-', linewidth=2, color='blue', label='MSG')
plt.plot(df_icon['datetime'], df_icon['label'], marker='o', linestyle='-', linewidth=2, color='orange', label='ICON')

plt.xlabel("Time (UTC)", fontdict={'fontsize': 16, 'fontweight': 'bold'})
plt.ylabel("Label", fontdict={'fontsize': 16, 'fontweight': 'bold'})
plt.title("Case Study Labels over Time", fontsize=16, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=14)
plt.yticks(fontsize=14)
plt.legend()
plt.grid(True)
plt.savefig(OUTPUT_PLOT, bbox_inches='tight')
