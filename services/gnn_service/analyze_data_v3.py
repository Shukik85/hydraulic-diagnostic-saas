"""
Advanced Analytics & Visualization for BIM Dataset (GNN input)
- Распределения признаков и целевых меток
- Анализ сбалансированности классов
- Поиск outliers, корреляций, примеры реальных и аномальных паттернов
- Фичи по временной деградации, каскадным сценариям
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from pathlib import Path

CSV_PATH = 'services/gnn_service/data/bim_simulated_balanced_v3.csv'
META_PATH = 'services/gnn_service/data/equipment_metadata.json'
OUT_DIR = 'services/gnn_service/data/analysis_report_v3/'
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# 1. DATA LOAD
print("[1] Загружаю данные...")
df = pd.read_csv(CSV_PATH)
print(f"[✓] Shape: {df.shape}")

# 2. Metadata
with open(META_PATH, 'r', encoding='utf-8') as f:
    meta = json.load(f)
components = meta['component_names']
colmap = meta['column_mapping']
faults = meta['fault_columns']

# 3. MAIN distribs, per-component
desc_stats = {}
for comp in components:
    comp_feats = colmap[comp]
    desc = df[comp_feats].describe().T
    desc_stats[comp] = desc.to_dict('index')
    plt.figure(figsize=(10,6))
    sns.boxplot(data=df[comp_feats], orient='h')
    plt.title(f"{comp} - feature distributions")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}{comp}_features_boxplot.png")
    plt.close()

# 4. Target/fault class balance
target_counts = {}
for comp in components:
    col = faults[comp]
    counts = df[col].value_counts().to_dict()
    target_counts[comp] = counts
    plt.figure(figsize=(4,2))
    sns.barplot(x=list(counts.keys()), y=list(counts.values()))
    plt.title(f"Target class {col}")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}{col}_class_balance.png")
    plt.close()

# 5. Corr matrix for all floats
plt.figure(figsize=(16,10))
corr = df.select_dtypes(include=[float, np.number]).corr()
sns.heatmap(corr, cmap='coolwarm', center=0, annot=False)
plt.title("Feature correlation matrix")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}correlation_matrix.png")
plt.close()

# 6. Outlier detection (simple 4 sigma)
outliers = {}
for comp in components:
    feats = colmap[comp]
    comp_outs = {}
    for feat in feats:
        z = (df[feat] - df[feat].mean()) / df[feat].std()
        n = int((np.abs(z)>4).sum())
        if n > 0:
            comp_outs[feat] = n
    if comp_outs:
        outliers[comp] = comp_outs
with open(f'{OUT_DIR}outliers.json', 'w', encoding='utf-8') as f:
    json.dump(outliers, f, indent=2)

# 7. Example time-based degradation scenario (флаг time-indexed)
if 'time' in df.columns:
    plt.figure(figsize=(14,5))
    for comp in components:
        feat = colmap[comp][0]  # например, главное давление/скорость
        plt.plot(df['time'], df[feat], label=comp)
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('first main sensor')
    plt.title('Degradation/Trend sample')
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}time_degradation_trend.png')
    plt.close()

# 8. Save overall statistics
df.info(buf=open(f'{OUT_DIR}df_info.txt','w'))
desc_stats['target_counts'] = target_counts
with open(f'{OUT_DIR}desc_stats.json', 'w', encoding='utf-8') as f:
    json.dump(desc_stats, f, indent=2)
print(f"[✓] Аналитика и визуализации сохранены в {OUT_DIR}")
