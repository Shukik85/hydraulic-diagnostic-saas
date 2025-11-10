"""
BIM Scenario Generator — Гибкая физическая генерация синтетических данных для GNN

- Использует физические нормы и компонентную структуру из equipment_metadata.json
- Позволяет смоделировать норму, деградации, мультиотказы, каскадные сбои
- Экспортирует датасет готовый для GNN pipeline
"""
import json
import numpy as np
import pandas as pd
import random
from pathlib import Path

META_PATH = 'services/gnn_service/data/equipment_metadata.json'
OUT_CSV = 'services/gnn_service/data/bim_simulated_balanced_v3.csv'
NUM_SAMPLES = 100_000

with open(META_PATH, 'r', encoding='utf-8') as f:
    meta = json.load(f)

NORMALS = meta['physical_norms']
COMPONENTS = meta['component_names']
COLUMNS = meta['column_mapping']
FAULTS = meta['fault_columns']
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

def sample_feature(param_spec):
    """Получить случайное значение в норме или за границей"""
    nominal = param_spec['nominal']
    delta = (param_spec['max'] - param_spec['min']) / 2
    return np.random.normal(loc=nominal, scale=delta/4)

def simulate_fault_state(param_spec, kind='over', severity=1.0):
    """Симуляция выхода за предел: over→max+delta, under→min-delta"""
    if kind == 'over':
        border = param_spec['max'] + severity * (param_spec['critical']-param_spec['max'] if param_spec['critical'] else 5)
        return np.random.uniform(param_spec['max']*1.01, border)
    elif kind == 'under':
        border = param_spec['min'] - severity * (param_spec['min']-param_spec['critical'] if param_spec['critical'] else 5)
        return np.random.uniform(border, param_spec['min']*0.99)
    return sample_feature(param_spec)

def generate_one(mode='normal', fault_comp=None, fault_type=None, severity=1.0):
    row = {}
    faults = {c: 0 for c in COMPONENTS}
    for comp in COMPONENTS:
        comp_params = NORMALS[comp]
        for idx, feat in enumerate(COLUMNS[comp]):
            par_keys = list(comp_params.keys())
            param = comp_params[par_keys[idx]]
            if mode == 'normal' or comp != fault_comp:
                val = sample_feature(param)
            else:
                if fault_type == 'over':
                    val = simulate_fault_state(param, 'over', severity)
                elif fault_type == 'under':
                    val = simulate_fault_state(param, 'under', severity)
                else:
                    val = sample_feature(param)
                faults[comp] = 1
            row[feat] = val
    for c in COMPONENTS:
        row[FAULTS[c]] = faults[c]
    return row

def generate_balanced_dataset(num_samples=NUM_SAMPLES, frac_normal=0.5, frac_single_fault=0.4, frac_multi_fault=0.1):
    out = []
    num_normal = int(num_samples * frac_normal)
    num_single = int(num_samples * frac_single_fault)
    num_multi = num_samples - num_normal - num_single
    # Норма
    for _ in range(num_normal):
        out.append(generate_one('normal'))
    # Одиночные поломки (равномерно по компонентам и типам)
    for i in range(num_single):
        comp = COMPONENTS[i % len(COMPONENTS)]
        ftype = random.choice(['over', 'under'])
        out.append(generate_one('fault', fault_comp=comp, fault_type=ftype, severity=random.uniform(0.8,1.5)))
    # Мультиотказы (2+ компоненты одновременно)
    for _ in range(num_multi):
        n = random.randint(2, 4)
        faulty_comps = random.sample(COMPONENTS, n)
        row = generate_one('normal')
        for fc in faulty_comps:
            ftype = random.choice(['over', 'under'])
            comp_params = NORMALS[fc]
            for idx, feat in enumerate(COLUMNS[fc]):
                par_keys = list(comp_params.keys())
                param = comp_params[par_keys[idx]]
                row[feat] = simulate_fault_state(param, ftype, severity=random.uniform(1,2))
            row[FAULTS[fc]] = 1
        out.append(row)
    return pd.DataFrame(out)

if __name__ == '__main__':
    data = generate_balanced_dataset()
    Path(OUT_CSV).parent.mkdir(exist_ok=True, parents=True)
    data.to_csv(OUT_CSV, index=False)
    print(f"[✓] BIM synthetic dataset generated: {OUT_CSV} | {data.shape}")
