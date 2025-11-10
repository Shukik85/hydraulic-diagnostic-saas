"""
BIM Синтетика — обновленный генератор: time-index, деградация, каскадные сценарии
- Добавляет столбец 'time' (если нет) для каждой строки
- Симулирует деградацию параметров для стареющих компонентов (отдельный сценарий)
- Генерирует базовый, деградирующий и каскадный режим по контролируемым правилам
"""
import json
import numpy as np
import pandas as pd
import random
from pathlib import Path

META_PATH = 'services/gnn_service/data/equipment_metadata.json'
OUT_CSV = 'services/gnn_service/data/bim_simulated_balanced_v3.csv'
NUM_SAMPLES = 100_000
DEGRAD_FRAC = 0.15   # сколько трендов внутридатасета тренировки
DEGRADE_STEPS = 500  # длина одного «износа» (будет сплайсовано)

with open(META_PATH, 'r', encoding='utf-8') as f:
    meta = json.load(f)
NORMALS = meta['physical_norms']
COMPONENTS = meta['component_names']
COLUMNS = meta['column_mapping']
FAULTS = meta['fault_columns']
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ===== Базовые вспомогательные =====
def sample_feature(param_spec):
    delta = (param_spec['max'] - param_spec['min']) / 2
    return np.random.normal(loc=param_spec['nominal'], scale=delta/4)

def simulate_fault_state(param_spec, kind='over', severity=1.0):
    if kind == 'over':
        border = param_spec['max'] + severity*(param_spec['critical']-param_spec['max'] if param_spec['critical'] else 5)
        return np.random.uniform(param_spec['max']*1.01, border)
    elif kind == 'under':
        border = param_spec['min'] - severity*(param_spec['min']-param_spec['critical'] if param_spec['critical'] else 5)
        return np.random.uniform(border, param_spec['min']*0.99)
    return sample_feature(param_spec)

def generate_one(mode='normal', fault_comp=None, fault_type=None, severity=1.0, t=0, degrade_comps=None):
    row = {}
    faults = {c: 0 for c in COMPONENTS}
    for comp in COMPONENTS:
        comp_params = NORMALS[comp]
        for idx, feat in enumerate(COLUMNS[comp]):
            par_keys = list(comp_params.keys())
            param = comp_params[par_keys[idx]]
            if mode == 'normal' or comp != fault_comp:
                if degrade_comps and comp in degrade_comps:
                    # Линейная или случайная деградация вверх/вниз
                    direction = random.choice([-1,1])
                    progress = t / DEGRADE_STEPS
                    shift = direction * progress * (param['max'] - param['min']) * random.uniform(0.02,0.08)
                    val = np.clip(param['nominal'] + shift, param['min'], param['critical'] if param['critical'] else param['max'])
                else:
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
    row['time'] = t
    return row


def generate_balanced_dataset(num_samples=NUM_SAMPLES, frac_normal=0.5, frac_single_fault=0.35, frac_multi_fault=0.1, frac_degrade=DEGRAD_FRAC):
    out = []
    num_normal = int(num_samples * frac_normal)
    num_single = int(num_samples * frac_single_fault)
    num_multi = int(num_samples * frac_multi_fault)
    num_degrade = num_samples - num_normal - num_single - num_multi
    # Базовая выборка
    for i in range(num_normal):
        out.append(generate_one('normal', t=i))
    for i in range(num_single):
        comp = COMPONENTS[i % len(COMPONENTS)]
        ftype = random.choice(['over', 'under'])
        out.append(generate_one('fault', fault_comp=comp, fault_type=ftype, severity=random.uniform(0.8,1.5), t=num_normal+i))
    for i in range(num_multi):
        n = random.randint(2, 4)
        faulty_comps = random.sample(COMPONENTS, n)
        row = generate_one('normal', t=num_normal+num_single+i)
        for fc in faulty_comps:
            ftype = random.choice(['over', 'under'])
            comp_params = NORMALS[fc]
            for idx, feat in enumerate(COLUMNS[fc]):
                par_keys = list(comp_params.keys())
                param = comp_params[par_keys[idx]]
                row[feat] = simulate_fault_state(param, ftype, severity=random.uniform(1,2))
            row[FAULTS[fc]] = 1
        out.append(row)
    # Серии возрастающей деградации
    for k in range(num_degrade//DEGRADE_STEPS):
        degrade_comps = random.sample(COMPONENTS, random.randint(1,len(COMPONENTS)))
        for t in range(DEGRADE_STEPS):
            out.append(generate_one('normal', t=len(out), degrade_comps=degrade_comps))
    return pd.DataFrame(out)

if __name__ == '__main__':
    data = generate_balanced_dataset()
    Path(OUT_CSV).parent.mkdir(exist_ok=True, parents=True)
    data.to_csv(OUT_CSV, index=False)
    print(f"[✓] BIM synthetic dataset (time, degradation, cascade): {OUT_CSV} | {data.shape}")
