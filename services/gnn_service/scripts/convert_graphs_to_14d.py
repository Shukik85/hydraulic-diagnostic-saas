import argparse
import json
import os

import torch

# Предполагаем, что EdgeSpec и EdgeType доступны.
# Если нет, нужно проверить импорты под твой проект.
# Обычно это выглядит так:
from gnn_service.data.edge_spec import EdgeSpec
from torch_geometric.data import Data
from tqdm import tqdm


def load_edge_specifications(specs_path):
    with open(specs_path) as f:
        data = json.load(f)

    # Convert list of dicts to dict by key (source_id, target_id)
    specs_map = {}
    for item in data:
        # Создаем ключ для быстрого поиска: "source->target"
        # Учитываем оба направления для двунаправленных
        key = f"{item['source_id']}->{item['target_id']}"
        specs_map[key] = EdgeSpec(**item)

        if item.get("flow_direction") == "bidirectional":
            rev_key = f"{item['target_id']}->{item['source_id']}"
            specs_map[rev_key] = EdgeSpec(**item)

    return specs_map


def get_mock_dynamic_features():
    """
    Возвращает заглушки динамических данных для генерации 14D вектора.
    В реальной жизни это берется из TimeSeries данных.
    """
    return {
        "flow_rate_lpm": 50.0,
        "pressure_drop_bar": 10.0,
        "temperature_delta_c": 5.0,
        "vibration_level_g": 0.1,
        "age_hours": 1000.0,
        "maintenance_score": 0.9,
    }


def convert_graph_to_14d(graph, edge_specs_map, component_ids, current_time=0.0):
    """
    Создает новый граф с 14D edge features (8 static + 6 dynamic).
    """

    # 1. Проверяем соответствие узлов
    num_nodes = graph.x.shape[0]
    if len(component_ids) != num_nodes:
        print(
            f"SKIP: Graph has {num_nodes} nodes, but {len(component_ids)} component_ids provided."
        )
        return None

    edge_index = graph.edge_index
    num_edges = edge_index.shape[1]

    new_edge_attrs = []
    valid_mask = []

    # 2. Итерируемся по всем ребрам
    for i in range(num_edges):
        src_idx = edge_index[0, i].item()
        dst_idx = edge_index[1, i].item()

        src_id = component_ids[src_idx]
        dst_id = component_ids[dst_idx]

        key = f"{src_id}->{dst_id}"

        if key not in edge_specs_map:
            # Если спецификации нет, пропускаем ребро или ставим дефолт
            # Для чистоты данных лучше пропустить, но это изменит топологию
            # В данном фиксе мы просто пометим как 'valid=False' и потом отфильтруем
            # Или выбросим ошибку, если строгая валидация
            valid_mask.append(False)
            continue

        edge_spec = edge_specs_map[key]

        # 3. Формируем 8D Static Features
        static_tensor = torch.tensor(
            [
                float(edge_spec.diameter_mm),
                float(edge_spec.length_m),
                float(edge_spec.pressure_rating_bar),
                float(edge_spec.get_age_hours(current_time)),
                float(edge_spec.cross_section_area_mm2),
                float(edge_spec.pressure_loss_coefficient),
                1.0 if edge_spec.material == "steel" else 0.0,  # Пример маппинга
                1.0 if edge_spec.edge_type == "high_pressure_hose" else 0.0,
            ],
            dtype=torch.float32,
        )

        # 4. Формируем 6D Dynamic Features (Mock)
        dyn = get_mock_dynamic_features()
        dynamic_tensor = torch.tensor(
            [
                dyn["flow_rate_lpm"],
                dyn["pressure_drop_bar"],
                dyn["temperature_delta_c"],
                dyn["vibration_level_g"],
                dyn["age_hours"],
                dyn["maintenance_score"],
            ],
            dtype=torch.float32,
        )

        # 5. Объединяем в 14D
        edge_attr_14d = torch.cat([static_tensor, dynamic_tensor])

        new_edge_attrs.append(edge_attr_14d)
        valid_mask.append(True)

    if not new_edge_attrs:
        return None

    # Собираем новый граф
    new_edge_attr = torch.stack(new_edge_attrs)
    valid_mask = torch.tensor(valid_mask, dtype=torch.bool)

    # Фильтруем edge_index по валидным ребрам (для которых нашли спецификацию)
    new_edge_index = edge_index[:, valid_mask]

    # Создаем объект Data
    new_graph = Data(x=graph.x, edge_index=new_edge_index, edge_attr=new_edge_attr, y=graph.y)

    return new_graph


def main():
    parser = argparse.ArgumentParser(description="Convert graphs to 14D edge features")
    parser.add_argument("--input", type=str, required=True, help="Input .pt file")
    parser.add_argument("--edge-specs", type=str, required=True, help="Edge specs JSON")
    parser.add_argument("--output", type=str, required=True, help="Output .pt file")
    parser.add_argument("--max-samples", type=int, default=None)
    # Обновленный default для нашего графа из 7 узлов
    parser.add_argument(
        "--component-ids",
        type=str,
        default="valve_main,pump_main_1,pump_main_2,cylinder_boom,cylinder_arm,cylinder_bucket,motor_swing",
        help="Comma-separated component IDs",
    )

    args = parser.parse_args()

    # Parse IDs
    component_ids = [cid.strip() for cid in args.component_ids.split(",")]
    print(f"Target Component IDs ({len(component_ids)}): {component_ids}")

    # Load Data
    specs_map = load_edge_specifications(args.edge_specs)
    print(f"Loaded {len(specs_map)} edge specifications.")

    print(f"Loading graphs from {args.input}...")
    graphs = torch.load(args.input, weights_only=False)
    if not isinstance(graphs, list):
        graphs = [graphs]

    print(f"Total graphs found: {len(graphs)}")

    if args.max_samples:
        graphs = graphs[: args.max_samples]
        print(f"Processing subset: {len(graphs)}")

    converted_graphs = []

    for g in tqdm(graphs, desc="Converting"):
        try:
            new_g = convert_graph_to_14d(g, specs_map, component_ids)
            if new_g is not None:
                converted_graphs.append(new_g)
        except Exception as e:
            print(f"Error converting graph: {e}")
            continue

    # Statistics
    print("\n" + "=" * 60)
    print("STATISTICS")
    print("=" * 60)

    if not converted_graphs:
        print("⚠️  WARNING: No graphs were successfully converted!")
        print(
            f"Check if component_ids match the graph nodes ({len(component_ids)} vs graph.num_nodes)"
        )
        return

    avg_nodes = sum(g.num_nodes for g in converted_graphs) / len(converted_graphs)
    avg_edges = sum(g.num_edges for g in converted_graphs) / len(converted_graphs)

    print(f"Successfully converted: {len(converted_graphs)} / {len(graphs)}")
    print(f"Avg Nodes: {avg_nodes:.1f}")
    print(f"Avg Edges: {avg_edges:.1f}")
    print(f"Edge Feature Shape: {converted_graphs[0].edge_attr.shape}")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(converted_graphs, args.output)
    print(f"\n✅ Saved to {args.output}")


if __name__ == "__main__":
    main()
