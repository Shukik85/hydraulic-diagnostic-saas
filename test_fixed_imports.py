"""
Test fixed imports.
"""

import os
import importlib.util


def test_import(module_path, module_name):
    """Test importing a module."""
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print(f"✅ {module_name} imported successfully")
        return True
    except Exception as e:
        print(f"❌ {module_name} import failed: {e}")
        return False


def main():
    """Test all imports."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    gnn_service_dir = os.path.join(current_dir, "gnn_service")

    modules_to_test = [
        ("config.py", "config"),
        ("prepare_bim_data.py", "prepare_bim_data"),
        ("model.py", "model"),
        ("dataset.py", "dataset"),
    ]

    print("Testing imports...")
    all_passed = True

    for file_name, module_name in modules_to_test:
        file_path = os.path.join(gnn_service_dir, file_name)
        if not test_import(file_path, module_name):
            all_passed = False

    if all_passed:
        print("\n🎉 All imports work! Use: python -m gnn_service.prepare_bim_data")
    else:
        print("\n❌ Some imports failed")


if __name__ == "__main__":
    main()
