# 🔧 Configuration Consolidation Guide

**Status**: ✅ Root `pyproject.toml` updated with all tool configurations

**Goal**: Single source of truth for Python tooling

---

## 📋 Current Situation

### ❌ Before: Scattered Configuration Files

```
📁 root/
├── pyproject.toml              (black, ruff basics)
├── services/gnn_service/
│   ├── pyproject.toml          (duplicate)
│   ├── pytest.ini              (pytest config)
│   ├── mypy.ini                (mypy config)
│   ├── ruff.toml               (7.3KB ruff config!)
│   ├── .coveragerc             (coverage config)
│   └── ...
```

**Problems**:
- 📌 Multiple sources of truth
- 🔄 Configuration sync issues
- 📚 Hard to maintain
- ⚠️ Inconsistent between tools
- 🐛 Easy to miss updates

### ✅ After: Centralized Configuration

```
📁 root/
├── pyproject.toml              (ALL tools configured)
├── services/gnn_service/
│   ├── pytest.ini              (DELETE)
│   ├── mypy.ini                (DELETE)
│   ├── ruff.toml               (DELETE)
│   ├── .coveragerc             (DELETE)
│   └── ...
```

**Benefits**:
- ✅ Single source of truth
- ✅ Easy to update
- ✅ IDE support (PyCharm, VS Code)
- ✅ CI/CD consistency
- ✅ No duplication

---

## 🚀 Migration Steps

### Step 1: Update ROOT `pyproject.toml` ✅

```bash
# Already done! Root pyproject.toml now contains:
✅ [tool.black]
✅ [tool.ruff]
✅ [tool.mypy]
✅ [tool.pytest.ini_options]
✅ [tool.coverage.run]
✅ [tool.coverage.report]
✅ [tool.isort]
✅ [tool.bandit]
```

### Step 2: Delete Local Config Files

```bash
cd services/gnn_service/

# Remove individual tool configs
rm pytest.ini          # ← pytest config now in root pyproject.toml
rm mypy.ini            # ← mypy config now in root pyproject.toml
rm ruff.toml           # ← ruff config now in root pyproject.toml
rm .coveragerc         # ← coverage config now in root pyproject.toml
rm pyproject.toml      # ← duplicate, not needed
```

### Step 3: Update Tools to Use Root Config

#### pytest
```bash
# OLD (from services/gnn_service/)
cd services/gnn_service
pytest tests/ -c pytest.ini

# NEW (from root)
cd root
pytest services/gnn_service/tests/ -c pyproject.toml
```

#### mypy
```bash
# OLD
cd services/gnn_service
mypy src/ --config-file=mypy.ini

# NEW
cd root
mypy services/gnn_service/src/ --config-file=pyproject.toml
```

#### ruff
```bash
# OLD
cd services/gnn_service
ruff check src/ --config=ruff.toml

# NEW
cd root
ruff check services/gnn_service/src/ --config=pyproject.toml
```

#### coverage
```bash
# OLD
cd services/gnn_service
pytest --cov=src --cov-config=.coveragerc

# NEW
cd root
pytest services/gnn_service/tests/ --cov=services/gnn_service/src --cov-config=pyproject.toml
```

### Step 4: Update CI/CD Pipelines

#### GitHub Actions
```yaml
# OLD
- name: Run Tests
  run: |
    cd services/gnn_service
    pytest tests/ -c pytest.ini

# NEW
- name: Run Tests
  run: |
    cd root_directory
    pytest services/gnn_service/tests/ -c pyproject.toml
```

### Step 5: Update IDE Settings

#### PyCharm
```
Settings → Languages & Frameworks → Python
├── Linting → Ruff
│   └── Config: pyproject.toml (root)
├── Type Checker → Mypy
│   └── Mypy path: (auto-detect from root)
├── Testing → pytest
│   └── pytest.ini: pyproject.toml (root)
```

#### VS Code
```json
{
  "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter",
    "editor.formatOnSave": true
  },
  "python.linting.ruffEnabled": true,
  "python.linting.ruffPath": "ruff",
  "python.testing.pytestEnabled": true,
  "python.testing.pytestPath": "pytest"
}
```

---

## 📊 Configuration Reference

### Tool Locations in Root `pyproject.toml`

| Tool | Section | Status |
|------|---------|--------|
| **black** | `[tool.black]` | ✅ Configured |
| **ruff** | `[tool.ruff]` + `[tool.ruff.lint]` + `[tool.ruff.format]` | ✅ Configured |
| **mypy** | `[tool.mypy]` + `[[tool.mypy.overrides]]` | ✅ Configured |
| **pytest** | `[tool.pytest.ini_options]` | ✅ Configured |
| **coverage** | `[tool.coverage.run]` + `[tool.coverage.report]` | ✅ Configured |
| **isort** | `[tool.isort]` | ✅ Configured |
| **bandit** | `[tool.bandit]` | ✅ Configured |

### Key Configuration Changes

```toml
# Python version upgraded
python_version = "3.14"           # from 3.11
target-version = ["py314"]        # from py311

# Test paths extended
testpaths = ["tests", "services/gnn_service/tests"]

# Coverage includes GNN service
source = ["src", "services/gnn_service/src"]

# Mypy strict mode fully enabled
disallow_untyped_defs = true
disallow_incomplete_defs = true
strict_equality = true
```

---

## ✅ Validation Checklist

### Before Cleanup

- [ ] Root `pyproject.toml` has all tool configurations
- [ ] All tools can run from root using `-c pyproject.toml`
- [ ] Tests pass with root config
- [ ] MyPy passes with root config
- [ ] Ruff passes with root config
- [ ] Coverage report generates from root config

### Cleanup Phase

- [ ] Backup local configs (git stash)
- [ ] Delete local config files
- [ ] Update CI/CD pipelines
- [ ] Update IDE settings
- [ ] Test all tools from root

### Post-Cleanup Validation

- [ ] `pytest` finds tests correctly
- [ ] Coverage report generated
- [ ] No config file conflicts
- [ ] IDE still recognizes Python settings
- [ ] CI/CD pipeline passes

---

## 🔍 Testing the Configuration

### From Root Directory

```bash
# All paths are relative to root

# Run tests
pytest services/gnn_service/tests/ -c pyproject.toml

# Type check
mypy services/gnn_service/src/ --config-file=pyproject.toml

# Lint
ruff check services/gnn_service/src/ --config=pyproject.toml

# Format
black services/gnn_service/src/ --config=pyproject.toml

# Coverage
pytest services/gnn_service/tests/ \
  --cov=services/gnn_service/src \
  --cov-config=pyproject.toml
```

---

## 🔗 Local Development Workflow

### If Working in `services/gnn_service/` Subdirectory

```bash
cd services/gnn_service/

# Point tools to parent directory config
pytest tests/ -c ../../pyproject.toml
mypy src/ --config-file=../../pyproject.toml
ruff check src/ --config=../../pyproject.toml
```

### Or Create a Symlink

```bash
cd services/gnn_service/
ln -s ../../pyproject.toml pyproject.toml

# Now tools find it automatically
pytest tests/
mypy src/
ruff check src/
```

---

## 🐛 Troubleshooting

### pytest not finding config
```bash
# Explicitly specify
pytest -c pyproject.toml tests/

# Or set environment variable
export PYTEST_ADDOPTS="-c pyproject.toml"
```

### mypy not finding overrides
```bash
# Make sure [[tool.mypy.overrides]] sections are present
# Check: grep -A5 "\[\[tool.mypy.overrides\]\]" pyproject.toml
```

### coverage not merging reports
```bash
# Set coverage config explicitly
pytest --cov-config=pyproject.toml --cov=...
```

### ruff conflicts with black
```bash
# Both configured to be compatible in pyproject.toml
# Should auto-resolve with line-length = 88
```

---

## 📚 Documentation References

- [PEP 621 - Python Project Metadata](https://peps.python.org/pep-0621/)
- [Ruff Configuration](https://docs.astral.sh/ruff/configuration/)
- [MyPy Configuration](https://mypy.readthedocs.io/en/stable/config_file.html)
- [Pytest Configuration](https://docs.pytest.org/en/latest/configuration.html)
- [Coverage.py Configuration](https://coverage.readthedocs.io/en/latest/config.html)

---

## 🎯 Next Steps

1. ✅ ROOT pyproject.toml created
2. 📝 Update CI/CD pipelines (GitHub Actions, GitLab CI, etc.)
3. 🗑️ Delete local config files
4. 🔧 Update IDE settings
5. ✔️ Run full test suite
6. 🚀 Deploy to staging

---

**Status**: Configuration consolidated into root `pyproject.toml`

**Last Updated**: December 16, 2025

**Maintainer**: ML Engineering Team
