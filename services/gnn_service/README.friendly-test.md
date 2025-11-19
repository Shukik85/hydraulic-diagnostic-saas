# Инструкция: только PyTorch/TorchScript

## Экспорт модели
```python
import torch
from your_model import UniversalGNN

model = UniversalGNN(...)
scripted = torch.jit.script(model)
scripted.save("/models/current/model.pt")
```

## Тестовый inference и деплой
- Использовать только .pt/.pth (torch.jit.script/trace)
- Любые динамические размеры входа поддерживаются нативно!

## Валидация модели
```python
import torch
m = torch.jit.load("/path/to/model.pt")
test_out = m(dummy_input)  # без ошибок
```

## GNN сервис и admin-endpoints
- Проверка только pt/pth, никакого ONNX
- Friendly test-inference (sandbox): через "/admin/model/test_inference" (только .pt/.pth)
- AB-тест: только через .pt/.pth модели

## AB-пайплайн не меняется
- Просто используйте путь до .pt/.pth для всех тестовых и прод моделей

## Документация/README
- Все инструкции для ONNX удалены, только PyTorch

---

**Все будущие GNN/ML модели — только в формате PyTorch!**

