# HW1 — аналитическая модель производительности небольшой CNN

## Окружение

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Ti, 16 GB (15 833 MiB доступно), compute capability 12.0, 70 SM |
| Драйвер | 595.84 |
| PyTorch | 2.11.0+cu130 (CUDA 13.0, cuDNN 9.19.0) |
| Python | 3.12.13 |
| ОС | Linux 7.0.0-34-generic, x86_64 |
| Модель | [models.py](models.py), 1 040 324 параметра (4,16 MB в FP32) |

Полный снимок окружения и протокола пишется в [results/env.json](results/env.json) при каждом запуске `measure.py`.

## Условия замеров

**Лимит мощности 250 W.** Все замеры в `results/` сняты с лимитом мощности, сниженным до 250 W
(`sudo nvidia-smi -pl 250`; штатный лимит карты 300 W) - иначе система перезагружается. При 300 W первый прогон сетки завершился с `CUDA error: unspecified launch failure` на S=112, B=128 и жёсткая перезагрузка машины. Лимит сбрасывается при перезагрузке, поэтому его нужно выставлять
перед каждым прогоном.    

Следствия для результатов:

- частоты SM под нагрузкой 2310–2392 MHz (паспортный boost — 2452 MHz), температура 46–71 °C;
- на больших конфигурациях (S ≥ 368, B ≥ 64) средняя мощность упирается в лимит: 243–261 W, медиана 248 W. Энергия там растёт ≈ пропорционально времени;
- мощность простоя GPU — 45,6 W; она входит в энергию (энергия измеряется для всего GPU).

**OOM не наблюдался.** Все 132 конфигурации отработали со статусом `OK`. Максимум памяти —
4366 MiB (S=512, B=256)

## Протокол

- `eval()` + `torch.inference_mode()`, FP32; `cudnn.benchmark = False`, TF32 выключен (cuDNN и matmul).
- Вход — случайный тензор `B × 3 × S × S`.
- Сетка: S ∈ {32, 64, 128, 224, 256, 384, 512} ∪ {112, 272, 368, 448}; B ∈ {1, 2, 4, …, 256} ∪ {33, 229, 252}.
  Случайные значения выбраны с `seed = 2026`. Точка считается валидационной (`is_validation = 1`),
  если случайно выбраны её S или B: 63 точки train, 69 validation.
- **Latency** — медиана wall-clock одного прохода (`synchronize` до и после), 10 прогревочных + до 100 повторов.
  Рядом записана медиана GPU-времени по CUDA events (`latency_gpu_s`).
- **Memory** — `torch.cuda.max_memory_allocated()` за один проход. Включает веса, входной тензор
- **Energy** — счётчик энергии NVML (`nvmlDeviceGetTotalEnergyConsumption`) на окне ≥ 2 с
  непрерывных проходов, делённый на число проходов.
- Перед сеткой — 3 с прогрева GPU; OOM ловится и записывается как `status = OOM`.

## Воспроизведение

```bash
pip install -r requirements.txt          # torch — сборка под CUDA
python measure.py                        # сетка -> results/measurements.csv, results/env.json
python measure.py --resume               # досчитать, если прогон прервался
python calibrate.py                      # фит theta -> results/theta.json
python plot_measurements.py              # графики -> results/figures/
python profile_model.py --S 224 --B 32   # трасса torch.profiler -> results/traces/
```

## Структура

```
hw1/
├── README.md
├── hw1_handwritten.pdf      # рукописные выводы формул (TODO)
├── models.py                # сеть
├── equations.py             # flops(), memory(), latency(), energy()
├── measure.py               # замеры
├── calibrate.py             # фит theta
├── plot_measurements.py     # графики
├── profile_model.py         # трассы torch.profiler по слоям
└── results/
    ├── measurements.csv
    ├── env.json
    ├── theta.json
    ├── traces/
    └── figures/
```

## Результаты

TODO

## Выводы

TODO
