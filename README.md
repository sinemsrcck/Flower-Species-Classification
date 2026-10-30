# Flower-Species-Classification
Flower species classification competition using PyTorch and transfer learning.

## Kurulum

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Veri setini indirip aç:

```bash
curl -L -o data/flower_data.tar.gz https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz
tar -xzf data/flower_data.tar.gz -C data/
curl -L -o data/cat_to_name.json https://raw.githubusercontent.com/udacity/aipnd-project/master/cat_to_name.json
```

Bu işlemler sonunda `data/train`, `data/valid`, `data/test` (her biri 102 sınıf
klasörü) ve `data/cat_to_name.json` oluşur. `data/` git'e eklenmez (`.gitignore`).

## Takım Görev Dağılımı

| Kişi | Görev | Teslimler |
|---|---|---|
| 1 | Ana Model ve Eğitim Altyapısı (EfficientNet-B0, veri hazırlama) | `dataset.py`, `efficientnet.ipynb`, `efficientnet_best.pth` |
| 2 | Model Optimizasyonu ve Final Entegrasyon (MobileNetV3-Large) | `mobilenet.ipynb`, `evaluate.py`, `predict.py`, `mobilenet_best.pth` |
| 3 | Genelleme ve Alternatif Model (CLIP) | `clip.ipynb`, `generalization_tests.py`, `results/` |
| 4 | Güçlü Model ve Performans Analizi (EfficientNetV2-S) | `efficientnetv2.ipynb`, `metrics.py`, `efficientnetv2_best.pth` |

### 3. Kişi — Genelleme ve Alternatif Model (CLIP)

CLIP (`openai/clip-vit-base-patch32`) ile zero-shot/few-shot sınıflandırma,
veri setinde olmayan türlerle genelleme testi, görüntü bozulmalarına
dayanıklılık testi ve ana modellerle karşılaştırma.

```bash
# Tüm deneyleri çalıştır (clip.ipynb'nin zero-shot/few-shot adımları ayrıca çalıştırılmalı)
jupyter nbconvert --to notebook --execute --inplace clip.ipynb

# OOD / bozulma / karşılaştırma testleri
python generalization_tests.py --task all
```

Sonuçlar `results/` altında: `clip_metrics.json`, `clip_few_shot_results.json`,
`ood_results.json`, `corruption_robustness.json/png`, `comparison_table.csv`,
`comparison_chart.png`, `GENERALIZATION_REPORT.md`.

Diğer kişilerin `results/<model>_metrics.json` dosyalarını (aynı şema:
`model_name`, `accuracy`, `inference_ms_per_image`, `model_size_mb`, `device`)
eklemesiyle `--task compare` tam karşılaştırma tablosunu otomatik üretir.

---

## 2. Kişi — MobileNetV3-Large ve Final Model Entegrasyonu

### MobileNetV3-Large

MobileNetV3-Large modeli, Oxford Flowers 102 veri seti üzerinde
transfer learning kullanılarak eğitildi.

Modelin PyTorch ve ONNX sürümleri karşılaştırıldı ve
CPU inference performansı değerlendirildi.

Sonuçlar:

| Metrik | Değer |
|---|---|
| Validation Accuracy | 96.94% |
| Test Accuracy | 94.87% |
| ONNX CPU Inference | 9.14 ms/image |
| ONNX Model Size | 16.80 MiB |

Test doğruluğu, 819 test görüntüsü üzerinde bağımsız olarak hesaplandı.

### Final Model Selection

Four models were compared using classification accuracy,
CPU inference latency, and model size.

| Model | Test Accuracy | CPU Latency | Model Size |
|---|---:|---:|---:|
| EfficientNet-B0 | 99.02% | 17.83 ms | 15.78 MiB |
| MobileNetV3-Large | 94.87% | 9.14 ms | 16.80 MiB |
| EfficientNetV2-S | 98.17% | 108.59 ms | 77.32 MiB |
| CLIP ViT-B/32 (zero-shot) | 66.54% | 56.58 ms | Not reported |

Note: Some accuracy values and CLIP CPU latency are teammate-reported.
CPU benchmarks were not all performed on identical hardware.

EfficientNet-B0 was selected as the final inference model
because of its high accuracy and efficient CPU performance.

### Final Inference Pipeline

The final prediction pipeline uses ONNX Runtime and supports
CPU-only inference.

Required files:

- `predict.py`
- `predict_shared.py`
- `shared_model_loader.py`
- `model_registry.json`
- `class_to_idx.json`
- `onnx/efficientnet_b0_fp32.onnx`

Run inference:

```bash
python predict.py --image path/to/flower.jpg --top-k 3