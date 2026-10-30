# Flower-Species-Classification
Flower species classification competition using PyTorch and transfer learning.

## Kurulum

> **Python sürümü:** 3.10–3.12 önerilir (`requirements.txt` kurulumu 3.11'de doğrulandı).
> `torch==2.8.0` Python 3.14 için kurulamıyor.

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
| EfficientNetV2-S | 98.17% | 108.59 ms | 77.32 MiB (ONNX fp32); teslim edilen `.pth`: 39.28 MB (fp16) |
| CLIP ViT-B/32 (zero-shot) | 66.54% | 56.58 ms | 577.08 MB |

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
python predict.py --image path/to/flower.jpg --top-k 3```

---

## 4. Kişi — EfficientNetV2-S ve Performans Analizi

ImageNet ağırlıklı **EfficientNetV2-S** (`torchvision`), ortak `dataset.py` ile
fine-tune edildi (gövde düşük LR, sınıflandırıcı 10× LR, AMP, label smoothing 0.1,
OneCycle LR, early stopping). Tüm adımlar ve çıktılar `efficientnetv2.ipynb` içinde;
metrik/puan fonksiyonları `metrics.py`'de.

**Model dosyası:** [efficientnetv2_best.pth](https://drive.google.com/file/d/1Pxuz-DyhDthYKstTQ_-KiJXFqRAU5Svq/view?usp=sharing)
(39.3 MB, fp16 ağırlıklar; checkpoint formatı MobileNetV3 ile aynı: `architecture`,
`class_to_idx`, `model_state_dict`, `precision`). Preprocessing: Resize 256 →
CenterCrop 224 → ImageNet mean/std. Sınıf sırası `dataset.py` ile aynı ("1"→0 … "102"→101).

### Hyperparameter denemeleri (`results/efficientnetv2_hparams.csv`)

| Learning rate | Batch size | Optimizer | Valid accuracy |
|---|---|---|---|
| **1e-3** | **32** | **AdamW** | **%99.39** |
| 1e-3 | 64 | AdamW | %99.14 |
| 1e-3 | 16 | AdamW | %98.90 |
| 1e-3 | 32 | Adam | %98.78 |
| 1e-4 | 32 | AdamW | %98.53 |
| 3e-4 | 32 | AdamW | %98.53 |
| 1e-2 | 32 | SGD | %98.29 |

Her ayar 3 epoch eğitildi; en iyi ayarla final eğitim yapıldı (18. epoch'ta early stopping).

### Test sonuçları (819 görüntü, `results/efficientnetv2_metrics.json`)

| Metrik | Değer |
|---|---|
| Test accuracy | **%98.17** (15 hata) |
| Top-5 accuracy | %100 |
| Precision / Recall / F1 (macro) | 0.980 / 0.983 / 0.979 |
| CPU inference (PyTorch, 1 thread, batch 1) | 119.6 ms/görüntü |
| Model boyutu | 39.3 MB |
| Tahmini yarışma puanı | **82.72** (accuracy 98.17, hız 41.8, boyut 89.1) |

Grafikler: `efficientnetv2_training.png`, `efficientnetv2_confusion_matrix.png`,
`efficientnetv2_errors.png`.

### CPU süresi ve boyut dezavantajı

EfficientNetV2-S ~20M parametreli büyük bir model. CPU'da **119.6 ms**, yani
yarışma sınırının (Tmax = 50 ms) **2 katından fazla**; hız puanı 41.8'e düşüyor.
Dosya **39.3 MB** ile Smax = 35 MB'ı aşıyor (boyut puanı 89.1). Fp16 kayıt boyutu
yarıya indirdi (fp32 ~78 MB olurdu), ama yine de sınırın üstünde.

Sonuç: accuracy yüksek olsa da (%98.17), **EfficientNet-B0 bu veri setinde hem daha
doğru (%99.02) hem çok daha hızlı ve küçük**. Büyük model ek doğruluk sağlamıyor;
EfficientNetV2-S bu yüzden final model değil, takımın üst-sınır referans modeli.

**Çözünürlük denemesi** (`results/efficientnetv2_resolution.csv`): Aynı model 192 px
girişle test edildiğinde accuracy %98.5'e çıktı, CPU süresi 87 ms'ye düştü ve puan
**86.86** oldu. Daha düşük çözünürlük bu veri setinde doğruluğu düşürmeden hızı artırıyor.

### Hata analizi

- **En zor tür camellia** (recall 0.56): 9 görüntünün 4'ü yanlış (mallow ×2,
  watercress, spring crocus). Renkleri ve yaprak formu çok çeşitli.
- Hatalar çoğunlukla **renk + şekil benzerliğinden** (corn poppy → watercress,
  californian poppy → buttercup) ve **alışılmadık açı/kadrajdan** (colt's foot → ball moss) kaynaklanıyor.
- 15 hatanın 10'unda model güveni %60'ın altında; top-5 accuracy %100.
- Train/valid eğrileri yakın, belirgin overfitting yok.
