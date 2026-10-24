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
