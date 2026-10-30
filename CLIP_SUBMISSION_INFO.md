# 3. Kişi (CLIP) — Model Bilgileri

Arkadaşımızın istediği karşılaştırma bilgileri. **Not:** CLIP, diğer modellerden
farklı olarak eğitilmiş bir model değil — hazır (pretrained) bir model olduğu
için "model dosyası" ve "sınıf eşlemesi" kısımları biraz farklı çalışıyor, aşağıda
açıklandı.

## 1. Model dosyası

Yerel bir `.pth` dosyası **yok** — CLIP, HuggingFace Hub'dan doğrudan indirilen
hazır (pretrained) bir model, hiç fine-tuning yapılmadı:

```python
from transformers import CLIPModel, CLIPProcessor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
```

Model kimliği: **`openai/clip-vit-base-patch32`**. Karşılaştırma ortamında bu
satırla otomatik indirilip yüklenebilir (internet gerektirir, ~605 MB).

## 2. Model mimarisi ve yükleme kodu

Repo'da `src/clip_utils.py` içinde hazır:
- `load_clip()` → model + processor + device yükler
- `build_class_prompts(cat_to_name)` → sınıf adlarını metin prompt'una çevirir
- `encode_text_prompts(...)`, `encode_images(...)` → embedding çıkarımı
- Tahmin mantığı: görüntü embedding'i ile 102 sınıfın metin embedding'i
  arasında cosine similarity, en yüksek olan seçilir (klasik sınıflandırıcı
  "head"i yok).

## 3. Preprocessing ayarları

CLIP'in kendi `CLIPProcessor`'ı tüm preprocessing'i otomatik yapıyor:

| Ayar | Değer |
|---|---|
| Resize | shortest edge → 224 |
| Center crop | 224 × 224 |
| Normalize mean | `[0.48145466, 0.4578275, 0.40821073]` |
| Normalize std | `[0.26862954, 0.26130258, 0.27577711]` |
| Renk uzayı | RGB |

**Dikkat:** Bu değerler ImageNet default'larından (`[0.485,0.456,0.406]` /
`[0.229,0.224,0.225]`) **farklı** — CLIP kendi eğitim verisine özel normalize
değerleri kullanıyor.

## 4. Sınıf eşlemesi

Geleneksel `class_to_idx` yok çünkü CLIP bir sınıflandırıcı head'i değil, metin
prompt'larıyla karşılaştırma yapıyor. Eşleme şöyle:

```python
class_ids = sorted(cat_to_name.keys(), key=int)  # ["1","2",...,"102"]
prompts = [f"a photo of a {cat_to_name[c]}, a type of flower" for c in class_ids]
```

Yani sınıf sırası `cat_to_name.json`'daki anahtarların **sayısal sıralamasıyla**
aynı (1, 2, 3, ..., 102) — ImageFolder'ın alfabetik sıralama tuzağı burada
geçerli değil, çünkü klasörlerden değil doğrudan `cat_to_name.json`'dan
okunuyor.

## 5. Accuracy / CPU inference sonuçları

(`results/clip_metrics.json`'dan, CPU'da — GPU'suz Intel i7 senaryosuna uygun)

| Metrik | Değer |
|---|---|
| Accuracy (zero-shot, test split, n=819) | **%66.54** |
| Inference hızı (CPU, ms/görüntü) | **84.30 ms** |
| Model boyutu | **577.08 MB** |
| Yarışma formülüyle final score (Tmax=50ms, Smax=35MB) | **55.66 / 100** |

**Özet:** CLIP, hiç eğitim verisi görmeden makul bir accuracy veriyor ama boyut
(577MB >> 35MB benchmark) ve hız (84ms >> 50ms benchmark) kriterlerinde çok
dezavantajlı. Yarışma teslimi için **önerilmiyor** — asıl değeri genelleme
(veri setinde olmayan türler) ve bozulma dayanıklılığı analizinde
(`results/GENERALIZATION_REPORT.md`).
