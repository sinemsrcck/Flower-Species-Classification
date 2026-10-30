# Genelleme ve Alternatif Model Raporu — CLIP (3. Kişi)

Bu rapor, SENG 481 çiçek sınıflandırma yarışması kapsamında 3. kişi görevi olan
"Genelleme ve Alternatif Model" çalışmasının yöntemini ve sonuçlarını özetler.
Tüm deneyler `clip.ipynb` ve `generalization_tests.py` içinde yeniden
üretilebilir; bu dosya sayısal bulguların ve yorumların özetidir.

## Yöntem

- **Model:** CLIP ViT-B/32 (`openai/clip-vit-base-patch32`, HuggingFace
  `transformers`), hiç fine-tuning yapılmadan.
- **Veri seti:** Udacity flower_data (Oxford 102 Flowers), `data/train`,
  `data/valid`, `data/test` — 102 sınıf, test split 819 görüntü.
- **Zero-shot sınıflandırma:** Her sınıf adı `"a photo of a {isim}, a type of
  flower"` şablonuna konup CLIP text encoder'dan geçirildi; görüntü embedding'i
  ile cosine similarity'si en yüksek olan sınıf seçildi.
- **Few-shot:** Dondurulmuş CLIP görüntü embedding'leri üzerinde, sınıf başına
  *k* ∈ {1, 2, 4, 8, 16} etiketli örnekle logistic regression (linear probe)
  eğitildi.
- **OOD (veri setinde olmayan türler):** 102 sınıfla adı/çakışan ailesi
  örtüşmeyen 8 tür (chrysanthemum, zinnia, gardenia, peony, hydrangea,
  lavender, begonia, jasmine) programatik olarak seçildi (bkz.
  `src/ood_species.py`), her biri için Wikimedia Commons'tan 4'er CC lisanslı
  görsel indirildi (`data/ood_samples/attribution.json`'da kaynak/lisans/yazar
  bilgisi mevcuttur).
- **Bozulma dayanıklılığı:** Test setinden rastgele 200 görüntü üzerinde 7
  bozulma türü (gaussian_noise, gaussian_blur, brightness, contrast, rotation,
  jpeg_compression, occlusion) × 5 şiddet seviyesi (1=hafif, 5=şiddetli)
  uygulanarak zero-shot accuracy ölçüldü (bkz. `src/corruptions.py`).
- **Metrikler:** accuracy, inference hızı (ms/görüntü, **CPU, 1 thread, 10
  ısınma + 100 ölçüm görüntüsü** — kişi 1 ve 4'ün `metrics.py`'deki
  `measure_cpu_latency()` yöntemiyle birebir aynı, takım genelinde CPU
  sayılarının karşılaştırılabilir olması için), model boyutu (parametre
  belleği, MB) — `results/clip_metrics.json` şeması diğer takım üyelerinin
  `evaluate.py`/`metrics.py` çıktısıyla aynıdır, böylece `--task compare` ile
  otomatik birleştirilebilir.
- **Yarışma skorlama formülü** (`Flower_Recognition_Competition_Guidelines_v1.pdf`'den):
  `FINAL SCORE = 0.60×Accuracy(%) + 0.25×Speed_score + 0.15×Size_score`, burada
  `Speed_score = min(100, Tmax/T × 100)` (Tmax=50 ms, GPU'suz Intel i7 + 16GB RAM
  üzerinde ölçülecek) ve `Size_score = min(100, Smax/S × 100)` (Smax=35 MB). Bu
  formül `clip.ipynb` ve `generalization_tests.py --task compare` içinde
  otomatik hesaplanıyor.
- **Ölçüm düzeltme geçmişi:** İlk denemede inference hızı yanlışlıkla Apple
  GPU'da (`mps`) ölçülmüştü; sonra CPU'ya çevrildi ama thread sayısı
  sabitlenmemişti (çoklu-thread, n=50). Kişi 4'ün `metrics.py` dosyasını
  incelerken takımın **1 thread, n=100, warmup=10** standardını kullandığı
  görüldü — CLIP de bu standarda göre yeniden ölçüldü. İlginç bulgu: 1 thread'de
  CLIP daha **hızlı** çıktı (56.58ms) çoklu-thread'li ölçüme (84.30ms) göre —
  küçük batch=1 işlemlerde thread senkronizasyon overhead'i çoklu-thread'i
  yavaşlatabiliyor. MPS değeri bilgi amaçlı `inference_ms_per_image_mps`
  alanında ayrıca saklanıyor.

## Sonuçlar

### Zero-shot / Few-shot (`results/clip_metrics.json`, `results/clip_few_shot_results.json`)

| Yöntem | Test accuracy |
|---|---|
| Zero-shot | **0.6654** |
| Few-shot k=1 | 0.5653 |
| Few-shot k=2 | 0.7155 |
| Few-shot k=4 | 0.7741 |
| Few-shot k=8 | 0.8046 |
| Few-shot k=16 | 0.8718 |

Zero-shot, hiçbir eğitim görmeden 102 ince taneli (fine-grained) çiçek sınıfı
arasında %66.5 accuracy'ye ulaşıyor — CLIP'in genel görsel-dil bilgisinin bu
alanda bile işe yaradığını gösteriyor. İlginç bir bulgu: **1-shot, zero-shot'un
altında kalıyor** (tek örnekle eğitilen doğrusal sınıflandırıcı gürültüye çok
duyarlı); 2-shot'tan itibaren zero-shot'u geçip 16-shot'ta %87.2'ye çıkıyor
(bkz. `results/few_shot_accuracy.png`).

### OOD türler (`results/ood_results.json`)

- 32 OOD görüntüsünün tamamında CLIP, 102 bilinen sınıftan görsel olarak en
  benzer olanını seçiyor (ör. chrysanthemum→oxeye daisy, gardenia→magnolia,
  begonia→moon orchid).
- Ortalama confidence: **OOD = 0.302**, **in-distribution = 0.334**. Fark
  küçük ama tutarlı — CLIP net bir "bilmiyorum" sinyali vermiyor, ama
  confidence'ı biraz düşürüyor. Pratik çıkarım: düşük bir confidence eşiği
  (~0.30 altı) "bu tür muhtemelen veri setinde yok" tespiti için zayıf ama
  kullanılabilir bir sinyal olabilir.

### Bozulma dayanıklılığı (`results/corruption_robustness.json/png`)

Clean accuracy (n=200): **0.6700**.

| Bozulma | Severity 1 | Severity 5 | Düşüş |
|---|---|---|---|
| rotation | 0.665 | 0.660 | ~yok |
| jpeg_compression | 0.665 | 0.650 | ~yok |
| gaussian_noise | 0.650 | 0.620 | hafif |
| occlusion | 0.645 | 0.530 | orta |
| brightness | 0.655 | 0.460 | yüksek |
| contrast | 0.640 | 0.400 | çok yüksek |
| gaussian_blur | 0.680 | 0.390 | çok yüksek |

CLIP, **döndürme** ve **JPEG sıkıştırma** artefaktlarına karşı neredeyse tam
dayanıklı; buna karşın **bulanıklık (blur)** ve **düşük kontrast** en şiddetli
seviyede accuracy'yi neredeyse yarıya indiriyor. Hocanın hazırlayacağı hidden
test setinde görüntü kalitesi (netlik, kontrast) CLIP performansı için
rotasyon/sıkıştırmadan çok daha kritik bir risk faktörü.

### Ana modellerle karşılaştırma (`results/comparison_table.csv/png`)

| Model | Accuracy (test) | ms/görüntü (CPU, 1 thread) | Boyut (MB) | Final Score* |
|---|---|---|---|---|
| CLIP-ViT-B/32 (zero-shot) | 0.6654 | 56.58 | 577.08 | **62.93** |
| EfficientNet-B0 (1. kişi, fine-tuned) | 0.9902 | 41.60 | 16.07 | **99.41** |
| MobileNetV3-Large (2. kişi, fine-tuned, ONNX) | 0.9487 | 9.14 | 16.81 | **96.92** |
| EfficientNetV2-S (4. kişi, fine-tuned) | 0.9817 | 119.62 | 39.28 | **82.72** |

\* `FINAL SCORE = 0.60×Accuracy(%) + 0.25×Speed_score + 0.15×Size_score` (Tmax=50ms, Smax=35MB — bkz. Yöntem). Dördü de kendi `results/<model>_metrics.json` dosyalarındaki accuracy/hız/boyut değerleriyle aynı formülle hesaplandı.

**Düzeltme notu:** Takımın ilk WhatsApp özetinde MobileNetV3 için "%96,94" test accuracy olarak paylaşılmıştı; kişi 2'nin resmi `mobilenetv3_test_metrics.json` dosyasına göre bu aslında **validation** accuracy'si — gerçek **test accuracy %94,87** (777/819 doğru). Tablo bu düzeltilmiş rakamla güncellendi.

Takımın dört modeli de artık `results/` altında — `generalization_tests.py --task
compare` her çalıştırıldığında tabloyu/grafiği otomatik günceller
(`results/comparison_table.csv`, `results/comparison_chart.png`).

CLIP'in model boyutu (577 MB), diğer üç modele (16-39 MB aralığı) kıyasla
**~15-36 kat** daha büyük. Yarışma formülüyle sıralama: **EfficientNet-B0
(99.41) > MobileNetV3 (96.92) > EfficientNetV2-S (82.72) > CLIP (62.93)**.
EfficientNetV2-S'in düşük skorunun sebebi CPU inference süresi (119.62ms,
`Tmax=50ms`'nin 2 katından fazla) — model boyutu (39.28 MB) ve accuracy
(%98.17) kendi başına iyi olsa da hız kriteri skoru ciddi düşürüyor. Bu
beklenen bir sonuç: EfficientNet-B0, MobileNetV3 ve EfficientNetV2-S bu 102
sınıf için özel olarak fine-tune edildi (MobileNetV3 ayrıca ONNX Runtime'a
dönüştürülerek CPU'da ekstra hızlandırılmış), CLIP ise hiç eğitilmeden genel
amaçlı bir model.

## Değerlendirme ve Öneriler

1. **CLIP, ana modellerle yarışmak için değil, tamamlayıcı bir referans/analiz
   aracı olarak değerli**: accuracy/hız/boyut üçlüsünde EfficientNet-B0 ve
   MobileNetV3 gibi fine-tune edilmiş hafif modellerin gerisinde kalması
   beklenir, ama hiç eğitim verisi olmadan makul bir taban çizgisi sunuyor.
2. **Genelleme gücü asıl katkısı**: veri setinde olmayan türlerde "çöküp"
   rastgele tahmin yapmak yerine görsel olarak mantıklı bir sınıfa yöneliyor —
   bu, eğitilmiş CNN'lerin (sadece 102 çıkışlı softmax) yapamayacağı bir şey.
3. **Hidden test seti için risk uyarısı**: eğer hocanın test görselleri düşük
   kaliteli/bulanık olursa, sadece ana modeller değil CLIP de ciddi performans
   kaybı yaşar — bu nedenle 1. ve 2. kişilerin augmentation stratejisinde blur
   ve kontrast varyasyonlarının yer alması önerilir.
4. **Few-shot bulgusu**: çok az etiketli veri olan senaryolarda (k≥2) CLIP
   tabanlı linear probe, hızlı ve ucuz bir alternatif olabilir; ama k=1 ile
   güvenilir değil.
5. **CLIP, takımın yarışmaya gönderilecek asıl modeli değil**: PDF'deki
   "students must submit a model that can be loaded and evaluated using the
   instructor's required interface" kuralı muhtemelen 2. kişinin
   `predict.py`/final paketine işaret ediyor. CLIP burada bir genelleme/analiz
   çalışması; yarışma formülüyle puanlandığında (62.93/100) takımın diğer üç
   modelinin gerisinde kalması beklenen ve sorunlu olmayan bir sonuç — CLIP'in
   katkısı accuracy/hız/boyut yarışında değil, genelleme ve dayanıklılık
   analizinde.
6. **Takımın dört modeli arasında final teslim için öneri**: yarışma
   skoruna göre sıralama EfficientNet-B0 (99.41) > MobileNetV3 (96.92) >
   EfficientNetV2-S (82.72) > CLIP (62.93). **EfficientNet-B0** en yüksek
   skoru alıyor (yüksek accuracy + düşük boyut + makul hız dengesi);
   **MobileNetV3 (ONNX)** çok yakın bir alternatif ve en düşük inference
   süresine sahip (9.14ms) — gerçek "Intel i7, no GPU" ortamında ONNX
   Runtime'ın ek hız avantajı riski azaltabilir. **EfficientNetV2-S** accuracy
   olarak güçlü (%98.17) ama CPU'da yavaş (119.62ms, `Tmax`'ın 2 katından
   fazla) kaldığı için final teslim için şu haliyle önerilmiyor — kişi 4'ün
   kendi `efficientnetv2_resolution.csv` denemesine göre 192px girdi
   çözünürlüğüyle hem accuracy (%98.5) hem hız (87ms) iyileşip final_score
   86.86'ya çıkıyor; bu konfigürasyon denenmeye değer.
