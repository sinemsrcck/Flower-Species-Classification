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
- **Metrikler:** accuracy, inference hızı (ms/görüntü, batch_size=1, warm-up
  sonrası ortalama), model boyutu (parametre belleği, MB) —
  `results/clip_metrics.json` şeması diğer takım üyelerinin `evaluate.py`
  çıktısıyla aynıdır, böylece `--task compare` ile otomatik birleştirilebilir.

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

| Model | Accuracy | ms/görüntü | Boyut (MB) |
|---|---|---|---|
| CLIP-ViT-B/32 (zero-shot) | 0.6654 | ~30-40 (MPS, run'lar arası değişken) | 577.08 |
| EfficientNet-B0 (1. kişi) | *bekleniyor* | | |
| MobileNetV3 (2. kişi) | *bekleniyor* | | |
| EfficientNetV2-S (4. kişi) | *bekleniyor* | | |

Diğer takım üyelerinin `results/<model>_metrics.json` dosyaları (aynı şema)
eklendiğinde, `python generalization_tests.py --task compare` tekrar
çalıştırılarak tam karşılaştırma tablosu ve grafiği otomatik üretilir.

CLIP'in model boyutu (577 MB), hafif mimarilere (tipik 15-20 MB) kıyasla
**~30 kat** daha büyük — zero-shot esnekliğinin (sıfır eğitim verisi
gerektirmemesi) bedeli, boyut ve hız kriterlerinde dezavantaj.

*Not: `inference_ms_per_image` değeri, MPS (Apple Silicon GPU) üzerinde
run'dan run'a ~10-15 ms değişkenlik gösterebiliyor (sistem yüküne bağlı);
güncel değer için `results/clip_metrics.json`'a bakın.*

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
