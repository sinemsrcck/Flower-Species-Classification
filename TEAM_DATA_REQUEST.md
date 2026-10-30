# Takım Arkadaşlarından İstenecekler (Model Entegrasyonu için)

Final paketi birleştirirken (karşılaştırma + `predict.py`) model dosyasıyla birlikte
aşağıdakilere ihtiyaç var. Bunlar olmadan model doğru yüklenip test edilemeyebilir.

## 1. Model dosyası
`.pth` dosyası (zaten teslim listesinde var).

## 2. Model mimarisi tanımı
`.pth` sadece ağırlıkları mı içeriyor (`state_dict`) yoksa `torch.save(model, ...)`
ile tüm model mi kaydedildi?

- Sadece `state_dict` ise: modeli yeniden oluşturan kod satırını paylaşın
  (örn. `efficientnet_b0(num_classes=102)`, hangi kütüphane/sürüm).

## 3. Preprocessing (inference için — training augmentation değil)

- Resize / crop boyutu (224x224 mi, başka bir şey mi?)
- Normalize mean/std değerleri:
  - ImageNet default mı: mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`
  - yoksa kendi hesapladığınız değerler mi?
- RGB sırası, başka bir ön işlem adımı var mı (center crop vb.)?

**Neden önemli:** Hocanın da belirttiği gibi en sık hata, inference preprocessing'in
training'dekiyle uyuşmaması.

## 4. Sınıf indeks → isim eşlemesi

Modelin çıkış nöronlarının hangi sırayla hangi sınıfa karşılık geldiğini gösteren
`class_to_idx` sözlüğü (veya eşdeğeri).

**Dikkat edilmesi gereken tuzak:** Train klasörünü PyTorch'un `ImageFolder`'ı ile
yüklediyseniz, klasör isimleri **alfabetik (string) sıraya** göre diziliyor —
yani "10", "11" gibi isimler "2"den önce geliyor. Bu, `cat_to_name.json`'daki
numaralarla elle eşleştirilmediyse modelin yanlış isimler döndürmesine yol açar
(sessiz bir hata, test sırasında fark edilmeyebilir).

## Özet — Paylaşılması Gerekenler

| # | Ne | Neden |
|---|---|---|
| 1 | `.pth` model dosyası | Teslim listesinde zaten var |
| 2 | Model mimarisi tanım kodu (state_dict ise) | Modeli yeniden oluşturmak için |
| 3 | Resize/crop boyutu + normalize mean/std | Preprocessing uyuşmazlığını önlemek için |
| 4 | `class_to_idx` eşlemesi | ImageFolder alfabetik sıralama tuzağını önlemek için |
