"""
Ortak veri yükleme kodu - Flower Competition (102 sınıf)

Sınıf sırası SAYISAL: klasör "1" -> 0, "2" -> 1, ..., "102" -> 101
Takımdaki herkes bu dosyayı kullanmalı ki etiketler aynı olsun.
"""
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# --- Sabit ayarlar ---
IMG_SIZE = 224        # modele giren görüntü boyutu
RESIZE_SIZE = 256     # kırpmadan önce küçültülen boyut
MEAN = [0.485, 0.456, 0.406]   # ImageNet ortalaması
STD = [0.229, 0.224, 0.225]    # ImageNet standart sapması


# --- 1) Sınıf sırası düzeltmesi ---
class NumericImageFolder(datasets.ImageFolder):
    """Normal ImageFolder sınıfları alfabetik sıralar ('1','10','100'...).
    Bu sınıf onları sayısal sıralar ('1','2','3'...)."""

    def find_classes(self, directory):
        classes = sorted(
            (d.name for d in Path(directory).iterdir() if d.is_dir()),
            key=int,
        )
        class_to_idx = {c: i for i, c in enumerate(classes)}
        return classes, class_to_idx


# --- 2) Görüntü dönüşümleri ---
def get_transforms(train: bool):
    if train:
        # Eğitim: her seferinde rastgele değişiklik (augmentation)
        return transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.3, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    # Valid / test / tahmin: rastgelelik YOK, her seferinde aynı sonuç
    return transforms.Compose([
        transforms.Resize(RESIZE_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


# --- 3) Herkesin çağıracağı tek fonksiyon ---
def get_dataloaders(data_dir, batch_size=32, num_workers=2):
    """data_dir içinde train/, valid/, test/ klasörleri olmalı.
    Döndürür: train_loader, valid_loader, test_loader, classes"""
    data_dir = Path(data_dir)

    train_ds = NumericImageFolder(data_dir / "train", transform=get_transforms(True))
    valid_ds = NumericImageFolder(data_dir / "valid", transform=get_transforms(False))
    test_ds = NumericImageFolder(data_dir / "test", transform=get_transforms(False))

    # Güvenlik kontrolü: üç bölümde de aynı 102 sınıf olmalı
    assert len(train_ds.classes) == 102, "102 sınıf bekleniyordu"
    assert train_ds.classes == valid_ds.classes == test_ds.classes, "Sınıflar uyuşmuyor"

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin)

    return train_loader, valid_loader, test_loader, train_ds.classes
