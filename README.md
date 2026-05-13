# Endüstriyel Kusur Tespiti

Transfer Öğrenmesi + MVTec AD Veri Seti üzerinde Anomali Tespiti  
ResNet50 özellik çıkartıcı · PatchCore bellek deposu · Gauss temel yöntemi karşılaştırması

---

## Genel Bakış

Bu proje, denetimsiz bir endüstriyel yüzey kusuru tespit sistemi uygular.  
Önceden eğitilmiş bir ResNet50, normal eğitim görüntülerinden yama seviyesi özelliklerini çıkarır ve bunları bir bellek deposunda (PatchCore) saklar. Test sırasında, anomali skorları tüm yamalar arasındaki maksimum en yakın komşu mesafesi olarak hesaplanır.

Karşılaştırma için klasik bir Gauss bulanıklığı + istatistik temel yöntemi dahil edilmiştir.

**MVTec AD üzerindeki Sonuçlar — tüm 15 kategorisi (gerçek veriler):**

| Kategori     | Temel AUROC | PatchCore AUROC | PatchCore F1 |
| ------------ | ----------- | --------------- | ------------ |
| şişe         | 0.351       | **0.998**       | 0.976        |
| kablo        | 0.537       | 0.843           | 0.584        |
| kapsül       | 0.612       | 0.754           | 0.463        |
| halı         | 0.196       | 0.933           | 0.893        |
| ızgara       | 0.942       | 0.617           | 0.100        |
| fındık       | 0.085       | **1.000**       | **1.000**    |
| deri         | 0.348       | **0.999**       | 0.974        |
| metal_somun  | 0.562       | 0.977           | 0.955        |
| hap          | 0.375       | 0.792           | 0.604        |
| vida         | **1.000**   | 0.808           | 0.636        |
| karo         | 0.381       | 0.872           | 0.682        |
| diş_fırçası  | 0.436       | 0.900           | 0.824        |
| transistör   | 0.603       | 0.865           | 0.686        |
| ağaç         | 0.580       | 0.997           | 0.974        |
| fermuvar     | 0.459       | 0.916           | 0.904        |
| **Ortalama** | **0.498**   | **0.885**       | **0.757**    |

---

## Proje Yapısı

```
Industrial-Defect-Detection/
├── config.py            # Tüm hiperparametreler ve yollar
├── data_loader.py       # Veri seti yükleme + kukla veri oluşturucu
├── baseline_method.py   # Gauss bulanıklığı temel detektörü
├── patchcore_model.py   # PatchCore anomali tespit modeli
├── evaluation.py        # Metrikler, çizimler, sonuç kaydetme
├── main.py              # Giriş noktası
├── requirements.txt
├── dataset/
│   ├── bottle/
│   ├── cable/
│   ├── ...              # 15 MVTec AD kategorisi
│   └── metal_nut/
│       ├── train/good/  # Normal eğitim görüntüleri
│       └── test/
│           ├── good/    # Normal test görüntüleri (etiket=0)
│           ├── bent/    # Kusur alt kategorileri (etiket=1)
│           └── ...
└── output/
    ├── results.csv                    # Tek kategori çalıştırması
    ├── all_categories_summary.csv     # Tüm kategoriler çalıştırması
    ├── memory_bank.npy
    ├── figures/
    │   ├── roc_curves.png
    │   ├── score_distributions.png
    │   └── model_comparison.png
    ├── bottle/                        # Kategori başına çıktılar
    │   ├── results.csv
    │   └── figures/
    └── ...
```

---

## Kurulum

```bash
pip install -r requirements.txt
```

> **GPU yok mu?** Kod CPU'da çalışır ancak önemli ölçüde daha yavaş olur.  
> Ücretsiz GPU erişimi için [Google Colab](https://colab.research.google.com) kullanın.

---

## Kullanım

### Sentetik veri ile hızlı test (indirme gerekli değil)

```bash
python main.py --dummy
```

Otomatik olarak 70 eğitim + 45 test görüntüsü oluşturur ve tam işlemi çalıştırır.

### Tek kategori (varsayılan: metal_nut)

```bash
python main.py
python main.py --few-shot   # ayrıca %30 az-şantiye senaryosu çalıştır
```

Kategoriyi değiştirmek için [config.py](config.py) dosyasında `DATA_CATEGORY` değerini ayarlayın.

### Tüm 15 kategori

```bash
python main.py --all-categories
```

Her MVTec AD kategorisinde Temel + PatchCore çalıştırır ve birleştirilmiş bir özeti `output/all_categories_summary.csv` dosyasına kaydeder. Her kategori ayrıca `output/` altında kendi alt klasörünü alır.

### CLI seçenekleri

| Bayrak             | Açıklama                                                        |
| ------------------ | --------------------------------------------------------------- |
| `--all-categories` | Tüm 15 MVTec AD kategorisinde çalıştır                          |
| `--dummy`          | MVTec AD yüklemek yerine sentetik görüntüler oluştur            |
| `--few-shot`       | Ayrıca PatchCore'u eğitim verilerinin %30'u ile çalıştır        |
| `--skip-baseline`  | Gauss temelini atla (zaman kaydeder)                            |
| `--coreset FLOAT`  | PatchCore için çekirdek alt örnekleme oranı (varsayılan: `0.1`) |

---

## Nasıl Çalışır

### 1. Transfer Öğrenmesi (ResNet50)

ImageNet'te önceden eğitilmiş ResNet50, dondurulmuş bir özellik çıkartıcı olarak kullanılır.  
`layer2` ve `layer3` çıkışları doku ve anlamsal yapı yakalamak için birleştirilir.

### 2. PatchCore

Normal eğitim setindeki tüm yama düzeyi özellikler bir bellek deposunda saklanır.  
Bir açgözlü çekirdek alt örnekleme adımı bellek depo boyutunu azaltır (varsayılan %10).  
Test anomali skoru = tüm görüntü yamaları üzerinde maksimum en yakın komşu mesafesi.

### 3. Gauss Temel Yöntemi

Her eğitim görüntüsü bulanıklaştırılır ve piksel başına ortalama/std haritası oluşturulur.  
Anomali skoru = ortalamadan piksel düzeyinde normalleştirilmiş ortalama mutlak sapma.

### 4. Az-Şantiye Senaryosu

PatchCore, yalnızca eğitim görüntülerinin %30'u kullanılarak yeniden çalıştırılır (~21 görüntü).  
Endüstri 4.0 ayarlarında ilgili hızlı uyum yeteneğini gösterir.

---

## Çıktı Dosyaları

| Dosya                                    | Açıklama                                                            |
| ---------------------------------------- | ------------------------------------------------------------------- |
| `output/results.csv`                     | AUROC, F1, Hassasiyet, Geri Çağırma (tek-kategori çalıştırması)     |
| `output/all_categories_summary.csv`      | Tüm 15 kategori için birleştirilmiş sonuçlar                        |
| `output/memory_bank.npy`                 | Kaydedilen PatchCore bellek deposu                                  |
| `output/figures/roc_curves.png`          | Tüm modeller için ROC eğrileri                                      |
| `output/figures/score_distributions.png` | Normal vs kusur skoru histogramları                                 |
| `output/figures/model_comparison.png`    | Bar grafiği: AUROC & F1 karşılaştırması                             |
| `output/<category>/`                     | Kategori başına sonuçlar ve şekiller (tüm-kategoriler çalıştırması) |

---

## Konfigürasyon

Veri seti kategorisini, modeli veya eşikleri değiştirmek için [config.py](config.py) dosyasını düzenleyin:

```python
DATA_CATEGORY = "metal_nut"   # MVTec AD kategorisi (tek-kategori modunda kullanılır)
MODEL_NAME    = "resnet50"
IMAGE_SIZE    = 224
BATCH_SIZE    = 32
DEVICE        = "cuda"        # veya "cpu"
```

Bayrağa dokunmadan farklı bir tek kategori çalıştırmak için:

```bash
# config.py'yi düzenleyin: DATA_CATEGORY = "carpet"
python main.py

# veya tüm 15'ini bir kerede çalıştırmak için --all-categories kullanın
python main.py --all-categories
```

---

## Gereklilikler

```
torch==1.13.1
torchvision==0.14.1
opencv-python==4.8.1.78
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
matplotlib==3.7.2
Pillow==10.0.0
tqdm==4.66.1
scipy==1.11.1
```
