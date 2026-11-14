# 🩺 Mobile-PolypNet: Lightweight Colon Polyp Segmentation Network for Low-Resource Settings

Implementasi lengkap model deep learning **Mobile-PolypNet** untuk segmentasi semantik polip kolon (*colorectal polyp*) berbasis arsitektur *lightweight autoencoder* untuk lingkungan bersumber daya komputasi terbatas (*low-resource / edge / mobile settings*).

Implementasi ini merujuk secara penuh pada publikasi ilmiah:
> **Paper:** *Mobile-PolypNet: Lightweight Colon Polyp Segmentation Network for Low-Resource Settings*  
> **Penulis:** Ranit Karmakar & Saeid Nooshabadi (*Department of Electrical and Computer Engineering, Michigan Technological University, USA*)  
> **Jurnal:** *Journal of Imaging 2022, 8(6), 169*  
> **DOI:** [https://doi.org/10.3390/jimaging8060169](https://doi.org/10.3390/jimaging8060169)  
> **Official Repo:** [https://github.com/rkarmaka/Mobile-PolypNet](https://github.com/rkarmaka/Mobile-PolypNet)

---

## 📌 1. Latar Belakang & Ringkasan Metode

Kanker Kolorektal (*Colorectal Cancer / CRC*) merupakan salah satu jenis kanker paling mematikan di dunia. Polip kolon merupakan lesi awal yang berpotensi berkembang menjadi ganas jika tidak terdeteksi sejak dini. Segmentasi otomatis pada citra endoskopi/kolonoskopi sangat krusial dalam membantu klinisi.

Namun, model segmentasi *state-of-the-art* (seperti PraNet, ResUNet++, SFA, dll.) memiliki jumlah parameter sangat besar (20M–25M) dan kebutuhan FLOPs yang masif (50G–220G), menyulitkan implementasi pada perangkat portabel/mobile di klinik dengan fasilitas komputasi terbatas.

### 💡 Inovasi Utama Mobile-PolypNet:
1. **Inverted Residual Bottleneck Blocks**: Mengadaptasi blok inverted residual dari MobileNetV2 dengan kombinasi *Pointwise Convolution* (1×1 ekspansi), *Depthwise Separable Convolution* (3×3), *Pointwise Convolution* (1×1 kontraksi), serta *Residual Skip Connection*.
2. **Efisiensi Parameter Ekstrem**: Hanya membutuhkan **233,001 parameter terlatih** (total **246,617 parameter**), sekitar **~88× lebih hemat parameter** dan **~17× lebih efisien komputasi** (~2.0 G FLOPs pada resolusi 224×224) dibanding model PraNet.
3. **DICE Loss Function**: Menggunakan fungsi objektif berbasis koefisien Sørensen–Dice (Persamaan 1 pada paper) untuk mengatasi ketidakseimbangan kelas (*foreground* polip vs *background* dinding usus).

---

## 🏛️ 2. Arsitektur Model

Arsitektur Mobile-PolypNet mengadopsi struktur *Autoencoder / U-Net backbone* dengan *Skip Connections*:

```
                        [ Citra Input (224×224×3) ]
                                     │
          [ Conv2D (32, 3×3) + BN + ReLU6 ] 
                         │
          [ DepthwiseConv2D (3×3) + BN + ReLU6 ] 
                         │
          [ PointwiseConv2D (32, 1×1) + BN (Linear) ]
                         │ (Fitur x: 224×224×32)
       ┌─────────────────┴─────────────────────────────────────────┐
       │ (Stride 2)                                                │ (Skip Connection)
[ Level 1 Encoder: 3x Bottleneck (e:48, c:8) ] (112×112)           │
       │ (Stride 2)                                                │
[ Level 2 Encoder: 3x Bottleneck (e:96, c:16) ] (56×56)            │
       │ (Stride 2)                                                │
[ Level 3 Encoder: 3x Bottleneck (e:144, c:32) ] (28×28)           │
       │ (Stride 2)                                                │
[ Level 4 Encoder: 3x Bottleneck (e:144, c:32) ] (14×14)           │
       │ (Stride 2)                                                │
[ Level 5 Bottleneck: 6x Bottleneck (e:144, c:32) ] (7×7)          │
       │                                                           │
       │ (ConvTranspose2D s=2) + Concat(c4)                        │
[ Level 4 Decoder: 2x Bottleneck (e:144, c:32) ] (14×14)           │
       │ (ConvTranspose2D s=2) + Concat(c3)                        │
[ Level 3 Decoder: 2x Bottleneck (e:96, c:16) ] (28×28)            │
       │ (ConvTranspose2D s=2) + Concat(c2)                        │
[ Level 2 Decoder: 2x Bottleneck (e:96, c:16) ] (56×56)            │
       │ (ConvTranspose2D s=2) + Concat(c1)                        │
[ Level 1 Decoder: 2x Bottleneck (e:48, c:8) ] (112×112)           │
       │ (ConvTranspose2D s=2) + Concat(x)                         │
[ Level 0 Decoder: 2x Bottleneck (e:48, c:8) ] (224×224) ◄─────────┘
       │
[ Conv2D (32, 3×3) + BN + ReLU6 ]
       │
[ Conv2D (16, 3×3) + BN + ReLU6 ]
       │
[ Conv2D (1, 3×3) + Sigmoid (Tanpa BN) ]
       │
                        [ Mask Prediksi (224×224×1) ]
```

### 📊 Verifikasi Parameter Model:
| Metrik Parameter | Hasil Implementasi | Target Resmi Paper (Table 4) | Status |
| :--- | :---: | :---: | :---: |
| **Trainable Parameters** | **233,001** | **233,001** | ✅ Identik 100% |
| **Non-Trainable Parameters** | **13,616** | **13,616** | ✅ Identik 100% |
| **Total Parameters** | **246,617** | **246,617** (~246 K) | ✅ Identik 100% |
| **Ukuran Bobot Model** | **~1.72 MB** | **~1.72 MB** | ✅ Identik |

---

## 📁 3. Struktur Direktori Proyek

```plaintext
Segmentasi Citra/
├── images/                                     # 1.000 citra kolonoskopi (.jpg)
│   ├── cju0qkwl35piu0993l0dewei2.jpg
│   └── ...
├── masks/                                      # 1.000 ground truth mask biner (.jpg)
│   ├── cju0qkwl35piu0993l0dewei2.jpg
│   └── ...
├── saved_models/                               # Bobot checkpoint terlatih
│   ├── mobile_polypnet_pretrained.weights.h5   # Bobot Keras 3 siap pakai (siap load_weights)
│   └── model_checkpoint_mobile-PolypNet/       # Checkpoint asli TensorFlow
├── main.ipynb                                  # Notebook implementasi lengkap end-to-end
├── requirements.txt                            # Daftar dependensi library Python
├── README.md                                   # Dokumentasi lengkap proyek
└── Mobile-PolypNet Lightweight Colon Polyp...pdf # File paper acuan
```

---

## ⚙️ 4. Instalasi & Persiapan Lingkungan

### Prasyarat:
- Python 3.10 – 3.13
- Git

### Langkah Instalasi:

1. **Buka terminal / PowerShell** pada direktori proyek:
   ```bash
   cd "D:\Kuliah\Akademik\Matkul\Semester 5\Pengolahan Citra\Segmentasi Citra"
   ```

2. **Buat dan aktifkan Virtual Environment (opsional namun disarankan)**:
   ```bash
   # Windows PowerShell
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install seluruh dependensi library**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Jalankan Jupyter Notebook**:
   ```bash
   jupyter lab
   # atau
   jupyter notebook
   ```
   Buka file `main.ipynb` dan pilih kernel Python yang sesuai.

---

## 🚀 5. Alur Kerja Notebook (`main.ipynb`)

Notebook `main.ipynb` disusun secara sistematis dan mandiri (*self-contained*):

1. **Konfigurasi Lingkungan**:
   - Random seed disetel ke `42` untuk menjamin reproduksibilitas.
   - Hyperparameter: `IMG_SIZE = 224`, `BATCH_SIZE = 8`, `LEARNING_RATE = 1e-3`, `EPOCHS = 50`.
2. **Pemuatan & Verifikasi Dataset**:
   - Membaca 1.000 citra dari folder `images/` dan 1.000 mask dari folder `masks/`.
   - Melakukan pembagian data: **80% Training**, **10% Validasi**, dan **10% Pengujian**.
3. **Data Preprocessing & Augmentasi**:
   - Normalisasi citra ke $[0.0, 1.0]$ dan binarisasi mask ke $\{0.0, 1.0\}$ pada resolusi $224 \times 224$.
   - Pipeline augmentasi paralel `tf.data`: rotasi acak antara $-90^\circ$ hingga $+90^\circ$, random flip horizontal/vertikal, dan variasi kecerahan ringan.
4. **Pembangunan Model Mobile-PolypNet**:
   - Mengonstruksi model dan memverifikasi parameter secara terprogram.
5. **Loss Function & Metrik**:
   - Menerapkan **DICE Loss** (Persamaan 1 paper) dan metrik komprehensif: DICE Score, mIoU (Jaccard Index), Sensitivity (Recall), Specificity, Precision, $F_2$ Score, dan MAE.
6. **Pelatihan & Fine-Tuning**:
   - Dilengkapi callback `ModelCheckpoint` (`best_mobile_polypnet.keras`), `EarlyStopping` (patience=25), dan `ReduceLROnPlateau`.
   - Mendukung opsi memuat bobot pretrained resmi (`saved_models/mobile_polypnet_pretrained.weights.h5`) untuk evaluasi langsung atau melanjutkan training.
7. **Evaluasi Kuantitatif Pengujian**:
   - Menguji seluruh citra pada Test Set independen dan mencetak tabel metrik performa klinis.
8. **Visualisasi Inferensi**:
   - Menampilkan visualisasi perbandingan 5 panel: Citra Asli, Ground Truth Mask, Probability Map, Binary Mask Prediksi, dan Overlay Kontur (Hijau = Ground Truth, Merah = Prediksi Model).

---

## 📈 6. Perbandingan Performa dengan Model Lain (Berdasarkan Paper)

Berdasarkan pengujian pada dataset Kvasir dalam paper (*Table 1 & Table 3*):

| Model | Parameter | Ukuran File | FLOPs (224×224) | DICE Score | mIoU | MAE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **U-Net** (MICCAI'15) | 7.85 M | 30.0 MB | 52.6 G | 0.818 | 0.746 | 0.055 |
| **U-Net++** (TMI'19) | 9.04 M | 34.6 MB | 112.6 G | 0.821 | 0.743 | 0.048 |
| **SFA** (MICCAI'19) | 25.59 M | 97.7 MB | 222.4 G | 0.723 | 0.611 | 0.075 |
| **PraNet** (MICCAI'20) | 20.52 M | 78.4 MB | 81.9 G | 0.898 | 0.840 | **0.030** |
| **Mobile-PolypNet (Ours)** | **246 K** | **1.72 MB** | **2.0 G** | **0.935** | **0.888** | 0.031 |

> 🌟 **Kesimpulan**: Mobile-PolypNet mengungguli model PraNet sebesar **+4.12% pada DICE score**, dengan jumlah parameter **~83× lebih sedikit** dan komputasi **~17× lebih efisien**.

---

## 📝 Sitasi

Jika menggunakan kode atau metode ini dalam riset/akademik, silakan merujuk pada sitasi paper asli:

```bibtex
@article{karmakar2022mobile,
  title={Mobile-PolypNet: Lightweight Colon Polyp Segmentation Network for Low-Resource Settings},
  author={Karmakar, Ranit and Nooshabadi, Saeid},
  journal={Journal of Imaging},
  volume={8},
  number={6},
  pages={169},
  year={2022},
  publisher={MDPI},
  doi={10.3390/jimaging8060169}
}
```
