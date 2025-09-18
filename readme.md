# Klasifikasi Tingkat Pendapatan Berdasarkan Data Demografi dengan Metode K-Nearest Neighbor

**Author**: Ferdiansyah Pratama

## Deskripsi

Proyek ini bertujuan untuk **memprediksi tingkat pendapatan individu** (≤50K atau >50K per tahun) berdasarkan data demografi dan pekerjaan. Dataset ini sering digunakan sebagai contoh untuk pembelajaran **Machine Learning** karena berisi berbagai variabel kategorikal dan numerik yang relevan.

Metode utama yang digunakan adalah **K-Nearest Neighbor (KNN)** dengan beberapa langkah **preprocessing data** seperti **handling missing values, encoding, scaling, dan balancing data**.

---

## Dataset

Dataset yang digunakan memiliki **45222 baris** dan **14 fitur awal**.
Target: **Income**

- `<=50K` → Low Income
- `>50K` → High Income

Beberapa atribut penting:

- **Numerik**: age, hours-per-week, capital-gain, capital-loss, educational-num, fnlwgt
- **Kategorikal**: workclass, education, marital-status, occupation, relationship, race, gender, native-country

---

## Preprocessing

Langkah-langkah preprocessing yang dilakukan:

1. **Handling Missing Values**

   - Nilai `nan` diganti dengan kategori `"Unknown"` atau median (untuk numerik).

2. **Simplifikasi Kategori**

   - Contoh: `Married-civ-spouse`, `Married-AF-spouse` → digabung jadi `"Married"`.

3. **Encoding**

   - Menggunakan **OneHotEncoding** untuk variabel kategorikal.

4. **Scaling**

   - Menggunakan **MinMaxScaler** agar semua fitur numerik berada pada skala yang sama (0–1).

5. **Handling Imbalance**

   - Menggunakan **SMOTE / RandomOverSampler** untuk menyeimbangkan distribusi target.

---

## Model: K-Nearest Neighbor

Parameter tuning menggunakan **RandomizedSearchCV** dengan parameter:

- `n_neighbors`: \[3, 5, 7, …, 19]
- `weights`: \['uniform', 'distance']
- `p`: \[1 (Manhattan), 2 (Euclidean)]

---

## Hasil Evaluasi

### Model Awal (tanpa tuning)

```
Accuracy  : 0.67
F1-Score  : 0.62 (macro)
```

### Model Setelah Tuning

```
Accuracy  : 0.70
F1-Score  : 0.63 (macro)
```

**Confusion Matrix:**

- Class `0 (Low Income)` → Precision 0.42, Recall 0.57
- Class `1 (High Income)` → Precision 0.84, Recall 0.74

---

## Kesimpulan

- Metode **KNN** cukup baik dalam memprediksi pendapatan dengan akurasi **70%** setelah tuning.
- Preprocessing (scaling & balancing) berpengaruh signifikan pada performa model.
- Kelas `>50K` lebih mudah diprediksi dibanding `<=50K`.

 **Saran:**
Mungkin untuk penelitian selanjutnya bisa dibandingkan dengan algoritma lain seperti **Random Forest, XGBoost, atau Logistic Regression**.

---

## Requirements

- Python 3.8+
- scikit-learn
- imbalanced-learn
- pandas
- numpy
- matplotlib
- seaborn
