> [!TIP]
> Kembali ke halaman utama, [Klik disini](https://github.com/AdityaR-AI/MLC/tree/main/)

# Naive Bayes Classification

### 1.0. Lakukan praktik dari https://youtu.be/Sj1ybuDDf9I?si=hCajHe1zasTQ9HGY , buat screenshot dengan nama kalian pada coding, kumpulkan dalam bentuk pdf, dari kegiatan ini:

#### 1.1. Pengenalan Bayes Theorem | Teori Bayes | Conditional Probability

**Baye’s Theorem**

Teorema Bayes, yang dinamai dari Thomas Bayes, adalah sebuah prinsip dalam statistik yang digunakan untuk menghitung probabilitas suatu kejadian berdasarkan informasi yang telah diketahui sebelumnya. Teorema ini sangat berguna dalam berbagai bidang, termasuk statistik, machine learning, dan pengambilan keputusan.

Baye’s Theorem menawarkan suatu formula untuk menghitung nilai probalibility  dari suatu event dengan memamfaatkan pengetahuan sebelumnya  dari kondisi terkait atau sering kali dikenal dengan istilah conditional probability

**Probabilitas Kondisional**
Probabilitas kondisional adalah probabilitas suatu kejadian A terjadi dengan syarat bahwa kejadian B telah terjadi. Ini dituliskan sebagai ( P(A | B) ). Konsep ini sangat penting dalam Teorema Bayes, karena Teorema Bayes pada dasarnya menghubungkan probabilitas kondisional dua kejadian.

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P4/pics/4a1.png?raw=true" alt="SS" width="30%"/>

**Di mana:**
  - ( P(A | B) ) adalah probabilitas kejadian A terjadi, dengan syarat bahwa B telah terjadi (probabilitas bersyarat).
  -	( P(B | A) ) adalah probabilitas kejadian B terjadi, dengan syarat bahwa A telah terjadi.
  -	( P(A) ) adalah probabilitas kejadian A secara keseluruhan.
  -	( P(B) ) adalah probabilitas kejadian B secara keseluruhan.

**Contoh**
Misalkan kita ingin mengetahui probabilitas seseorang menderita penyakit A (A) setelah mendapatkan hasil tes positif (B). Kita perlu mengetahui beberapa informasi:
  -	Probabilitas seseorang menderita penyakit A, ( P(A) ).
  -	Probabilitas hasil tes positif jika seseorang menderita penyakit A, ( P(B | A) ).
  -	Probabilitas hasil tes positif secara keseluruhan, ( P(B) ).

Dengan informasi tersebut, kita dapat menggunakan Teorema Bayes untuk menghitung ( P(A | B) ).

### 1.2. Pengenalan Naive Bayes Classification

**Pengenalan Naive Bayes Classification**

Naive Bayes Classification adalah salah satu metode klasifikasi yang berbasis pada Teorema Bayes. Metode ini sangat populer dalam machine learning dan statistik karena kesederhanaannya, kecepatan dalam pelatihan, serta kemampuannya untuk memberikan hasil yang baik meskipun dengan asumsi yang sederhana.

Studi Kasus 1

Terdapat 1 warung dengan 3 menu dengan nilai probabilitas dimasing2 menu yang didatangi 2 pelanggan:

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P4/pics/4a2.png?raw=true" alt="SS" width="40%"/>

**Misi:** Prediksi siapa pelanggan yang memesan jika pesanannya bakso dan lumpia

### 1.3. Pengenalan Prior Probability

**Prior Probability P(y)** adalah nilai probability dari kemunculan suatu nilai target label tertentu tanpa memperhatikan nilai feature nya.
Untuk kasus asep dan joko disini kita bisa asumsikan bahwa peluang pemesanan yang dilakukan asep (0.1 + 0.8 + 0.1 = 1) dan joko (0.5 + 0.2 + 3 = 0.5) seimbang, oleh karena prior dari kasus ini diekpresikan sebagai berikut:
  -	P(Asep) = 0.5 berarti 50%
  -	P(Joko) = 0.5 Berarti 50%

### 1.4. Pengenalan Likelihood

**Likelihood P(X|y)** adalah probalility kemunculan nilai fitur tertentu bila diketahui kemunculan nilai target labelnya.
Dalam konteks prediksi pemesanan asep dan joko Likelihood diekpresikan sebagai berikut:
  -	Asep, nilai probability kemunculan pemesanan lumpia dan bakso bila diketahui Asep pemesannya

P(lumpia, bakso | Asep) = (0,1 x 0,8)
			    = 0,08

   - Joko, nilai probability kemunculan pemesanan lumpia dan bakso bila diketahui Joko pemesannya

P(lumpia, bakso | Joko) = (0,1 x 0,8)
			    = 0,08

### 1.5. Pengenalan Evidence | Normalizer

Evidence atau Normalizer P(X) adalah total akumulasi dari hasil perkalian antara likelihood dengan priornya.
Untuk konteks probability kemunculan pesanan lumpia dan bakso dapat diekpresikan sebagai berikut:

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P4/pics/4a3.png?raw=true" alt="SS" width="48%"/>

### 1.6. Pengenalan Posterior Probability

Posterior Probability P(y|X) merupakan nilai probability kemunculan suatu class atau target label dengan diketahui kemunculan sekumpulan nilai feature nya

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P4/pics/4a4.png?raw=true" alt="SS" width="35%"/>

Berikut adalah Probability Asep sebagai pemesan jika diketahui persanannya adlah lumpia dan bakso begitu pun sebaliknya

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P4/pics/4a5.png?raw=true" alt="SS" width="42%"/>

Seperti yang bisa dilihat untuk kasus ini nilai posterior Asep lebih tinggi, **maka Naïve Bayes akan mengklasifikasikan asep sebagai pemesannya.**

### 1.7. Studi kasus dan implementasi Naive Bayes

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P4/pics/4a6.png?raw=true" alt="SS" width="35%"/>

**Misi:** Prediksi siapa pelanggan yang memesan bila diketahui pesanannya adalah **bakso dan siomay.**

**Posterior Probability P(y|X)**

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P4/pics/4a7.png?raw=true" alt="SS" width="42%"/>

Berdasarkan hasil **Joko adalah pemesannya**

### Praktik

> Persiapan Dataset | Wisconsin Breast Cancer Dataset

**Load Dataset**

```python
print ('Aditya Rimandi Putra')
print ('41155050210030\n') 

# Mengimpor dataset kanker payudara dari pustaka sklearn
from sklearn.datasets import load_breast_cancer

# Mencetak deskripsi dataset kanker payudara
print(load_breast_cancer().DESCR)
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P4/pics/4b1.png?raw=true" alt="SS" width="80%"/>

```python
print ('Aditya Rimandi Putra')
print ('41155050210030\n') 

# Menampilkan dokumentasi fungsi load_breast_cancer
# load_breast_cancer?

# Memuat data fitur (X) dan label (y) dari dataset kanker payudara
X, y = load_breast_cancer(return_X_y=True)

# Menampilkan bentuk (dimensi) dari data fitur X
X.shape
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P4/pics/4b2.png?raw=true" alt="SS" width="30%"/>

**Training & Testing**

```python
print ('Aditya Rimandi Putra')
print ('41155050210030\n') 

# Mengimpor fungsi train_test_split dari pustaka sklearn.model_selection
from sklearn.model_selection import train_test_split

# Membagi dataset menjadi data pelatihan (X_train, y_train) dan data pengujian (X_test, y_test)
X_train, X_test, y_train, y_test = train_test_split(
    X,                # Data fitur
    y,                # Data label
    test_size=0.2,   # Menggunakan 20% data sebagai data pengujian
    random_state=0   # Menetapkan seed untuk memastikan hasil yang konsisten
)

# Mencetak bentuk (dimensi) dari data pelatihan
print(f'X_train shape {X_train.shape}')

# Mencetak bentuk (dimensi) dari data pengujian
print(f'X_test shape {X_test.shape}')
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P4/pics/4b3.png?raw=true" alt="SS" width="30%"/>

> Implementasi Naive Bayes Classification dengan Scikit-Learn

**Naive Bayes dengan Scikit Learn**

```python
print ('Aditya Rimandi Putra')
print ('41155050210030\n') 

# Mengimpor GaussianNB dari pustaka sklearn.naive_bayes
from sklearn.naive_bayes import GaussianNB

# Mengimpor fungsi accuracy_score dari pustaka sklearn.metrics
from sklearn.metrics import accuracy_score

# Membuat objek model Gaussian Naive Bayes
model = GaussianNB()

# Melatih model menggunakan data pelatihan (X_train dan y_train)
model.fit(X_train, y_train)

# Menggunakan model untuk memprediksi label dari data pengujian (X_test)
y_pred = model.predict(X_test)

# Menghitung dan mengembalikan akurasi dari prediksi model
accuracy_score(y_test, y_pred)
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P4/pics/4b4.png?raw=true" alt="SS" width="30%"/>

```python
print ('Aditya Rimandi Putra')
print ('41155050210030\n') 

# Menghitung akurasi dengan metode score dari model
model.score(X_test, y_test)
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P4/pics/4b5.png?raw=true" alt="SS" width="30%"/>

> [!TIP]
> Kembali ke halaman utama, [Klik disini](https://github.com/AdityaR-AI/MLC/tree/main/)














