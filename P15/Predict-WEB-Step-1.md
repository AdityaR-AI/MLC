## Aplikasi Deteksi Diabetes berbasis Web

### Membuat Model

Klik link dibawah ini untuk kembali ke menu utama:

> [!TIP]
> Kembali ke halaman utama, [Klik disini](https://github.com/AdityaR-AI/MLC/tree/main/)

Silahkan klik link dibawah ini untuk menuju Step yang ingin dilihat:

> [!NOTE]
> Step 1 - Membuat Model [Pages Link](https://github.com/AdityaR-AI/MLC/tree/main/P15/Predict-WEB-Step-1.md)

> [!NOTE]
> Step 2 - Mengaplikasikan Model ke Web [Pages Link](https://github.com/AdityaR-AI/MLC/tree/main/P15/Predict-WEB-Step-2.md)

> [!NOTE]
> Step 3 - Deploy Hosting [Pages Link](https://github.com/AdityaR-AI/MLC/tree/main/P15/Predict-WEB-Step-3.md)

> [!NOTE]
> **Link Hosting:** [https://adityari.pythonanywhere.com/](https://adityari.pythonanywhere.com/)

1.	**Download dataset**

Download dataset di [Pages Link](https://github.com/heriistantoo/save-sklearn)

2.	**Import library**

```python
# Mencetak informasi identitas
print('Nama: Aditya Rimandi Putra')  # Mencetak nama lengkap
print('NPM : 41155050210030\n')  # Mencetak Nomor Pokok Mahasiswa (NPM)
 
# Mengimpor library pandas dan memberikan alias 'pd'
# pandas digunakan untuk manipulasi dan analisis data terstruktur
import pandas as pd
 
# Mengimpor library numpy dan memberikan alias 'np'
# numpy digunakan untuk komputasi numerik, terutama array dan matriks
import numpy as np
 
# Mengimpor library scikit-learn (sklearn)
# sklearn menyediakan alat untuk machine learning seperti klasifikasi, regresi, dll.
import sklearn
 
# Mengimpor library flask
# flask adalah framework web ringan untuk membangun aplikasi web dan API
import flask
 
# Mengimpor KNeighborsClassifier dari sklearn.neighbors
# KNeighborsClassifier adalah algoritma klasifikasi berbasis K-Nearest Neighbors (KNN)
from sklearn.neighbors import KNeighborsClassifier
 
# Mengimpor accuracy_score dari sklearn.metrics
# accuracy_score digunakan untuk menghitung akurasi prediksi model
from sklearn.metrics import accuracy_score
 
# Mengimpor GridSearchCV dari sklearn.model_selection
# GridSearchCV digunakan untuk mencari parameter terbaik model dengan validasi silang
from sklearn.model_selection import GridSearchCV

```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P15/pic/15a1.png?raw=true" alt="SS" width="25%"/>

3.	**Cek versi library**

```python
# Mencetak informasi identitas
print('Nama: Aditya Rimandi Putra')  # Mencetak nama lengkap
print('NPM : 41155050210030\n')  # Mencetak Nomor Pokok Mahasiswa (NPM)
 
# Mencetak versi dari library flask yang terinstal
# flask.__version__ mengembalikan string yang berisi versi flask
print(flask.__version__)
 
# Mencetak versi dari library scikit-learn (sklearn) yang terinstal
# sklearn.__version__ mengembalikan string yang berisi versi scikit-learn
print(sklearn.__version__)

```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P15/pic/15a2.png?raw=true" alt="SS" width="25%"/>
<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P15/pic/15a3.png?raw=true" alt="SS" width="10%"/>

*ini adalah version library untuk flask dan scikit-learn (sklearn), buat requirements.txt dan masukan data version ini:
```
flask==3.0.3
scikit-learn==1.5.1
```
*ini adalah library yang kita butuhkan sampai deploy, walau flask sendiri tidak digunakan saat membuat model
*walau ada library selain flask dan sklearn yang diimport diwalau, namun karena itu akan include penginstalan saat sklearn (scikit-learn) diinstal jadi tidak perlu mendata library lain

4.	**Membaca dataset**

```python
# Mencetak informasi identitas
print('Nama: Aditya Rimandi Putra')  # Mencetak nama lengkap
print('NPM : 41155050210030\n')  # Mencetak Nomor Pokok Mahasiswa (NPM)
 
# Membaca file CSV 'diabetes_latih.csv' ke dalam DataFrame pandas
# DataFrame adalah struktur data tabel yang digunakan untuk menyimpan dan memanipulasi data
df = pd.read_csv('./diabetes_latih.csv')
 
# Mengambil semua nilai dari DataFrame dan menyimpannya dalam variabel X_train
# df.values mengembalikan array numpy yang berisi semua data dalam DataFrame
X_train = df.values
 
# Menghapus kolom ke-8 (indeks 8) dari array X_train
# np.delete digunakan untuk menghapus elemen dari array numpy
# axis=1 menunjukkan bahwa penghapusan dilakukan pada kolom (bukan baris)
X_train = np.delete(X_train, 8, axis=1)
 
# Mengambil nilai dari kolom 'Outcome' dalam DataFrame dan menyimpannya dalam variabel y_train
# Kolom 'Outcome' biasanya berisi label/target untuk data latih
y_train = df['Outcome'].values
 
# Membaca file CSV 'diabetes_uji.csv' ke dalam DataFrame pandas
# File ini biasanya berisi data uji yang akan digunakan untuk menguji model
df = pd.read_csv('./diabetes_uji.csv')
 
# Mengambil semua nilai dari DataFrame dan menyimpannya dalam variabel X_test
X_test = df.values
 
# Menghapus kolom ke-8 (indeks 8) dari array X_test
# Kolom ini dihapus karena diasumsikan sebagai kolom target/label, bukan fitur
X_test = np.delete(X_test, 8, axis=1)
 
# Mengambil nilai dari kolom 'Outcome' dalam DataFrame dan menyimpannya dalam variabel y_test
# Kolom 'Outcome' berisi label/target untuk data uji
y_test = df['Outcome'].values

```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P15/pic/15a4.png?raw=true" alt="SS" width="25%"/>

5.   **Membuat model KNN**
```python
# Mencetak informasi identitas
print('Nama: Aditya Rimandi Putra')  # Mencetak nama lengkap
print('NPM : 41155050210030\n')  # Mencetak Nomor Pokok Mahasiswa (NPM)
 
# Membuat model K-Nearest Neighbors (KNN) dengan jumlah tetangga (n_neighbors) = 3
# KNeighborsClassifier adalah algoritma klasifikasi berbasis jarak terdekat
knn_clf = KNeighborsClassifier(n_neighbors=3)
 
# Melatih model KNN menggunakan data latih (X_train) dan label latih (y_train)
# Proses ini akan mengajarkan model untuk mengenali pola dalam data
knn_clf.fit(X_train, y_train)
 
# Memprediksi label untuk data uji (X_test) menggunakan model KNN yang sudah dilatih
# Hasil prediksi disimpan dalam variabel y_pred
y_pred = knn_clf.predict(X_test)
 
# Menghitung akurasi prediksi dengan membandingkan label sebenarnya (y_test) dan label prediksi (y_pred)
# accuracy_score mengembalikan nilai akurasi dalam bentuk desimal
# round(..., 3) digunakan untuk membulatkan nilai akurasi hingga 3 angka di belakang koma
round(accuracy_score(y_test, y_pred), 3)

```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P15/pic/15a5.png?raw=true" alt="SS" width="25%"/>
<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P15/pic/15a6.png?raw=true" alt="SS" width="10%"/>

6.	**Mencari parameter neighbor terbaik**

```python
# Mencetak informasi identitas
print('Nama: Aditya Rimandi Putra')  # Mencetak nama lengkap
print('NPM : 41155050210030\n')  # Mencetak Nomor Pokok Mahasiswa (NPM)
 
# Membuat dictionary param_grid yang berisi parameter yang akan diuji
# 'n_neighbors': np.arange(1, 201) berarti kita akan mencoba nilai n_neighbors dari 1 hingga 200
param_grid = {'n_neighbors': np.arange(1, 201)}
 
# Membuat objek GridSearchCV untuk mencari parameter terbaik
# GridSearchCV akan mencoba semua kombinasi parameter yang diberikan dalam param_grid
# KNeighborsClassifier() adalah model yang akan di-tuning
# cv=3 berarti menggunakan validasi silang (cross-validation) dengan 3 fold
# scoring='accuracy' berarti metrik evaluasi yang digunakan adalah akurasi
knn_clf = GridSearchCV(KNeighborsClassifier(), param_grid, cv=3, scoring='accuracy')
 
# Melatih model GridSearchCV menggunakan data latih (X_train) dan label latih (y_train)
# Proses ini akan mencoba semua kombinasi parameter dan memilih yang terbaik berdasarkan akurasi
knn_clf.fit(X_train, y_train)
 
# Mengambil parameter terbaik yang ditemukan oleh GridSearchCV
# best_params_ mengembalikan dictionary yang berisi parameter terbaik
knn_clf.best_params_

```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P15/pic/15a7.png?raw=true" alt="SS" width="25%"/>
<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P15/pic/15a8.png?raw=true" alt="SS" width="10%"/>

7.	**Membuat model KNN dengan hasil tuning neighbor**
```python
# Mencetak informasi identitas
print('Nama: Aditya Rimandi Putra')  # Mencetak nama lengkap
print('NPM : 41155050210030\n')  # Mencetak Nomor Pokok Mahasiswa (NPM)
 
# Membuat model K-Nearest Neighbors (KNN) dengan jumlah tetangga (n_neighbors) = 9
# KNeighborsClassifier adalah algoritma klasifikasi berbasis jarak terdekat
# n_neighbors=9 berarti model akan menggunakan 9 tetangga terdekat untuk melakukan klasifikasi
knn_clf = KNeighborsClassifier(n_neighbors=9)
 
# Melatih model KNN menggunakan data latih (X_train) dan label latih (y_train)
# Proses ini akan mengajarkan model untuk mengenali pola dalam data
knn_clf.fit(X_train, y_train)
 
# Memprediksi label untuk data uji (X_test) menggunakan model KNN yang sudah dilatih
# Hasil prediksi disimpan dalam variabel y_pred
y_pred = knn_clf.predict(X_test)
 
# Menghitung akurasi prediksi dengan membandingkan label sebenarnya (y_test) dan label prediksi (y_pred)
# accuracy_score mengembalikan nilai akurasi dalam bentuk desimal
# round(..., 3) digunakan untuk membulatkan nilai akurasi hingga 3 angka di belakang koma
round(accuracy_score(y_test, y_pred), 3)

```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P15/pic/15a9.png?raw=true" alt="SS" width="25%"/>
<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P15/pic/15a10.png?raw=true" alt="SS" width="10%"/>

8.	**1.8	Ekspor model ke pickle**
```python
# Mencetak informasi identitas
print('Nama: Aditya Rimandi Putra')  # Mencetak nama lengkap
print('NPM : 41155050210030\n')  # Mencetak Nomor Pokok Mahasiswa (NPM)
 
# Mengimpor library pickle
# pickle adalah library Python yang digunakan untuk serialisasi dan deserialisasi objek Python
# Serialisasi adalah proses mengubah objek Python menjadi format biner yang dapat disimpan atau ditransfer
import pickle
 
# Membuka file 'knn_pickle' dalam mode write binary ('wb')
# File ini akan digunakan untuk menyimpan model KNN yang sudah dilatih
# 'wb' berarti file dibuka untuk ditulis dalam format biner
with open('knn_pickle', 'wb') as r:
    # Menyimpan model KNN (knn_clf) ke dalam file 'knn_pickle' menggunakan pickle.dump
    # Proses ini disebut serialisasi, di mana objek Python diubah menjadi format biner dan disimpan ke file
    pickle.dump(knn_clf, r)


```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P15/pic/15a11.png?raw=true" alt="SS" width="25%"/>

File pickle akan tersedia difolder setelah eskpor model berhasil
```
/ipynb folder
  |-- knn_pickle 
```

Model pickle ini adalah model yang akan diaplikasikan ke web

9.	**Load pickle model yang sudah dilatih**

```python
# Mencetak informasi identitas
print('Nama: Aditya Rimandi Putra')  # Mencetak nama lengkap
print('NPM : 41155050210030\n')  # Mencetak Nomor Pokok Mahasiswa (NPM)
 
# Membuka file 'knn_pickle' dalam mode read binary ('rb')
with open('knn_pickle', 'rb') as r:
    # Memuat model KNN dari file menggunakan pickle.load
    loaded_model = pickle.load(r)
 
# Sekarang loaded_model adalah model KNN yang sudah dilatih dan siap digunakan

```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P15/pic/15a12.png?raw=true" alt="SS" width="25%"/>

10.	**lihat akurasi model yang diload**

```python
# Mencetak informasi identitas
print('Nama: Aditya Rimandi Putra')  # Mencetak nama lengkap
print('NPM : 41155050210030\n')  # Mencetak Nomor Pokok Mahasiswa (NPM)
 
# Menggunakan model KNN yang sudah dimuat (knnp) untuk memprediksi label dari data uji (X_test)
# Hasil prediksi disimpan dalam variabel y_pred
y_pred = knnp.predict(X_test)
 
# Menghitung akurasi prediksi dengan membandingkan label sebenarnya (y_test) dan label prediksi (y_pred)
# accuracy_score mengembalikan nilai akurasi dalam bentuk desimal
# round(..., 3) digunakan untuk membulatkan nilai akurasi hingga 3 angka di belakang koma
round(accuracy_score(y_test, y_pred), 3)

```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P15/pic/15a13.png?raw=true" alt="SS" width="25%"/>
<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P15/pic/15a14.png?raw=true" alt="SS" width="10%"/>

11. **Ekspor menjadi joblib**

```python
# Mencetak informasi identitas
print('Nama: Aditya Rimandi Putra')  # Mencetak nama lengkap
print('NPM : 41155050210030\n')  # Mencetak Nomor Pokok Mahasiswa (NPM)
 
# Mengimpor library joblib
# joblib adalah library Python yang digunakan untuk serialisasi dan deserialisasi objek Python
# joblib sering digunakan untuk menyimpan model machine learning karena lebih efisien untuk objek besar
import joblib
 
# Menyimpan model KNN (knn_clf) ke dalam file 'knn_joblib' menggunakan joblib.dump
# Proses ini disebut serialisasi, di mana objek Python diubah menjadi format biner dan disimpan ke file
# joblib lebih efisien daripada pickle untuk objek besar seperti model machine learning
joblib.dump(knn_clf, 'knn_joblib')

```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P15/pic/15a15.png?raw=true" alt="SS" width="25%"/>
<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P15/pic/15a16.png?raw=true" alt="SS" width="20%"/>

12.	**Load model joblib**

```python
# Mencetak informasi identitas
print('Nama: Aditya Rimandi Putra')  # Mencetak nama lengkap
print('NPM : 41155050210030\n')  # Mencetak Nomor Pokok Mahasiswa (NPM)
 
# Memuat model KNN dari file 'knn_joblib' menggunakan joblib.load
# joblib.load membaca file biner dan mengembalikan objek Python yang disimpan di dalamnya
# Dalam hal ini, objek yang dimuat adalah model KNN yang sudah dilatih
knnjl = joblib.load('knn_joblib')

```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P15/pic/15a17.png?raw=true" alt="SS" width="25%"/>

13.	**Cek akurasi joblib**

```python
# Mencetak informasi identitas
print('Nama: Aditya Rimandi Putra')  # Mencetak nama lengkap
print('NPM : 41155050210030\n')  # Mencetak Nomor Pokok Mahasiswa (NPM)
 
# Menggunakan model KNN yang sudah dimuat (knnjl) untuk memprediksi label dari data uji (X_test)
# Hasil prediksi disimpan dalam variabel y_pred
y_pred = knnjl.predict(X_test)
 
# Menghitung akurasi prediksi dengan membandingkan label sebenarnya (y_test) dan label prediksi (y_pred)
# accuracy_score mengembalikan nilai akurasi dalam bentuk desimal
# round(..., 3) digunakan untuk membulatkan nilai akurasi hingga 3 angka di belakang koma
round(accuracy_score(y_test, y_pred), 3)

```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P15/pic/15a18.png?raw=true" alt="SS" width="25%"/>
<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P15/pic/15a19.png?raw=true" alt="SS" width="10%"/>

Silahkan klik link dibawah ini untuk menuju Step yang ingin dilihat:

> [!NOTE]
> Step 1 - Membuat Model [Pages Link](https://github.com/AdityaR-AI/MLC/tree/main/P15/Predict-WEB-Step-1.md)

> [!NOTE]
> Step 2 - Mengaplikasikan Model ke Web [Pages Link](https://github.com/AdityaR-AI/MLC/tree/main/P15/Predict-WEB-Step-2.md)

> [!NOTE]
> Step 3 - Deploy Hosting [Pages Link](https://github.com/AdityaR-AI/MLC/tree/main/P15/Predict-WEB-Step-3.md)

> [!NOTE]
> **Link Hosting:** [https://adityari.pythonanywhere.com/](https://adityari.pythonanywhere.com/)

Klik link dibawah ini untuk kembali ke menu utama:

> [!TIP]
> Kembali ke halaman utama, [Klik disini](https://github.com/AdityaR-AI/MLC/tree/main/)









