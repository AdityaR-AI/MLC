## Classification dengan KNN | K-Nearest Neighbours

> Materi ini terbagi menjadi 2 Part, berikut linknya:

Silahkan klik link dibawah ini tuntuk menuju tugas yang inign dilihat:

> [!NOTE]
> Part 1 - Classification dengan KNN | K-Nearest Neighbours [Pages Link](https://github.com/AdityaR-AI/MLC/blob/main/P5/K%20Nearest%20Neigbor%20&%20Support%20Vector%20Machine_I.md)

> [!NOTE]
> Part 2 - Support Vector Machine Classification | SVM [Pages Link](https://github.com/AdityaR-AI/MLC/tree/main/P5/K%20Nearest%20Neigbor%20%26%20Support%20Vector%20Machine_II.md)

Klik link dibawah ini untuk kembali ke menu utama:

> [!TIP]
> Kembali ke halaman utama, [Klik disini](https://github.com/AdityaR-AI/MLC/tree/main/)

### 1.0. K-Nearest Neighbours (KNN). Lakukan praktik dari https://youtu.be/4zARMcgc7hA?si=x6RoHQXFF4NY76X8 , buat screenshot dengan nama kalian pada coding, kumpulkan dalam bentuk pdf, dari kegiatan ini:

**Classification dengan KNN (K Nearest Neighbours)** 
  - KNN adalah model machine learning yang dapat digunakan untuk melakukan prediksi berdasarkan kedekatan karakteristik dengan sejumlah tetangga terdekat. 
  - Prediksi yang dilakukan dapat diterapkan baik pada classification maupun regression tasks. 
Referensi: https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm

### 1.1. Persiapan sample dataset

```python
import pandas as pd  # Mengimpor pustaka pandas untuk manipulasi data

# Mencetak nama dan NPM
print ('Nama: Aditya Rimandi Putra')
print ('NPM : 41155050210030\n')

# Membuat dictionary yang berisi data sensus
sensus = {
    'tinggi': [158, 170, 183, 191, 155, 163, 180, 158, 178],  # Daftar tinggi badan
    'berat': [64, 86, 84, 80, 49, 59, 67, 54, 67],  # Daftar berat badan
    'jk': [  # Daftar jenis kelamin
        'pria', 'pria', 'pria', 'pria', 'wanita', 'wanita', 'wanita', 'wanita', 'wanita'
    ]
}

# Mengonversi dictionary menjadi DataFrame pandas
sensus_df = pd.DataFrame(sensus)

# Menampilkan DataFrame yang telah dibuat
sensus_df
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P5/pics/5a3.png?raw=true" alt="SS" width="25%"/>

### 1.2. Visualisasi dataset

```python
import matplotlib.pyplot as plt  # Mengimpor pustaka matplotlib untuk visualisasi data

# Mencetak nama dan NPM
print ('Nama: Aditya Rimandi Putra')
print ('NPM : 41155050210030\n')

# Membuat figure dan axes untuk plot
fig, ax = plt.subplots()

# Mengelompokkan data berdasarkan jenis kelamin dan membuat scatter plot untuk setiap kelompok
for jk, d in sensus_df.groupby('jk'):
    ax.scatter(d['tinggi'], d['berat'], label=jk)  # Membuat scatter plot untuk tinggi dan berat

# Menambahkan legend pada plot di posisi kiri atas
plt.legend(loc='upper left')

# Menambahkan judul pada plot
plt.title('Sebaran Data Tinggi Badan, Berat Badan, dan Jenis Kelamin')

# Menambahkan label untuk sumbu X
plt.xlabel('Tinggi Badan (cm)')

# Menambahkan label untuk sumbu Y
plt.ylabel('Berat Badan (kg)')

# Menampilkan grid pada plot
plt.grid(True)

# Menampilkan plot
plt.show()
```
<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P5/pics/5a4.png?raw=true" alt="SS" width="55%"/>

### 1.3. Pengantar classification dengan K-Nearest Neighbours | KNN

Setelah kita memahami konteks dataset dan juga permasalahanya, kita akan coba menerapkan KNN atau K-Nearest Neighbours untuk melakukan klasifikasi jenis kelamin berdasarkan data tinggi dan berat badan, sesuai dengan namanya model mesin learning yang satu ini akan melakukan prediksi dalam kasus ini adalah prediksi gender / prediksi jenis kelamin berdasarkan kemiripan karakteristik atau features dengan dataset yang kita miliki, KNN juga termasuk salah satu model machine-learning dasar yang wajib dikuasai.

### 1.4. Preprocessing dataset dengan Label Binarizer

```python
import numpy as np  # Mengimpor pustaka numpy untuk manipulasi array dan operasi numerik

# Mencetak nama dan NPM
print ('Nama: Aditya Rimandi Putra')
print ('NPM : 41155050210030\n')

# Mengonversi kolom 'tinggi' dan 'berat' dari DataFrame menjadi array numpy
X_train = np.array(sensus_df[['tinggi', 'berat']])

# Mengonversi kolom 'jk' dari DataFrame menjadi array numpy
y_train = np.array(sensus_df['jk'])

# Mencetak array X_train yang berisi data tinggi dan berat
print(f'X_train: \n{X_train}\n')

# Mencetak array y_train yang berisi data jenis kelamin
print(f'y_train: {y_train}')
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P5/pics/5a5.png?raw=true" alt="SS" width="75%"/>

```python
from sklearn.preprocessing import LabelBinarizer  # Mengimpor LabelBinarizer dari pustaka sklearn untuk mengubah label menjadi format biner

# Mencetak nama dan NPM
print ('Nama: Aditya Rimandi Putra')
print ('NPM : 41155050210030\n')

# Membuat objek LabelBinarizer
lb = LabelBinarizer()

# Mengubah label y_train menjadi format biner (one-hot encoding)
y_train = lb.fit_transform(y_train)

# Mencetak hasil y_train yang telah diubah
print(f'y train: \n{y_train}')
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P5/pics/5a6.png?raw=true" alt="SS" width="25%"/>

```python
# Mencetak nama dan NPM
print ('Nama: Aditya Rimandi Putra')
print ('NPM : 41155050210030\n')

# Mengubah bentuk array y_train menjadi satu dimensi
y_train = y_train.flatten()

# Mencetak hasil y_train yang telah diubah
print (f'y_train: {y_train}')
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P5/pics/5a7.png?raw=true" alt="SS" width="30%"/>

### 1.5. Training KNN Classification Model

```python
from sklearn.neighbors import KNeighborsClassifier  # Mengimpor KNeighborsClassifier dari pustaka sklearn untuk klasifikasi K-Nearest Neighbors

# Mencetak nama dan NPM
print ('Nama: Aditya Rimandi Putra')
print ('NPM : 41155050210030\n')

# Menentukan jumlah tetangga terdekat yang akan digunakan
K = 3

# Membuat model K-Nearest Neighbors dengan jumlah tetangga K
model = KNeighborsClassifier(n_neighbors=K)

# Melatih model dengan data pelatihan (X_train dan y_train)
model.fit(X_train, y_train)
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P5/pics/5a8.png?raw=true" alt="SS" width="35%"/>

### 1.6. Prediksi dengan KNN Classification Model

```python
# Mencetak nama dan NPM
print ('Nama: Aditya Rimandi Putra')
print ('NPM : 41155050210030\n')

# Mendefinisikan tinggi badan dan berat badan
tinggi_badan = 155  # dalam sentimeter
berat_badan = 70    # dalam kilogram

# Membuat array numpy dari tinggi dan berat badan, dan mengubah bentuknya menjadi 2D
X_new = np.array([tinggi_badan, berat_badan]).reshape(1, -1)

# Menampilkan X_new
X_new
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P5/pics/5a9.png?raw=true" alt="SS" width="30%"/>

```python
# Mencetak nama dan NPM
print ('Nama: Aditya Rimandi Putra')
print ('NPM : 41155050210030\n')

# Menggunakan model yang sudah dilatih untuk memprediksi label berdasarkan data baru
y_new = model.predict(X_new)

# Menampilkan hasil prediksi
y_new
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P5/pics/5a10.png?raw=true" alt="SS" width="30%"/>

```python
# Mengembalikan label yang diprediksi ke bentuk aslinya
lb.inverse_transform(y_new)
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P5/pics/5a11.png?raw=true" alt="SS" width="30%"/>


### 1.7. Visualisasi Nearest Neighbours

```python
import matplotlib.pyplot as plt  # Mengimpor pustaka matplotlib untuk visualisasi data

# Mencetak nama dan NPM
print ('Nama: Aditya Rimandi Putra')
print ('NPM : 41155050210030\n')

# Membuat figure dan axes untuk plot
fig, ax = plt.subplots()

# Mengelompokkan data berdasarkan jenis kelamin dan membuat scatter plot untuk setiap kelompok
for jk, d in sensus_df.groupby("jk"):
    ax.scatter(d['tinggi'], d['berat'], label=jk)  # Membuat scatter plot untuk tinggi dan berat berdasarkan jenis kelamin

# Menambahkan titik misterius (data baru) ke plot
plt.scatter(
    tinggi_badan,  # Tinggi badan dari data baru
    berat_badan,   # Berat badan dari data baru
    marker='s',    # Menggunakan bentuk persegi untuk titik
    color='red',   # Mengatur warna titik menjadi merah
    label='misterius'  # Memberikan label untuk titik misterius
)

# Menambahkan legend pada plot di posisi kiri atas
plt.legend(loc='upper left')

# Menambahkan judul pada plot
plt.title('Sebaran Data Tinggi Badan, Berat Badan, dan Jenis Kelamin')

# Menambahkan label untuk sumbu X
plt.xlabel('Tinggi Badan (cm)')

# Menambahkan label untuk sumbu Y
plt.ylabel('Berat Badan (kg)')

# Menampilkan grid pada plot
plt.grid(True)

# Menampilkan plot
plt.show()
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P5/pics/5a12.png?raw=true" alt="SS" width="55%"/>


### 1.8. Kalkulasi jarak dengan Euclidean Distance

```python
# Mencetak nama dan NPM
print ('Nama: Aditya Rimandi Putra')
print ('NPM : 41155050210030\n')

# Membuat array numpy dari tinggi badan dan berat badan
misterius = np.array([tinggi_badan, berat_badan])  # Mengonversi tinggi dan berat badan ke dalam array numpy

# Menampilkan array misterius yang berisi data baru
misterius  # Menampilkan nilai dari array misterius

```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P5/pics/5a13.png?raw=true" alt="SS" width="39%"/>

```python
# Menampilkan variabel x_train
X_train  # Menampilkan nilai dari variabel x_train
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P5/pics/5a14.png?raw=true" alt="SS" width="30%"/>

```python
from scipy.spatial.distance import euclidean  # Mengimpor fungsi euclidean dari pustaka scipy untuk menghitung jarak

# Mencetak nama dan NPM
print ('Nama: Aditya Rimandi Putra')
print ('NPM : 41155050210030\n')

# Menghitung jarak Euclidean antara titik misterius dan setiap titik dalam X_train
data_jarak = [euclidean(misterius, d) for d in X_train]  # Menggunakan list comprehension untuk menghitung jarak

# Menampilkan daftar jarak yang telah dihitung
data_jarak  # Menampilkan nilai dari data_jarak
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P5/pics/5a15.png?raw=true" alt="SS" width="35%"/>

```python
# Mencetak nama dan NPM
print ('Nama: Aditya Rimandi Putra')
print ('NPM : 41155050210030\n')

# Menambahkan kolom baru 'jarak' ke DataFrame sensus_df yang berisi data jarak yang telah dihitung
sensus_df['jarak'] = data_jarak  # Menyimpan jarak Euclidean ke dalam DataFrame

# Mengurutkan DataFrame berdasarkan kolom 'jarak'
sensus_df_sorted = sensus_df.sort_values(['jarak'])  # Mengurutkan DataFrame berdasarkan jarak

# Menampilkan DataFrame yang telah diurutkan
sensus_df_sorted  # Menampilkan hasil DataFrame yang telah diurutkan
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P5/pics/5a16.png?raw=true" alt="SS" width="39%"/>

### 1.9. Evaluasi KNN Classification Model | Persiapan testing set

**Testing Set**

```python
# Mencetak nama dan NPM
print ('Nama: Aditya Rimandi Putra')
print ('NPM : 41155050210030\n')

# Mendefinisikan data pengujian (X_test) sebagai array numpy
X_test = np.array([[168, 65], [180, 96], [160, 52], [169, 67]])
# Mendefinisikan label pengujian (y_test) dengan mengubah label kategori menjadi angka
y_test = lb.transform(np.array(['pria', 'pria', 'wanita', 'wanita'])).flatten()

# Mencetak X_test
print(f'X_test: \n{X_test}\n')
# Mencetak y_test
print(f'y_test: \n{y_test}')
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P5/pics/5a17.png?raw=true" alt="SS" width="39%"/>

**Prediksi Terhadap Testing Set**  

```python
# Mencetak nama dan NPM
print ('Nama: Aditya Rimandi Putra')
print ('NPM : 41155050210030\n')

# Menggunakan model untuk memprediksi label berdasarkan data pengujian
y_pred = model.predict(X_test)  # Menggunakan model yang telah dilatih untuk membuat prediksi

# Menampilkan hasil prediksi
y_pred  # Menampilkan nilai dari y_pred
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P5/pics/5a18.png?raw=true" alt="SS" width="39%"/>

### 1.10. Evaluasi model dengan accuracy score

```python
# Mencetak nama dan NPM
print ('Nama: Aditya Rimandi Putra')
print ('NPM : 41155050210030\n')

# Mengimpor fungsi accuracy_score dari sklearn.metrics
from sklearn.metrics import accuracy_score

# Menghitung akurasi model dengan membandingkan label yang sebenarnya dan prediksi
acc = accuracy_score(y_test, y_pred)

# Mencetak nilai akurasi
print(f'accuracy: {acc}')
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P5/pics/5a19.png?raw=true" alt="SS" width="39%"/>

### 1.11. Evaluasi model dengan precision score

```python
# Mencetak nama dan NPM
print ('Nama: Aditya Rimandi Putra')
print ('NPM : 41155050210030\n')

# Mengimpor fungsi precision_score dari sklearn.metrics
from sklearn.metrics import precision_score

# Menghitung presisi model dengan membandingkan label yang sebenarnya dan prediksi
prec = precision_score(y_test, y_pred)

# Mencetak nilai presisi
print(f'Precision: {prec}')
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P5/pics/5a20.png?raw=true" alt="SS" width="35%"/>

### 1.12. Evaluasi model dengan recall score 

```python
# Mencetak nama dan NPM
print ('Nama: Aditya Rimandi Putra')
print ('NPM : 41155050210030\n')

# Mengimpor fungsi recall_score dari sklearn.metrics
from sklearn.metrics import recall_score

# Menghitung recall model dengan membandingkan label yang sebenarnya dan prediksi
rec = recall_score(y_test, y_pred)

# Mencetak nilai recall
print(f'recall: {rec}') 
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P5/pics/5a21.png?raw=true" alt="SS" width="39%"/>

### 1.13. Evaluasi model dengan F1 score

```python
# Mencetak nama dan NPM
print ('Nama: Aditya Rimandi Putra')
print ('NPM : 41155050210030\n')

# Mengimpor fungsi f1_score dari sklearn.metrics
from sklearn.metrics import f1_score

# Menghitung F1-score model dengan membandingkan label yang sebenarnya dan prediksi
f1 = f1_score(y_test, y_pred)

# Mencetak nilai F1-score
print(f'f1_score: {f1}')
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P5/pics/5a22.png?raw=true" alt="SS" width="39%"/>

### 1.14. Evaluasi model dengan classification report

```python
# Mencetak nama dan NPM
print ('Nama: Aditya Rimandi Putra')
print ('NPM : 41155050210030\n')

# Mengimpor fungsi classification_report dari sklearn.metrics
from sklearn.metrics import classification_report

# Menghasilkan laporan klasifikasi dengan membandingkan label yang sebenarnya dan prediksi
cls_report = classification_report(y_test, y_pred)

# Mencetak laporan klasifikasi
print(f'Classification Report: \n{cls_report}')
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P5/pics/5a23.png?raw=true" alt="SS" width="50%"/>

### 1.15. Evaluasi model dengan Mathews Correlation Coefficient

```python
# Mencetak nama dan NPM
print ('Nama: Aditya Rimandi Putra')
print ('NPM : 41155050210030\n')

# Mengimpor fungsi matthews_corrcoef dari sklearn.metrics
from sklearn.metrics import matthews_corrcoef

# Menghitung Matthews Correlation Coefficient (MCC) model
mcc = matthews_corrcoef(y_test, y_pred)

# Mencetak nilai MCC
print(f'MCC: {mcc}')
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P5/pics/5a24.png?raw=true" alt="SS" width="30%"/>

Klik link dibawah ini untuk kembali ke menu utama:

> [!TIP]
> Kembali ke halaman utama, [Klik disini](https://github.com/AdityaR-AI/MLC/tree/main/)








