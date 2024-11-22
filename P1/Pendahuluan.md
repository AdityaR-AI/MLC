Klik link dibawah ini untuk kembali ke menu utama:

> [!TIP]
> Kembali ke halaman utama, [Klik disini](https://github.com/AdityaR-AI/MLC/tree/main/)

## Pendahuluan

### 1.0. Instalasi Jupyter Noterbook, Lakukan download dan instalasi:
  
> Saya disini menggunakan Anaconda daripada menginstal Jupyter langsung karena dialamnya sudah terdapat beragam Library dan Jupyter itu sendiri
  1.	Kunjungi https://www.anaconda.com/download

![SS](https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P1/pics/1a1.png?raw=true)

### 1.1. Jupyter Notebook (https://jupyter.org/), dan Library python seperti NumPy, SciPy, Pandas, Matplotlib, Seaborn, Scikit-learn.
  1. Cek apakah Jupyter sudah tersedia, buka anaconda prompt atau shell lalu ketik perintah seperti Digambar

![SS](https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P1/pics/1a2.png?raw=true)

![SS](https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P1/pics/1a3.png?raw=true)

  2. Cek apakah library tersebut sudah tersedia saat instalasi Anaconda selesai

![SS](https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P1/pics/1a4.png?raw=true)

### 1.2. Tuliskan nama dan nomor NPM anda pada Jupiter Notebook.
  1. untuk menambahkan Jupiter Notebook klik File > New > Notebook atau bisa juga buka VSCode newfile namai ektensinya .ipynb, dan disini saya pake VSCode
  2. Buka VSCodel alu dibagian kanan atas pilih Python Environtment dari Anaconda

![SS](https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P1/pics/1a5.png?raw=true)

  3. Buat file baru dengan nama ektensi .ipynb lalu isi Nama dan NPM

```python
# Menentukan nama dan NPM
nama = "Aditya Rimandi Putra"  
npm = "41155050210030"  

# Menampilkan nama dan NPM
print(f"Nama: {nama}")
print(f"NPM: {npm}")
```

### 1.3. Buat screenshot sebagai jawaban nomor 1 di Tugas Pertemuan 1!

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P1/pics/1a6.png?raw=true" alt="SS" width="300"/>

### 2.0. Menggunakan Google Collab, Lakukan

### 2.1. Gunakan Google Colab (https://colab.research.google.com/).

  1. Buka https://colab.research.google.com/

### 2.2. Tuliskan nama dan nomor NPM anda pada Google Colab.

 1. Buat Notebook bar

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P1/pics/1b1.png?raw=true" alt="SS" width="80%"/>

  2. Tulis Nama dan NPM

```python
# Menentukan nama dan NPM
nama = "Aditya Rimandi Putra"  
npm = "41155050210030"  

# Menampilkan nama dan NPM
print(f"Nama: {nama}")
print(f"NPM: {npm}")
```

### 2.3. Buat screenshot sebagai jawaban nomor 2 di Tugas Pertemuan 1!

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P1/pics/1b2.png?raw=true" alt="SS" width="50%"/>

### 3.0. Buatlah akun di https://www.kaggle.com
	
  1. Buka Kaggle
  2. Pilih Sign Up, dan ikuti langkahnya	
  3. Akun berhasil dibuat

### 3.1. Buat screenshot sebagai jawaban nomor 3 di Tugas Pertemuan 1!

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P1/pics/1c1.png?raw=true" alt="SS" width="80%"/>

### 4.0. Buatlah akun di https://github.com
	
  1. Bukaa Github
  2. Pilih Sign Up, dan ikuti langkahnya
  3. Akun berhasil dibuat
 
### 4.1. Buat screenshot sebagai jawaban nomor 4 di Tugas Pertemuan 1!

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P1/pics/1d1.png?raw=true" alt="SS" width="80%"/>

### 5.0. Lakukan praktek dari https://youtu.be/mSO2hJln0OY?feature=shared . Praktek tersebut yaitu:

### 5.1. Load sample dataset

```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

# Mengimpor dataset iris dari sklearn
from sklearn.datasets import load_iris

# Memuat dataset iris
iris = load_iris()

# Menampilkan dataset iris
iris
```
<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P1/pics/1e1.png?raw=true" alt="SS" width="60%"/>

```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

iris.keys()
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P1/pics/1e2.png?raw=true" alt="SS" width="90%"/>

### 5.2. Metadata | Deskripsi dari sample dataset

Referensi Iris flower dataset: https://en.wikipedia.org/wiki/Iris_flower_data_set 

```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

# Menampilkan deskripsi dari dataset iris
print(iris.DESCR)
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P1/pics/1e3.png?raw=true" alt="SS" width="90%"/>

### 5.3. Explanatory & Response Variables | Features & Target

  Explanatory Variable (Features)


```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

# Mendefinisikan X sebagai data dari dataset iris
X = iris.data

# Menampilkan bentuk dari X
X.shape

# Menampilkan data X (opsional, bisa di-uncomment jika ingin melihat data)
# X
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P1/pics/1e4.png?raw=true" alt="SS" width="30%"/>

```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

# Mendefinisikan X sebagai data dari dataset iris
X = iris.data

# Menampilkan bentuk dari X (baris ini di-comment)
# X.shape

# Menampilkan data X
X
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P1/pics/1e5.png?raw=true" alt="SS" width="50%"/>

  Response Variable (Target)

```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

# Mendefinisikan y sebagai target dari dataset iris
y = iris.target

# Menampilkan bentuk dari y
y.shape

# Menampilkan data y (opsional, bisa di-uncomment jika ingin melihat data)
# y
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P1/pics/1e6.png?raw=true" alt="SS" width="30%"/>

```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

# Mendefinisikan y sebagai target dari dataset iris
y = iris.target

# Menampilkan bentuk dari y
# y.shape

# Menampilkan data y 
y
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P1/pics/1e7.png?raw=true" alt="SS" width="80%"/>

### 5.4. Feature & Target Names

```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

# Mendefinisikan feature_names sebagai nama fitur dari dataset iris
feature_names = iris.feature_names

# Menampilkan nama fitur
feature_names
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P1/pics/1e8.png?raw=true" alt="SS" width="50%"/>

```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

# Mendefinisikan target_names sebagai nama target dari dataset iris
target_names = iris.target_names

# Menampilkan nama target
target_names
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P1/pics/1e9.png?raw=true" alt="SS" width="70%"/>

### 5.5. Visualisasi Data

Visualisasi Sepal Length & Width


```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

# Mengimpor library yang diperlukan
import matplotlib.pyplot as plt

# Mengambil hanya dua fitur pertama (Sepal length dan Sepal width)
X = X[:, :2]

# Menentukan batas untuk sumbu x dan y
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

# Membuat scatter plot
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.grid(True)
plt.show()
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P1/pics/1e10.png?raw=true" alt="SS" width="60%"/>

### 5.6. Training Set & Testing Set

```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

# Mengimpor library yang diperlukan
from sklearn.model_selection import train_test_split

# Membagi dataset menjadi set pelatihan dan set pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Menampilkan ukuran set pelatihan dan set pengujian
print(f'X train: {X_train.shape}')
print(f'X test: {X_test.shape}')
print(f'y train: {y_train.shape}')
print(f'y test: {y_test.shape}')
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P1/pics/1e11.png?raw=true" alt="SS" width="40%"/>

### 5.7. Load sample dataset sebagai Pandas Data Frame

```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

# Mengimpor library yang diperlukan
from sklearn.datasets import load_iris

# Memuat dataset iris sebagai DataFrame
iris = load_iris(as_frame=True)

# Mengambil fitur dari dataset iris ke dalam DataFrame
iris_features_df = iris.data

# Menampilkan DataFrame fitur
iris_features_df
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P1/pics/1e12.png?raw=true" alt="SS" width="70%"/>

### 6.0. Lakukan praktek dari https://youtu.be/tiREcHrtDLo?feature=shared  . Praktek tersebut yaitu:

### 6.1. Persiapan dataset | Loading & splitting dataset

  Load Sample Dataset: Iris Dataset
  
```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

# Mengimpor library yang diperlukan
from sklearn.datasets import load_iris

# Memuat dataset iris
iris = load_iris()

# Mengambil fitur dari dataset iris
X = iris.data

# Mengambil target dari dataset iris
y = iris.target
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P1/pics/1e13.png?raw=true" alt="SS" width="40%"/>

  Splitting Dataset: Tarinning and Testing Sets

```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

# Mengimpor fungsi untuk membagi dataset
from sklearn.model_selection import train_test_split

# Membagi dataset menjadi set pelatihan dan set pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P1/pics/1e14.png?raw=true" alt="SS" width="40%"/>

### 6.2. Training model Machine Learning

**Training Model**

  -	Di Scikit-learn, model machine learning dibuat dari kelas yang disebut estimator. Ini seperti cetakan kue yang digunakan untuk membuat kue dengan bentuk yang sama berulang kali. Estimator ini akan menentukan jenis model machine learning yang akan kita gunakan.
  
  -	Setiap estimator memiliki dua metode utama: fit() dan predict(). 
  
    -	fit(): Metode ini digunakan untuk melatih model. Kita memberikan data latih (X_train dan y_train) kepada model, dan model akan belajar pola-pola dalam data tersebut. Proses ini mirip seperti seorang guru yang mengajarkan siswa dengan memberikan contoh-contoh soal.
    
    -	predict(): Setelah model dilatih, kita bisa menggunakan metode predict() untuk melakukan prediksi. Kita memberikan data baru (X_test) kepada model, dan model akan memprediksi label atau kelas untuk data tersebut. Ini seperti ketika kita memberikan soal ujian kepada siswa yang sudah belajar, dan siswa tersebut akan berusaha menjawab soal tersebut berdasarkan apa yang telah dipelajarinya.

```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

# Mengimpor KNeighborsClassifier dari sklearn
from sklearn.neighbors import KNeighborsClassifier

# Membuat model KNN dengan 3 tetangga terdekat
model = KNeighborsClassifier(n_neighbors=3)

# Melatih model dengan data pelatihan
model.fit(X_train, y_train)
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P1/pics/1e15.png?raw=true" alt="SS" width="40%"/>

### 6.3. Evaluasi model Machine Learning

```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

# Mengimpor fungsi untuk menghitung akurasi
from sklearn.metrics import accuracy_score

# Melakukan prediksi pada data pengujian
y_pred = model.predict(X_test)

# Menghitung akurasi model
acc = accuracy_score(y_test, y_pred)

# Mencetak akurasi
print(f'Accuracy: {acc}')
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P1/pics/1e16.png?raw=true" alt="SS" max-width="40%"/>

### 6.4. Pemanfaatan trained model machine learning

 ```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

# Mendefinisikan data baru untuk prediksi
data_baru = [[5, 5, 3, 2],
             [2, 4, 3, 5]]

# Melakukan prediksi pada data baru
preds = model.predict(data_baru)

# Menampilkan hasil prediksi
preds
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P1/pics/1e17.png?raw=true" alt="SS" max-width="40%"/>

 ```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

# Mengonversi prediksi kelas numerik menjadi nama spesies
pred_species = [iris.target_names[p] for p in preds]

# Mencetak hasil prediksi
print(f'Hasil Prediksi: {pred_species}')
```
<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P1/pics/1e18.png?raw=true" alt="SS" width="60%"/>

### 6.5. Deploy model Machine Learning | Dumping dan Loading model Machine Learning

Dumping Model


 ```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

# Mengimpor pustaka joblib untuk menyimpan model
import joblib

# Menyimpan model ke dalam file
joblib.dump(model, 'iris_classifier_knn.joblib')
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P1/pics/1e19.png?raw=true" alt="SS" width="40%"/>

  Loading model Machine Learning dari File joblib

 ```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

# Memuat model dari file
production_model = joblib.load('iris_classifier_knn.joblib')
```
<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P1/pics/1e20.png?raw=true" alt="SS" width="40%"/>

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P1/pics/1emiss.png?raw=true" alt="SS" width="20%"/>

### 7.0. Lakukan praktek dari https://youtu.be/smNnhEd26Ek?feature=shared  . Praktek tersebut yaitu:

### 7.1. Persiapan sample dataset

 ```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

# Mengimpor pustaka yang diperlukan
import numpy as np
from sklearn import preprocessing

# Mendefinisikan data contoh sebagai array NumPy
sample_data = np.array([[2.1, -1.9, 5.5],
                         [-1.5, 2.4, 3.5],
                         [0.5, -7.9, 5.6],
                         [5.9, 2.3, -5.8]])

# Menampilkan data contoh
sample_data
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P1/pics/1e21.png?raw=true" alt="SS" width="40%"/>

 ```python
# Mendapatkan bentuk dari sample_data
sample_data.shape
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P1/pics/1e22.png?raw=true" alt="SS" width="30%"/>

### 7.2. Teknik data preprocessing 1: binarization

 ```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

# Menampilkan sample_data
sample_data
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P1/pics/1e23.png?raw=true" alt="SS" width="40%"/>

 ```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

# Membuat objek Binarizer dengan threshold 0.5
preprocessor = preprocessing.Binarizer(threshold=0.5)

# Menerapkan binarization pada sample_data
binarised_data = preprocessor.transform(sample_data)

# Menampilkan hasil binarization
binarised_data
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P1/pics/1e24.png?raw=true" alt="SS" width="40%"/>

### 7.3. Teknik data preprocessing 2: scaling

 ```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

# Menampilkan sample_data
sample_data
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P1/pics/1e25.png?raw=true" alt="SS" width="40%"/>

 ```python
# Membuat objek MinMaxScaler
preprocessor = preprocessing.MinMaxScaler(feature_range=(0, 1))

# Menghitung parameter untuk scaling
preprocessor.fit(sample_data)

# Menerapkan transformasi untuk menormalisasi data
scaled_data = preprocessor.transform(sample_data)
scaled_data
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P1/pics/1e26.png?raw=true" alt="SS" width="60%"/>

 ```python
# Alternatif: Menghitung dan menerapkan scaling dalam satu langkah
scaled_data = preprocessor.fit_transform(sample_data)
scaled_data
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P1/pics/1e27.png?raw=true" alt="SS" width="60%"/>

### 7.4. Teknik data preprocessing 3: normalization

  L1 Normalisasi: Deviasi Absolut Terkecil 
  
  Referensi: https://en.wikipedia.org/wiki/Least_absolute_deviations

 ```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

# Menampilkan sample_data
sample_data
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P1/pics/1e28.png?raw=true" alt="SS" width="40%"/>

 ```python
# Melakukan L1 normalisasi pada data
l1_normalised_data = preprocessing.normalize(sample_data, norm='l1')

# Menampilkan hasil normalisasi
print(l1_normalised_data)
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P1/pics/1e29.png?raw=true" alt="SS" width="40%"/>

  L2 Normalisasi (Least Squares) 

  Referensi: https://en.wikipedia.org/wiki/Least_squares 

 ```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

# Menampilkan sample_data
sample_data
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P1/pics/1e30.png?raw=true" alt="SS" width="40%"/>

 ```python
# Melakukan L2 normalisasi pada data
l2_normalised_data = preprocessing.normalize(sample_data, norm='l2')

# Menampilkan hasil normalisasi
print(l2_normalised_data)
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P1/pics/1e31.png?raw=true" alt="SS" width="40%"/>

Klik link dibawah ini untuk kembali ke menu utama:

> [!TIP]
> Kembali ke halaman utama, [Klik disini](https://github.com/AdityaR-AI/MLC/tree/main/)










