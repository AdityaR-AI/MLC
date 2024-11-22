
> Materi ini terbagi menjadi 2 Part, berikut linknya:

Silahkan klik link dibawah ini tuntuk menuju tugas yang inign dilihat:

> [!NOTE]
> Part 1 - Classification dengan KNN | K-Nearest Neighbours [Pages Link](https://github.com/AdityaR-AI/MLC/tree/main/P5/K%20Nearest%20Neigbor%20&%20Support%20Vector%20Machine_I.md)

> [!NOTE]
> Part 2 - Support Vector Machine Classification | SVM [Pages Link](https://github.com/AdityaR-AI/MLC/tree/main/P5/K%20Nearest%20Neigbor%20%26%20Support%20Vector%20Machine_II.md)

Klik link dibawah ini untuk kembali ke menu utama:

> [!TIP]
> Kembali ke halaman utama, [Klik disini](https://github.com/AdityaR-AI/MLC/tree/main/)

## Classification dengan KNN | K-Nearest Neighbours

### 2.0. Support Vector Machine (SVM). Lakukan praktik dari https://youtu.be/z69XYXpvVrE?si=KR_hDSlwjGIMcT0w , buat screenshot dengan nama kalian pada coding, kumpulkan dalam bentuk pdf, dari kegiatan ini:

**Klasifikasi dengan Support Vector Machine (SVM)**

Support Vector Machine (SVM) adalah salah satu algoritma machine learning yang paling kuat dan populer untuk klasifikasi, baik linear maupun non-linear. SVM bekerja dengan mencari hyperplane yang optimal untuk memisahkan data ke dalam kelas-kelas yang berbeda.

### 2.1. Pengenalan Decision Boundary & Hyperplane

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P5/pics/5bn1.png?raw=true" alt="SS" width="38%"/>

  1. **Pendahuluan Konsep SVM**
  - Bagian ini menjelaskan beberapa konsep dasar untuk membantu kita memahami mekanisme kerja dari Support Vector Machine (SVM).
  - Kita akan melihat contoh kasus klasifikasi dengan dua kelas, yaitu kelas hitam dan kelas putih.
  - Kedua kelas ini memiliki dua fitur, yaitu x1x_1x1 dan x2x_2x2, yang ditampilkan sebagai skala untuk masing-masing fitur.

  2. **Decision Boundary dalam Klasifikasi**
  - Tujuan kita adalah menarik garis lurus (garis linier) yang dapat memisahkan kedua kelas tersebut.
  - Dalam tugas klasifikasi, garis yang memisahkan kelas-kelas ini disebut Decision Boundary.
  - Pada contoh ini, terdapat tiga garis linier sebagai pilihan boundary: H1 (hijau), H2 (biru), dan H3 (merah).

  3. **Memilih Decision Boundary Terbaik**
  - Di antara garis H1, H2, dan H3, kita perlu memilih yang terbaik sebagai Decision Boundary.
  - Analisis Setiap Garis:
    - H1: Jelas bahwa H1 tidak efektif memisahkan kedua kelas, sehingga tidak cocok sebagai boundary.
    -	H2 dan H3: Kedua garis ini dapat memisahkan kelas, tetapi kita harus memilih yang terbaik.
  - Kesimpulan: H3 dipilih sebagai Decision Boundary terbaik karena memiliki Margin yang lebih besar dibandingkan H2.

  4. **Pengenalan Hyperplane**
  - Istilah Hyperplane biasa digunakan dalam SVM untuk mewakili Decision Boundary.
  - Dalam kasus ini, kita memiliki dua fitur, sehingga menghasilkan plot dua dimensi dan Decision Boundary berupa garis linier.
  - Dimensi Berbeda:
    - Jika hanya ada satu fitur, Decision Boundary akan berupa satu titik atau ambang nilai.
    - Dengan tiga fitur, boundary menjadi bidang datar.
    - Dengan empat atau lebih fitur, boundary menjadi bidang multidimensi yang dikenal sebagai Hyperplane.
  - Dalam SVM, untuk mempermudah, setiap Decision Boundary disebut sebagai Hyperplane.

  5. **Tujuan SVM**
  - Tujuan utama dari SVM adalah mencari Decision Boundary yang dapat memisahkan kelas dengan baik.
  - SVM berusaha mencari Decision Boundary dengan margin terbesar untuk memisahkan kelas dengan lebih baik.

### 2.2. Pengenalan Support Vector & Maximum Margin

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P5/pics/5bn2.png?raw=true" alt="SS" width="33%"/>

  1. **Definisi Margin dalam SVM**
  - Margin adalah jarak terdekat antara decision boundary dengan anggota dari kelas yang ingin dipisahkan.
  - Margin ditentukan untuk memaksimalkan pemisahan antara dua kelas, dalam contoh ini adalah kelas biru dan kelas hijau.

  2. **Contoh Kasus untuk Memahami Margin**
  -	Kita memiliki dua kelas: kelas biru dan kelas hijau.
  -Setiap kelas diwakili oleh dua fitur, yaitu x1 dan x2, yang diukur dalam skala tertentu.

  3. **Decision Boundary dan Margin**
  - Dalam ilustrasi ini, garis lurus merah adalah decision boundary yang memisahkan kelas biru dan hijau.
  - Area berwarna kuning yang mengapit decision boundary adalah margin.
  - Margin ini didapatkan dari jarak terdekat antara decision boundary dengan titik-titik dari kelas yang berdekatan.

  4. **Support Vector**
  - Titik-titik yang berada paling dekat dengan decision boundary dan menentukan posisi margin disebut support vector.
  - Pada contoh ini, terdapat tiga titik yang bertindak sebagai support vector:
    - Support Vector 1: Titik biru pertama yang dekat dengan boundary.
    - Support Vector 2: Titik biru kedua yang dekat dengan boundary.
    - Support Vector 3: Titik hijau yang dekat dengan boundary.
  - Support vector ini adalah titik-titik dari masing-masing kelas yang paling dekat dengan decision boundary.

  5. **Maximum Margin**
  - Support Vector Machine (SVM) bertujuan untuk memilih decision boundary berdasarkan margin terbesar, yang dikenal dengan istilah Maximum Margin.
  - Dengan memilih maximum margin, SVM mampu memisahkan kelas dengan lebih baik dan memberikan margin terluas antara kelas.

  6. **Kesimpulan Dasar SVM**
  - SVM berfokus pada penentuan decision boundary dengan maksimum margin, dibantu oleh support vector untuk memaksimalkan jarak pemisahan.
  - Support vector adalah anggota kelas yang berperan penting dalam menentukan batas pemisahan ini.

### 2.3. Pengenalan kondisi Linearly Inseparable dan Kernel Tricks

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P5/pics/5bn3.png?raw=true" alt="SS" width="90%"/>

  1. **Penggunaan Decision Boundary Linear**
  - Pada contoh-contoh sebelumnya, kita menggunakan garis lurus atau garis linear sebagai decision boundary untuk memisahkan dua kelas.
  - Namun, ada beberapa kasus dimana pemisahan kelas tidak bisa dilakukan dengan garis linear.

  2. **Contoh Kasus Linearly Inseparable**
  - Dalam contoh ini, terdapat dua kelas: kelas titik dan kelas X.
  - Data ini memiliki dua fitur, sehingga ketika di-plot, menghasilkan plot 2 dimensi seperti yang terlihat.
  - Pada kasus ini, garis linear tidak bisa memisahkan kedua kelas dengan baik. Kondisi seperti ini disebut linear inseparable.

  3. **Proyeksi ke Dimensi Lebih Tinggi**
  - Untuk mengatasi masalah linear inseparable, SVM memproyeksikan data ke dimensi yang lebih tinggi.
  - Misalnya, data yang awalnya berada di dua dimensi dapat diproyeksikan ke tiga dimensi.
  - Setelah diproyeksikan, data menjadi lebih mudah dipisahkan menggunakan decision boundary berbentuk bidang datar.

  4. **Pentingnya Decision Boundary di Dimensi Tinggi**
  - Pada proyeksi tiga dimensi, kita dapat menggunakan bidang datar sebagai decision boundary untuk memisahkan kelas X dengan kelas titik.
  - Teknik ini membuat pemisahan kelas menjadi lebih mudah dibandingkan jika tetap berada di dua dimensi.

  5. **Kenaikan Beban Komputasi**
  - Memproyeksikan data ke dimensi yang lebih tinggi dapat meningkatkan beban komputasi.
  - Untuk mengatasi hal ini, SVM menggunakan teknik efisien yang disebut Kernel Tricks.
  
  6. **Kernel Tricks dalam SVM**
  - Kernel Tricks memungkinkan SVM untuk melakukan proyeksi ke dimensi lebih tinggi tanpa harus benar-benar melakukan perhitungan di dimensi tersebut, sehingga menghemat komputasi.
  - SVM menawarkan beberapa jenis kernel seperti polinomial, sigmoid, dan Radial Basis Function (RBF).

  7. **Kesimpulan**
  - Penggunaan support vector dan kernel tricks dalam pembentukan decision boundary adalah alasan mengapa model ini dinamakan Support Vector Machine (SVM).
  - Penjelasan ini mencakup konsep dasar SVM yang penting untuk dipahami.

### 2.4. Pengenalan MNIST Handwritten Digits Dataset

```python
# Mencetak nama dan NPM
print ('Nama: Aditya Rimandi Putra')
print ('NPM : 41155050210030\n')

# Mengimpor fungsi fetch_openml dari sklearn.datasets
from sklearn.datasets import fetch_openml

# Mengambil dataset MNIST dari OpenML
X, y = fetch_openml('mnist_784', data_home='./dataset/mnist', return_X_y=True)

# Memeriksa bentuk (shape) dari data fitur X
X.shape
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P5/pics/5b1.png?raw=true" alt="SS" width="25%"/>

```python
# Mengimpor pustaka yang diperlukan
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

# Inisialisasi posisi subplot
pos = 1

# Mengonversi X menjadi array NumPy (jika belum)
X = np.array(X)

# Menampilkan 8 gambar pertama dari dataset MNIST
for data in X[:8]:
    plt.subplot(1, 8, pos)  # Membuat subplot dengan 1 baris dan 8 kolom
    plt.imshow(data.reshape((28, 28)), cmap=cm.Greys_r)  # Mengubah data menjadi gambar 28x28 dan menggunakan colormap grayscale
    plt.axis('off')  # Mematikan sumbu
    pos += 1  # Meningkatkan posisi untuk subplot berikutnya

# Menampilkan gambar
plt.show()

# Menampilkan label untuk 8 gambar pertama
y[:8]
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P5/pics/5b2.png?raw=true" alt="SS" width="85%"/>

```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

# Menggunakan 1000 contoh pertama untuk data pelatihan
# X_train = X[:60000]
# y_train = y[:60000]

# Mengambil data untuk pengujian dari indeks 60000 hingga akhir
# X test = X[60000:]
#y_test = y[60000:]

X_train = X[:1000]  # Mengambil 1000 elemen pertama dari X untuk data pelatihan
y_train = y[:1000]  # Mengambil 1000 elemen pertama dari y untuk label pelatihan

# Menggunakan sisa data untuk data pengujian
X_test = X[69000:]  # Mengambil elemen dari indeks 69000 hingga akhir dari X untuk data pengujian
y_test = y[69000:]  # Mengambil elemen dari indeks 69000 hingga akhir dari y untuk label pengujian
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P5/pics/5b3.png?raw=true" alt="SS" width="25%"/>

### 2.5. Klasifikasi dengan Support Vector Classifier | SVC

```python
# Mengimpor pustaka SVM dari scikit-learn
from sklearn.svm import SVC

# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

# Membuat model SVM dengan random_state untuk reproduktifitas
model = SVC(random_state=0)

# Melatih model dengan data pelatihan
model.fit(X_train, y_train)
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P5/pics/5b4.png?raw=true" alt="SS" width="25%"/>

```python
# Mengimpor fungsi untuk evaluasi klasifikasi dari scikit-learn
from sklearn.metrics import classification_report

# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

# Menggunakan model untuk memprediksi label dari data pengujian
y_pred = model.predict(X_test)

# Mencetak laporan klasifikasi yang menunjukkan metrik evaluasi
print(classification_report(y_test, y_pred))
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P5/pics/5b5.png?raw=true" alt="SS" width="72%"/>

### 2.6. Hyperparameter Tuning dengan Grid Search

```python
# Mengimpor GridSearchCV dari scikit-learn
from sklearn.model_selection import GridSearchCV

# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

# Menentukan parameter yang akan diuji dalam pencarian grid
parameters = {
    'kernel': ['rbf', 'poly', 'sigmoid'],  # Jenis kernel yang akan diuji
    'C': [0.5, 1, 10, 100],                 # Parameter regularisasi
    'gamma': ['scale', 1, 0.1, 0.01, 0.001] # Parameter gamma untuk kernel RBF
}

# Membuat objek GridSearchCV
grid_search = GridSearchCV(
    estimator=SVC(random_state=0),  # Estimator yang digunakan
    param_grid=parameters,           # Parameter yang akan diuji
    n_jobs=6,                        # Jumlah pekerjaan yang akan dijalankan secara paralel
    verbose=1,                       # Menampilkan informasi proses
    scoring='accuracy'               # Metode evaluasi yang digunakan
)

# Melatih model dengan pencarian grid
grid_search.fit(X_train, y_train)
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P5/pics/5b6.png?raw=true" alt="SS" width="60%"/>

```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

# Mencetak skor terbaik dari pencarian grid
print(f'Best Score: {grid_search.best_score_}')

# Mengambil parameter terbaik dari model terbaik
best_params = grid_search.best_estimator_.get_params()

# Mencetak parameter terbaik
print(f'Best Parameters:')
for param in parameters:
    print(f'\t{param}: {best_params[param]}')
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P5/pics/5b7.png?raw=true" alt="SS" width="29%"/>

### 2.7. Evaluasi Model

```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

# Menggunakan model terbaik dari pencarian grid untuk memprediksi label data pengujian
y_pred = grid_search.predict(X_test)

# Mencetak laporan klasifikasi untuk evaluasi performa model
print(classification_report(y_test, y_pred))
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P5/pics/5b8.png?raw=true" alt="SS" width="82%"/>

Klik link dibawah ini untuk kembali ke menu utama:

> [!TIP]
> Kembali ke halaman utama, [Klik disini](https://github.com/AdityaR-AI/MLC/tree/main/)








