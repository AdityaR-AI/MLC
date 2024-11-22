## Decision Tree & Random Fores

### Decision Tree Classification

> Materi ini terbagi menjadi 2 Part, berikut linknya:

Silahkan klik link dibawah ini tuntuk menuju tugas yang inign dilihat:

> [!NOTE]
> Part 1 - Decision Tree Classification [Pages Link](https://github.com/AdityaR-AI/MLC/tree/main/P3/Decision%20Tree%20%26%20Random%20Fores_I.md)

> [!NOTE]
> Part 2 - Random Forest Classification [Pages Link](https://github.com/AdityaR-AI/MLC/tree/main/P3/Decision%20Tree%20%26%20Random%20Forest_II.md)

Klik link dibawah ini untuk kembali ke menu utama:

> [!TIP]
> Kembali ke halaman utama, [Klik disini](https://github.com/AdityaR-AI/MLC/tree/main/)

### 1.0. Lakukan praktik dari https://youtu.be/5wwXKtLkyqs?si=fn88eveu_qbCC6b3 , buat screenshot dengan nama kalian pada coding, kumpulkan dalam bentuk pdf, dari kegiatan ini: 

### 1.1. Pengenalan komponen Decision Tree: root, node, leaf

Decision Tree adalah salah satu algoritma populer dalam machine learning, yang strukturnya menyerupai pohon bercabang untuk memecahkan masalah klasifikasi atau regresi. Tiga komponen utama dalam Decision Tree adalah Terminology: **root node**, **internal node**, dan **leaf node**. 

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P3/3a1.png?raw=true" alt="SS" width="62%"/>

  -	**Root Node**: Ini adalah node pertama yang mewakili fitur utama atau pertanyaan pertama yang membagi dataset menjadi subset lebih kecil. Dalam diagram di atas, root node adalah pertanyaan "Fear technology?" yang memulai proses pengambilan keputusan.
  -	**Internal Node**: Ini adalah node yang berada di antara root dan leaf node, mewakili keputusan berdasarkan fitur tambahan. Misalnya, pertanyaan "Your dad rich?" dan "Care about privacy?" adalah internal node yang mempersempit klasifikasi lebih lanjut.
  -	**Leaf Node**: Ini adalah node terminal yang tidak memiliki cabang lebih lanjut dan mewakili hasil akhir dari proses klasifikasi. Di diagram, ikon-ikon seperti Apple, Chrome, Ubuntu, dan Windows adalah leaf node yang mewakili hasil dari pengambilan keputusan berdasarkan input fitur.

Selain itu, terdapat beberapa algoritma lain yang sering digunakan dalam pembuatan Decision Tree seperti **ID3, C4.5, dan C5.0**. Algoritma-algoritma ini memiliki cara berbeda dalam memilih fitur pada setiap node, tetapi prinsip dasarnya tetap sama: membagi dataset untuk menghasilkan keputusan yang optimal.

### 1.2. Pengenalan Gini Impurity

Gini Impurity adalah salah satu metode yang digunakan untuk mengukur seberapa murni suatu node pada Decision Tree. Nilai Gini Impurity memberikan indikasi seberapa sering elemen yang dipilih secara acak dari kumpulan data akan diklasifikasikan secara salah jika berdasarkan distribusi kelas dari data tersebut.

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P3/3a6.png?raw=true" alt="SS" width="95%"/>

Pada gambar di atas, contoh visualisasi menunjukkan dua kelompok titik biru dan hijau yang dipisahkan oleh garis vertikal. Dalam analisis ini, terdapat dua ruas yang dihitung, yaitu **Ruas Kiri dan Ruas Kanan**, untuk mendapatkan nilai impurity mereka masing-masing. Berikut adalah langkah-langkah perhitungan Gini Impurity:
  1. **Ruas Kiri:**
     - Semua titik di ruas kiri adalah biru, sehingga probabilitas kelas biru P(biru) adalah 1.
     -	Menggunakan rumus G = 1 - Σ Pi<sup>2</sup>, karena hanya ada satu kelas, Gini Impurity untuk ruas kiri adalah 0.

  2. **Ruas Kanan:**
     -	Di sini terdapat dua kelas, yaitu biru dan hijau. Probabilitas untuk kelas biru adalah 1/6 dan untuk kelas hijau adalah 5/6.
     -	Dengan memasukkan nilai probabilitas ke dalam rumus Gini Impurity, kita mendapatkan Gini Impurity untuk ruas kanan sebesar 0.278.

  3. **Average Gini Impurity:**
     -	Untuk menghitung rata-rata Gini Impurity, kita mengambil proporsi data di ruas kiri dan kanan, lalu mengalikan dengan Gini Impurity dari masing-masing ruas.
     -	Nilai akhir dari Average Gini Impurity adalah 0.1668.
Konsep ini penting dalam pembuatan Decision Tree karena digunakan untuk menentukan seberapa baik suatu fitur membagi dataset pada setiap node. Fitur yang menghasilkan Gini Impurity terendah dianggap sebagai pemisah terbaik pada tahap tersebut.

### 1.3. Pengenalan Information Gain
Information Gain (IG) merupakan salah satu konsep penting dalam pembentukan Decision Tree. Information Gain mengukur pengurangan ketidakpastian (impurity) setelah dataset dibagi berdasarkan fitur tertentu. Dengan kata lain, Information Gain digunakan untuk menentukan seberapa baik suatu fitur membagi data pada node Decision Tree.

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P3/3a7.png?raw=true" alt="SS" width="40%"/>

Pada gambar di atas, konsep Information Gain dijelaskan dengan perhitungan menggunakan Gini Impurity:
  -	**Gini Impurity sebelum pemisahan:** Gini Impurity awal untuk node atas adalah 0.6779. Ini menunjukkan tingkat ketidakpastian sebelum data dipisahkan.
  -	**Gini Impurity setelah pemisahan:** Setelah data dipisahkan, kita menghitung rata-rata Gini Impurity dari dua node hasil pemisahan. Dalam

contoh ini, Average Gini Impurity dari dua node hasil pemisahan adalah 0.1668.
Untuk mendapatkan Information Gain, kita cukup mengurangi nilai Average Gini Impurity dari Gini Impurity awal:

**Information Gain** = 0.6779 - 0.1668 = 0.51

Nilai ini menunjukkan bahwa pemisahan tersebut mengurangi ketidakpastian sebesar 0.51, yang berarti fitur yang digunakan untuk pemisahan ini cukup efektif dalam memisahkan data menjadi kelompok-kelompok yang lebih homogen.

### 1.4. Membangun Decision Tree
ujuan utama dari membangun decision tree adalah untuk menemukan pola atau aturan klasifikasi yang paling efektif dalam memisahkan kelas-kelas dalam data. Untuk itu, atribut-atribut yang memberikan informasi terbaik dalam memisahkan data dipilih untuk membuat percabangan dalam pohon. Metode evaluasi seperti information gain atau Gini Impurity sering digunakan untuk menentukan seberapa baik suatu atribut memisahkan kelas-kelas.

Proses ini terdiri dari:
  -	Membuat keputusan
  -	Masukkan beberapa pertimbangan untuk mengambil keputusan
  -	Melakukan pengujian
  -	Membuat Kesimpulan

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P3/3a8.png?raw=true" alt="SS" width="80%"/>

Sekarang kita akan masuk pada langkah-langkah membangun decision tree, yang dijelaskan dalam gambar.

**Penjelasan Konten "Membangun Decision Tree"**

> **1. Tabel Data**

Gambar ini dimulai dengan contoh dataset yang terdiri dari 5 baris, dengan kolom sebagai berikut:
  -	Color (Warna buah): Green (Hijau), Yellow (Kuning), Red (Merah).
  -	Diameter (Ukuran buah, dalam satuan numerik).
  -	Label (Jenis buah): Apple, Grape, Lemon.
  -	
Dataset ini menjadi dasar untuk membangun decision tree yang akan mengklasifikasikan jenis buah berdasarkan warna dan ukuran.

> **2. Proses Pemilihan Fitur (Splitting Criteria)**

Di sebelah kanan, ditampilkan kemungkinan pertanyaan yang dapat digunakan untuk memisahkan data, seperti:
  -	Apakah warna == Hijau?
  -	Apakah warna == Kuning?
  -	Apakah warna == Merah?
  -	Apakah diameter <= 1?
  -	Apakah diameter <= 3?

Langkah ini penting dalam membangun decision tree, di mana kita harus memilih atribut atau fitur yang paling baik dalam memisahkan kelas-kelas yang ada. Pemilihan atribut dilakukan berdasarkan information gain atau metrik lainnya yang mengukur seberapa baik pemisahan dilakukan. Dalam gambar, node biru menunjukkan titik pemisahan data dengan fitur yang memberikan information gain tertinggi.

> **3. Perhitungan Gini Impurity**

Di bagian bawah gambar, ditampilkan perhitungan Gini Impurity untuk dataset awal, yang digunakan untuk mengukur ketidakmurnian kelas sebelum dilakukan pemisahan (split). Rumus Gini Impurity adalah sebagai berikut:

  - > G = 1 - (P(apple)<sup>2</sup> + P(grape)<sup>2</sup> + P(lemon)<sup>2</sup>)

Di mana P(apple), P(grape), dan P(lemon) adalah probabilitas kemunculan masing-masing kelas dalam dataset. Untuk dataset ini:
  -	Probabilitas kemunculan apple adalah 2/5,
  -	Probabilitas grape adalah 2/5, dan
  -	Probabilitas lemon adalah 1/5.

Setelah perhitungan, nilai Gini Impurity adalah 0.63. Semakin kecil nilai Gini, semakin baik data terpisah dengan bersih ke dalam kelas-kelas yang berbeda. Gini Impurity digunakan sebagai dasar untuk memilih fitur terbaik untuk split, dengan tujuan meminimalkan ketidakmurnian (impurity).

Dengan langkah-langkah ini, decision tree dibangun dengan memisahkan data secara bertahap hingga setiap cabang berakhir pada klasifikasi tertentu (misalnya, buah apple, grape, atau lemon).

### 1.5. Persiapan dataset: Iris Dataset

```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

# Mengimpor fungsi load_iris dari modul sklearn.datasets
from sklearn.datasets import load_iris

# Memuat dataset iris dan membaginya menjadi fitur (X) dan label target (y)
# Parameter return_X_y=True menunjukkan bahwa kita ingin data dalam dua variabel terpisah
X, y = load_iris(return_X_y=True)

# Mencetak dimensi dari array fitur X
# X.shape mengembalikan tuple yang berisi (jumlah sampel, jumlah fitur)
print(f'Dimensi Feature: {X.shape}')

# Mencetak kelas unik yang terdapat dalam array target y
# set(y) membuat himpunan dari label kelas unik yang ada dalam array target
print(f'Class: {set(y)}')

print ('\nNama: Aditya Rimandi Putra')
print ('NPM : 41155050210030')
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P3/pics/3a1.png?raw=true" alt="SS" width="30%"/>

```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

# Mengimpor fungsi train_test_split dari modul sklearn.model_selection
from sklearn.model_selection import train_test_split

# Membagi dataset menjadi data pelatihan (X_train, y_train) dan data pengujian (X_test, y_test)
# test_size=0.3 menunjukkan bahwa 30% dari data akan digunakan sebagai data pengujian
# random_state=0 memastikan bahwa pembagian data bersifat reproduktif (hasil yang sama setiap kali dijalankan)
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.3, 
                                                    random_state=0)

```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P3/pics/3a2.png?raw=true" alt="SS" width="30%"/>

### 1.6. Training model Decision Tree Classifier

```python
print ('\nNama: Aditya Rimandi Putra')
print ('NPM : 41155050210030')

# Mengimpor kelas DecisionTreeClassifier dari modul sklearn.tree
from sklearn.tree import DecisionTreeClassifier

# Membuat model pohon keputusan dengan kedalaman maksimum 4
# max_depth=4 membatasi kedalaman pohon untuk menghindari overfitting
model = DecisionTreeClassifier(max_depth=4)

# Melatih model menggunakan data pelatihan (X_train) dan label target (y_train)
# Proses ini akan menyesuaikan model dengan pola yang ada dalam data pelatihan
model.fit(X_train, y_train)
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P3/pics/3a3.png?raw=true" alt="SS" width="45%"/>

### 1.7. Visualisasi model Decision Tree

```python
print ('\nNama: Aditya Rimandi Putra')
print ('NPM : 41155050210030')

# Mengimpor pustaka matplotlib.pyplot untuk visualisasi grafik
import matplotlib.pyplot as plt 

# Mengimpor modul tree dari sklearn untuk memplot pohon keputusan
from sklearn import tree

# Mengatur resolusi gambar yang dihasilkan menjadi 85 DPI
plt.rcParams['figure.dpi'] = 85

# Mengatur ukuran subplots menjadi 10x10 inci
plt.subplots(figsize=(10, 10)) 

# Memplot pohon keputusan yang telah dilatih dengan model
# fontsize=10 mengatur ukuran font untuk label dalam plot
tree.plot_tree(model, fontsize=10) 

# Menampilkan plot yang telah dibuat
plt.show()
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P3/pics/3a4.png?raw=true" alt="SS" width="90%"/>

### 1.8. Evaluasi model Decision Tree

```python
print ('\nNama: Aditya Rimandi Putra')
print ('NPM : 41155050210030')

# Mengimpor fungsi classification_report dari modul sklearn.metrics
from sklearn.metrics import classification_report

# Menggunakan model untuk memprediksi label target pada data pengujian (X_test)
y_pred = model.predict(X_test)

# Mencetak laporan klasifikasi yang menunjukkan metrik evaluasi model
# Laporan ini mencakup precision, recall, dan f1-score untuk setiap kelas
print(classification_report(y_test, y_pred))
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P3/pics/3a5.png?raw=true" alt="SS" width="70%"/>

Klik link dibawah ini untuk kembali ke menu utama:

> [!TIP]
> Kembali ke halaman utama, [Klik disini](https://github.com/AdityaR-AI/MLC/tree/main/)








