## Random Forest Classification

> Materi ini terbagi menjadi 2 Part, berikut linknya:

Silahkan klik link dibawah ini tuntuk menuju tugas yang inign dilihat:

> [!NOTE]
> Part 1 - Decision Tree Classification [Pages Link](https://github.com/AdityaR-AI/MLC/tree/main/P3/Decision%20Tree%20%26%20Random%20Fores_I.md)

> [!NOTE]
> Part 2 - Random Forest Classification [Pages Link](https://github.com/AdityaR-AI/MLC/tree/main/P3/Decision%20Tree%20%26%20Random%20Forest_II.md)

### 2.0. Lakukan praktik dari https://youtu.be/yKovaQ6tyV8?si=HnHG6kcoCsDwvo_0 , buat screenshot dengan nama kalian pada coding, kumpulkan dalam bentuk pdf, dari kegiatan ini: 

### 2.1. Proses training model Machine Learning secara umum

Dalam machine learning, proses training adalah salah satu langkah terpenting untuk membangun model yang dapat membuat prediksi yang akurat. Training adalah tahap di mana model belajar dari data untuk mengenali pola-pola yang ada sehingga dapat digunakan untuk melakukan klasifikasi, prediksi, atau estimasi pada data yang belum pernah dilihat sebelumnya. Algoritma yang dilatih pada data ini akan menyesuaikan parameter-parameternya secara bertahap hingga mencapai hasil yang optimal.

Setiap model machine learning memerlukan dua hal utama dalam proses training: data yang digunakan untuk melatih model, dan data yang menjadi target atau label dari proses tersebut. Setelah model dilatih, model tersebut bisa digunakan untuk memprediksi hasil dari data baru yang tidak diketahui labelnya.

Selanjutnya, kita akan melihat tahapan proses training model secara umum yang dijelaskan dalam gambar.

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P3/3b1.png?raw=true" alt="SS" width="50%"/>

**Langkah-langkah Proses Training Model**
  1. **Training Set**

Proses dimulai dengan training set, yang terdiri dari dua bagian:
  -	X_train: Fitur-fitur input dari data. Fitur-fitur ini bisa berupa berbagai jenis data seperti angka, kategori, atau teks yang merepresentasikan karakteristik objek yang ingin kita pelajari.
  -	y_train: Label atau target yang ingin diprediksi oleh model. Misalnya, dalam klasifikasi buah, y_train mungkin berupa jenis buah seperti apple, grape, atau lemon.

Data ini merupakan dasar yang digunakan oleh algoritma machine learning untuk mengenali pola dan hubungan antara fitur dan label.
  
  2. **Model**

Setelah dataset disiapkan, X_train dan y_train dimasukkan ke dalam algoritma machine learning untuk membangun model. Model ini mencoba belajar dari pola dalam data pelatihan untuk menemukan cara terbaik dalam mengklasifikasikan atau memprediksi hasil yang benar berdasarkan masukan fitur.

Algoritma ini melalui iterasi proses pembelajaran, di mana ia memperbaiki asumsi dan hubungannya berdasarkan kesalahan yang ditemukan selama latihan.

  3. **Trained Model**

Setelah proses pelatihan selesai, hasilnya adalah model yang telah dilatih, disebut trained model. Model ini berisi semua informasi yang telah dipelajari dari dataset pelatihan dan siap untuk digunakan.

Model ini sekarang mampu memprediksi atau mengklasifikasikan data baru berdasarkan pola-pola yang telah dipelajari.

  4. **X_new**
     
Pada tahap selanjutnya, model yang sudah dilatih digunakan untuk memprediksi data baru yang belum pernah dilihat sebelumnya. Data baru ini disebut X_new.

Misalnya, jika model tersebut digunakan untuk memprediksi jenis buah berdasarkan ukuran dan warna, maka X_new bisa berisi ukuran dan warna buah yang tidak diketahui jenisnya.

  5. **y_pred**

Hasil dari prediksi model pada data baru ini disebut y_pred. Ini adalah hasil prediksi dari model yang sudah dilatih sebelumnya. Jika modelnya akurat, maka nilai y_pred akan mendekati atau sama dengan label yang benar untuk data baru tersebut.

Sebagai contoh, jika model dilatih untuk mengklasifikasikan jenis buah, maka y_pred bisa berupa prediksi bahwa buah baru adalah apple, grape, atau lemon.

Dengan memahami langkah-langkah ini, kita bisa melihat bagaimana model machine learning dilatih dari data dan bagaimana ia digunakan untuk membuat prediksi pada data baru. Proses ini merupakan dasar dari berbagai jenis algoritma, termasuk Random Forest, yang menggunakan banyak pohon keputusan untuk memperkuat akurasi prediksi.

### 2.2. Pengenalan Ensemble Learning

Ensemble Learning adalah teknik dalam machine learning di mana beberapa model digabungkan untuk melakukan prediksi secara lebih akurat daripada model individual. Tujuannya adalah untuk mengurangi kesalahan prediksi dan meningkatkan performa model secara keseluruhan. Dengan menggabungkan kekuatan dari beberapa model, ensemble dapat mengurangi variabilitas (variance), bias, atau ketidakseimbangan yang mungkin ada pada model tunggal.

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P3/3b2.png?raw=true" alt="SS" width="78%"/>

Referensi: https://en.wikipedia.org/wiki/Ensemble_learning

**Langkah-langkah Ensemble Learning:**

  1. **Training Set**

Proses dimulai dengan sebuah training set yang berisi fitur input (X_train) dan label target (y_train). Sama seperti pada model machine learning standar, dataset ini digunakan untuk melatih model agar dapat memprediksi output yang diinginkan.

  2. **Training Multiple Models**

Dalam ensemble learning, training set yang sama digunakan untuk melatih beberapa model. Pada gambar di atas, tiga jenis model digunakan:
  - KNN (K-Nearest Neighbors)
  -	SVM (Support Vector Machine)
  -	Decision Tree

Setiap model ini akan belajar dari data yang sama tetapi dengan algoritma yang berbeda. Karena algoritma yang berbeda memiliki cara yang berbeda dalam menemukan pola dari data, mereka menghasilkan prediksi yang berbeda.
  3. **Trained Models**

Setelah proses training, setiap model menghasilkan trained model masing-masing, yang sudah "belajar" dari data. Model yang terlatih ini siap untuk memprediksi hasil berdasarkan data baru (X_new).

  4. **Prediksi Model**

Setiap trained model akan membuat prediksi pada dataset baru (X_new). Pada gambar, kita melihat tiga prediksi yang dihasilkan:
  -	y_pred 1 dari model KNN
  - y_pred 2 dari model SVM
  - y_pred 3 dari model Decision Tree

  5. **Mean or Mode (Penggabungan Prediksi)**
     
Karena ketiga model memberikan hasil prediksi yang berbeda-beda, kita perlu menggabungkannya untuk mendapatkan prediksi final. Teknik penggabungan yang paling umum digunakan dalam ensemble learning adalah dengan menghitung rata-rata (mean) atau modus (mode) dari hasil prediksi. Dengan cara ini, kita bisa mendapatkan hasil prediksi yang lebih stabil dan akurat.

  6. **y_pred final**

Hasil akhir dari prediksi gabungan disebut sebagai y_pred final. Ini adalah prediksi yang dihasilkan dari penggabungan output tiga model yang berbeda, dan dianggap lebih akurat karena menggabungkan informasi dari berbagai perspektif algoritma.
Manfaat Ensemble Learning
  -	Mengurangi Varians dan Bias: Dengan menggabungkan beberapa model, ensemble dapat mengurangi kesalahan yang mungkin disebabkan oleh variabilitas data atau bias pada satu model.
  -	Meningkatkan Akurasi: Hasil prediksi dari beberapa model sering kali lebih baik daripada menggunakan model tunggal.
  -	Stabilitas Model: Ensemble learning membuat model lebih tahan terhadap perubahan kecil pada dataset, yang dikenal sebagai robustness.
Metode ensemble learning ini sering digunakan dalam berbagai aplikasi, termasuk classification dan regression, untuk mendapatkan hasil yang lebih baik dan lebih andal.

### 2.3. Pengenalan Bootstrap Aggregating | Bagging

Bagging, singkatan dari Bootstrap Aggregating, adalah metode ensemble learning yang menggabungkan beberapa model prediksi untuk meningkatkan akurasi prediksi dan mengurangi varians. Prinsipnya adalah dengan membangun banyak model dengan data bootstrap dari dataset asli, kemudian menggabungkan prediksi mereka untuk mendapatkan prediksi akhir.

**Bagaimana cara kerja Bagging?**
  1. **Bootstrap Sampling:** Dataset asli diambil secara acak dengan pengembalian, menciptakan dataset baru yang disebut bootstrap sample. Setiap bootstrap sample memiliki ukuran yang sama dengan dataset asli, tetapi beberapa data mungkin muncul lebih dari sekali, sementara yang lain mungkin tidak muncul sama sekali.
  
  2. **Membangun Model:** Model prediksi (misalnya, pohon keputusan, regresi linier) dilatih pada setiap bootstrap sample yang berbeda. Ini menghasilkan sejumlah model yang independen satu sama lain.
  
  3. **Menggabungkan Prediksi:** Untuk mendapatkan prediksi akhir, prediksi dari semua model yang dilatih dikombinasikan. Cara penggabungannya tergantung pada jenis masalah:
     - Untuk klasifikasi, prediksi dari semua model dikumpulkan dan prediksi kelas dengan jumlah suara terbanyak dipilih.
       Untuk regresi, prediksi dari semua model dirata-ratakan.

**Proses Bagging**

Gambar di bawah ini menggambarkan proses Bagging (Bootstrap Aggregating) untuk membangun model ensemble pada Random Forest. 

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P3/3b3.png?raw=true" alt="SS" width="70%"/>

Referensi: https://en.wikipedia.org/wiki/Bootstrap_aggregating
  1. **Training Set:** Dimulai dengan dataset pelatihan asli (X_train, y_train).
  2. **Random Sampling with Replacement:** Dataset asli diambil sampelnya secara acak dengan penggantian (bootstrap) untuk menciptakan beberapa kumpulan dataset baru yang disebut "Bag". Setiap Bag berisi data yang dipilih secara acak dengan kemungkinan data yang sama untuk dipilih berulang kali.
  3. **Model Training:** Model yang sama (dalam kasus ini, "Model") dilatih pada setiap Bag yang berbeda.
  4. **Prediction:** Setiap model dilatih pada Bag yang berbeda kemudian menghasilkan prediksi (X_new) untuk data baru.
  5. **Mean or Mode:** Prediksi dari semua model digabungkan menggunakan mean atau mode (tergantung pada jenis tugas) untuk mendapatkan prediksi akhir (y_pred).

### 2.4. Pengenalan Random Forest | Hutan Acak 

Mengenal Random Forest: Hutan Acak untuk Prediksi yang Lebih Akurat

Random Forest adalah teknik pembelajaran mesin yang memanfaatkan kekuatan banyak decision tree. Bayangkan memiliki sekelompok pohon yang masing-masing membuat prediksi berdasarkan data yang berbeda, dan kemudian "menghitung suara" untuk menentukan jawaban akhir. Inilah konsep dasar dari Random Forest.

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P3/3b4.png?raw=true" alt="SS" width="78%"/>

Referensi: https://en.wikipedia.org/wiki/Random_forest

**Bagaimana Random Forest Bekerja?**
  1. **Pembentukan Bagging (Bootstrap Aggregating):**
  - Data pelatihan dibagi menjadi beberapa subset (bag) secara acak. Setiap bag mungkin berisi beberapa data yang sama, seperti mengambil sampel dengan penggantian.
  - Misalnya, kita punya 100 data, lalu dibagi menjadi 5 bag, setiap bag berisi 20 data.

  2. **Features Randomness (Acak Fitur):**
  - Selain pembagian data, setiap decision tree dalam Random Forest juga menggunakan subset fitur secara acak.
  - Ini membantu mencegah model menjadi terlalu sensitif terhadap fitur tertentu dan meningkatkan generalisasi model.
  - Misalnya, jika ada 10 fitur dalam data, setiap decision tree hanya akan menggunakan 5 fitur secara acak.

  3. **Pembuatan Pohon Keputusan (Decision Tree):**
  - Setiap bag dibentuk, kemudian decision tree dibentuk untuk setiap bag.
  - Setiap decision tree dilatih pada data dalam bag-nya dan subset fitur yang acak.
  - Karena setiap pohon dilatih dengan data dan fitur yang berbeda, hasilnya pun akan berbeda.

  4. **Penggabungan Prediksi (Aggregation):**
  - Setelah semua decision tree terlatih, model Random Forest akan menggunakan mean (rata-rata) atau mode (nilai yang paling sering) dari prediksi setiap pohon untuk membuat prediksi akhir.
  - Jika kita punya 5 decision tree, maka prediksi final adalah rata-rata dari 5 prediksi pohon.

### 2.5. Persiapan dataset | Iris Flower Dataset

```python
print ('Nama: Aditya Rimandi Putra')
print ('NPM : 41155050210030\n')

# Mengimpor fungsi load_iris dari modul sklearn.datasets untuk memuat dataset Iris
from sklearn.datasets import load_iris

# Memuat dataset Iris dan memisahkan fitur (X) dan label target (y)
# return_X_y=True mengembalikan fitur dan label sebagai dua variabel terpisah
X, y = load_iris(return_X_y=True)

# Mencetak dimensi dari fitur yang dimuat
# X.shape memberikan ukuran dari array fitur
print(f'Dimensi Feature: {X.shape}')

# Mencetak kelas yang ada dalam label target
# set(y) digunakan untuk mendapatkan himpunan unik dari kelas
print(f'Class: {set(y)}')
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P3/pics/3b1.png?raw=true" alt="SS" width="30%"/>

```python
print ('Nama: Aditya Rimandi Putra')
print ('NPM : 41155050210030\n')

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

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P3/pics/3b2.png?raw=true" alt="SS" width="30%"/>

### 2.6. Implementasi Random Forest Classifier dengan Scikit Learn

```python
print ('Nama: Aditya Rimandi Putra')
print ('NPM : 41155050210030\n')

# Mengimpor kelas RandomForestClassifier dari modul sklearn.ensemble
from sklearn.ensemble import RandomForestClassifier

# Membuat model Random Forest dengan 100 estimator (pohon keputusan)
# n_estimators=100 menentukan jumlah pohon dalam hutan acak
# random_state=0 digunakan untuk memastikan hasil yang konsisten pada setiap run
model = RandomForestClassifier(n_estimators=100, random_state=0)

# Melatih model menggunakan data pelatihan (X_train) dan label target (y_train)
# Proses ini akan menyesuaikan model dengan pola yang ada dalam data pelatihan
model.fit(X_train, y_train)
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P3/pics/3b3.png?raw=true" alt="SS" width="40%"/>

### 2.7. Evaluasi model  dengan Classification Report

```python
print ('Nama: Aditya Rimandi Putra')
print ('NPM : 41155050210030\n')

# Mengimpor fungsi classification_report dari modul sklearn.metrics
from sklearn.metrics import classification_report

# Menggunakan model yang telah dilatih untuk memprediksi label target pada data pengujian (X_test)
y_pred = model.predict(X_test)

# Mencetak laporan klasifikasi yang menunjukkan metrik evaluasi model
# Laporan ini mencakup precision, recall, dan f1-score untuk setiap kelas
# y_test adalah label target sebenarnya, sedangkan y_pred adalah label target yang diprediksi oleh model
print(classification_report(y_test, y_pred))
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P3/pics/3b4.png?raw=true" alt="SS" width="60%"/>






