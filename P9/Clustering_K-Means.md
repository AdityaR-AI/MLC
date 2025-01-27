Klik link dibawah ini untuk kembali ke menu utama:

> [!TIP]
> Kembali ke halaman utama, [Klik disini](https://github.com/AdityaR-AI/MLC/tree/main/)

## Clustering dengan K-Means

### Apa itu Clustering?
**Clustering** adalah teknik dalam pembelajaran mesin yang digunakan untuk mengelompokkan data ke dalam grup atau klaster berdasarkan kesamaan karakteristik atau fitur di dalamnya. Tujuannya adalah untuk mengelompokkan objek yang memiliki sifat atau pola yang serupa sehingga objek dalam satu klaster lebih mirip satu sama lain daripada dengan objek di klaster lainnya. Clustering adalah teknik **unsupervised learning**, yang berarti tidak memerlukan label atau target yang sudah diketahui sebelumnya.

Contoh penerapan clustering termasuk:
  -	**Segmentasi pelanggan** dalam marketing untuk mengidentifikasi kelompok konsumen dengan kebiasaan yang serupa.
  -	**Pengenalan pola** dalam pengolahan citra atau analisis teks untuk menemukan pola tersembunyi.
  -	**Pemrosesan biologis** untuk mengelompokkan gen atau sampel berdasarkan ekspresi gen.

### Algoritma Apa Saja yang Bisa Digunakan pada Clustering?
Berikut adalah beberapa algoritma yang umum digunakan dalam clustering
1. **K-Means**
  -	Salah satu algoritma clustering yang paling populer. K-Means membagi data ke dalam k klaster dengan cara meminimalkan jarak antar data dalam satu klaster dan pusat klaster (centroid). K-Means bekerja dengan iteratif untuk memperbarui posisi centroid dan membagi ulang data berdasarkan jaraknya ke centroid terdekat.
2. **Hierarchical Clustering**
  -	Algoritma ini mengelompokkan data secara hierarkis (bertingkat), membentuk struktur seperti pohon yang disebut dendrogram. Ada dua pendekatan utama:
    -	Agglomerative (bottom-up): Setiap data mulai sebagai klaster terpisah dan kemudian digabungkan berdasarkan kedekatannya.
    -	Divisive (top-down): Dimulai dengan satu klaster besar yang berisi semua data, lalu dibagi menjadi klaster-klaster lebih kecil.
3. **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**
  -	Algoritma ini mengelompokkan data berdasarkan kerapatan titik data di sekitar mereka. DBSCAN sangat baik untuk menemukan klaster dengan bentuk yang tidak teratur dan dapat mengidentifikasi noise (data yang tidak termasuk dalam klaster).
4. **Gaussian Mixture Model (GMM)**
  -	GMM adalah pendekatan probabilistik yang mengasumsikan bahwa data dihasilkan dari campuran distribusi Gaussian. Setiap klaster dianggap sebagai distribusi Gaussian, dan model ini mengasumsikan bahwa setiap titik data memiliki probabilitas untuk menjadi bagian dari klaster tertentu.
5. **Mean Shift**
  -	Algoritma ini mencari area dengan kerapatan titik data yang tinggi dan memindahkan titik ke arah area tersebut (shift). Ini dapat menemukan klaster dengan bentuk yang lebih fleksibel dibandingkan K-Means.
6. **Spectral Clustering**
  -	Algoritma ini menggunakan informasi tentang eigenvalues dan eigenvectors dari matriks hubungan antar data untuk membagi data ke dalam klaster. Spectral clustering sering digunakan untuk data yang memiliki struktur non-linear.

### Perbedaan Clustering dan Klasifikasi
<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P9/pic/11a1.png?raw=true" alt="SS" width="80%"/>
<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P9/pic/11a2.png?raw=true" alt="SS" width="80%"/>

### Tipe Clustering
<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P9/pic/11a3.png?raw=true" alt="SS" width="80%"/>
  
- **Hirarki Clustering**
<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P9/pic/11a4.png?raw=true" alt="SS" width="80%"/>

-  **Non-Hirarki Clustering**
<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P9/pic/11a5.png?raw=true" alt="SS" width="80%"/>

### Pada Dokumentasi Ini Saya Hanya Akan Mempraktekan K-Means
Pada dokumen ini, kita akan fokus untuk mempraktikkan algoritma **K-Means**. K-Means dipilih karena kesederhanaannya, efisiensinya dalam menangani dataset besar, dan kemampuannya untuk memberikan hasil yang cukup baik pada banyak jenis data.

### Apa itu K-Means?
<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P9/pic/11a6.png?raw=true" alt="SS" width="80%"/>

### Langkah-langkah K-Means
<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P9/pic/11a7.png?raw=true" alt="SS" width="80%"/>
<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P9/pic/11a8.png?raw=true" alt="SS" width="80%"/>

### Formula K-Means
<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P9/pic/11a9.png?raw=true" alt="SS" width="80%"/>

### Kelebihan dan Kekurangan K-Means
<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P9/pic/11a10.png?raw=true" alt="SS" width="80%"/>

## Praktik Clustering dengan K-Means
<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P9/pic/11a11.png?raw=true" alt="SS" width="80%"/>

### Langkah 1: Import Konten
```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')  # Mencetak nama lengkap
print('NPM : 41155050210030\n')       # Mencetak Nomor Pokok Mahasiswa (NPM)
 
import pandas as pd  # Mengimpor library pandas untuk manipulasi data
import os           # Mengimpor library os untuk berinteraksi dengan sistem file
 
# Menentukan path ke file konsumen.csv
folder_path = os.path.join(os.getcwd(), 'content')  # Menggabungkan path saat ini dengan folder 'content'
file_path = os.path.join(folder_path, 'konsumen.csv')  # Menggabungkan folder_path dengan nama file 'konsumen.csv'
```
<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P9/pic/11b1.png?raw=true" alt="SS" width="35%"/>

```python
# Path ke folder 'content'
folder_path = './content'  # Menentukan path relatif ke folder 'content', sesuaikan dengan lokasi folder Anda
 
# Memeriksa apakah folder ada
if os.path.exists(folder_path):  # Mengecek apakah folder dengan path yang ditentukan ada
    # Menampilkan isi folder
    files = os.listdir(folder_path)  # Mengambil daftar file dan folder yang ada di dalam folder 'content'
    print(f"Isi folder '{folder_path}':")  # Mencetak pesan yang menunjukkan isi folder
    for file in files:  # Melakukan iterasi untuk setiap file dalam daftar
        print(file)  # Mencetak nama file
else:
    print(f"Folder '{folder_path}' tidak ditemukan.")  # Menampilkan pesan jika folder tidak ditemukan

```
<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P9/pic/11b2.png?raw=true" alt="SS" width="35%"/>

### Langkah 2: Import Library
```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')  # Mencetak nama lengkap
print('NPM : 41155050210030\n')       # Mencetak Nomor Pokok Mahasiswa (NPM)
 
# Mengimpor library yang akan digunakan
import warnings  # Mengimpor library untuk mengelola peringatan
import time      # Mengimpor library untuk fungsi terkait waktu
import matplotlib.pyplot as plt  # Mengimpor library untuk visualisasi data
import numpy as np  # Mengimpor library untuk operasi numerik dan array
import pandas as pd  # Mengimpor library untuk manipulasi dan analisis data
from sklearn.cluster import KMeans  # Mengimpor algoritma KMeans dari library scikit-learn untuk 
```
<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P9/pic/11b3.png?raw=true" alt="SS" width="35%"/>

### Langkah 3: Menyiapkan Dataset
```python
1.	# Mencetak nama dan NPM
2.	print('Nama: Aditya Rimandi Putra')  # Mencetak nama lengkap
3.	print('NPM : 41155050210030\n')       # Mencetak Nomor Pokok Mahasiswa (NPM)
4.	 
5.	# Menyiapkan data dan memanggil dataset
6.	dataset = pd.read_csv('content/konsumen.csv')  # Membaca file CSV 'konsumen.csv' dari folder 'content' dan menyimpannya dalam variabel dataset
7.	dataset.keys()  # Menampilkan kunci (nama kolom) dari dataset yang telah dimuat

```
<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P9/pic/11b4.png?raw=true" alt="SS" width="40%"/>

```python
# Menampilkan 5 baris data pertama dari dataset tersebut
dataku = pd.DataFrame(dataset)  # Mengonversi dataset menjadi DataFrame dan menyimpannya dalam variabel 'dataku'
dataku.head()  # Menampilkan 5 baris pertama dari DataFrame 'dataku'

```
<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P9/pic/11b5.png?raw=true" alt="SS" width="35%"/>

```python
# Konversi ke data array
X = np.asarray(dataset)  # Mengonversi DataFrame 'dataset' menjadi array NumPy dan menyimpannya dalam variabel 'X'
print(X)  # Mencetak array 'X' ke layar

```
<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P9/pic/11b6.png?raw=true" alt="SS" width="35%"/>

```python
# Menampilkan data dalam bentuk scatter plot
plt.scatter(X[:, 0], X[:, 1], label='True Position')  # Membuat scatter plot dengan sumbu X dari kolom pertama dan sumbu Y dari kolom kedua array 'X'
plt.xlabel("Gaji")  # Menambahkan label untuk sumbu X
plt.ylabel("Pengeluaran")  # Menambahkan label untuk sumbu Y
plt.title("Grafik Konsumen")  # Menambahkan judul untuk grafik
plt.show()  # Menampilkan grafik scatter plot

```
<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P9/pic/11b7.png?raw=true" alt="SS" width="50%"/>

### Langkah 4: Menggunakan Library K-Means untuk Clustering
```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')  # Mencetak nama
print('NPM : 41155050210030\n')  # Mencetak NPM
 
# Mengabaikan peringatan dari sklearn (dapat diaktifkan jika diperlukan)
# warnings.filterwarnings("ignore", category=User Warning, module="sklearn")
 
# Inisialisasi variabel untuk menyimpan OMP_NUM_THREADS dengan waktu eksekusi terendah
lowest_time = float('inf')  # Set waktu awal sebagai yang sangat besar
omp_value = None  # Menyimpan jumlah thread yang menghasilkan waktu terendah
 
# Percobaan untuk berbagai jumlah thread
for threads in [1, 4, 8, 16]:  # Menguji jumlah thread yang berbeda
    os.environ["OMP_NUM_THREADS"] = str(threads)  # Mengatur jumlah thread untuk OpenMP
    kmeans = KMeans(n_clusters=2)  # Inisialisasi model KMeans dengan 2 cluster
    start_time = time.time()  # Mulai pencatatan waktu
    kmeans.fit(X)  # Latih model KMeans dengan data X
    exec_time = time.time() - start_time  # Hitung waktu eksekusi
    print(f"OMP_NUM_THREADS={threads}, Waktu Eksekusi: {exec_time:.2f} detik")  # Mencetak waktu eksekusi
 
    # Cek apakah waktu eksekusi saat ini lebih rendah dari waktu sebelumnya
    if exec_time < lowest_time:  # Jika waktu eksekusi saat ini lebih rendah
        lowest_time = exec_time  # Update waktu terendah
        omp_value = threads  # Simpan jumlah thread yang menghasilkan waktu terendah
 
# Optimal OMP_NUM_THREADS
os.environ["OMP_NUM_THREADS"] = str(omp_value)  # Mengonversi omp_value menjadi string dan mengatur jumlah thread optimal
 
# Mengaktifkan K Means dengan jumlah K = 2
kmeans = KMeans(n_clusters=2)  # Inisialisasi model KMeans dengan 2 cluster
kmeans.fit(X)  # Latih model KMeans dengan data X

```
<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P9/pic/11b8.png?raw=true" alt="SS" width="50%"/>

```python
# Menampilkan nilai centroid
# yang dihasilkan oleh algoritma KMeans
print(kmeans.cluster_centers_)  # Mencetak nilai centroid dari setiap cluster yang dihasilkan oleh 
```
<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P9/pic/11b9.png?raw=true" alt="SS" width="40%"/>

```python
# Menampilkan label untuk setiap data point
print(kmeans.labels_)  # Mencetak label yang diberikan oleh model KMeans untuk setiap data point dalam dataset

```
<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P9/pic/11b10.png?raw=true" alt="SS" width="45%"/>

### Langkah 5: Menampilkan Output
```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')  # Mencetak nama
print('NPM : 41155050210030\n')  # Mencetak NPM
 
# Plot data point
# Memvisualisasikan bagaimana data telah dikelompokkan (di-klasterisasi).
plt.scatter(X[:,0], X[:,1], c=kmeans.labels_, cmap='rainbow')  # Membuat scatter plot dari data point dengan warna berdasarkan label cluster
plt.xlabel("Gaji")  # Menambahkan label sumbu X
plt.ylabel("Pengeluaran")  # Menambahkan label sumbu Y
plt.title("Grafik Konsumen")  # Menambahkan judul pada grafik
plt.show()  # Menampilkan grafik

```
<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P9/pic/11b11.png?raw=true" alt="SS" width="35%"/>
<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P9/pic/11b12.png?raw=true" alt="SS" width="60%"/>

```python
# Plot data point
# Memvisualisasikan bagaimana data telah dikelompokkan (di-klasterisasi).
# Menampilkan centroid dengan warna hitam.
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='rainbow')  # Membuat scatter plot dari data point dengan warna berdasarkan label cluster
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='black')  # Menambahkan centroid ke plot dengan warna hitam
plt.xlabel("Gaji")  # Menambahkan label sumbu X
plt.ylabel("Pengeluaran")  # Menambahkan label sumbu Y
plt.title("Grafik Konsumen")  # Menambahkan judul pada grafik
plt.show()  # Menampilkan grafik

```
<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P9/pic/11b12.png?raw=true" alt="SS" width="60%"/>

Klik link dibawah ini untuk kembali ke menu utama:

> [!TIP]
> Kembali ke halaman utama, [Klik disini](https://github.com/AdityaR-AI/MLC/tree/main/)




































