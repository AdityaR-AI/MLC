Klik link dibawah ini untuk kembali ke menu utama:

> [!TIP]
> Kembali ke halaman utama, [Klik disini](https://github.com/AdityaR-AI/MLC/tree/main/)

## Percepteron

# Model 1

1. **Membuat table data logika and 2 input**

```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')  # Mencetak nama
print('NPM : 41155050210030\n')  # Mencetak NPM
 
#table logika AND 2 input
"""
X1 X2 y
1  1  1
1  0  0
0  0  0
0  0  0
"""

```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P11/pic/11a1.png?raw=true" alt="SS" width="40%"/>

2.	**Memanggil 2 model keras.models.Sequential dan model.compile**
```python
# Mencetak informasi identitas
print('Nama: Aditya Rimandi Putra')  # Mencetak nama lengkap
print('NPM : 41155050210030\n')  # Mencetak Nomor Pokok Mahasiswa (NPM)
 
# Penggunaan Machine Learning dengan TensorFlow/Keras
import tensorflow as tf  # Mengimpor library TensorFlow
from tensorflow import keras  # Mengimpor modul Keras dari TensorFlow
import numpy as np  # Mengimpor library NumPy untuk pengolahan numerik
 
# Definisikan model Machine Learning
model = keras.Sequential([
    keras.Input(shape=(2,)),  # Lapisan Input dengan 2 dimensi input
    keras.layers.Dense(units=1)  # Lapisan Dense dengan 1 neuron output
])
 
# Kompilasi model dengan optimizer dan fungsi loss
model.compile(optimizer='sgd', loss='mean_squared_error')
#   - optimizer='sgd' menggunakan algoritma optimasi Stochastic Gradient Descent (SGD)
#   - loss='mean_squared_error' menggunakan fungsi loss Mean Squared Error (MSE)

```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P11/pic/11a2.png?raw=true" alt="SS" width="25%"/>

3.	**Melakukan set data dari tabel data yang telah dibuat**
```python
# Mencetak informasi identitas
print('Nama: Aditya Rimandi Putra')  # Mencetak nama lengkap
print('NPM : 41155050210030\n')  # Mencetak Nomor Pokok Mahasiswa (NPM)
 
# Definisikan data input dan target
xs = np.array([
    [1,1], [1,0], [0,1], [0,0]
], dtype=int)  # Data input (X1 dan X2)
ys = np.array(
    [1, 0, 0, 0], dtype=int
)  # Data target (y)

```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P11/pic/11a3.png?raw=true" alt="SS" width="25%"/>

4.	**Memanggil tampilan arstitektur awal**
```python
# Mencetak informasi identitas
print('Nama: Aditya Rimandi Putra')  # Mencetak nama lengkap
print('NPM : 41155050210030\n')  # Mencetak Nomor Pokok Mahasiswa (NPM)
 
# Tampilkan arsitektur model awal
model.summary()
#   - model.summary() digunakan untuk menampilkan informasi tentang arsitektur model
#   - Informasi yang ditampilkan meliputi jumlah lapisan, jenis lapisan, jumlah parameter, dan lain-lain

```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P11/pic/11a4.png?raw=true" alt="SS" width="25%"/>

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P11/pic/11a5.png?raw=true" alt="SS" width="50%"/>

5.	**Melakukan pengecekan model bobot awal**
```python
# Mencetak informasi identitas
print('Nama: Aditya Rimandi Putra')  # Mencetak nama lengkap
print('NPM : 41155050210030\n')  # Mencetak Nomor Pokok Mahasiswa (NPM)
 
# Cek bobot model awal
weights = model.get_weights()  # Mendapatkan bobot model awal
weights  # Menampilkan bobot model awal
#   - model.get_weights() digunakan untuk mendapatkan bobot model awal
#   - Bobot model awal adalah nilai awal yang digunakan oleh model sebelum dilakukan pelatihan

```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P11/pic/11a6.png?raw=true" alt="SS" width="35%"/>

6.	**Melakukan Training**
```python
# Mencetak informasi identitas
print('Nama: Aditya Rimandi Putra')  # Mencetak nama lengkap
print('NPM : 41155050210030\n')  # Mencetak Nomor Pokok Mahasiswa (NPM)
 
# Pelatihan model
model.fit(xs, ys, epochs=1000)  # Melakukan pelatihan model dengan data xs dan ys selama 1000 epoch
#   - model.fit() digunakan untuk melakukan pelatihan model
#   - xs adalah data input yang digunakan untuk pelatihan
#   - ys adalah data target yang digunakan untuk pelatihan
#   - epochs=1000 menentukan jumlah epoch yang digunakan untuk pelatihan

```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P11/pic/11a7.png?raw=true" alt="SS" width="40%"/>

7.	**Melakukan Testing**
```python
# Mencetak informasi identitas
print('Nama: Aditya Rimandi Putra')  # Mencetak nama lengkap
print('NPM : 41155050210030\n')  # Mencetak Nomor Pokok Mahasiswa (NPM)
 
# Uji model
data = np.array([[1, 1]])  # Data input yang digunakan untuk uji
answer = model.predict(data)  # Melakukan prediksi dengan model
print(answer)  # Menampilkan hasil prediksi
#   - model.predict() digunakan untuk melakukan prediksi dengan model
#   - data adalah data input yang digunakan untuk uji
#   - hasil prediksi adalah output yang dihasilkan oleh model berdasarkan data input

```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P11/pic/11a8.png?raw=true" alt="SS" width="35%"/>

8.	**Melakukan pengecekan model bobot Kembali**
```python
# Mencetak informasi identitas
print('Nama: Aditya Rimandi Putra')  # Mencetak nama lengkap
print('NPM : 41155050210030\n')  # Mencetak Nomor Pokok Mahasiswa (NPM)
 
# Cek bobot model setelah pelatihan
weights = model.get_weights()  # Mendapatkan bobot model setelah pelatihan
weights  # Menampilkan bobot model setelah pelatihan
#   - model.get_weights() digunakan untuk mendapatkan bobot model setelah pelatihan
#   - Bobot model setelah pelatihan adalah nilai bobot yang telah diupdate setelah proses pelatihan

```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P11/pic/11a9.png?raw=true" alt="SS" width="35%"/>

### Model 2
1.	**Membuat table data**
```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')  # Mencetak nama
print('NPM : 41155050210030\n')  # Mencetak NPM
 
#table data
"""
w1 = 2
w2 = 4
 
x1 x2 y
2  3  16
4  1  12
5  4  28
7  5  34
8  2  24
2  1  8
4  9  44
8  2  24
7  1  18
6  5  32
1  1  6
3  2  14
"""

```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P11/pic/11a10.png?raw=true" alt="SS" width="90%"/>

2.	**Memanggil 2 model keras.models.Sequential dan model.compile dan set data** 
```python
# Mencetak informasi identitas
print('Nama: Aditya Rimandi Putra')  # Mencetak nama lengkap
print('NPM : 41155050210030\n')  # Mencetak Nomor Pokok Mahasiswa (NPM)
 
# Definisikan model baru
model2 = keras.Sequential([
    keras.Input(shape=(2,)),  # Lapisan Input dengan 2 dimensi input
    keras.layers.Dense(units=1)  # Lapisan Dense dengan 1 neuron output
])  # Model baru dengan 1 lapisan Dense dan 1 neuron output
#   - units=1 menentukan jumlah neuron output
#   - input_shape=[2] menentukan jumlah input
 
# Kompilasi model baru
model2.compile(optimizer='sgd', loss='mean_squared_error')
#   - optimizer='sgd' menentukan algoritma optimasi yang digunakan
#   - loss='mean_squared_error' menentukan fungsi loss yang digunakan
 
# Definisikan data input dan target
xs = np.array(
    [
        [2, 3], [4, 1], [5, 4], [7, 5], [8, 2], [2,1],
        [4, 9], [8, 2], [7, 1], [6, 5], [1, 1], [3, 2]
    ]
)  # Data input
ys = np.array(
    [16, 12, 28, 34, 24, 8, 44, 24, 18, 32, 6, 14]
)  # Data target
#   - xs dan ys adalah data yang digunakan untuk pelatihan model

```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P11/pic/11a11.png?raw=true" alt="SS" width="25%"/>

3.	**Melakukan pengecekan model bobot awal**
```python
# Mencetak informasi identitas
print('Nama: Aditya Rimandi Putra')  # Mencetak nama lengkap
print('NPM : 41155050210030\n')  # Mencetak Nomor Pokok Mahasiswa (NPM)
 
# Cek bobot model awal
weights = model2.get_weights()  # Mendapatkan bobot model awal
weights  # Menampilkan bobot model awal
#   - model2.get_weights() digunakan untuk mendapatkan bobot model awal
#   - Bobot model awal adalah nilai bobot yang digunakan oleh model sebelum dilakukan pelatihan

```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P11/pic/11a12.png?raw=true" alt="SS" width="35%"/>

4.	**Melakukan training**
```python
# Mencetak informasi identitas
print('Nama: Aditya Rimandi Putra')  # Mencetak nama lengkap
print('NPM : 41155050210030\n')  # Mencetak Nomor Pokok Mahasiswa (NPM)
 
# Pelatihan model
model2.fit(xs, ys, epochs=1000)  # Melakukan pelatihan model dengan data xs dan ys selama 1000 epoch
#   - model2.fit() digunakan untuk melakukan pelatihan model
#   - xs adalah data input yang digunakan untuk pelatihan
#   - ys adalah data target yang digunakan untuk pelatihan
#   - epochs=1000 menentukan jumlah epoch yang digunakan untuk pelatihan

```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P11/pic/11a13.png?raw=true" alt="SS" width="40%"/>

5.	**Melakukan pengetesan**
```python
# Mencetak informasi identitas
print('Nama: Aditya Rimandi Putra')  # Mencetak nama lengkap
print('NPM : 41155050210030\n')  # Mencetak Nomor Pokok Mahasiswa (NPM)
 
# Uji model
data = np.array([[1, 3]])  # Data input yang digunakan untuk uji
answer = model2.predict(data)  # Melakukan prediksi dengan model
print(answer)  # Menampilkan hasil prediksi
#   - model2.predict() digunakan untuk melakukan prediksi dengan model
#   - data adalah data input yang digunakan untuk uji
#   - hasil prediksi adalah output yang dihasilkan oleh model berdasarkan data input
 
# Seharusnya hasil prediksi adalah 14
#   - 1*2 + 3*4 = 14 adalah perhitungan yang seharusnya dilakukan oleh model

```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P11/pic/11a14.png?raw=true" alt="SS" width="30%"/>

6.	**Melakukan pengecekan model bobot Kembali**
```python
# Mencetak informasi identitas
print('Nama: Aditya Rimandi Putra')  # Mencetak nama lengkap
print('NPM : 41155050210030\n')  # Mencetak Nomor Pokok Mahasiswa (NPM)
 
# Cek bobot model setelah pelatihan
weights = model2.get_weights()  # Mendapatkan bobot model setelah pelatihan
weights  # Menampilkan bobot model setelah pelatihan
#   - model2.get_weights() digunakan untuk mendapatkan bobot model setelah pelatihan
#   - Bobot model setelah pelatihan adalah nilai bobot yang telah diupdate setelah proses pelatihan

```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P11/pic/11a15.png?raw=true" alt="SS" width="30%"/>

7.	**Testing Kembali bobot dan data**
```python
# Mencetak informasi identitas
print('Nama: Aditya Rimandi Putra')  # Mencetak nama lengkap
print('NPM : 41155050210030\n')  # Mencetak Nomor Pokok Mahasiswa (NPM)
 
# Uji model
data = np.array([[3, 3]])  # Data input yang digunakan untuk uji
answer = model2.predict(data)  # Melakukan prediksi dengan model
print(answer)  # Menampilkan hasil prediksi
#   - model2.predict() digunakan untuk melakukan prediksi dengan model
#   - data adalah data input yang digunakan untuk uji
#   - hasil prediksi adalah output yang dihasilkan oleh model berdasarkan data input
 
# Seharusnya hasil prediksi adalah 18
#   - 3*2 + 3*4 = 18 adalah perhitungan yang seharusnya dilakukan oleh model

```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P11/pic/11a16.png?raw=true" alt="SS" width="30%"/>


```python
# Mencetak informasi identitas
print('Nama: Aditya Rimandi Putra')  # Mencetak nama lengkap
print('NPM : 41155050210030\n')  # Mencetak Nomor Pokok Mahasiswa (NPM)
 
# Cek bobot model setelah pelatihan
weights = model2.get_weights()  # Mendapatkan bobot model setelah pelatihan
weights  # Menampilkan bobot model setelah pelatihan
#   - model2.get_weights() digunakan untuk mendapatkan bobot model setelah pelatihan
#   - Bobot model setelah pelatihan adalah nilai bobot yang telah diupdate setelah proses pelatihan

```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P11/pic/11a17.png?raw=true" alt="SS" width="30%"/>

Klik link dibawah ini untuk kembali ke menu utama:

> [!TIP]
> Kembali ke halaman utama, [Klik disini](https://github.com/AdityaR-AI/MLC/tree/main/)
