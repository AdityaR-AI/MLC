Klik link dibawah ini untuk kembali ke menu utama:

> [!TIP]
> Kembali ke halaman utama, [Klik disini](https://github.com/AdityaR-AI/MLC/tree/main/)

## Backpropagation

1.	**Import Library dan Dataset**
```python
# Mencetak informasi identitas
print('Nama: Aditya Rimandi Putra')  # Mencetak nama lengkap
print('NPM : 41155050210030\n')  # Mencetak Nomor Pokok Mahasiswa (NPM)
 
# Mengimpor library yang dibutuhkan
import pandas as pd  # Library untuk pengolahan data
import numpy as np  # Library untuk pengolahan numerik
 
# Membaca data dari file CSV
emas = pd.read_csv('dataemas.csv')  # Membaca data dari file 'dataemas.csv' ke dalam DataFrame
 
# Mencetak data yang telah dibaca
print(emas)  # Mencetak isi DataFrame 'emas'

```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P12/pic/12a1.png?raw=true" alt="SS" width="60%"/>

2.	**Membuat dataframe, fitur, dan target**
```python
# Mencetak informasi identitas
print('Nama: Aditya Rimandi Putra')  # Mencetak nama lengkap
print('NPM : 41155050210030\n')  # Mencetak Nomor Pokok Mahasiswa (NPM)
 
# Membuat DataFrame dari data emas
data = pd.DataFrame(emas, columns=['Open', 'High', 'Low', 'Price'])  # Membuat DataFrame dengan kolom 'Open', 'High', 'Low', dan 'Price'
 
# Memisahkan data menjadi fitur (x) dan target (y)
x = data.iloc[:,0:3].values  # Memilih kolom 'Open', 'High', dan 'Low' sebagai fitur (x)
y = data.iloc[:,-1].values  # Memilih kolom 'Price' sebagai target (y)
 
# Mencetak data fitur (x) dan target (y)
print(x)  # Mencetak nilai fitur (x)
print(y)  # Mencetak nilai target (y)

```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P12/pic/12a2.png?raw=true" alt="SS" width="50%"/>

3.	**Membagi data yang telah dibuat menjadi data Training dan Testing**
```python
# Mencetak informasi identitas
print('Nama: Aditya Rimandi Putra')  # Mencetak nama lengkap
print('NPM : 41155050210030\n')  # Mencetak Nomor Pokok Mahasiswa (NPM)
 
# Mengimpor library untuk membagi data training dan testing
from sklearn.model_selection import train_test_split  # Library untuk membagi data
 
# Membagi data menjadi data training dan testing
# Digunakan untuk membagi data menjadi dua bagian, yaitu data training dan data testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
# x_train: data fitur training
# x_test: data fitur testing
# y_train: data target training
# y_test: data target testing
# test_size=0.2: proporsi data testing dari total data (20% dari total data)
# random_state=0: nilai acak untuk memastikan hasil yang sama

```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P12/pic/12a3.png?raw=true" alt="SS" width="25%"/>

4.	**Memulai Training**
```python
# Mencetak informasi identitas
print('Nama: Aditya Rimandi Putra')  # Mencetak nama lengkap
print('NPM : 41155050210030\n')  # Mencetak Nomor Pokok Mahasiswa (NPM)
 
# Mengimpor library TensorFlow untuk membuat model jaringan saraf
import tensorflow as tf  # Library untuk membuat model jaringan saraf
 
# Membuat model jaringan saraf dengan arsitektur Sequential
model = tf.keras.models.Sequential()  # Membuat model dengan arsitektur Sequential
 
# Menambahkan lapisan Dense (Fully Connected) ke model
model.add(tf.keras.layers.Dense(units=3, activation='relu'))  # Lapisan Dense pertama dengan 3 neuron dan aktivasi ReLU
model.add(tf.keras.layers.Dense(units=9, activation='linear'))  # Lapisan Dense kedua dengan 9 neuron dan aktivasi Linear
model.add(tf.keras.layers.Dense(units=1))  # Lapisan Dense ketiga dengan 1 neuron (output)
 
# Mengkompilasi model dengan fungsi loss dan optimizer
model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(0.001))  # Menggunakan fungsi loss Mean Absolute Error dan optimizer Adam dengan learning rate 0.001
 
# Melatih model dengan data training
model.fit(x_train, y_train, epochs=800, batch_size=128)  # Melatih model dengan data training selama 800 epoch dan batch size 128

```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P12/pic/12a4.png?raw=true" alt="SS" width="35%"/>

5.	**Tampilkan prediksi nilai target**
```python
# Mencetak informasi identitas
print('Nama: Aditya Rimandi Putra')  # Mencetak nama lengkap
print('NPM : 41155050210030\n')  # Mencetak Nomor Pokok Mahasiswa (NPM)
 
# Mengimpor library Matplotlib untuk membuat plot
import matplotlib.pyplot as plt  # Library untuk membuat plot
 
# Memprediksi nilai target untuk data testing
print(model.predict(x_test))  # Memprediksi nilai target untuk data testing menggunakan model yang telah dilatih

```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P12/pic/12a5.png?raw=true" alt="SS" width="30%"/>

6.	**Membuat plot untuk membandingkan nilai target sebenarnya dan nilai target yang diprediksi**
```python
# Mencetak informasi identitas
print('Nama: Aditya Rimandi Putra')  # Mencetak nama lengkap
print('NPM : 41155050210030\n')  # Mencetak Nomor Pokok Mahasiswa (NPM)
 
# Membuat plot untuk membandingkan nilai target sebenarnya dan nilai target yang diprediksi
plt.plot(y_test, model.predict(x_test), 'b', label='Data Hasil Prediksi')  # Membuat plot untuk nilai target sebenarnya dan nilai target yang diprediksi
plt.title('Harga emas')  # Menambahkan judul ke plot
plt.legend()  # Menambahkan legenda ke plot
plt.show()  # Menampilkan plot

```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P12/pic/12a6.png?raw=true" alt="SS" width="30%"/>

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P12/pic/12a7.png?raw=true" alt="SS" width="50%"/>

Klik link dibawah ini untuk kembali ke menu utama:

> [!TIP]
> Kembali ke halaman utama, [Klik disini](https://github.com/AdityaR-AI/MLC/tree/main/)











