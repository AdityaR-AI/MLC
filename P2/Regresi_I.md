
> Materi ini terbagi menjadi 3 Part, berikut linknya:

Silahkan klik link dibawah ini tuntuk menuju tugas yang inign dilihat:

> [!NOTE]
> Part 1 - Simple Linear Regression dengan Scikit-Learn [Pages Link](https://github.com/AdityaR-AI/MLC/tree/main/P2/Regresi_I.md)

> [!NOTE]
> Part 2 - Multiple Linear Regression & Polynomial Regression [Pages Link](https://github.com/AdityaR-AI/MLC/tree/main/P2/Regresi_II.md)

> [!NOTE]
> Part 3 - Logistic Regression pada Binary Classification Task [Pages Link](https://github.com/AdityaR-AI/MLC/tree/main/P2/Regresi_III.md)

Klik link dibawah ini untuk kembali ke menu utama:

> [!TIP]
> Kembali ke halaman utama, [Klik disini](https://github.com/AdityaR-AI/MLC/tree/main/)

## Simple Linear Regression dengan Scikit-Learn

### 1.0. Lakukan praktek dari https://youtu.be/lcjq7-2zMSA?si=f4jWJR6lY8y0BZKl  dan buat screen shot hasil run. Praktek tersebut yaitu:

### 1.1. Sample dataset

> Simple Linear Regression memodelkan hubungan antara sebuah response variable dengan sebuah explanatory variable sebagai suatu garis lurus (linear)
> Referensi: https://en.wikipedia.org/wiki/Simple_linear_regression

```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

# Mengimpor pustaka pandas dengan alias pd
import pandas as pd

# Membuat dictionary yang berisi data diameter dan harga pizza
pizza = {
    'diameter': [6, 8, 10, 14, 18],  # Daftar diameter pizza dalam inci
    'harga': [7, 9, 13, 17.5, 18]    # Daftar harga pizza dalam dolar
}

# Mengonversi dictionary menjadi DataFrame menggunakan pandas
pizza_df = pd.DataFrame(pizza)

# Menampilkan DataFrame pizza_df
pizza_df
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P2/pics/2a1.png?raw=true" alt="SS" width="40%"/>

### 1.2. Visualisasi dataset

```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

# Mengimpor pustaka matplotlib.pyplot dengan alias plt untuk visualisasi data
import matplotlib.pyplot as plt

# Membuat plot sebar (scatter plot) menggunakan DataFrame pizza_df
# 'kind' menentukan jenis plot, 'x' dan 'y' menentukan kolom yang akan digunakan untuk sumbu
pizza_df.plot(kind='scatter', x='diameter', y='harga')

# Menambahkan judul pada plot
plt.title('Perbandingan Diameter dan Harga Pizza')

# Menambahkan label untuk sumbu x
plt.xlabel('Diameter (inch)')

# Menambahkan label untuk sumbu y
plt.ylabel('Harga (dollar)')

# Mengatur batas sumbu x dari 0 hingga 25
plt.xlim(0, 25)

# Mengatur batas sumbu y dari 0 hingga 25
plt.ylim(0, 25)

# Menampilkan grid pada plot untuk memudahkan pembacaan
plt.grid(True)

# Menampilkan plot
plt.show()
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P2/pics/2a2.png?raw=true" alt="SS" width="60%"/>

### 1.3. Transformasi dataset

**Penyesuaian Dataset**

```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

# Mengimpor pustaka numpy dengan alias np untuk manipulasi array
import numpy as np

# Mengonversi kolom 'diameter' dari DataFrame pizza_df menjadi array numpy
X = np.array(pizza_df['diameter'])

# Mengonversi kolom 'harga' dari DataFrame pizza_df menjadi array numpy
y = np.array(pizza_df['harga'])

# Menampilkan array X yang berisi diameter pizza
print(f'x: {X}')

# Menampilkan array y yang berisi harga pizza
print(f'y: {y}')
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P2/pics/2a3.png?raw=true" alt="SS" width="40%"/>

```python
# Mengubah bentuk array X menjadi dua dimensi
X = X.reshape(-1, 1)

# Menampilkan bentuk (shape) dari array X yang baru
X.shape
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P2/pics/2a4.png?raw=true" alt="SS" width="10%"/>

```python
# Menampilkan isi dari array X setelah diubah bentuk
X
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P2/pics/2a5.png?raw=true" alt="SS" width="35%"/>


### 1.4. Training Simple Linear Regression Model

```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

# Mengimpor kelas LinearRegression dari modul sklearn.linear_model
from sklearn.linear_model import LinearRegression

# Membuat instance dari model LinearRegression
model = LinearRegression()

# Melatih model dengan data pelatihan (X dan y)
# X mewakili fitur input, sedangkan y mewakili variabel target
model.fit(X, y)
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P2/pics/2a6.png?raw=true" alt="SS" width="35%"/>

### 1.5. Visualisasi Simple Linear Regression Model | Penjelasan persamaan garis linear

```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

# Membuat array numpy dengan dua elemen dan mengubah bentuknya menjadi dua dimensi
X_vis = np.array([0, 25]).reshape(-1, 1)

# Menggunakan model untuk memprediksi nilai berdasarkan input X_vis
y_vis = model.predict(X_vis)
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P2/pics/2a7.png?raw=true" alt="SS" width="35%"/>

```python
# Membuat scatter plot dari data X dan y
plt.scatter(X, y)

# Menambahkan garis plot untuk prediksi berdasarkan X_vis dan y_vis
plt.plot(X_vis, y_vis, '-r')

# Menambahkan judul dan label pada sumbu
plt.title('Perbandingan Diameter dan Harga Pizza')
plt.xlabel('Diameter (inch)')
plt.ylabel('Harga (dollar)')

# Mengatur batas sumbu x dan y
plt.xlim(0, 25)
plt.ylim(0, 25)

# Menampilkan grid
plt.grid(True)

# Menampilkan plot
plt.show()
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P2/pics/2a8.png?raw=true" alt="SS" width="45%"/>

Formula Linear Regression: y = a + βx

  - y: response variable
  - x: explanatory variable
  - a: intercept
  - β: slope

```python
# Menampilkan nilai intercept dan slope dari model regresi
print(f'intercept: {model.intercept_}')
print(f'slope: {model.coef_}')
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P2/pics/2a9.png?raw=true" alt="SS" width="40%"/>

### 1.6. Kalkulasi nilai slope

**Mencari nilai slope**

Nilai slope pada Linear Regression bisa diperoleh dengan memanfaatkan formula
berikut:

> β = cou(x,y) / var(x)

```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

# Menampilkan nilai dari X dan y
print(f'X:\n{X}\n')
print(f'X flatten: {X.flatten()}\n')
print(f'y: {y}')
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P2/pics/2a10.png?raw=true" alt="SS" width="35%"/>

**Variance**

```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

# Menghitung varians dari X yang telah diratakan
variance_x = np.var(X.flatten(), ddof=1)
print(f'variance: {variance_x}')
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P2/pics/2a11.png?raw=true" alt="SS" width="35%"/>

**Covaraince**

```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

# Menghitung matriks kovarians antara X yang telah diratakan dan y
np.cov(X.flatten(), y)
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P2/pics/2a12.png?raw=true" alt="SS" width="35%"/>

```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

# Menghitung kovarians antara X dan y
covariance_xy = np.cov(X.transpose(), y)[0][1]
print(f'covariance: {covariance_xy}')
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P2/pics/2a13.png?raw=true" alt="SS" width="35%"/>

**Slope**

```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

# Menghitung kemiringan (slope) dari regresi linier
slope = covariance_xy / variance_x
print(f'slope: {slope}')
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P2/pics/2a14.png?raw=true" alt="SS" width="35%"/>

### 1.7. Kalkukasi nilai intercept

**Mencari nilai intercept**

Nilai intercept pada Linear Regression bisa diperoleh dengan memanfaatkan formula berikut:

> α = ȳ – βx

```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

# Menghitung intercept dari regresi linier
intercept = np.mean(y) - slope * np.mean(X)
print(f'intercept: {intercept}')
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P2/pics/2a15.png?raw=true" alt="SS" width="40%"/>

### 1.8. Prediksi harga pizza dengan Simple Linear Regression Model

```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

# Mendefinisikan diameter pizza dan mereshape menjadi array 2D
diameter_pizza = np.array([12, 20, 23]).reshape(-1, 1)
diameter_pizza
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P2/pics/2a16.png?raw=true" alt="SS" width="35%"/>

```python
# Menggunakan model untuk memprediksi harga berdasarkan diameter pizza
prediksi_harga = model.predict(diameter_pizza)
prediksi_harga
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P2/pics/2a17.png?raw=true" alt="SS" width="60%"/>

```python
# Menampilkan diameter pizza dan prediksi harga
for dmtr, hrg in zip(diameter_pizza, prediksi_harga):
    print(f'Diameter: {dmtr[0]} prediksi harga: {hrg}')
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P2/pics/2a18.png?raw=true" alt="SS" width="60%"/>

### 1.9. Evaluasi model dengan Coefficient of Determination | R Squared

**Training & Testing Dataset**

```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

# Mendefinisikan data pelatihan untuk X dan y
X_train = np.array([6, 8, 10, 14, 18]).reshape(-1, 1)
y_train = np.array([7, 9, 13, 17.5, 18])

# Mendefinisikan data pengujian untuk X dan y
X_test = np.array([8, 9, 11, 16, 12]).reshape(-1, 1)
y_test = np.array([11, 8.5, 15, 18, 11])
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P2/pics/2a19.png?raw=true" alt="SS" width="35%"/>

**Training Simple Liniear Regression Modelt**

```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

# Membuat objek model regresi linier
model = LinearRegression()

# Melatih model menggunakan data pelatihan
model.fit(X_train, y_train)
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P2/pics/2a20.png?raw=true" alt="SS" width="35%"/>

**Evaluasi Linear Regression Model dengan Coefficient of Determination atau R-squared (R2)**

Referensi: https://en.wikipedia.org/wiki/Coefficient_of_determination

```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

# Mengimpor fungsi r2_score dari pustaka scikit-learn
from sklearn.metrics import r2_score 

# Menggunakan model untuk memprediksi nilai pada data pengujian
y_pred = model.predict(X_test)

# Menghitung nilai R-squared untuk mengevaluasi kinerja model
r_squared = r2_score(y_test, y_pred)

# Menampilkan nilai R-squared
print(f'R-squared: {r_squared}')
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P2/pics/2a21.png?raw=true" alt="SS" width="35%"/>

**Mencari nilai R-squared (R<sup>2</sup>)**

SS<sub>res</sub> = Σ<sub>i=1</sub><sup>n</sup> (y<sub>i</sub> - f(x<sub>i</sub>))<sup>2</sup> <br>

SS<sub>tot</sub> = Σ<sub>i=1</sub><sup>n</sup> (y<sub>i</sub> - ȳ)<sup>2</sup> <br>

SS<sub>res</sub>

```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

# Menghitung jumlah kuadrat residu (ss_res)
ss_res = sum([(y_i - model.predict(x_i.reshape(-1, 1))[0])**2
               for x_i, y_i in zip(X_test, y_test)])

# Menampilkan nilai ss_res
print(f'ss_res: {ss_res}')
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P2/pics/2a22.png?raw=true" alt="SS" width="35%"/>

SS<sub>tot</sub>

```python
# Menghitung rata-rata dari nilai yang sebenarnya
mean_y = np.mean(y_test)

# Menghitung jumlah kuadrat total (ss_tot)
ss_tot = sum([(y_i - mean_y)**2 for y_i in y_test])

# Menampilkan nilai ss_tot
print(f'ss_tot: {ss_tot}')
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P2/pics/2a23.png?raw=true" alt="SS" width="15%"/>

R<sup>2</sup>

```python
# Menghitung nilai R-squared
r_squared = 1 - (ss_res / ss_tot)

# Menampilkan nilai R-squared
print(f'R-squared: {r_squared}')
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P2/pics/2a24.png?raw=true" alt="SS" width="35%"/>

Klik link dibawah ini untuk kembali ke menu utama:

> [!TIP]
> Kembali ke halaman utama, [Klik disini](https://github.com/AdityaR-AI/MLC/tree/main/)



