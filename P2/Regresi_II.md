# Regresi

## Multiple Linear Regression & Polynomial Regression

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

### 2.0. Lakukan praktek dari https://youtu.be/nWJUJenAyB8?si=BQDzWwrMnr8jtzpV  dan buat screen shot hasil run. Praktek tersebut yaitu:

### 2.1. Persiapan sample dataset

**Training Dataset**

```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

import pandas as pd  # Mengimpor pustaka pandas untuk manipulasi data

# Mendefinisikan data pizza dalam bentuk dictionary
pizza = {
    'diameter': [6, 8, 10, 14, 18],  # Diameter pizza dalam inci
    'n_topping': [2, 1, 0, 2, 0],    # Jumlah topping pada setiap pizza
    'harga': [7, 9, 13, 17.5, 18]    # Harga pizza dalam dolar
}

# Membuat DataFrame dari dictionary
train_pizza_df = pd.DataFrame(pizza)

# Menampilkan DataFrame
train_pizza_df
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P2/pics/2b1.png?raw=true" alt="SS" width="35%"/>

**Testing Dataset**

```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

# Mendefinisikan data pizza untuk DataFrame pengujian
pizza = {
    'diameter': [8, 9, 11, 16, 12],  # Diameter pizza dalam inci
    'n_topping': [2, 0, 2, 2, 0],    # Jumlah topping pada setiap pizza
    'harga': [11, 8.5, 15, 18, 11]    # Harga pizza dalam dolar
}

# Membuat DataFrame dari dictionary
test_pizza_df = pd.DataFrame(pizza)

# Menampilkan DataFrame
test_pizza_df
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P2/pics/2b2.png?raw=true" alt="SS" width="35%"/>

### 2.2. Preprocessing dataset

```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

import numpy as np  # Mengimpor pustaka NumPy untuk manipulasi array

# Mengonversi DataFrame pelatihan menjadi array NumPy
X_train = np.array(train_pizza_df[['diameter', 'n_topping']])  # Fitur: diameter dan jumlah topping
y_train = np.array(train_pizza_df['harga'])  # Target: harga

# Menampilkan array fitur dan target untuk data pelatihan
print(f'X_train:\n{X_train}\n')
print(f'y_train: {y_train}')
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P2/pics/2b3.png?raw=true" alt="SS" width="40%"/>

```python
# Mengonversi DataFrame pengujian menjadi array NumPy
X_test = np.array(test_pizza_df[['diameter', 'n_topping']])  # Fitur: diameter dan jumlah topping
y_test = np.array(test_pizza_df['harga'])  # Target: harga

# Menampilkan array fitur dan target untuk data pengujian
print(f'X_test:\n{X_test}\n')
print(f'y_test: {y_test}')
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P2/pics/2b4.png?raw=true" alt="SS" width="40%"/>

### 2.3. Pengenalan Multiple Linear Regression | Apa itu Multiple Linear Regression?

Multiple Linear Regression merupakan generalisasi dari Simple Linear Regression yang memungkinkan untuk menggunakan beberapa explanatory variables.

> y = a + β₁x1 + β2x2 + ... + β<sub>n</sub>x<sub>n</sub>

Referensi: https://en.wikipedia.org/wiki/Linear_regression

```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

from sklearn.linear_model import LinearRegression  # Mengimpor kelas LinearRegression dari pustaka scikit-learn
from sklearn.metrics import r2_score  # Mengimpor fungsi r2_score untuk menghitung koefisien determinasi

# Membuat model regresi linier
model = LinearRegression()

# Melatih model dengan data pelatihan
model.fit(X_train, y_train)

# Memprediksi harga pizza menggunakan data pengujian
y_pred = model.predict(X_test)

# Menghitung dan mencetak nilai R-squared
print(f'r_squared: {r2_score(y_test, y_pred)}')
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P2/pics/2b5.png?raw=true" alt="SS" width="30%"/>

### 2.4. Pengenalan Polynomial Regression | Apa itu Polynomial Regression?

> Polynomial Regression memodelkan hubungan antara independent variable X dan dependent variable y sebagai derajat polynomial dalam x.

Referensi: https://en.wikipedia.org/wiki/Polynomial_regression

**Preproccesing Dataset**

```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

# Mengonversi kolom 'diameter' dari DataFrame pelatihan menjadi array NumPy
X_train = np.array(train_pizza_df['diameter']).reshape(-1, 1)  # Fitur: diameter

# Mengonversi kolom 'harga' dari DataFrame pelatihan menjadi array NumPy
y_train = np.array(train_pizza_df['harga'])  # Target: harga

# Menampilkan array fitur dan target untuk data pelatihan
print(f'X_train:\n{X_train}\n')
print(f'y_train: {y_train}')
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P2/pics/2b6.png?raw=true" alt="SS" width="30%"/>

### 2.5. Quadratic Polynomial Regression

> y = α + β<sub>1</sub> x + β<sub>2</sub> x<sup>2</sup>

**Polynomial Features**

```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

from sklearn.preprocessing import PolynomialFeatures  # Mengimpor kelas PolynomialFeatures dari pustaka scikit-learn

# Membuat objek PolynomialFeatures dengan derajat 2 (kuadratik)
quadratic_feature = PolynomialFeatures(degree=2)

# Mengubah fitur X_train menjadi fitur kuadratik
X_train_quadratic = quadratic_feature.fit_transform(X_train)

# Menampilkan fitur kuadratik
print(f'X_train_quadratic:\n{X_train_quadratic}\n')
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P2/pics/2b7.png?raw=true" alt="SS" width="20%"/>

**Training Model**

```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

# Membuat model regresi linier
model = LinearRegression()

# Melatih model dengan fitur kuadratik dan target harga
model.fit(X_train_quadratic, y_train)
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P2/pics/2b8.png?raw=true" alt="SS" width="25%"/>

**Visualisasi Model**

```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

import matplotlib.pyplot as plt  # Mengimpor pustaka Matplotlib untuk visualisasi

# Membuat array X_vis yang berisi nilai dari 0 hingga 25 dengan 100 titik
X_vis = np.linspace(0, 25, 100).reshape(-1, 1)

# Mengubah X_vis menjadi fitur kuadratik menggunakan objek quadratic_feature
X_vis_quadratic = quadratic_feature.transform(X_vis)

# Menggunakan model untuk memprediksi harga berdasarkan fitur kuadratik
y_vis_quadratic = model.predict(X_vis_quadratic)

# Membuat scatter plot untuk data pelatihan
plt.scatter(X_train, y_train, label='Data Pelatihan', color='blue')

# Menambahkan garis prediksi ke plot
plt.plot(X_vis, y_vis_quadratic, '-r', label='Prediksi Kuadratik')

# Menambahkan judul dan label
plt.title('Perbandingan Diameter dan Harga Pizza')
plt.xlabel('Diameter (inch)')
plt.ylabel('Harga (dollar)')

# Mengatur batas sumbu
plt.xlim(0, 25)
plt.ylim(0, 25)

# Menambahkan grid untuk kemudahan visualisasi
plt.grid(True)

# Menampilkan legenda
plt.legend()

# Menampilkan plot
plt.show()
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P2/pics/2b9.png?raw=true" alt="SS" width="45%"/>

### 2.6. Linear Regression vs Quadratic Polynomial Regression vs Cubic Polynomial Regression

```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

#Training Set
# Membuat scatter plot untuk data pelatihan
plt.scatter(X_train, y_train)

#linier
# Model Regresi Linier
model = LinearRegression()  # Membuat model regresi linier
model.fit(X_train, y_train)  # Melatih model dengan data pelatihan
X_vis = np.linspace(0, 25, 100).reshape(-1, 1)  # Membuat array untuk visualisasi
y_vis = model.predict(X_vis)  # Memprediksi harga menggunakan model linier
plt.plot(X_vis, y_vis, '--r', label='linear')  # Menambahkan garis prediksi linier

#Quadratic
# Model Regresi Kuadratik
quadratic_feature = PolynomialFeatures(degree=2)  # Membuat fitur kuadratik
X_train_quadratic = quadratic_feature.fit_transform(X_train)  # Mengubah data pelatihan
model = LinearRegression()  # Membuat model regresi linier baru
model.fit(X_train_quadratic, y_train)  # Melatih model dengan fitur kuadratik
X_vis_quadratic = quadratic_feature.transform(X_vis)  # Mengubah data untuk visualisasi
y_vis = model.predict(X_vis_quadratic)  # Memprediksi harga menggunakan model kuadratik
plt.plot(X_vis, y_vis, '--g', label='quadratic')  # Menambahkan garis prediksi kuadratik

#Cubic
# Model Regresi Kubik
cubic_feature = PolynomialFeatures(degree=3)  # Membuat fitur kubik
X_train_cubic = cubic_feature.fit_transform(X_train)  # Mengubah data pelatihan
model = LinearRegression()  # Membuat model regresi linier baru
model.fit(X_train_cubic, y_train)  # Melatih model dengan fitur kubik
X_vis_cubic = cubic_feature.transform(X_vis)  # Mengubah data untuk visualisasi
y_vis = model.predict(X_vis_cubic)  # Memprediksi harga menggunakan model kubik
plt.plot(X_vis, y_vis, '--y', label='cubic')  # Menambahkan garis prediksi kubik

# Menambahkan judul dan label
plt.title('Perbandingan Diameter dan Harga Pizza')
plt.xlabel('Diameter (inch)')
plt.ylabel('Harga (dollar)')

# Menampilkan legenda dan pengaturan sumbu
plt.legend()
plt.xlim(0, 25)
plt.ylim(0, 25)
plt.grid(True)

# Menampilkan plot
plt.show()
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P2/pics/2b10.png?raw=true" alt="SS" width="50%"/>

Klik link dibawah ini untuk kembali ke menu utama:

> [!TIP]
> Kembali ke halaman utama, [Klik disini](https://github.com/AdityaR-AI/MLC/tree/main/)








