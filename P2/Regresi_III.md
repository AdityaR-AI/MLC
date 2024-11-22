## Regresi

### Logistic Regression pada Binary Classification Task

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

### 3.0. Lakukan praktek dari https://youtu.be/oe7DW4rSH1o?si=H-PZJ9rs9-Kab-Ln  dan buat screen shot hasil run. Praktek tersebut yaitu:

### 3.1. Formula dasar pembentuk Logistic Regression | Fungsi Sigmoid

**Simple Linear Regression**
  - y = α + βx
  - g(x) = α + βx 

**Multiple Linear Regression**
  - y = α + β₁x₁ + β₂x₂ + ... + βₙxₙ
  - g(X) = α + βX

**Logistic Regression**
  - g(X) = sigmoid(a + ẞX)
  - sigmoid(x) = 1 / (1+exp(-x))

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P2/a.png?raw=true" alt="SS" width="35%"/>

### 3.2. Persiapan dataset | SMS Spam Collection Dataset

```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

import pandas as pd  # Mengimpor pustaka pandas untuk manipulasi data

# Membaca dataset dari file CSV
df = pd.read_csv('./dataset/SMSSpamCollection',
                 sep='\t',  # Menentukan pemisah kolom sebagai tab
                 header=None,  # Tidak ada baris header dalam file
                 names=['label', 'sms'])  # Menentukan nama kolom

# Menampilkan 5 baris pertama dari DataFrame
df.head()
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P2/pics/2c1.png?raw=true" alt="SS" width="45%"/>

```python
# Menghitung jumlah setiap label (spam dan ham)
df['label'].value_counts()
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P2/pics/2c2.png?raw=true" alt="SS" width="28%"/>

### 3.3. Pembagian training dan testing set

```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

from sklearn.preprocessing import LabelBinarizer  # Mengimpor LabelBinarizer untuk mengubah label menjadi format biner
X = df['sms'].values  # Mengambil nilai SMS dari DataFrame
y = df['label'].values  # Mengambil nilai label dari DataFrame

# Membuat objek LabelBinarizer dan mengubah label menjadi format biner
lb = LabelBinarizer()
y = lb.fit_transform(y).ravel()  # Transformasi dan meratakan array hasil

# Menampilkan kelas yang ada setelah binarisasi
lb.classes_
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P2/pics/2c3.png?raw=true" alt="SS" width="28%"/>

```python
from sklearn.model_selection import train_test_split  # Mengimpor fungsi untuk membagi dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)  # Membagi data menjadi set pelatihan dan pengujian

# Menampilkan data pelatihan
print(X_train, '\n')
print(y_train)
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P2/pics/2c4.png?raw=true" alt="SS" width="95%"/>

### 3.4. Feature extraction dengan TF-IDF

```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

from sklearn.feature_extraction.text import TfidfVectorizer  # Mengimpor TfidfVectorizer untuk mengubah teks menjadi representasi numerik

# Membuat objek TfidfVectorizer dengan mengabaikan kata-kata umum (stop words)
vectorizer = TfidfVectorizer(stop_words='english')

# Mengubah data pelatihan menjadi representasi TF-IDF
X_train_tfidf = vectorizer.fit_transform(X_train)

# Mengubah data pengujian menjadi representasi TF-IDF menggunakan vektor yang sama
X_test_tfidf = vectorizer.transform(X_test)

# Menampilkan representasi TF-IDF dari data pelatihan
print(X_train_tfidf)
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P2/pics/2c5.png?raw=true" alt="SS" width="32%"/>

### 3.5. Binary Classification dengan Logistic Regression

```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

from sklearn.linear_model import LogisticRegression  # Mengimpor LogisticRegression dari scikit-learn

# Membuat objek model Logistic Regression
model = LogisticRegression()

# Melatih model menggunakan data pelatihan TF-IDF dan label
model.fit(X_train_tfidf, y_train)

# Memprediksi label untuk data pengujian TF-IDF
y_pred = model.predict(X_test_tfidf)

# Menampilkan 5 prediksi pertama beserta SMS yang sesuai
for pred, sms in zip(y_pred[:5], X_test[:5]):
    print(f'PRED: {pred} SMS: {sms}\n')
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P2/pics/2c6.png?raw=true" alt="SS" width="95%"/>


### 3.6. Evaluation Metrics pada Binary Classification Task

> Terdiri dari

<ul>
  <li>Confusion Matrix</li>
  <li>Accuracy</li>
  <li>Precission & Recall</li>
  <li>F1 Score</li>
  <li>ROC</li>
</ul>

<strong>Dengan Terminologi Dasar</strong> 
<ul> 
  <li>True Positive (TP)</li> 
  <li>True Negative (TN)</li> 
  <li>False Positive (FP)</li> 
  <li>False Negative (FN)</li> 
</ul>

### 3.7. Pengenalan Confusion Matrix

```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

from sklearn.metrics import confusion_matrix  # Mengimpor fungsi confusion_matrix dari scikit-learn

# Membuat matriks kebingungan (confusion matrix) berdasarkan label sebenarnya dan prediksi
matrix = confusion_matrix(y_test, y_pred)

# Menampilkan matriks kebingungan
matrix
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P2/pics/2c7.png?raw=true" alt="SS" width="30%"/>

```python
# Mengambil nilai TN, FP, FN, TP dari matriks kebingungan
tn, fp, fn, tp = matrix.ravel()

# Mencetak nilai TN, FP, FN, TP
print(f'TN: {tn}')
print(f'FP: {fp}')
print(f'FN: {fn}')
print(f'TP: {tp}')
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P2/pics/2c8.png?raw=true" alt="SS" width="10%"/>

```python
import matplotlib.pyplot as plt  # Mengimpor pustaka matplotlib untuk visualisasi

# Menampilkan matriks kebingungan sebagai gambar
plt.matshow(matrix)  # Menggunakan matshow untuk menampilkan matriks kebingungan

plt.colorbar()  # Menambahkan bar warna untuk menunjukkan skala nilai
plt.title('Confusion Matrix')  # Menambahkan judul pada grafik
plt.ylabel('True label')  # Menambahkan label pada sumbu y
plt.xlabel('Predicted label')  # Menambahkan label pada sumbu x
plt.show()  # Menampilkan grafik
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P2/pics/2c9.png?raw=true" alt="SS" width="35%"/>

### 3.8. Pengenalan Accuracy Score

Accuracy (Akurasi) merupakan pengukuran proporsi prediksi yang benar dari total prediksi. Rumus akurasi adalah:

> Akurasi = TP+TN / TP+TN+FP+FN = (Jumlah prediksi benar) / (Jumlah total prediksi)

Contohnya, jika model Anda memprediksi 100 data dan 90 di antaranya benar, maka akurasinya adalah 90%.

Referensi: https://en.wikipedia.org/wiki/Accuracy_and_precision

```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

from sklearn.metrics import accuracy_score  # Mengimpor fungsi accuracy_score dari scikit-learn

# Menghitung dan menampilkan akurasi model
accuracy = accuracy_score(y_test, y_pred)
accuracy
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P2/pics/2c10.png?raw=true" alt="SS" width="25%"/>

### 3.9. Pengenalan Precision dan Recall

**Precission & Recall**

Selain menggunakan accuracy, performa dari suatu classifier umumnya juga diukur berdasarkan nilai Precission dan Recall.

Referensi: https://en.wikipedia.org/wiki/Precision_and_recall

**Precission or Positive Predictive Value (PPV)**

> Precission = TP / TP+FP

Referensi: https://en.wikipedia.org/wiki/Positive_and_negative_predictive_values

```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

from sklearn.metrics import precision_score  # Mengimpor fungsi precision_score dari scikit-learn

# Menghitung dan menampilkan presisi model
precision = precision_score(y_test, y_pred)
precision
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P2/pics/2c11.png?raw=true" alt="SS" width="25%"/>

**Recall or True Positive Rate (TPR) or Sensitivity**

> Recall = TP / TP+FN

Referensi: https://en.wikipedia.org/wiki/Sensitivity_and_specificity

```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

from sklearn.metrics import recall_score  # Mengimpor fungsi recall_score dari scikit-learn

# Menghitung dan menampilkan recall model
recall = recall_score(y_test, y_pred)
recall
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P2/pics/2c12.png?raw=true" alt="SS" width="25%"/>

### 3.10. Pengenalan F1 Score | F1 Measure

**F1-Score**

F1-score atau F1-measure adalah harmonic mean dari precission dan recall.

> F1 score = precission × recall / precission + recall

Referensi: https://en.wikipedia.org/wiki/F-score

```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

from sklearn.metrics import f1_score  # Mengimpor fungsi f1_score dari scikit-learn

# Menghitung dan menampilkan F1-score model
f1_score(y_test, y_pred)

```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P2/pics/2c13.png?raw=true" alt="SS" width="25%"/>

### 3.11. Pengenalan ROC | Receiver Operating Characteristic

**ROC: Receiver Operating Characteristic**

ROC menawarkan visualisasi terhadap performa dari classifier dengan membandingkan nilai Recall (TPR) dan nilai Fallout (FPR)

> fallout = FP / TN+FP

Referensi: https://en.wikipedia.org/wiki/Receiver_operating_characteristic

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P2/b.png?raw=true" alt="SS" width="40%"/>

```python
# Mencetak nama dan NPM
print('Nama: Aditya Rimandi Putra')
print('NPM : 41155050210030\n')

from sklearn.metrics import roc_curve, auc  # Mengimpor fungsi roc_curve dan auc dari scikit-learn
import matplotlib.pyplot as plt  # Mengimpor matplotlib untuk visualisasi

# Menghitung probabilitas prediksi dari model
prob_estimates = model.predict_proba(X_test_tfidf)

# Menghitung False Positive Rate (FPR), True Positive Rate (TPR), dan threshold
fpr, tpr, threshold = roc_curve(y_test, prob_estimates[:, 1])

# Menghitung Area Under the Curve (AUC)
nilai_auc = auc(fpr, tpr)

# Memvisualisasikan ROC Curve
plt.plot(fpr, tpr, 'b', label=f'AUC={nilai_auc:.2f}')  # Plot ROC Curve
plt.plot([0, 1], [0, 1], 'r--', label='Random Classifier')  # Garis acuan untuk classifier acak
plt.title('ROC: Receiver Operating Characteristic')  # Judul plot
plt.xlabel('Fallout or False Positive Rate')  # Label sumbu X
plt.ylabel('Recall or True Positive Rate')  # Label sumbu Y
plt.legend()  # Menampilkan legenda
plt.show()  # Menampilkan plot
```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P2/pics/2c14.png?raw=true" alt="SS" width="60%"/>

> [!TIP]
> Kembali ke halaman utama, [Klik disini](https://github.com/AdityaR-AI/MLC/tree/main/)


