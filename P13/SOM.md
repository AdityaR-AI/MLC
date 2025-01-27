Klik link dibawah ini untuk kembali ke menu utama:

> [!TIP]
> Kembali ke halaman utama, [Klik disini](https://github.com/AdityaR-AI/MLC/tree/main/)

## Self Organizing Maps (SOM)

1.	**Pengolahan Data dan Persiapan Library**
```python

# Mencetak informasi identitas
print('Nama: Aditya Rimandi Putra')  # Mencetak nama lengkap
print('NPM : 41155050210030\n')  # Mencetak Nomor Pokok Mahasiswa (NPM)
 
# Mengimpor library pandas untuk pengolahan data
import pandas as pd
 
# Mengimpor library StandardScaler dari scikit-learn untuk melakukan normalisasi data
from sklearn.preprocessing import StandardScaler
 
# Menginstal library simpsom menggunakan pip
!pip install simpsom
 
# Mengimpor library simpsom untuk pengolahan data
import simpsom as sps


```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P13/pic/13a1.png?raw=true" alt="SS" width="85%"/>

2.	**Mempersiapkan Data untuk Analisis**
```python

# Mencetak informasi identitas
print('Nama: Aditya Rimandi Putra')  # Mencetak nama lengkap
print('NPM : 41155050210030\n')  # Mencetak Nomor Pokok Mahasiswa (NPM)
 
# Mengunduh dataset dari URL yang diberikan
url = 'https://raw.githubusercontent.com/kokocamp/vlog119/refs/heads/main/vlog119.csv'
vlog139 = pd.read_csv(url)  # Membaca dataset dari URL ke dalam DataFrame
 
# Memilih fitur yang akan digunakan untuk analisis
X = vlog139[['gpa','gmat','work_experience','admitted']]  # Fitur independen
y = vlog139['admitted']  # Fitur dependen (target)
 
# Membuat objek StandardScaler untuk melakukan normalisasi data
scaler = StandardScaler()
 
# Melakukan normalisasi data menggunakan StandardScaler
data = scaler.fit_transform(pd.DataFrame(X))  # Normalisasi fitur independen
labels = scaler.fit_transform(pd.DataFrame(y))  # Normalisasi fitur dependen (target)
 
# Mencetak hasil normalisasi data
print(data)


```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P13/pic/13a2.png?raw=true" alt="SS" width="45%"/>

3.	**Membangun dan Melatih Model Self-Organizing Map (SOM)**
```python

# Mencetak informasi identitas
print('Nama: Aditya Rimandi Putra')  # Mencetak nama lengkap
print('NPM : 41155050210030\n')  # Mencetak Nomor Pokok Mahasiswa (NPM)
 
# Membuat model Self-Organizing Map (SOM)
net = sps.SOMNet(10, 10, data)  # Membuat jaringan SOM dengan ukuran 10x10 dan data yang telah dinormalisasi
 
# Melakukan pelatihan model SOM
net.train(train_algo='batch', epochs=1000, start_learning_rate=0.01)  # Melakukan pelatihan dengan algoritma batch, 1000 epoch, dan learning rate awal 0.01
 
# Visualisasi hasil model SOM
net.nodes_graph()  # Mencetak grafik node SOM
net.diff_graph()  # Mencetak grafik perbedaan antara node SOM
net.project(data, labels=labels)  # Mencetak hasil projeksi data ke dalam SOM
net.cluster(data)  # Mencetak hasil clustering data menggunakan SOM


```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P13/pic/13a3.png?raw=true" alt="SS" width="40%"/>
<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P13/pic/13a4.png?raw=true" alt="SS" width="60%"/>
<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P13/pic/13a5.png?raw=true" alt="SS" width="30%"/>

Klik link dibawah ini untuk kembali ke menu utama:

> [!TIP]
> Kembali ke halaman utama, [Klik disini](https://github.com/AdityaR-AI/MLC/tree/main/)
