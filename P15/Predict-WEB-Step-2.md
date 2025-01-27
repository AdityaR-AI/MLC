## Aplikasi Deteksi Diabetes berbasis Web

### Mengaplikasikan Model ke Web

Klik link dibawah ini untuk kembali ke menu utama:

> [!TIP]
> Kembali ke halaman utama, [Klik disini](https://github.com/AdityaR-AI/MLC/tree/main/)

Silahkan klik link dibawah ini untuk menuju Step yang ingin dilihat:

> [!NOTE]
> Step 1 - Membuat Model [Pages Link](https://github.com/AdityaR-AI/MLC/tree/main/P15/Predict-WEB-Step-1.md)

> [!NOTE]
> Step 2 - Mengaplikasikan Model ke Web [Pages Link](https://github.com/AdityaR-AI/MLC/tree/main/P15/Predict-WEB-Step-2.md)

> [!NOTE]
> Step 3 - Deploy Hosting [Pages Link](https://github.com/AdityaR-AI/MLC/tree/main/P15/Predict-WEB-Step-3.md)

> [!NOTE]
> **Link Hosting:** [https://adityari.pythonanywhere.com/](https://adityari.pythonanywhere.com/)

1. **Buka anaconda prompt**
<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P15/pic/15b1.png?raw=true" alt="SS" width="25%"/>

2.	**Pergi ke direktori web akan disimpan**

**Cd [directory route]**

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P15/pic/15b2.png?raw=true" alt="SS" width="65%"/>

3.	**Instal virtualenv dan aktifkan, paste kode berikut ke anaconda prompt**

```python
pip install virtualenv
 
virtualenv env
 
env\Scripts\activate.bat

```
<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P15/pic/15b3.png?raw=true" alt="SS" width="65%"/>
<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P15/pic/15b4.png?raw=true" alt="SS" width="65%"/>
<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P15/pic/15b5.png?raw=true" alt="SS" width="65%"/>

4.	**Instal library yang dibutuhkan sesuaikan dengan versi saat membuat model.**

```python
pip install Flask==3.0.3 Flask-SQLAlchemy
 
pip install scikit-learn==1.5.1

```
<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P15/pic/15b6.png?raw=true" alt="SS" width="75%"/>
<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P15/pic/15b7.png?raw=true" alt="SS" width="75%"/>

5.	**Donwload template web**

pergi ke [Klik disini](https://github.com/heriistantoo/flaskdiabet) dan download template, simpan di folder project dan sesuaikan struktur menjadi:

```
/project
  |-- app.py
  |-- requirements.txt
  |-- /static    (CSS, JS)
  |-- /templates (HTML files)
  |-- knn_pickle (model yang sudah dipickle) #ganti dengan pickle yg sudah dibuat sendiri    distep 1
```

6.	**Sesuaikan requirement sesuai dengan yang digunakan untuk membuat model**

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P15/pic/15b8.png?raw=true" alt="SS" width="35%"/>

7.	**sesuaikan template html dan py**

**index.html**
```html
{% extends 'base.html' %}
 
{% block head %}
<title>Tugas Pertemuan 15 - Machine Learning</title>
{% endblock %}
 
{% block body %}
<div class="container-contact100">
    <div class="wrap-contact100">
        <span class="label-input100">
            Nama : Aditya Rimandi Putra
            <br>
              NPM      : 41155050210030
        </span>
        <form class="contact100-form validate-form" action="/" method="POST">
             
            <span class="contact100-form-title">
                Deteksi Diabetes!
            </span>
 
            <div class="wrap-input100 validate-input">
                <span class="label-input100">Banyak Melahirkan</span>
                <input class="input100" type="text" placeholder="Masukan jumlah melahirkan" name="melahirkan" id="melahirkan" required="">
                <span class="focus-input100"></span>
            </div>
 
            <div class="wrap-input100 validate-input">
                <span class="label-input100">Kadar Glukosa</span>
                <input class="input100" type="text" placeholder="Masukan kadar glukosa" name="glukosa" id="glukosa" required="">
                <span class="focus-input100"></span>
            </div>
 
            <div class="wrap-input100 validate-input">
                <span class="label-input100">Tekanan Darah</span>
                <input class="input100" type="text" placeholder="Masukan tekanan darah" name="darah" id="darah" required="">
                <span class="focus-input100"></span>
            </div>
 
            <div class="wrap-input100 validate-input">
                <span class="label-input100">Tebal Kulit</span>
                <input class="input100" type="text" placeholder="Masukan ketebalan kulit" name="kulit" id="kulit" required="">
                <span class="focus-input100"></span>
            </div>
 
            <div class="wrap-input100 validate-input">
                <span class="label-input100">Kadar Insulin</span>
                <input class="input100" type="text" placeholder="Masukan kadar insulin" name="insulin" id="insulin" required="">
                <span class="focus-input100"></span>
            </div>
 
            <div class="wrap-input100 validate-input">
                <span class="label-input100">BMI</span>
                <input class="input100" type="text" placeholder="Masukan BMI" name="bmi" id="bmi" required="">
                <span class="focus-input100"></span>
            </div>
 
            <div class="wrap-input100 validate-input">
                <span class="label-input100">Riwayat Diabetes</span>
                <input class="input100" type="text" placeholder="Masukan derajat diabetes keturunan" name="riwayat" id="riwayat" required="">
                <span class="focus-input100"></span>
            </div>
 
            <div class="wrap-input100 validate-input">
                <span class="label-input100">Umur</span>
                <input class="input100" type="text" placeholder="Masukan umur" name="umur" id="umur" required="">
                <span class="focus-input100"></span>
            </div>
 
            <div class="container-contact100-form-btn">
                <div class="wrap-contact100-form-btn">
                    <div class="contact100-form-bgbtn"></div>
                    <button class="contact100-form-btn">
                        <span>
                            Prediksi
                            <i class="fa fa-long-arrow-right m-l-7" aria-hidden="true"></i>
                        </span>
                    </button>
                </div>
            </div>
 
        </form>
    </div>
</div>
 
{% endblock %}


```
**hasil.html**
```html
{% extends 'base.html' %}
 
{% block head %}
<title>Tugas Pertemuan 15 - Machine Learning</title>
{% endblock %}
 
{% block body %}
<div class="container-contact100">
    <div class="wrap-contact100">
        <span class="label-input100">
            Nama : Aditya Rimandi Putra
            <br>
              NPM      : 41155050210030
        </span>
        <form class="contact100-form validate-form" action="/" method="GET">
            {% if finalData == 1 %}
            <span class="contact100-form-title">
                POSITIF DIABETES
            </span>
 
            {% else %}
            <span class="contact100-form-title">
                NEGATIF DIABETES
            </span>
 
            {% endif %}
 
            <div class="container-contact100-form-btn">
                <div class="wrap-contact100-form-btn">
                    <div class="contact100-form-bgbtn"></div>
                    <button class="contact100-form-btn">
                        <span>
                            Selesai
                        </span>
                    </button>
                </div>
            </div>
             
        </form>
    </div>
</div>
 
{% endblock %}


```
**app.py**
```python
# Mencetak nama dan NIM
print('Aditya Rimandi Putra')
print('41155050210030')
 
# Mengimpor library yang diperlukan
from flask import Flask, render_template, request, redirect  # Flask untuk membuat aplikasi web
import pickle  # Untuk memuat model yang disimpan
import sklearn  # Library machine learning
import numpy as np  # Untuk komputasi numerik
import os  # Untuk mengelola path file
 
# Membuat instance Flask
app = Flask(__name__)
 
# Mendefinisikan route utama ('/') dengan metode POST dan GET
@app.route('/', methods=['POST', 'GET'])
def index():
    # Jika metode request adalah POST (form dikirim)
    if request.method == 'POST':
        # Tentukan path lengkap ke file knn_pickle
        # os.path.join digunakan untuk menggabungkan direktori saat ini dengan nama file
        pickle_file_path = os.path.join(os.path.dirname(__file__), 'knn_pickle')
         
        # Memuat model KNN dari file knn_pickle
        with open(pickle_file_path, 'rb') as r:
            model = pickle.load(r)
 
        # Mengambil input dari form HTML
        melahirkan = float(request.form['melahirkan'])  # Jumlah melahirkan
        glukosa    = float(request.form['glukosa'])    # Kadar glukosa
        darah      = float(request.form['darah'])      # Tekanan darah
        kulit      = float(request.form['kulit'])      # Ketebalan kulit
        insulin    = float(request.form['insulin'])    # Kadar insulin
        bmi        = float(request.form['bmi'])        # Indeks massa tubuh (BMI)
        riwayat    = float(request.form['riwayat'])    # Riwayat diabetes dalam keluarga
        umur       = float(request.form['umur'])       # Usia
 
        # Menyiapkan data untuk prediksi
        # Mengubah input menjadi array numpy
        datas = np.array((melahirkan, glukosa, darah, kulit, insulin, bmi, riwayat, umur))
        # Mengubah bentuk array menjadi (1, 8) karena model mengharapkan input 2D
        datas = np.reshape(datas, (1, -1))
 
        # Melakukan prediksi menggunakan model
        isDiabetes = model.predict(datas)
 
        # Render template hasil.html dan kirim hasil prediksi (isDiabetes) ke template
        return render_template('hasil.html', finalData=isDiabetes)
     
    # Jika metode request adalah GET (halaman dimuat pertama kali)
    else:
        # Render template index.html (form input)
        return render_template('index.html')
     
# Menjalankan aplikasi Flask
if __name__ == "__main__": #`ctrl + /` if __main__ saat deploy di pythonanywhere karena tidak dibutuhkan
    app.run(debug=True)  # debug=True untuk mode pengembangan


```

8.	**Jalankan server**
```python
python app.py

```
<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P15/pic/15b9.png?raw=true" alt="SS" width="65%"/>

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P15/pic/15b10.png?raw=true" alt="SS" width="35%"/>

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P15/pic/15b11.png?raw=true" alt="SS" width="35%"/>

Silahkan klik link dibawah ini untuk menuju Step yang ingin dilihat:

> [!NOTE]
> Step 1 - Membuat Model [Pages Link](https://github.com/AdityaR-AI/MLC/tree/main/P15/Predict-WEB-Step-1.md)

> [!NOTE]
> Step 2 - Mengaplikasikan Model ke Web [Pages Link](https://github.com/AdityaR-AI/MLC/tree/main/P15/Predict-WEB-Step-2.md)

> [!NOTE]
> Step 3 - Deploy Hosting [Pages Link](https://github.com/AdityaR-AI/MLC/tree/main/P15/Predict-WEB-Step-3.md)

> [!NOTE]
> **Link Hosting:** [https://adityari.pythonanywhere.com/](https://adityari.pythonanywhere.com/)

Klik link dibawah ini untuk kembali ke menu utama:

> [!TIP]
> Kembali ke halaman utama, [Klik disini](https://github.com/AdityaR-AI/MLC/tree/main/)
