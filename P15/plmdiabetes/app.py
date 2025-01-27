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
if __name__ == "__main__":
    app.run(debug=True)  # debug=True untuk mode pengembangan