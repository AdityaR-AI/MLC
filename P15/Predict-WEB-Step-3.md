## Aplikasi Deteksi Diabetes berbasis Web

### Deploy Hosting

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

1.	**Pergi ke [https://www.pythonanywhere.com/](https://www.pythonanywhere.com/) dan daftar**
<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P15/pic/15c1.png?raw=true" alt="SS" width="55%"/>

2.	**pergi ke menu Web Apps**
<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P15/pic/15c2.png?raw=true" alt="SS" width="25%"/>

3.	**add new web app > flask > python versi terbaru**
<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P15/pic/15c3.png?raw=true" alt="SS" width="40%"/>

4.	**Buat file project jadi zip**
<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P15/pic/15c4.png?raw=true" alt="SS" width="55%"/>

5.	**Pergi ke menu file, arahkan ke folder mysite/ , lalu upload file zip tadi**
[https://www.pythonanywhere.com/user/AdityaRI/files/home/AdityaRI/mysite](https://www.pythonanywhere.com/user/AdityaRI/files/home/AdityaRI/mysite)

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P15/pic/15c5.png?raw=true" alt="SS" width="20%"/>

6.	**Pergi ke Dashboard, klik ‘bash’ pada ‘New Console’**
<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P15/pic/15c6.png?raw=true" alt="SS" width="25%"/>
<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P15/pic/15c7.png?raw=true" alt="SS" width="25%"/>

7.	**pada terminal bash masukan kode berikut**

```python
#unzip project file
unzip ~/mysite/project.zip -d ~/mysite/
 
#buat environment
python3.10 -m venv /home/AdityaRI/mysite/env
 
#aktifkan environment
source /home/AdityaRI/mysite/env/bin/activate
 
#install requirement
pip install -r /home/AdityaRI/mysite/requirements.txt

```

8.	**setting halaman web**

Pergi ke halaman web

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P15/pic/15c8.png?raw=true" alt="SS" width="25%"/>

Dibagian code, klik `/var/www/adityari_pythonanywhere_com_wsgi.py`
<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P15/pic/15c9.png?raw=true" alt="SS" width="55%"/>

Ganti flash_app menjadi app

```python
import sys
path = '/home/AdityaRI/mysite'
if path not in sys.path:
   sys.path.append(path)
 
from app import app as application

```

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P15/pic/15c10.png?raw=true" alt="SS" width="35%"/>
<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P15/pic/15c11.png?raw=true" alt="SS" width="55%"/>

Klik reload

<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P15/pic/15c12.png?raw=true" alt="SS" width="35%"/>

9.	**Tes halaman,** [https://adityari.pythonanywhere.com/](https://adityari.pythonanywhere.com/)
<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P15/pic/15c13.png?raw=true" alt="SS" width="60%"/>
<img src="https://raw.githubusercontent.com/AdityaR-AI/MLC/main/P15/pic/15c14.png?raw=true" alt="SS" width="60%"/>   

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

