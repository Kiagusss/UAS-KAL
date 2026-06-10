# YouTube Comment Scraper, Preprocessing, dan Benchmark SVM/LSTM/IndoBERT

Repositori ini menyediakan script Python untuk mengunduh komentar dari YouTube, melakukan preprocessing teks dalam tiga profil berbeda, dan membandingkan performa **SVM**, **LSTM**, serta **IndoBERT** pada dataset yang sudah diberi label.

---

## Daftar Isi

1. [Prasyarat & Instalasi](#1-prasyarat--instalasi)
2. [Langkah Mendapatkan YouTube API Key](#2-langkah-mendapatkan-youtube-api-key)
3. [Panduan Scraping Data (`scraper.py`)](#3-panduan-scraping-data-scraperpy)
4. [Panduan Preprocessing Data (`preprocessing.py`)](#4-panduan-preprocessing-data-preprocessingpy)
5. [Panduan Labeling Otomatis InSet](#5-panduan-labeling-otomatis-inset)
6. [Panduan Training dan Benchmark](#6-panduan-training-dan-benchmark)
7. [Prediksi Sentimen Teks Baru](#7-prediksi-sentimen-teks-baru)
8. [Mengapa Profil Preprocessing Berbeda?](#8-mengapa-profil-preprocessing-berbeda)
9. [Urutan Command dari Awal sampai Akhir](#9-urutan-command-dari-awal-sampai-akhir)

---

## 1. Prasyarat & Instalasi

Pastikan Anda telah menginstal Python (versi 3.8 ke atas disarankan) di komputer Anda.

### Langkah Instalasi:

1. Buka terminal atau Command Prompt (CMD).
2. Arahkan ke direktori project:
   ```bash
   cd c:\xampp\htdocs\KA-Lanjutan
   ```
3. Instal semua library yang dibutuhkan menggunakan file `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

_(Opsional)_ Jika Anda ingin menjalankan simulasi tokenisasi IndoBERT untuk membandingkan hasil token sebelum dan sesudah pembersihan, pastikan Anda juga menginstal PyTorch:

```bash
pip install torch
```

---

## 2. Langkah Mendapatkan YouTube API Key

Untuk mengambil komentar menggunakan API resmi YouTube, Anda memerlukan kunci API (API Key) dari Google Cloud Console:

1. Buka [Google Cloud Console](https://console.cloud.google.com/).
2. Login menggunakan akun Google Anda.
3. Buat Proyek Baru (**New Project**) dengan mengklik dropdown proyek di pojok kiri atas, lalu klik "New Project". Beri nama bebas (misal: `YouTube-Scraper`).
4. Di bilah pencarian atas, ketik **"YouTube Data API v3"** lalu klik pada hasil yang muncul.
5. Klik tombol **Enable** (Aktifkan) untuk proyek Anda.
6. Setelah diaktifkan, masuk ke menu **Credentials** (Kredensial) di bilah navigasi sebelah kiri.
7. Klik tombol **+ Create Credentials** di bagian atas, kemudian pilih **API Key**.
8. Kunci API Anda akan dibuat (berupa string panjang berisi karakter acak). Salin kunci tersebut dan simpan dengan aman.

---

## 3. Panduan Scraping Data (`scraper.py`)

Script `scraper.py` digunakan untuk mengunduh semua komentar utama beserta balasannya dari video YouTube tertentu.

### Parameter Script:

- `--api_key` : Kunci API YouTube Anda. (Bisa dikosongkan jika Anda sudah mengaturnya di Environment Variable `YOUTUBE_API_KEY`).
- `--video` : URL lengkap video YouTube atau Video ID (contoh: `https://www.youtube.com/watch?v=xxxxxx` atau cukup `xxxxxx`).
- `--limit` : Jumlah maksimum komentar yang ingin diunduh (opsional, jika kosong akan mengambil semua komentar).
- `--no_replies` : Tambahkan flag ini jika Anda **tidak** ingin menyertakan balasan (replies) komentar.
- `--output` : Nama file hasil output CSV (default: `comments_raw.csv`).

### Contoh Perintah Menjalankan:

```bash
python scraper.py --api_key "API_KEY_ANDA_DI_SINI" --video "https://www.youtube.com/watch?v=8mGf7k3p1uM" --limit 3000 --output comments_raw.csv
```

---

## 4. Panduan Preprocessing Data (`preprocessing.py`)

Setelah mendapatkan file `comments_raw.csv`, Anda bisa membuat tiga versi teks sekaligus:

- `svm` untuk model klasik yang biasanya lebih cocok dengan teks yang lebih bersih dan lowercase.
- `lstm` untuk model sekuensial berbasis embedding dan urutan kata.
- `indobert` untuk Transformer yang tetap membutuhkan konteks kalimat dan tanda baca penting.

### Parameter Script:

- `--input` : File CSV hasil scraping (default: `comments_raw.csv`).
- `--output` : File CSV hasil pembersihan (default: `comments_cleaned.csv`).
- `--profile` : Pilih `all`, `svm`, `lstm`, atau `indobert`.
- `--lowercase` : Paksa semua teks menjadi huruf kecil.
- `--remove_punct` : Paksa hapus semua tanda baca.
- `--slang_file` : File CSV tambahan berisi pemetaan slang buatan Anda (opsional).
- `--test_tokenizer` : Flag untuk mensimulasikan tokenisasi IndoBERT.

### Contoh Perintah Menjalankan:

```bash
python preprocessing.py --input comments_raw.csv --output comments_cleaned.csv --profile all --test_tokenizer
```

Hasil `--profile all` akan menambahkan kolom `cleaned_svm`, `cleaned_lstm`, dan `cleaned_indobert` pada CSV.

## 5. Panduan Labeling Otomatis InSet

Untuk memberi label sentimen secara otomatis, project ini memakai lexicon InSet dari repository [fajri91/InSet](https://github.com/fajri91/InSet). InSet menyediakan daftar kata/frasa positif dan negatif beserta bobotnya.

### Cara Pakai

```bash
python label_with_inset.py --input comments_cleaned.csv --output comments_labeled_inset.csv
```

Script ini akan:

- membaca kolom teks dari CSV
- mencocokkan kata/frasa ke lexicon InSet
- menjumlahkan skor sentimen
- memberi label `positive`, `negative`, atau `neutral`

Hasil output akan menambahkan kolom:

- `inset_score`
- `inset_positive_hits`
- `inset_negative_hits`
- `label`

Setelah itu, file hasil labeling bisa langsung dipakai untuk benchmark.

Jika Anda ingin langsung menjalankan alur lengkap tanpa manual, gunakan:

```bash
python run_inset_pipeline.py --input comments_raw.csv
```

Pipeline ini akan:

- memberi label InSet ke data mentah
- membersihkan ulang hasil labeling
- menjalankan benchmark SVM, LSTM, dan IndoBERT

## 6. Panduan Training dan Benchmark

Script per model tersedia sebagai:

- `train_svm.py`
- `train_lstm.py`
- `train_indobert.py`

Untuk membandingkan semuanya sekaligus, gunakan:

```bash
python benchmark_models.py --input comments_cleaned.csv --label_column label --output_dir model_outputs
```

Catatan penting: benchmark membutuhkan kolom `label` di CSV. Jika Anda belum punya label manual, jalankan dulu `label_with_inset.py` untuk membuat label otomatis dari lexicon InSet.

## 7. Prediksi Sentimen Teks Baru

Setelah model dilatih, Anda bisa memakai skrip prediksi untuk memberi sentimen pada teks baru atau CSV baru.

Prediksi satu teks:

```bash
python predict_sentiment.py --model_type svm --model_path model_outputs --text "film ini bagus banget"
```

Prediksi batch dari CSV:

```bash
python predict_sentiment.py --model_type indobert --model_path model_outputs --input comments_cleaned.csv --output predictions.csv
```

Jika CSV sudah berisi teks yang telah dibersihkan, tambahkan `--skip_cleaning`.

## 8. Mengapa Profil Preprocessing Berbeda?

Untuk SVM dan LSTM, preprocessing bisa lebih agresif karena model tidak memahami konteks sekuat Transformer. Untuk IndoBERT, preprocessing harus lebih hati-hati agar struktur kalimat tetap utuh.

Dalam Machine Learning tradisional (seperti Naive Bayes, SVM, atau Random Forest), kita terbiasa melakukan **Stemming** (mengubah kata ke kata dasar) dan **Stopwords Removal** (menghapus kata sambung). Namun, untuk model berbasis **Transformer seperti IndoBERT**, aturan tersebut **TIDAK BERLAKU** atau bahkan **DILARANG** karena beberapa alasan berikut:

### 1. Tanpa Stemming (No Stemming)

- **Mengapa?** BERT mempelajari relasi semantik antar kata dengan memperhatikan imbuhan (seperti _di-_, _ke-_, _me-_, _-kan_). Kata _"makanan"_ (kata benda) dan _"memakan"_ (kata kerja aktif) memiliki makna kontekstual yang sangat berbeda. Jika kita melakukan stemming menjadi _"makan"_, BERT akan kehilangan informasi tata bahasa yang krusial tersebut.
- **Solusi:** Kita membiarkan kata apa adanya. Tokenizer BERT (WordPiece) akan otomatis memotong kata menjadi sub-kata (subwords) jika kata tersebut tidak ada di kamus dasar (misal: `memakan` -> `me` + `##makan`).

### 2. Tanpa Menghapus Stopword secara Agresif (No Stopwords Removal)

- **Mengapa?** BERT adalah model kontekstual dua arah (bidirectional) yang sangat bergantung pada struktur kalimat utuh. Kata-kata seperti _"yang"_, _"di"_, _"dan"_, atau _"tidak"_ memberikan informasi struktur kalimat dan arah hubungan antar kata. Menghapus stopword akan merusak struktur kalimat alami dan menurunkan akurasi model BERT.
- **Solusi:** Stopword dibiarkan tetap ada dalam teks.

### 3. Pentingnya Normalisasi Slang & Singkatan (Slang Normalization)

- **Mengapa?** IndoBERT dilatih (pre-trained) menggunakan korpus bahasa Indonesia formal (seperti Wikipedia Indonesia dan artikel berita). Ketika dihadapkan pada teks media sosial/YouTube yang penuh singkatan (seperti _yg_, _dgn_, _bgt_, _gk_), tokenizer IndoBERT akan memecah singkatan tersebut menjadi potongan sub-kata yang tidak bermakna atau bahkan menghasilkan token `[UNK]` (Unknown).
- **Solusi:** Kita mengubah kata slang menjadi kata formal/baku sebelum masuk ke tokenizer IndoBERT (misal: _yg_ -> _yang_). Ini sangat membantu IndoBERT dalam mengenali kata dengan tepat dan memaksimalkan pemahaman kontekstualnya.

### 4. Penanganan Emoji & Tanda Baca

- **Emoji:** Sebagian besar model IndoBERT tidak dilatih dengan karakter emoji, sehingga emoji hanya akan dibaca sebagai token `[UNK]`. Oleh karena itu, emoji dihapus.
- **Tanda baca:** Tanda tanya `?`, tanda seru `!`, titik `.`, dan koma `,` dipertahankan karena BERT memanfaatkan tanda-tanda ini untuk mendeteksi batas kalimat (sentence boundaries) dan intonasi teks.

## 9. Urutan Command dari Awal sampai Akhir

Berikut urutan command yang bisa dijalankan di terminal dari awal sampai akhir.

### Opsi 1: Alur manual lengkap

1. Masuk ke folder project.

```bash
cd c:\xampp\htdocs\KA-Lanjutan
```

2. Instal dependensi.

```bash
pip install -r requirements.txt
```

3. Ambil data komentar YouTube.

```bash
python scraper.py --api_key "API_KEY_ANDA_DI_SINI" --video "https://www.youtube.com/watch?v=8mGf7k3p1uM" --limit 3000 --output comments_raw.csv
```

4. Lakukan preprocessing awal.

```bash
python preprocessing.py --input comments_raw.csv --output comments_cleaned.csv --profile all
```

5. Beri label sentimen otomatis dengan InSet.

```bash
python label_with_inset.py --input comments_cleaned.csv --output comments_labeled_inset.csv
```

6. Preprocessing ulang hasil labeling.

```bash
python preprocessing.py --input comments_labeled_inset.csv --output comments_labeled_cleaned.csv --profile all
```

7. Jalankan benchmark SVM, LSTM, dan IndoBERT.

```bash
python benchmark_models.py --input comments_labeled_cleaned.csv --label_column label --output_dir model_outputs
```

8. Prediksi sentimen untuk teks baru.

```bash
python predict_sentiment.py --model_type svm --model_path model_outputs --text "film ini bagus banget"
```

### Opsi 2: Jalankan pipeline lengkap sekaligus

Jika ingin langsung dari data mentah sampai benchmark, gunakan:

```bash
python run_inset_pipeline.py --input comments_raw.csv
```

Pipeline ini akan:

- memberi label InSet ke data mentah
- melakukan cleaning ulang
- menjalankan benchmark SVM, LSTM, dan IndoBERT

Jika Anda hanya ingin hasil prediksi dari model yang sudah dilatih, langsung pakai `predict_sentiment.py`.
