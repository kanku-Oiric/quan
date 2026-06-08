import os
import re                               # FIX A: pindah ke atas, bukan di dalam fungsi
import json                             # FIX B: pindah ke atas, bukan di dalam fungsi
import warnings
from google import genai
from pypdf import PdfReader
from pydantic import BaseModel          # FIX C: pindah ke atas, bukan di dalam fungsi

warnings.filterwarnings("ignore")

# =====================================================================
# KONSTANTA — FIX D: gak ada lagi magic number tersebar di fungsi
# =====================================================================
GEMINI_MODEL      = "gemini-2.5-flash"
PDF_CHAR_LIMIT    = 8000               # FIX 5: dinaikkan dari 4000 → 8000, mudah diubah di sini
FILENAME_MAX_LEN  = 100

# =====================================================================
# SCHEMA PYDANTIC — FIX C: definisi di level modul, bukan di-recreate
#                           setiap kali analyze_and_link() dipanggil
# =====================================================================
class ObsidianNoteSchema(BaseModel):
    judul: str
    konten: str

class AIResearcher:
    def __init__(self, api_key: str, obsidian_path: str):
        self.client = genai.Client(api_key=api_key)
        self.obsidian_path = obsidian_path

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Ekstrak teks dari file PDF secara lokal, dibatasi PDF_CHAR_LIMIT karakter."""
        if not os.path.exists(pdf_path):
            return ""
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
                # FIX 5: stop lebih awal kalau sudah melewati limit,
                # daripada ekstrak semua baru dipotong di akhir
                if len(text) >= PDF_CHAR_LIMIT:
                    break
            return text[:PDF_CHAR_LIMIT]
        except Exception as e:
            print(f"Error ekstrak PDF: {e}")
            return ""

    def analyze_and_link(
        self,
        discord_chat: str,
        pdf_path: str = None,
        file_relevan_nama: str = "",
        file_relevan_konten: str = "",
        daftar_semua_judul: list = None
    ) -> dict | None:
        """
        Pipeline utama: bangun prompt → kirim ke Gemini → parse JSON.
        Return dict {judul, konten} atau None jika gagal.
        """
        format_judul_saja = (
            "\n".join([f"- {j}" for j in daftar_semua_judul])
            if daftar_semua_judul else "Belum ada file."
        )

        pdf_context = ""
        if pdf_path:
            print(f"--> [Researcher] Mengekstrak teks dari PDF...")
            pdf_context = self.extract_text_from_pdf(pdf_path)

        prompt = f"""
Lu adalah AI Knowledge Architect dan Strategic Risk Analyst untuk Obsidian Second Brain.
Fokus: mental models, probabilistic thinking, dan sistem dinamik.
Tugas lu: SINTESIS dan EKSEKUSI — bukan cari, bukan jelaskan dari awal.

ANALOGI WAJIB PEGANG: Kalau inputnya "belajar matematika pertambahan",
output lu bukan "sejarah notasi matematika". Output lu adalah:
"langkah 1: tulis angkanya, langkah 2: tambahkan, langkah 3: cek hasilnya".
Fundamental tetap ada tapi minimal — porsi TERBESAR adalah cara cepat pakai/eksekusi.

=========================================
[DATA SOURCE — DISIAPKAN TURBOVEC, JANGAN CARI SENDIRI]

WHITELIST LINK (HANYA ini yang boleh jadi [[Link]], tidak ada pengecualian):
{format_judul_saja}

KONTEKS RELEVAN DARI VAULT:
Judul: {file_relevan_nama}
Isi:
\"\"\"{file_relevan_konten}\"\"\"

INPUT BARU:
\"\"\"{discord_chat}
{pdf_context}\"\"\"
=========================================
[GUARDRAILS — HARD CONSTRAINTS]
1. ZERO HALLUCINATION LINKING: Tulis [[Nama]] HANYA jika nama itu ada persis di WHITELIST di atas.
   Jika tidak ada → tulis nama biasa tanpa [[ ]]. Lebih baik tidak ada link daripada link palsu.
2. NO EXTRA SEARCHING: Jangan karang fakta di luar konteks Turbovec + Input Baru.
   Kalau tidak ada datanya → tulis "data tidak tersedia di vault."
3. ZERO FLUFF: Langsung ke inti. Tidak ada kalimat pembuka, tidak ada "Tentu saja!", tidak ada basa-basi.
4. IMPLEMENTATION ANCHOR: Setiap klaim harus bisa langsung dieksekusi atau diverifikasi.
   Hindari statement seperti "hal ini penting" tanpa menjelaskan BAGAIMANA melakukannya.

=========================================
[OUTPUT BLUEPRINT — IKUTI PERSIS STRUKTUR INI]

---
tags:
  - strategic-synthesis
  - operational-model
---

# 🧠 [Judul Maks 6 Kata — Fokus pada AKSInya]

> [!abstract] Executive Summary
> Kalimat 1: Apa mekanisme intinya (bukan definisi — mekanisme).
> Kalimat 2: Satu langkah konkret untuk langsung mulai menggunakannya.

## 🛠️ 1. Mechanism & Fast Route
**Cara kerja intinya (1-3 kalimat maks):**
[Jelaskan mekanisme — bukan sejarah, bukan teori lengkap]

**Jalur tercepat untuk pakai/eksekusi:**
- Langkah 1: [aksi konkret]
- Langkah 2: [aksi konkret]  
- Langkah 3: [aksi konkret]
[Maks 5 langkah. Kalau lebih, potong yang tidak esensial.]

## 🎯 2. Implementation Model
**Situasi kapan ini dipakai:**
[1 kalimat — kondisi spesifik trigger penggunaan]

**Output yang dihasilkan:**
[Bentuk konkret hasilnya — file, keputusan, dokumen, tindakan]

**Indikator berhasil (done ≠ perfect):**
- [ ] [kriteria verifikasi 1]
- [ ] [kriteria verifikasi 2]

**Timeline realistis:**
[Estimasi waktu untuk satu siklus eksekusi penuh]

## ⚡ 3. Risks & Mitigation
**Risiko Utama:** [Blind-spot atau titik kegagalan terbesar — spesifik ke dokumen/input ini]

**Mitigasi A (Taktis — bisa dilakukan sekarang):**
[Aksi konkret untuk reduce risiko dalam 1 sesi kerja]

**Mitigasi B (Sistemik — kalau A tidak cukup):**
[Perubahan proses atau struktur jangka panjang]

## 🔗 4. Network Synthesis
**Koneksi ke vault:** [[{file_relevan_nama}]]
[1-2 kalimat — jelaskan keterkaitan OPERASIONAL, bukan hanya tematik.
Contoh yang benar: "Konsep X di note ini menjadi input langsung untuk langkah 2 di [[file_relevan_nama]]"
Contoh yang salah: "Note ini berkaitan dengan [[file_relevan_nama]] karena membahas topik serupa"]

**Link tambahan yang relevan (hanya dari WHITELIST):**
[Jika ada — sebutkan nama dan kenapa spesifik relevan]
"""

        print("--> [Researcher] Mengirim prompt ke Gemini API...")
        response = self.client.models.generate_content(
            model=GEMINI_MODEL,                         # FIX D: pakai konstanta
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": ObsidianNoteSchema  # FIX C: schema dari level modul
            }
        )

        try:
            # FIX B: json sudah di-import di atas — tidak perlu import ulang di sini
            clean_text = response.text.strip()
            result = json.loads(clean_text)
            return result
        except Exception as e:
            print(f"❌ Error parsing JSON: {e}")
            print(f"Raw Response yang bikin error:\n{response.text}")
            return None

    def save_to_obsidian(self, judul: str, konten: str) -> str:
        """Simpan catatan ke Obsidian vault. Return path file jika sukses, string kosong jika gagal."""
        # FIX A: re sudah di-import di atas — tidak perlu import ulang di sini
        judul_bersih = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '-', judul)
        judul_bersih = judul_bersih.strip('. ')
        judul_bersih = judul_bersih[:FILENAME_MAX_LEN]  # FIX D: pakai konstanta

        filename = f"{judul_bersih}.md"
        full_path = os.path.join(self.obsidian_path, filename)

        print(f"--> [Obsidian] Mencoba nulis ke: {full_path}")   # FIX E: hapus komentar debug lama

        try:
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(konten)
            print(f"✅ [Obsidian] File berhasil ditulis: {filename}")
            return full_path
        except Exception as e:
            print(f"Gagal menyimpan ke Obsidian: {e}")
            return ""