import os
import discord
import numpy as np
import warnings
from discord.ext import commands
from sentence_transformers import SentenceTransformer
from turbovec import TurboQuantIndex
from researcher import AIResearcher
from dotenv import load_dotenv  # Library buat load file .env
from datetime import datetime, timedelta, timezone

# Sembunyikan warning bawaan Hugging Face
warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
load_dotenv()

DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OBSIDIAN_VAULT_PATH = os.getenv("OBSIDIAN_VAULT_PATH")

# Validasi awal biar bot gak running dalam keadaan buta/kosong
if not DISCORD_BOT_TOKEN or not GEMINI_API_KEY or not OBSIDIAN_VAULT_PATH:
    print("❌ ERROR: Variabel di file .env belum diisi lengkap! Periksa file .env lu.")
    exit()
# =====================================================================
# CONFIG TAMBAHAN: AMBIL ID CHANNEL KHUSUS DARI .ENV
# =====================================================================
# Masukkan ID Channel khusus PDF lu di file .env dengan nama CHANNEL_PDF_ID
TARGET_CHANNEL_ID = os.getenv("CHANNEL_PDF_ID") 

if TARGET_CHANNEL_ID:
    TARGET_CHANNEL_ID = int(TARGET_CHANNEL_ID)
else:
    print("⚠️ WARNING: CHANNEL_PDF_ID belum di-set di .env. Fitur auto-download belum aktif.")
# =====================================================================
# INITIALIZATION (LOKAL MEMORI & CLOUD BRAIN)
# =====================================================================
print("--> [Bot] Menginisialisasi Model Embedding & Turbovec Lokal...")
model = SentenceTransformer('all-MiniLM-L6-v2')
index = TurboQuantIndex(dim=384, bit_width=4)

print("--> [Bot] Menginisialisasi AI Researcher (Gemini API)...")
researcher = AIResearcher(api_key=GEMINI_API_KEY, obsidian_path=OBSIDIAN_VAULT_PATH)

# List untuk simpan metadata catatan di RAM secara lokal
metadata_lokal = []

# Setup Discord Bot Intents
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# =====================================================================
# FUNGSI RE-INDEX LOKAL (BIAR TURBOVEC SELALU UPDATE)
# =====================================================================
def refresh_local_vector_index():
    """Membaca folder Obsidian dan memasukkannya ke Turbovec lokal (0 Token Cloud)"""
    global metadata_lokal, index
    metadata_lokal = []
    all_vectors = []
    
    # Reset index biar gak duplikat saat dipanggil ulang
    index = TurboQuantIndex(dim=384, bit_width=4)
    
    if not os.path.exists(OBSIDIAN_VAULT_PATH):
        print(f"❌ Error: Folder Obsidian tidak ditemukan di jalur: {OBSIDIAN_VAULT_PATH}")
        return False
        
    for root, dirs, files in os.walk(OBSIDIAN_VAULT_PATH):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if not content.strip():
                            continue
                        
                        # Generate embedding lokal pakai CPU laptop lu (Gratis)
                        vector = model.encode(content)
                        all_vectors.append(vector)
                        metadata_lokal.append({"title": file, "content": content})
                except Exception as e:
                    print(f"Error baca {file}: {e}")
                    
    if all_vectors:
        all_vectors_numpy = np.array(all_vectors).astype(np.float32)
        index.add(all_vectors_numpy)
        print(f"✅ [Turbovec] Sukses index {len(all_vectors)} file secara lokal.")
        return True
    return False

@bot.event
async def on_ready():
    print("--> [Bot] Menyinkronkan catatan Obsidian ke Turbovec...")
    refresh_local_vector_index()
    print(f"✅ Bot Discord siap digunakan! Login sebagai: {bot.user}")

# =====================================================================
# PERINTAH UTAMA: !brain
# =====================================================================
@bot.command(name="brain")
@commands.cooldown(1, 30, commands.BucketType.user) 
async def process_to_brain(ctx, *, argumen_chat: str = ""):
    """Perintah untuk memproses chat/PDF menjadi Knowledge Graph di Obsidian"""
    
    async with ctx.typing():
        pdf_local_path = None
        
        # 1. Cek Lampiran PDF dari Discord
        if ctx.message.attachments:
            attachment = ctx.message.attachments[0]
            if attachment.filename.endswith(".pdf"):
                pdf_local_path = os.path.join(os.getcwd(), attachment.filename)
                await attachment.save(pdf_local_path)
        
        if not argumen_chat and not pdf_local_path:
            await ctx.send("❌ Kasih teks obrolan atau lampirin PDF dong setelah ketik `!brain`!")
            return

        # ===================================================
        # INI BAGIAN YANG ILANG DARI KODE LU BRO wkwkwk
        # ===================================================
        query_text = argumen_chat if argumen_chat else f"Analisis dokumen: {attachment.filename}"
        
        # 2. PROSES FILTERING LOKAL PAKAI TURBOVEC
        file_relevan_nama = "Tidak ada"
        file_relevan_konten = "Tidak ada konteks lama yang mirip."
        
        query_vector = model.encode(query_text)
        query_vector_fixed = np.array([query_vector]).astype(np.float32)
        
        if len(metadata_lokal) > 0:
            scores, indices = index.search(query_vector_fixed, k=1)
            if indices.size > 0 and len(indices[0]) > 0:
                best_match_idx = indices[0][0]
                if best_match_idx < len(metadata_lokal):
                    file_relevan_nama = metadata_lokal[best_match_idx]['title']
                    file_relevan_konten = metadata_lokal[best_match_idx]['content']

        daftar_semua_judul = [m['title'] for m in metadata_lokal]

        await ctx.send("🧠 *Turbovec berhasil menyaring memori. Menghubungi Gemini cloud...*")
        
        # 3. LEMPAR DATA KE GEMINI API (Dengan Anti-Crash)
        hasil = None
        try:
            hasil = researcher.analyze_and_link(
                discord_chat=query_text,
                pdf_path=pdf_local_path,
                file_relevan_nama=file_relevan_nama,
                file_relevan_konten=file_relevan_konten,
                daftar_semua_judul=daftar_semua_judul
            )
        except Exception as e:
            await ctx.send(f"⚠️ *Waduh, server Gemini lagi ngambek (Overload/Error):* `{e}`")
        
        # Hapus file PDF sementara dari lokal disk
        if pdf_local_path and os.path.exists(pdf_local_path):
            os.remove(pdf_local_path)

        # 4. SIMPAN HASIL KE OBSIDIAN
        if hasil:
            path_tersimpan = researcher.save_to_obsidian(hasil['judul'], hasil['konten'])
            if path_tersimpan:
                refresh_local_vector_index()
                
                embed = discord.Embed(
                    title="🎯 Knowledge Graph Berhasil Diperbarui!",
                    color=discord.Color.green()
                )
                embed.add_field(name="📄 Judul File Baru", value=f"`{hasil['judul']}.md`", inline=False)
                embed.add_field(name="🔗 Direferensikan Dari", value=f"`{file_relevan_nama}`", inline=True)
                await ctx.send(embed=embed)
            else:
                await ctx.send("❌ Gagal menulis catatan ke folder Obsidian.")

# =====================================================================
# PERINTAH TAMBAHAN: !sync_time (Membaca PDF Berdasarkan Rentang Hari)
# =====================================================================
@bot.command(name="sync_time")
@commands.has_permissions(administrator=True)
async def sync_channel_by_time(ctx, jumlah_hari: int = 7):
    """
    Cara pakai di Discord: !sync_time 7
    (Bot bakal otomatis nyari semua PDF yang diupload dalam 7 hari terakhir)
    """
    if TARGET_CHANNEL_ID and ctx.channel.id != TARGET_CHANNEL_ID:
        await ctx.send("❌ Perintah ini cuma bisa dijalankan di channel khusus PDF lu!")
        return

    # Hitung waktu batas mundur (pake timezone UTC/WIB biar sinkron sama Discord)
    waktu_sekarang = datetime.now(timezone.utc)
    waktu_batas_awal = waktu_sekarang - timedelta(days=jumlah_hari)
    
    waktu_lokal_print = waktu_batas_awal.astimezone() # Buat tampilan di chat biar gampang dibaca
    await ctx.send(f"⏳ *Auto-System: Menyapu sejarah sejak tanggal [{waktu_lokal_print.strftime('%Y-%m-%d %H:%M')}] ({jumlah_hari} hari terakhir)...*")
    
    count_pdf_diproses = 0

    # Panggil history dengan parameter 'after' (hanya ambil pesan setelah waktu_batas_awal)
    async_history = ctx.channel.history(after=waktu_batas_awal, oldest_first=True)

    async for message in async_history:
        if message.author == bot.user:
            continue

        if message.attachments:
            attachment = message.attachments[0]
            if attachment.filename.endswith(".pdf"):
                print(f"--> [Time-Sync] Menemukan PDF dalam rentang waktu: {attachment.filename}")
                count_pdf_diproses += 1
                
                await ctx.send(f"📥 *Processing [{count_pdf_diproses}]: `{attachment.filename}`...*")
                
                # Download file sementara
                pdf_local_path = os.path.join(os.getcwd(), attachment.filename)
                await attachment.save(pdf_local_path)
                
                query_text = message.content if message.content else f"Analisis dokumen rentang waktu: {attachment.filename}"
                
                # --- FILTERING LOKAL PAKAI TURBOVEC ---
                file_relevan_nama = "Tidak ada"
                file_relevan_konten = "Tidak ada konteks lama yang mirip."
                
                query_vector = model.encode(query_text)
                query_vector_fixed = np.array([query_vector]).astype(np.float32)
                
                if len(metadata_lokal) > 0:
                    scores, indices = index.search(query_vector_fixed, k=1)
                    if indices.size > 0 and len(indices[0]) > 0:
                        best_match_idx = indices[0][0]
                        if best_match_idx < len(metadata_lokal):
                            file_relevan_nama = metadata_lokal[best_match_idx]['title']
                            file_relevan_konten = metadata_lokal[best_match_idx]['content']
                
                daftar_semua_judul = [m['title'] for m in metadata_lokal]
                # --------------------------------------
                
                # Kirim ke Gemini
                hasil = researcher.analyze_and_link(
                    discord_chat=query_text,
                    pdf_path=pdf_local_path,
                    file_relevan_nama=file_relevan_nama,
                    file_relevan_konten=file_relevan_konten,
                    daftar_semua_judul=daftar_semua_judul
                )
                
                if os.path.exists(pdf_local_path):
                    os.remove(pdf_local_path)
                
                if hasil:
                    path_tersimpan = researcher.save_to_obsidian(hasil['judul'], hasil['konten'])
                    if path_tersimpan:
                        refresh_local_vector_index()
                
                # Jeda aman anti-rate limit
                import asyncio
                await asyncio.sleep(3)

    await ctx.send(f"✨ **SINKRONISASI BERDASARKAN WAKTU SELESAI!** Sukses memasukkan `{count_pdf_diproses}` file PDF dari {jumlah_hari} hari terakhir.")
# =====================================================================
# EVENT LISTENER: AUTO PILOT (Fungsi Baru untuk Channel Khusus)
# =====================================================================
@bot.event
async def on_message(message):
    # 1. Cegah bot memproses pesannya sendiri biar gak looping infinity
    if message.author == bot.user:
        return

    # 2. Cek apakah pesan ini masuk ke channel khusus PDF yang sudah ditentukan
    if TARGET_CHANNEL_ID and message.channel.id == TARGET_CHANNEL_ID:
        
        # 3. Cek apakah ada file yang dilampirkan
        if message.attachments:
            attachment = message.attachments[0]
            
            # 4. Filter strictly hanya untuk file PDF
            if attachment.filename.endswith(".pdf"):
                print(f"--> [Auto-Pilot] Nemu PDF masuk di channel target: {attachment.filename}")
                
                # Kirim indikator typing di Discord biar user tahu bot lagi kerja
                async with message.channel.typing():
                    await message.channel.send(f"📥 *Auto-System: Mendeteksi `{attachment.filename}`. Memulai ekstraksi dan indexing...*")
                    
                    # Simpan PDF secara lokal sementara waktu
                    pdf_local_path = os.path.join(os.getcwd(), attachment.filename)
                    await attachment.save(pdf_local_path)
                    
                    # BARIS 204 HARUSNYA DI SINI (Menjorok ke dalam memakai variabel 'message.content')
                    query_text = message.content if message.content else f"Analisis dokumen ilmiah: {attachment.filename}"
                    
                    # --- PIPELINE DIET TOKEN TURBOVEC ---
                    file_relevan_nama = "Tidak ada"
                    file_relevan_konten = "Tidak ada konteks lama yang mirip."
                    
                    query_vector = model.encode(query_text)
                    query_vector_fixed = np.array([query_vector]).astype(np.float32)
                    
                    if len(metadata_lokal) > 0:
                        scores, indices = index.search(query_vector_fixed, k=1)
                        if indices.size > 0 and len(indices[0]) > 0:
                            best_match_idx = indices[0][0]
                            if best_match_idx < len(metadata_lokal):
                                file_relevan_nama = metadata_lokal[best_match_idx]['title']
                                file_relevan_konten = metadata_lokal[best_match_idx]['content']
                    
                    daftar_semua_judul = [m['title'] for m in metadata_lokal]
                    # --------------------------------------
                    
                    # Kirim ke Gemini API via Researcher
                    hasil = researcher.analyze_and_link(
                        discord_chat=query_text,
                        pdf_path=pdf_local_path,
                        file_relevan_nama=file_relevan_nama,
                        file_relevan_konten=file_relevan_konten,
                        daftar_semua_judul=daftar_semua_judul
                    )
                    
                    # Hapus PDF lokal sementara setelah selesai di-ekstrak
                    if os.path.exists(pdf_local_path):
                        os.remove(pdf_local_path)
                        
                    # Simpan ke harddisk Obsidian jika Gemini sukses memproses
                    if hasil:
                        path_tersimpan = researcher.save_to_obsidian(hasil['judul'], hasil['konten'])
                        if path_tersimpan:
                            refresh_local_vector_index() # Sinkronisasi ulang memori lokal
                            
                            embed = discord.Embed(
                                title="📥 Auto-Index Sukses!",
                                description=f"File PDF berhasil diproses otomatis tanpa command.",
                                color=discord.Color.blue()
                            )
                            embed.add_field(name="📄 Hasil Catatan Baru", value=f"`{hasil['judul']}.md`", inline=False)
                            embed.add_field(name="🔗 Kaitan Memori", value=f"`{file_relevan_nama}`", inline=True)
                            await message.channel.send(embed=embed)
                        else:
                            await message.channel.send("❌ Auto-System gagal menulis ke Obsidian.")
                    else:
                        await message.channel.send("❌ Auto-System gagal memproses via Gemini API.")
                        
    # CRITICAL: Baris di bawah ini WAJIB sejajar dengan if pertama di dalam on_message
    await bot.process_commands(message)

# Jalankan bot Discord lu (taruh paling bawah file)
bot.run(DISCORD_BOT_TOKEN)