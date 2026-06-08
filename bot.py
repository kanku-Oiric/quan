import os
import asyncio
import discord
import numpy as np
import warnings
from discord.ext import commands
from sentence_transformers import SentenceTransformer
from turbovec import TurboQuantIndex
from researcher import AIResearcher
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
from feeder import auto_feed_to_obsidian

warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
load_dotenv()

DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OBSIDIAN_VAULT_PATH = os.getenv("OBSIDIAN_VAULT_PATH")

if not DISCORD_BOT_TOKEN or not GEMINI_API_KEY or not OBSIDIAN_VAULT_PATH:
    print("❌ ERROR: Variabel di file .env belum diisi lengkap! Periksa file .env lu.")
    exit()

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

metadata_lokal = []

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# =====================================================================
# FUNGSI RE-INDEX LOKAL (synchronous — selalu panggil via run_in_executor)
# =====================================================================
def refresh_local_vector_index():
    global metadata_lokal, index
    metadata_lokal = []
    all_vectors = []
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

# =====================================================================
# HELPER — CARI KONTEKS RELEVAN
# =====================================================================
def cari_konteks_relevan(query_text: str) -> tuple[str, str, list]:
    file_relevan_nama = "Tidak ada"
    file_relevan_konten = "Tidak ada konteks lama yang mirip."

    if len(metadata_lokal) > 0:
        query_vector = model.encode(query_text)
        query_vector_fixed = np.array([query_vector]).astype(np.float32)

        scores, indices = index.search(query_vector_fixed, k=1)
        if indices.size > 0 and len(indices[0]) > 0:
            best_match_idx = indices[0][0]
            if best_match_idx < len(metadata_lokal):
                file_relevan_nama = metadata_lokal[best_match_idx]['title']
                file_relevan_konten = metadata_lokal[best_match_idx]['content']

    daftar_semua_judul = [m['title'] for m in metadata_lokal]
    return file_relevan_nama, file_relevan_konten, daftar_semua_judul

# =====================================================================
# HELPER — PIPELINE PENUH (TURBOVEC → GEMINI → OBSIDIAN)
# =====================================================================
async def proses_dan_simpan(query_text: str, pdf_local_path: str = None) -> tuple[dict | None, str]:
    loop = asyncio.get_running_loop()

    # Step 1: Cari konteks lokal (sinkronus tapi cepat)
    file_relevan_nama, file_relevan_konten, daftar_semua_judul = cari_konteks_relevan(query_text)

    # Step 2: Kirim ke Gemini
    # ─────────────────────────────────────────────────────────────────
    # BUG ASLI ADA DI SINI. analyze_and_link() pakai httpx synchronous
    # client di balik layar (via google-genai SDK). Tanpa run_in_executor,
    # ini block seluruh event loop asyncio selama 10–60 detik sambil nunggu
    # response Gemini → Discord heartbeat mati → WARNING "heartbeat blocked".
    #
    # Fix: wrap di run_in_executor sama persis seperti save + reindex di bawah.
    # ─────────────────────────────────────────────────────────────────
    hasil = None
    try:
        hasil = await loop.run_in_executor(
            None,
            lambda: researcher.analyze_and_link(
                discord_chat=query_text,
                pdf_path=pdf_local_path,
                file_relevan_nama=file_relevan_nama,
                file_relevan_konten=file_relevan_konten,
                daftar_semua_judul=daftar_semua_judul
            )
        )
    except Exception as e:
        print(f"❌ [Gemini] Error: {e}")
        return None, file_relevan_nama

    # Step 3 & 4: Simpan + re-index di thread terpisah
    if hasil:
        path_tersimpan = await loop.run_in_executor(
            None,
            lambda: researcher.save_to_obsidian(hasil['judul'], hasil['konten'])
        )
        if path_tersimpan:
            await loop.run_in_executor(None, refresh_local_vector_index)
        else:
            return None, file_relevan_nama

    return hasil, file_relevan_nama
async def news_feeder_loop():
    await bot.wait_until_ready()
    loop = asyncio.get_running_loop()
    
    while not bot.is_closed():
        print("⚡ [Auto-Pilot Feeder] Memulai background scraping berita...")
        try:
            # 💡 Jalankan scraping web di thread terpisah biar gak blocking
            await loop.run_in_executor(None, auto_feed_to_obsidian) 
            
            # 💡 Jalankan re-indexing lokal di thread terpisah juga
            print("🧠 [Auto-Pilot Feeder] Sinkronisasi memori baru ke Turbovec...")
            await loop.run_in_executor(None, refresh_local_vector_index)
            
        except Exception as e:
            print(f"❌ Feeder error: {e}")
            
        # Jeda waktu looping (6 jam)
        await asyncio.sleep(21600)
# =====================================================================
# ON READY
# =====================================================================
@bot.event
async def on_ready():
    print("--> [Bot] Menyinkronkan catatan Obsidian ke Turbovec...")
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, refresh_local_vector_index)
    print(f"✅ Bot Discord siap digunakan! Login sebagai: {bot.user}")

# =====================================================================
# ON READY & BACKGROUND TASK TRIGGER
# =====================================================================
@bot.event
async def on_ready():
    print("--> [Bot] Menyinkronkan catatan Obsidian ke Turbovec...")
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, refresh_local_vector_index)
    print(f"✅ Bot Discord siap digunakan! Login sebagai: {bot.user}")
    
    # 🔥 FIX: Daftarkan background task feeder di sini pake loop aktif
    print("📅 [System] Mengaktifkan Auto-Pilot Feeder loop...")
    loop.create_task(news_feeder_loop())
# =====================================================================
# PERINTAH UTAMA: !brain
# =====================================================================
@bot.command(name="brain")
@commands.cooldown(1, 30, commands.BucketType.user)
async def process_to_brain(ctx, *, argumen_chat: str = ""):
    async with ctx.typing():
        pdf_local_path = None
        attachment = None

        if ctx.message.attachments:
            attachment = ctx.message.attachments[0]
            if attachment.filename.endswith(".pdf"):
                pdf_local_path = os.path.join(os.getcwd(), attachment.filename)
                await attachment.save(pdf_local_path)

        if not argumen_chat and not pdf_local_path:
            await ctx.send("❌ Kasih teks obrolan atau lampirin PDF dong setelah ketik `!brain`!")
            return

        query_text = argumen_chat if argumen_chat else f"Analisis dokumen: {attachment.filename}"

        await ctx.send("🧠 *Turbovec berhasil menyaring memori. Menghubungi Gemini cloud...*")

        hasil, file_relevan_nama = await proses_dan_simpan(query_text, pdf_local_path)

        if pdf_local_path and os.path.exists(pdf_local_path):
            os.remove(pdf_local_path)

        if hasil:
            embed = discord.Embed(
                title="🎯 Knowledge Graph Berhasil Diperbarui!",
                color=discord.Color.green()
            )
            embed.add_field(name="📄 Judul File Baru", value=f"`{hasil['judul']}.md`", inline=False)
            embed.add_field(name="🔗 Direferensikan Dari", value=f"`{file_relevan_nama}`", inline=True)
            await ctx.send(embed=embed)
        else:
            await ctx.send("⚠️ *Waduh, Gemini gagal atau error. Coba lagi ya!*")

# =====================================================================
# PERINTAH TAMBAHAN: !sync_time
# =====================================================================
@bot.command(name="sync_time")
@commands.has_permissions(administrator=True)
async def sync_channel_by_time(ctx, jumlah_hari: int = 7):
    if TARGET_CHANNEL_ID and ctx.channel.id != TARGET_CHANNEL_ID:
        await ctx.send("❌ Perintah ini cuma bisa dijalankan di channel khusus PDF lu!")
        return

    waktu_sekarang = datetime.now(timezone.utc)
    waktu_batas_awal = waktu_sekarang - timedelta(days=jumlah_hari)
    waktu_lokal_print = waktu_batas_awal.astimezone()
    await ctx.send(
        f"⏳ *Auto-System: Menyapu sejarah sejak [{waktu_lokal_print.strftime('%Y-%m-%d %H:%M')}] "
        f"({jumlah_hari} hari terakhir)...*"
    )

    count_pdf_diproses = 0
    async_history = ctx.channel.history(after=waktu_batas_awal, oldest_first=True)

    async for message in async_history:
        if message.author == bot.user:
            continue

        if message.attachments:
            attachment = message.attachments[0]
            if attachment.filename.endswith(".pdf"):
                print(f"--> [Time-Sync] Menemukan PDF: {attachment.filename}")
                count_pdf_diproses += 1

                await ctx.send(f"📥 *Processing [{count_pdf_diproses}]: `{attachment.filename}`...*")

                pdf_local_path = os.path.join(os.getcwd(), attachment.filename)
                await attachment.save(pdf_local_path)

                query_text = (
                    message.content if message.content
                    else f"Analisis dokumen rentang waktu: {attachment.filename}"
                )

                hasil, _ = await proses_dan_simpan(query_text, pdf_local_path)

                if os.path.exists(pdf_local_path):
                    os.remove(pdf_local_path)

                if not hasil:
                    await ctx.send(f"⚠️ Gagal memproses `{attachment.filename}`. Lanjut ke file berikutnya...")

                await asyncio.sleep(3)

    await ctx.send(
        f"✨ **SINKRONISASI SELESAI!** "
        f"Sukses memasukkan `{count_pdf_diproses}` file PDF dari {jumlah_hari} hari terakhir."
    )

# =====================================================================
# EVENT LISTENER: AUTO PILOT
# =====================================================================
@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    if TARGET_CHANNEL_ID and message.channel.id == TARGET_CHANNEL_ID:
        if message.attachments:
            attachment = message.attachments[0]
            if attachment.filename.endswith(".pdf"):
                print(f"--> [Auto-Pilot] Nemu PDF masuk: {attachment.filename}")

                async with message.channel.typing():
                    await message.channel.send(
                        f"📥 *Auto-System: Mendeteksi `{attachment.filename}`. "
                        f"Memulai ekstraksi dan indexing...*"
                    )

                    pdf_local_path = os.path.join(os.getcwd(), attachment.filename)
                    await attachment.save(pdf_local_path)

                    query_text = (
                        message.content if message.content
                        else f"Analisis dokumen ilmiah: {attachment.filename}"
                    )

                    hasil, file_relevan_nama = await proses_dan_simpan(query_text, pdf_local_path)

                    if os.path.exists(pdf_local_path):
                        os.remove(pdf_local_path)

                    if hasil:
                        embed = discord.Embed(
                            title="📥 Auto-Index Sukses!",
                            description="File PDF berhasil diproses otomatis tanpa command.",
                            color=discord.Color.blue()
                        )
                        embed.add_field(name="📄 Hasil Catatan Baru", value=f"`{hasil['judul']}.md`", inline=False)
                        embed.add_field(name="🔗 Kaitan Memori", value=f"`{file_relevan_nama}`", inline=True)
                        await message.channel.send(embed=embed)
                    else:
                        await message.channel.send("❌ Auto-System gagal memproses via Gemini API.")

    await bot.process_commands(message)

bot.run(DISCORD_BOT_TOKEN)