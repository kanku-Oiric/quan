import os
import requests
import trafilatura
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
OBSIDIAN_VAULT_PATH = os.getenv("OBSIDIAN_VAULT_PATH")

def fetch_news_urls_from_rss():
    """
    Contoh sederhana narik RSS feed gratisan.
    Lu bisa ganti bagian ini pakai API Inoreader kalau mau lebih advance & multi-source.
    """
    # Contoh RSS feed CNBC Internasional - Urusan Bisnis/Ekonomi
    rss_url = "https://search.cnbc.com/rs/search/all/view.rss?partnerId=2000&keywords=geopolitics"
    urls = []
    
    try:
        from xml.etree import ElementTree
        response = requests.get(rss_url, timeout=10)
        if response.status_code == 200:
            root = ElementTree.fromstring(response.content)
            for item in root.findall('.//item')[:3]:  # Ambil 3 berita terbaru aja biar gak overload
                link = item.find('link').text
                urls.append(link)
    except Exception as e:
        print(f"❌ Gagal ambil RSS: {e}")
    return urls

def convert_url_to_clean_markdown(url):
    """
    Fungsi sakti pake trafilatura buat download web, 
    bersihin iklan, dan convert langsung ke Markdown (.md)
    """
    print(f"--> Mendownload & membersihkan: {url}")
    downloaded = trafilatura.fetch_url(url)
    
    if downloaded is None:
        return None
        
    # extract() dengan output_format='markdown' bakal ngasilin teks .md yang super rapi!
    markdown_content = trafilatura.extract(
        downloaded, 
        output_format='markdown',
        include_links=True,
        include_images=False
    )
    return markdown_content

def auto_feed_to_obsidian():
    links = fetch_news_urls_from_rss()
    
    for link in links:
        content = convert_url_to_clean_markdown(link)
        if not content:
            continue
            
        # Bikin judul file berdasarkan timestamp + clean string
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        file_name = f"News-{timestamp}.md"
        full_path = os.path.join(OBSIDIAN_VAULT_PATH, file_name)
        
        # Tambahin frontmatter biar rapi di Obsidian lu
        frontmatter = f"---\ntags:\n  - auto-news\n  - intelligence-stream\nsource: {link}\ndate: {datetime.now().strftime('%Y-%m-%d')}\n---\n\n"
        
        try:
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(frontmatter + content)
            print(f"✅ Berita berhasil disimpan ke Obsidian: {file_name}")
            
            # 💡 DISINI LU BISA PANGGIL LOGIKA GEMINI LU!
            # Contoh: jalankan fungsi researcher.analyze_and_link() buat file baru ini 
            # supaya langsung dicari korelasinya dengan catatan lama di Turbovec.
            
        except Exception as e:
            print(f"❌ Gagal nulis file: {e}")

if __name__ == "__main__":
    if not OBSIDIAN_VAULT_PATH:
        print("Set dulu .env lu bro!")
    else:
        print("⚡ Memulai sinkronisasi berita otomatis...")
        auto_feed_to_obsidian()