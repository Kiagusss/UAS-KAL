import os
import argparse
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from tqdm import tqdm

def parse_youtube_url(url_or_id):
    """
    Ekstrak Video ID dari URL YouTube atau kembalikan ID jika input sudah berupa ID.
    """
    if "youtube.com/watch?v=" in url_or_id:
        return url_or_id.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url_or_id:
        return url_or_id.split("youtu.be/")[1].split("?")[0]
    else:
        return url_or_id

def get_video_comments(api_key, video_id, max_results=None, include_replies=True):
    """
    Mengambil komentar dari video YouTube menggunakan YouTube Data API v3.
    """
    youtube = build('youtube', 'v3', developerKey=api_key)
    comments_data = []
    
    print(f"[*] Memulai pengambilan komentar untuk Video ID: {video_id}...")
    
    # Inisialisasi request pertama
    try:
        request = youtube.commentThreads().list(
            part="snippet,replies",
            videoId=video_id,
            maxResults=100, # Maksimum per request
            textFormat="plainText" # Kita ambil plain text agar tidak mengandung tag HTML
        )
        
        pbar = tqdm(desc="Mengunduh komentar", unit=" thread")
        
        while request and (max_results is None or len(comments_data) < max_results):
            response = request.execute()
            
            for item in response.get('items', []):
                # 1. Ambil top-level comment
                top_comment = item['snippet']['topLevelComment']['snippet']
                comment_id = item['snippet']['topLevelComment']['id']
                
                comments_data.append({
                    'comment_id': comment_id,
                    'parent_id': None,
                    'author': top_comment.get('authorDisplayName', 'Unknown'),
                    'text': top_comment.get('textOriginal', ''),
                    'like_count': top_comment.get('likeCount', 0),
                    'published_at': top_comment.get('publishedAt', ''),
                    'updated_at': top_comment.get('updatedAt', ''),
                    'reply_count': item['snippet'].get('totalReplyCount', 0),
                    'is_reply': False
                })
                
                # 2. Ambil replies jika diaktifkan dan tersedia
                if include_replies and 'replies' in item:
                    for reply_item in item['replies']['comments']:
                        reply = reply_item['snippet']
                        comments_data.append({
                            'comment_id': reply_item['id'],
                            'parent_id': comment_id,
                            'author': reply.get('authorDisplayName', 'Unknown'),
                            'text': reply.get('textOriginal', ''),
                            'like_count': reply.get('likeCount', 0),
                            'published_at': reply.get('publishedAt', ''),
                            'updated_at': reply.get('updatedAt', ''),
                            'reply_count': 0, # Reply tidak memiliki reply lagi di YouTube
                            'is_reply': True
                        })
                
                pbar.update(1)
                
                # Cek jika melewati batas maksimum yang diminta user
                if max_results and len(comments_data) >= max_results:
                    break
            
            # Paginasi ke halaman berikutnya
            request = youtube.commentThreads().list_next(request, response)
            
        pbar.close()
        
    except HttpError as e:
        print(f"\n[!] Terjadi kesalahan API: {e}")
        if e.resp.status == 403:
            print("[!] Silakan periksa apakah API Key Anda valid atau kuota harian Anda telah habis.")
        elif e.resp.status == 404:
            print("[!] Video tidak ditemukan. Pastikan Video ID atau URL benar.")
        return None
    except Exception as e:
        print(f"\n[!] Terjadi kesalahan: {e}")
        return None
        
    # Trim list jika melebihi batas akibat penambahan reply
    if max_results and len(comments_data) > max_results:
        comments_data = comments_data[:max_results]
        
    df = pd.DataFrame(comments_data)
    return df

def main():
    parser = argparse.ArgumentParser(description="YouTube Comment Scraper")
    parser.add_argument("--api_key", required=False, help="YouTube Data API v3 Key")
    parser.add_argument("--video", required=True, help="Video ID atau URL lengkap YouTube")
    parser.add_argument("--limit", type=int, default=None, help="Maksimum jumlah komentar yang diambil")
    parser.add_argument("--no_replies", action="store_true", help="Jangan ambil balasan komentar (replies)")
    parser.add_argument("--output", default="comments_raw.csv", help="Nama file hasil output (CSV)")
    
    args = parser.parse_args()
    
    # Mengambil API Key dari argumen atau Environment Variable
    api_key = args.api_key or os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        print("[!] ERROR: API Key tidak ditemukan. Silakan masukkan via --api_key atau set env YOUTUBE_API_KEY.")
        # Kita minta input manual jika dijalankan secara interaktif dan tidak ada argumen
        api_key = input("Masukkan YouTube API Key Anda: ").strip()
        if not api_key:
            return
            
    video_id = parse_youtube_url(args.video)
    
    df = get_video_comments(
        api_key=api_key,
        video_id=video_id,
        max_results=args.limit,
        include_replies=not args.no_replies
    )
    
    if df is not None and not df.empty:
        # Buat direktori output jika belum ada
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        df.to_csv(args.output, index=False, encoding='utf-8')
        print(f"[+] Berhasil mengunduh {len(df)} komentar.")
        print(f"[+] Hasil disimpan ke: {os.path.abspath(args.output)}")
    else:
        print("[!] Tidak ada komentar yang berhasil diambil.")

if __name__ == "__main__":
    main()
