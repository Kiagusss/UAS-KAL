import sys
from preprocessing import clean_text_for_bert, DEFAULT_SLANG_DICT

def run_test():
    # Mengatur output encoding terminal ke UTF-8 agar aman mencetak emoji pada Windows
    if sys.stdout.encoding != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except AttributeError:
            pass
            
    test_comments = [
        "Wahhh keren bgt videonya!!! 😍 Ditunggu kelanjutannya ya kk... @admin #mantap",
        "yg bener aja?? gk bs dipahami dgn mudah sih, tp ok lah bwt pemula. link video: https://youtube.com/watch?v=12345",
        "sy udh nonton smpe hbs, trs udh sy share jg ke klrga & tmn2 bnyak bgt informasinya. thx ya &lt;3",
        "Keren abisss!!! Gak sia-sia nonton dr awal smpe akhir, makasih bgt ya kk. Jd makin paham.",
    ]
    
    print("="*60)
    print("DEMO PREPROCESSING KOMENTAR UNTUK INDOBERT")
    print("="*60)
    
    for i, comment in enumerate(test_comments, 1):
        cleaned = clean_text_for_bert(comment, slang_dict=DEFAULT_SLANG_DICT, lowercase=False)
        print(f"\n[Sampel {i}]")
        print(f"Asli   : {comment}")
        print(f"Bersih : {cleaned}")
        
    print("\n" + "="*60)
    
    # Coba jalankan tokenisasi jika modul transformers terinstall
    try:
        from transformers import AutoTokenizer
        print("[*] Mencoba memuat tokenizer IndoBERT untuk simulasi tokenisasi...")
        tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
        
        sample_cleaned = clean_text_for_bert(test_comments[0], slang_dict=DEFAULT_SLANG_DICT)
        print(f"\nTeks Bersih: {sample_cleaned}")
        print(f"Tokens     : {tokenizer.tokenize(sample_cleaned)}")
    except ImportError:
        print("[!] Info: Library 'transformers' belum diinstal, simulasi tokenisasi dilewati.")
        print("[!] Jalankan 'pip install -r requirements.txt' untuk menginstalnya.")
    except Exception as e:
        print(f"[!] Error saat memuat tokenizer: {e}")

if __name__ == "__main__":
    run_test()
