import urllib.request as r, os, shutil

BASE = "https://raw.githubusercontent.com/GHTassinari/ProjetoRAG-IA2-Backend/main"
ROOT = "/home/site/wwwroot"
UUID = "217faceb-e088-4dc8-accf-a7d0924c48d5"

shutil.rmtree(ROOT + "/chroma_db", ignore_errors=True)
os.makedirs(ROOT + "/chroma_db/" + UUID, exist_ok=True)

files = [
    ("chroma_db/chroma.sqlite3", ROOT + "/chroma_db/chroma.sqlite3"),
    ("chroma_db/" + UUID + "/data_level0.bin", ROOT + "/chroma_db/" + UUID + "/data_level0.bin"),
    ("chroma_db/" + UUID + "/header.bin", ROOT + "/chroma_db/" + UUID + "/header.bin"),
    ("chroma_db/" + UUID + "/length.bin", ROOT + "/chroma_db/" + UUID + "/length.bin"),
    ("chroma_db/" + UUID + "/link_lists.bin", ROOT + "/chroma_db/" + UUID + "/link_lists.bin"),
    ("gemini_functions.py", ROOT + "/gemini_functions.py"),
    ("init_db.py", ROOT + "/init_db.py"),
]

for src, dst in files:
    r.urlretrieve(BASE + "/" + src, dst)
    print("OK:", src)

print("Tudo atualizado!")
