# -*- coding: utf-8 -*-
"""
Indexe le fichier texte fastText (cc.fr.300.vec, ~4,5 Go) : pour chaque mot,
la position (octet) de sa ligne dans le fichier. build_lexicon.py s'en sert
pour lire directement les vecteurs voulus sans reparcourir tout le fichier.

Ecrit sources/fasttext_index.tsv : mot<TAB>offset, dans l'ordre du fichier
(fastText trie son vocabulaire par frequence decroissante).

Usage : python index_fasttext.py [chemin du .vec]
"""

import sys
import time

VEC_PATH = sys.argv[1] if len(sys.argv) > 1 else "backups/04/applyVectorsToXML/cc.fr.300.vec"
OUT_PATH = "sources/fasttext_index.tsv"


def main():
    t0 = time.time()
    count = 0
    with open(VEC_PATH, "rb") as f, open(OUT_PATH, "w", encoding="utf-8") as out:
        f.readline()  # en-tete : nombre de mots, dimensions
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            word = line[: line.index(b" ")].decode("utf-8", errors="replace")
            if "\t" in word or "\n" in word:
                continue
            out.write(f"{word}\t{offset}\n")
            count += 1
            if count % 200000 == 0:
                print(f"{count} mots, {time.time() - t0:.0f}s", flush=True)
    print(f"Termine : {count} mots en {time.time() - t0:.0f}s -> {OUT_PATH}")


if __name__ == "__main__":
    main()
