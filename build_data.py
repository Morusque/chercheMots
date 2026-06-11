# -*- coding: utf-8 -*-
"""
Convertit words_database.xml (source maitresse, ~217 Mo) en deux fichiers compacts
charges par index.html :

  - words_meta.json : metadonnees de tous les mots, en tableaux paralleles
        words[i], lemmes[i] ("" si identique au mot), natures[i] (index dans
        natureNames), genres[i], nombres[i], syllables[i] (0 si inconnu),
        freqs[i] (-1 si inconnue), prons[i] (prononciations separees par "|")
  - vectors.bin : pour chaque mot, vecteur semantique normalise puis quantifie
        en Int8. Layout : Float32[count] (echelle par mot, little-endian)
        suivi de Int8[count*dims]. cosinus(a,b) ~= scaleA*scaleB*dot(qA,qB).
        Echelle 0 = pas de vecteur.

Usage : python build_data.py
La validation comparant classements exacts et quantifies s'execute a la fin.
"""

import json
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

XML_PATH = "words_database.xml"
META_PATH = "words_meta.json"
VEC_PATH = "vectors.bin"
DIMS = 300

TEST_WORDS = ["musique", "chanson", "amour", "mer", "acide"]
TOP_N = 30  # taille des classements compares pour la validation


def parse_xml():
    words, lemmes, natures, genres, nombres, syllables, freqs, prons = (
        [], [], [], [], [], [], [], []
    )
    nature_names = []
    nature_index = {}
    vector_rows = []

    t0 = time.time()
    count = 0
    for _, elem in ET.iterparse(XML_PATH, events=("end",)):
        if elem.tag != "word":
            continue

        word = elem.get("word") or ""
        lemme = elem.get("lemme") or ""
        if lemme == word:
            lemme = ""
        nature = elem.get("nature") or ""
        if nature not in nature_index:
            nature_index[nature] = len(nature_names)
            nature_names.append(nature)

        syll = elem.get("syllables")
        freq = elem.get("freq")

        pron_list = [
            p.text.strip()
            for p in elem.iter("pronunciation")
            if p.text and p.text.strip()
        ]

        vec_node = elem.find("vectors")
        if vec_node is not None and vec_node.text and vec_node.text.strip():
            vec = np.array(vec_node.text.split(), dtype=np.float32)
            if vec.shape[0] != DIMS:
                sys.exit(f"Vecteur de taille inattendue ({vec.shape[0]}) pour '{word}'")
        else:
            vec = None

        words.append(word)
        lemmes.append(lemme)
        natures.append(nature_index[nature])
        genres.append(elem.get("genre") or "")
        nombres.append(elem.get("nombre") or "")
        syllables.append(int(syll) if syll else 0)
        freqs.append(round(float(freq), 4) if freq else -1)
        prons.append("|".join(pron_list))
        vector_rows.append(vec)

        count += 1
        if count % 10000 == 0:
            print(f"  {count} mots... ({time.time() - t0:.0f}s)")
        elem.clear()

    print(f"Parse termine : {count} mots en {time.time() - t0:.0f}s")
    meta = {
        "version": 1,
        "count": count,
        "dims": DIMS,
        "natureNames": nature_names,
        "words": words,
        "lemmes": lemmes,
        "natures": natures,
        "genres": genres,
        "nombres": nombres,
        "syllables": syllables,
        "freqs": freqs,
        "prons": prons,
    }
    return meta, vector_rows


def quantize(vector_rows):
    count = len(vector_rows)
    scales = np.zeros(count, dtype=np.float32)
    quantized = np.zeros((count, DIMS), dtype=np.int8)
    normalized = np.zeros((count, DIMS), dtype=np.float32)  # pour la validation

    for i, vec in enumerate(vector_rows):
        if vec is None:
            continue
        norm = float(np.linalg.norm(vec))
        if norm == 0:
            continue
        unit = vec / norm
        normalized[i] = unit
        maxabs = float(np.max(np.abs(unit)))
        if maxabs == 0:
            continue
        q = np.clip(np.rint(unit / maxabs * 127), -127, 127).astype(np.int8)
        quantized[i] = q
        scales[i] = maxabs / 127.0

    return scales, quantized, normalized


def validate(meta, scales, quantized, normalized):
    print("\n--- Validation de la quantification ---")
    words = meta["words"]
    index_of = {}
    for i, w in enumerate(words):
        index_of.setdefault(w, i)

    # cosinus exacts vs quantifies
    deq = quantized.astype(np.float32) * scales[:, None]  # vecteurs dequantifies

    worst_overlap = 1.0
    for test_word in TEST_WORDS:
        i = index_of.get(test_word)
        if i is None or scales[i] == 0:
            print(f"  '{test_word}' absent ou sans vecteur, ignore")
            continue

        exact = normalized @ normalized[i]
        approx = deq @ deq[i]

        err = np.abs(exact - approx)
        top_exact = set(np.argsort(-exact)[:TOP_N])
        top_approx = set(np.argsort(-approx)[:TOP_N])
        overlap = len(top_exact & top_approx) / TOP_N
        worst_overlap = min(worst_overlap, overlap)
        print(
            f"  '{test_word}': top-{TOP_N} identique a {overlap*100:.0f}%, "
            f"erreur cosinus max {err.max():.5f}, moyenne {err.mean():.5f}"
        )

    if worst_overlap < 0.9:
        print("ATTENTION : la quantification degrade les classements au-dela de 10%.")
    else:
        print("OK : classements preserves.")


def main():
    meta, vector_rows = parse_xml()
    scales, quantized, normalized = quantize(vector_rows)

    with open(VEC_PATH, "wb") as f:
        f.write(scales.astype("<f4").tobytes())
        f.write(quantized.tobytes())
    print(f"Ecrit {VEC_PATH} ({(scales.nbytes + quantized.nbytes) / 1e6:.1f} Mo)")

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, separators=(",", ":"))
    import os
    print(f"Ecrit {META_PATH} ({os.path.getsize(META_PATH) / 1e6:.1f} Mo)")

    validate(meta, scales, quantized, normalized)


if __name__ == "__main__":
    main()
