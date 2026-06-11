# -*- coding: utf-8 -*-
"""
Convertit words_database.xml (source maitresse, ~217 Mo) en deux fichiers compacts
charges par index.html, en enrichissant au passage les donnees :

  - words_meta.json : metadonnees de tous les mots, en tableaux paralleles
        words[i], lemmes[i] ("=" si identique au mot, "" si inconnu),
        natures[i] (index dans natureNames), genres[i], nombres[i],
        syllables[i] (0 si inconnu), freqs[i] (-1 si inconnue),
        prons[i] (prononciations separees par "|")
  - vectors.bin : pour chaque mot, vecteur semantique normalise puis quantifie
        en Int8. Layout : Float32[count] (echelle par mot, little-endian)
        suivi de Int8[count*dims]. cosinus(a,b) ~= scaleA*scaleB*dot(qA,qB).
        Echelle 0 = pas de vecteur.

Enrichissement des prononciations manquantes (62% de la base a l'origine) :
  1. cmudict_fr.txt (lecture UTF-8 correcte, le matching d'origine avait
     des problemes d'encodage)
  2. Lexique383 (lexique_phon.csv, extrait de backups/05/Lexique383.xlsb) :
     notation SAMPA convertie vers l'alphabet du projet par une table de
     correspondance derivee automatiquement des ~40k mots communs aux deux
     sources (vote majoritaire par phoneme)
  3. Propagation : meme orthographe, pluriels reguliers en -s/-x (s final
     muet), participes e/ee/es/ees depuis l'infinitif en -er (son identique)
Les lemmes manquants sont aussi completes depuis Lexique.

Usage : python build_data.py
La validation comparant classements exacts et quantifies s'execute a la fin.
"""

import csv
import io
import json
import os
import sys
import time
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

import numpy as np

XML_PATH = "words_database.xml"
CMUDICT_PATH = "cmudict_fr.txt"
LEXIQUE_PATH = "lexique_phon.csv"
META_PATH = "words_meta.json"
VEC_PATH = "vectors.bin"
DIMS = 300

# nature (XML) -> categories grammaticales correspondantes (Lexique)
NATURE_TO_CGRAM = {
    "noun": {"NOM"},
    "verb": {"VER", "AUX"},
    "adjective": {"ADJ"},
    "adverb": {"ADV"},
}

TEST_WORDS = ["musique", "chanson", "amour", "mer", "acide"]
TOP_N = 30  # taille des classements compares pour la validation


def norm(word):
    return unicodedata.normalize("NFC", word).lower()


def load_cmudict():
    """mot -> liste de prononciations (chaines de phonemes separes par des espaces)"""
    prons = defaultdict(list)
    with io.open(CMUDICT_PATH, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith(";;;"):
                continue
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                w = norm(parts[0])
                if parts[1] not in prons[w]:
                    prons[w].append(parts[1])
    return prons


def load_lexique():
    """mot -> liste de lignes {phon, cgram, lemme, genre, nombre, freq, nbsyll}"""

    def as_float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    rows = defaultdict(list)
    with io.open(LEXIQUE_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            w = norm(str(row["1_ortho"]))
            if not w:
                continue
            films = as_float(row.get("9_freqfilms2"))
            livres = as_float(row.get("10_freqlivres"))
            # la frequence du XML d'origine = moyenne films/livres de Lexique
            freq = round((films + livres) / 2, 4) if films is not None and livres is not None else None
            nbsyll = as_float(row.get("24_nbsyll"))
            rows[w].append({
                "phon": str(row["2_phon"]),
                "cgram": str(row["4_cgram"]),
                "lemme": norm(str(row["3_lemme"])) if row["3_lemme"] else "",
                "genre": str(row.get("5_genre") or ""),
                "nombre": str(row.get("6_nombre") or ""),
                "freq": freq,
                "nbsyll": int(nbsyll) if nbsyll else 0,
            })
    return rows


def derive_phoneme_mapping(cmu, lexique):
    """Derive la table SAMPA (1 char) -> phoneme du projet par vote majoritaire
    sur les mots communs aux deux sources dont les longueurs concordent."""
    votes = defaultdict(Counter)
    for w, pron_list in cmu.items():
        rows = lexique.get(w)
        if not rows:
            continue
        tokens = pron_list[0].split()
        for row in rows:
            phon = row["phon"]
            if len(phon) == len(tokens):
                for c, t in zip(phon, tokens):
                    votes[c][t] += 1
                break
    mapping = {}
    for c, counter in votes.items():
        token, n = counter.most_common(1)[0]
        if n >= 5:  # ignore le bruit des alignements accidentels
            mapping[c] = token
    print(f"Table phonetique derivee : {len(mapping)} caracteres mappes")
    return mapping


def lexique_pron(phon, mapping):
    """Convertit une transcription Lexique en prononciation du projet,
    ou None si un caractere n'est pas couvert par la table."""
    tokens = []
    for c in phon:
        t = mapping.get(c)
        if t is None:
            return None
        tokens.append(t)
    return " ".join(tokens)


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
        if lemme:
            lemme = "=" if lemme == word else lemme
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
        elem.clear()

    print(f"Parse XML : {count} mots en {time.time() - t0:.0f}s")
    meta = {
        "version": 3,
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


def enrich(meta):
    cmu = load_cmudict()
    lexique = load_lexique()
    mapping = derive_phoneme_mapping(cmu, lexique)

    words = meta["words"]
    prons = meta["prons"]
    lemmes = meta["lemmes"]
    natures = meta["natures"]
    nature_names = meta["natureNames"]
    count = meta["count"]

    def lexique_rows_for(i):
        """Lignes Lexique du mot, celles de la bonne categorie d'abord."""
        rows = lexique.get(norm(words[i]), [])
        wanted = NATURE_TO_CGRAM.get(nature_names[natures[i]], set())
        matching = [r for r in rows if r["cgram"] in wanted]
        return matching or rows

    missing_initial = sum(1 for p in prons if not p)

    # 1. cmudict puis Lexique
    from_cmu = from_lex = 0
    for i in range(count):
        if prons[i]:
            continue
        w = norm(words[i])
        if w in cmu:
            prons[i] = "|".join(cmu[w])
            from_cmu += 1
            continue
        converted = []
        for row in lexique_rows_for(i):
            p = lexique_pron(row["phon"], mapping)
            if p and p not in converted:
                converted.append(p)
        if converted:
            prons[i] = "|".join(converted)
            from_lex += 1

    # 2. propagation
    by_spelling = defaultdict(list)
    for i in range(count):
        if prons[i]:
            for p in prons[i].split("|"):
                if p not in by_spelling[norm(words[i])]:
                    by_spelling[norm(words[i])].append(p)

    prop = 0
    for i in range(count):
        if prons[i]:
            continue
        w = norm(words[i])
        found = by_spelling.get(w)  # meme orthographe (autre nature)
        if not found and w[-1:] in ("s", "x"):
            found = by_spelling.get(w[:-1])  # pluriel regulier, s/x muet
        if not found:
            for suffix in ("ées", "és", "ée", "é"):  # participes d'un verbe en -er
                if w.endswith(suffix):
                    found = by_spelling.get(w[: -len(suffix)] + "er")
                    break
        if found:
            prons[i] = "|".join(found)
            prop += 1

    still_missing = sum(1 for p in prons if not p)
    print("Prononciations manquantes :", missing_initial, "->", still_missing,
          f"(cmudict +{from_cmu}, Lexique +{from_lex}, propagation +{prop})")

    # 3. attributs manquants depuis Lexique (lemme, frequence, syllabes, genre, nombre)
    freqs = meta["freqs"]
    syllables = meta["syllables"]
    genres = meta["genres"]
    nombres = meta["nombres"]
    fixed = Counter()
    for i in range(count):
        rows = None
        if not lemmes[i] or freqs[i] < 0 or not syllables[i] or not genres[i] or not nombres[i]:
            rows = lexique_rows_for(i)
        if not rows:
            continue
        for row in rows:
            if not lemmes[i] and row["lemme"]:
                lemmes[i] = "=" if row["lemme"] == norm(words[i]) else row["lemme"]
                fixed["lemme"] += 1
            if freqs[i] < 0 and row["freq"] is not None:
                freqs[i] = row["freq"]
                fixed["freq"] += 1
            if not syllables[i] and row["nbsyll"]:
                syllables[i] = row["nbsyll"]
                fixed["syllabes"] += 1
            if not genres[i] and row["genre"]:
                genres[i] = row["genre"]
                fixed["genre"] += 1
            if not nombres[i] and row["nombre"]:
                nombres[i] = row["nombre"]
                fixed["nombre"] += 1
    print("Attributs completes depuis Lexique :",
          ", ".join(f"{k} +{v}" for k, v in fixed.items()) or "aucun")


def quantize(vector_rows):
    count = len(vector_rows)
    scales = np.zeros(count, dtype=np.float32)
    quantized = np.zeros((count, DIMS), dtype=np.int8)
    normalized = np.zeros((count, DIMS), dtype=np.float32)  # pour la validation

    for i, vec in enumerate(vector_rows):
        if vec is None:
            continue
        norm_ = float(np.linalg.norm(vec))
        if norm_ == 0:
            continue
        unit = vec / norm_
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
    enrich(meta)
    scales, quantized, normalized = quantize(vector_rows)

    with open(VEC_PATH, "wb") as f:
        f.write(scales.astype("<f4").tobytes())
        f.write(quantized.tobytes())
    print(f"Ecrit {VEC_PATH} ({(scales.nbytes + quantized.nbytes) / 1e6:.1f} Mo)")

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, separators=(",", ":"))
    print(f"Ecrit {META_PATH} ({os.path.getsize(META_PATH) / 1e6:.1f} Mo)")

    validate(meta, scales, quantized, normalized)


if __name__ == "__main__":
    main()
