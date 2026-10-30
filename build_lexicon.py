# -*- coding: utf-8 -*-
"""
Reconstruit la base de mots (words_meta.json + vectors.bin, meme format que
build_data.py) a partir de ressources lexicales ouvertes, sans passer par
words_database.xml :

  - GLAFF 1.2.2 (Wiktionnaire, CC BY-SA 3.0) : formes, categories, genre,
    nombre, prononciations API, frequences (corpus Frantext, LM10, FrWaC)
  - Morphalou 3.1 (ATILF, LGPL-LR) : formes flechies, dont les composes a
    trait d'union et l'orthographe d'avant 1990, absents de GLAFF
  - Lexique 3.83 (lexique_phon.csv) : frequences films/livres par categorie
    (l'echelle de l'ancienne base), prononciations de secours
  - cmudict_fr.txt, puis gruut (phonetisation automatique, pip install gruut[fr]),
    puis l'ancienne base : derniers recours pour la prononciation
  - fastText cc.fr.300 : vecteurs semantiques (lancer index_fasttext.py avant)
  - corrections.tsv : corrections manuelles, appliquees en dernier

Une entree = une forme + une nature. Les homographes sont donc separes :
"est" nom (/ɛst/) et "est" verbe (/ɛ/) ont chacun leur prononciation et leur
frequence.

Sont retenues les formes qui ont un vecteur fastText et une frequence d'au moins
MIN_FREQ (Lexique, sinon corpus GLAFF, sinon estimee d'apres le rang fastText),
plus toutes celles connues de Lexique ou de l'ancienne base, et les mots
grammaticaux.

Format de sortie (lu par index.html) :
  - words_meta.json : tableaux paralleles words, lemmes ("=" : identique au mot),
    natures (indices dans natureNames), genres, nombres ("" : les deux ou sans
    objet), syllables, freqs (occurrences par million), prons (variantes
    separees par "|"), vecIndex (ligne du vecteur, -1 : aucun)
  - vectors.bin : Float32[vectorCount] (echelles) puis les vecteurs quantifies
    sur vectorBits bits (4 bits : deux composantes par octet, +8)

Usage : python build_lexicon.py
Les sources sont dans sources/ (voir sources/LISEZMOI.txt).
"""

import csv
import json
import math
import os
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

GLAFF_PATH = "sources/glaff/glaff-1.2.2.txt"
MORPHALOU_PATH = "sources/morphalou/Morphalou3.1_CSV.csv"
LEXIQUE_PATH = "lexique_phon.csv"
CMUDICT_PATH = "cmudict_fr.txt"
FASTTEXT_VEC = "backups/04/applyVectorsToXML/cc.fr.300.vec"
FASTTEXT_INDEX = "sources/fasttext_index.tsv"
CORRECTIONS_PATH = "corrections.tsv"
OLD_META_PATH = "backups/13_avant_reconstruction/words_meta.json"
META_PATH = "words_meta.json"
VEC_PATH = "vectors.bin"
REPORT_PATH = "rapport_construction.txt"
GRUUT_CACHE = "sources/gruut_cache.json"
DIMS = 300
# 4 bits : vecteurs deux fois plus legers, ~90 % des 30 plus proches voisins
# inchanges ; 8 bits : classements quasi identiques aux vecteurs d'origine
VECTOR_BITS = 4

# Seuil de frequence (occurrences par million de mots) en dessous duquel une forme
# n'est gardee que si Lexique ou l'ancienne base la connaissent. Le rapport de
# construction donne le nombre d'entrees pour d'autres seuils.
MIN_FREQ = 0.01
# Forme presente dans GLAFF mais jamais vue dans ses corpus (FrWaC : 1,6 milliard
# de mots) : sa frequence est au plus de l'ordre de 1/1600 par million.
GLAFF_ZERO_CAP = 0.0006

NATURES = ["noun", "adjective", "adverb", "verb", "other"]
NOUN, ADJ, ADV, VERB, OTHER = range(5)

TEST_WORDS = ["musique", "chanson", "amour", "mer", "acide"]
TOP_N = 30

LETTERS = "a-zàâäçéèêëîïôöûùüÿœæñ"
VALID_FORM = re.compile(f"^[{LETTERS}](?:[{LETTERS}'-]*[{LETTERS}])?$")

report_lines = []


def log(*parts):
    line = " ".join(str(p) for p in parts)
    print(line, flush=True)
    report_lines.append(line)


def norm(word):
    return unicodedata.normalize("NFC", word).replace("’", "'").strip()


# --- Prononciations : conversion vers l'alphabet du projet -------------------

VOWELS = {"aa", "ai", "an", "au", "ee", "ei", "eu", "ii", "in", "oe", "on", "oo", "ou", "un", "uu"}

CONSONANTS = {c: c + c for c in "bdfgklmnpstvz"}

IPA_TO_TOKEN = {
    **CONSONANTS,
    "a": "aa", "ɑ": "aa", "ɛ": "ai", "ɔ": "au", "ə": "ee", "e": "ei", "ø": "eu",
    "i": "ii", "œ": "oe", "o": "oo", "u": "ou", "y": "uu",
    "ɑ̃": "an", "ã": "an", "ɛ̃": "in", "ẽ": "in", "ɔ̃": "on", "õ": "on", "œ̃": "un",
    "ʃ": "ch", "ɡ": "gg", "ɲ": "gn", "ñ": "gn", "ʒ": "jj", "c": "kk", "x": "kk",
    "ŋ": "ng", "ʁ": "rr", "r": "rr", "ɥ": "uy", "w": "ww", "j": "yy",
}
# separateurs de syllabes, liaisons, allongements ; h : variante "h aspire"
IPA_IGNORED = set(".ːˈˌ‿' -​h")

MORPHALOU_TO_TOKEN = {
    **CONSONANTS,
    "a": "aa", "A": "aa", "e": "ei", "E": "ai", "E/": "ai", "@": "ee", "2": "eu", "6": "eu",
    "9": "oe", "i": "ii", "o": "oo", "O": "au", "O/": "au", "u": "ou", "y": "uu",
    "a~": "an", "e~": "in", "o~": "on", "9~": "un",
    "j": "yy", "w": "ww", "H": "uy", "S": "ch", "Z": "jj", "J": "gn", "N": "ng", "R": "rr", "r": "rr",
}

LEXIQUE_TO_TOKEN = {
    **CONSONANTS,
    "a": "aa", "i": "ii", "y": "uu", "u": "ou", "o": "oo", "O": "au", "e": "ei", "E": "ai",
    "°": "ee", "2": "eu", "9": "oe", "5": "in", "1": "un", "@": "an", "§": "on",
    "8": "uy", "j": "yy", "w": "ww", "N": "gn", "G": "ng", "R": "rr", "S": "ch", "Z": "jj", "x": "kk",
}


def ipa_to_pron(ipa):
    """Une transcription API -> "tt ei" ; None si un symbole est inconnu."""
    tokens = []
    chars = unicodedata.normalize("NFD", ipa)
    i = 0
    while i < len(chars):
        cluster = chars[i]
        i += 1
        while i < len(chars) and unicodedata.combining(chars[i]):
            cluster += chars[i]
            i += 1
        if cluster in IPA_IGNORED:
            continue
        token = IPA_TO_TOKEN.get(unicodedata.normalize("NFC", cluster))
        if token is None:
            # diacritique sans incidence (ex. devoisement) : on garde la base
            token = IPA_TO_TOKEN.get(cluster[0]) if "̃" not in cluster else None
        if token is None:
            return None
        tokens.append(token)
    return " ".join(tokens) if tokens else None


def table_to_pron(symbols, table):
    tokens = []
    for s in symbols:
        t = table.get(s)
        if t is None:
            return None
        tokens.append(t)
    return " ".join(tokens) if tokens else None


def add_unique(lst, values):
    for v in values:
        if v and v not in lst:
            lst.append(v)


# --- Entrees ------------------------------------------------------------------

class Entry:
    __slots__ = ("lemmas", "genres", "nombres", "infinitive",
                 "prons_glaff", "prons_lexique", "prons_morphalou",
                 "freq_lexique", "freq_glaff", "in_glaff", "sources")

    def __init__(self):
        self.lemmas = Counter()
        self.genres = set()     # "m", "f", "" (= sans genre / les deux)
        self.nombres = set()
        self.infinitive = False
        self.prons_glaff = []
        self.prons_lexique = []
        self.prons_morphalou = []
        self.freq_lexique = None
        self.freq_glaff = None
        self.in_glaff = False
        self.sources = set()


entries = defaultdict(Entry)   # (forme, nature) -> Entry


def gender_of(code):
    return {"m": "m", "f": "f", "masculine": "m", "feminine": "f"}.get(code, "")


def number_of(code):
    return {"s": "s", "p": "p", "singular": "s", "plural": "p"}.get(code, "")


# --- GLAFF --------------------------------------------------------------------

def glaff_nature(tag):
    c = tag[0]
    if c == "N":
        return NOUN
    if c == "A":
        return ADJ if tag[1:2] == "f" else OTHER
    if c == "V":
        return VERB
    if c == "R":
        return ADV
    return OTHER


def load_glaff():
    t0 = time.time()
    rows = kept = 0
    with open(GLAFF_PATH, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("|")
            rows += 1
            form = norm(p[0])
            if not VALID_FORM.match(form):
                continue
            tag, lemma = p[1], norm(p[2])
            nature = glaff_nature(tag)
            e = entries[(form, nature)]
            e.sources.add("glaff")
            e.in_glaff = True
            kept += 1

            if nature == NOUN:
                g, n = tag[2:3], tag[3:4]
            elif nature == ADJ:
                g, n = tag[3:4], tag[4:5]
            elif nature == VERB:
                if tag[2] == "n":
                    e.infinitive = True
                if tag[2] == "p" and tag[3] == "s":   # participe passe : Vmps-sm
                    g, n = tag[6:7], tag[5:6]
                else:
                    g = n = ""
            else:   # pronoms, determinants : Pp3msn-, Dk-mp--
                g, n = tag[3:4], tag[4:5]
            e.genres.add(gender_of(g))
            e.nombres.add(number_of(n))

            freq = sum(float(p[k]) for k in (6, 10, 14)) / 3   # frequences relatives de la forme
            e.freq_glaff = (e.freq_glaff or 0) + freq
            e.lemmas[lemma] += freq + 1e-6
            add_unique(e.prons_glaff, (ipa_to_pron(v) for v in p[3].split(";") if v))
    log(f"GLAFF : {rows} lignes, {kept} retenues ({time.time() - t0:.0f}s)")


# --- Morphalou ------------------------------------------------------------------

MORPHALOU_NATURES = {
    "Nom commun": NOUN, "Adjectif qualificatif": ADJ, "Adverbe": ADV, "Verbe": VERB,
}


def load_morphalou():
    t0 = time.time()
    rows = kept = 0
    started = False
    lemma_cols = None
    with open(MORPHALOU_PATH, encoding="utf-8") as f:
        for line in f:
            if not started:   # en-tete descriptif jusqu'a la ligne des noms de colonnes
                started = line.startswith("GRAPHIE;ID;")
                continue
            p = line.rstrip("\n").split(";")
            if len(p) < 18:
                continue
            if p[0]:
                lemma_cols = p[:9]
            rows += 1
            if "abréviation" in lemma_cols[3]:
                continue
            form = norm(p[9])
            if not VALID_FORM.match(form):
                continue
            nature = MORPHALOU_NATURES.get(lemma_cols[2], OTHER)
            lemma = norm(re.sub(r"^(se |s')", "", lemma_cols[0]))
            e = entries[(form, nature)]
            e.sources.add("morphalou")
            kept += 1
            if nature == VERB:
                if p[12] == "infinitive":
                    e.infinitive = True
                if p[12] == "participle" and p[14] == "past":
                    e.genres.add(gender_of(p[13]))
                    e.nombres.add(number_of(p[11]))
                else:
                    e.genres.add("")
                    e.nombres.add("")
            else:
                genre = p[13] if p[13] not in ("-", "") else lemma_cols[5]
                e.genres.add(gender_of(genre))
                e.nombres.add(number_of(p[11]))
            e.lemmas[lemma] += 1e-7   # departage seulement, GLAFF/Lexique priment
            add_unique(e.prons_morphalou, (
                table_to_pron(v.split(), MORPHALOU_TO_TOKEN)
                for v in p[16].split(" OU ") if v.strip()
            ))
    log(f"Morphalou : {rows} formes, {kept} retenues ({time.time() - t0:.0f}s)")


# --- Lexique ----------------------------------------------------------------------

def lexique_nature(cgram):
    return {"NOM": NOUN, "ADJ": ADJ, "ADV": ADV, "VER": VERB, "AUX": VERB}.get(cgram, OTHER)


def load_lexique():
    rows = kept = 0
    with open(LEXIQUE_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows += 1
            form = norm(row["1_ortho"])
            if not VALID_FORM.match(form):
                continue
            nature = lexique_nature(row["4_cgram"])
            e = entries[(form, nature)]
            e.sources.add("lexique")
            kept += 1
            try:
                freq = (float(row["9_freqfilms2"]) + float(row["10_freqlivres"])) / 2
            except ValueError:
                freq = 0.0
            e.freq_lexique = (e.freq_lexique or 0) + freq
            lemma = norm(row["3_lemme"])
            if lemma:
                e.lemmas[lemma] += freq + 1e-3
                if nature == VERB and lemma == form:
                    e.infinitive = True
            if nature != VERB or row["5_genre"]:
                e.genres.add(gender_of(row["5_genre"]))
            if nature != VERB or row["6_nombre"]:
                e.nombres.add(number_of(row["6_nombre"]))
            add_unique(e.prons_lexique, [table_to_pron(row["2_phon"], LEXIQUE_TO_TOKEN)])
    log(f"Lexique : {rows} lignes, {kept} retenues")


def merge_ligature_variants():
    """Lexique (et les corpus de GLAFF) ecrivent "coeur" ; GLAFF et Morphalou "cœur".
    Chaque graphie sans ligature est fusionnee dans la forme a ligature quand
    celle-ci existe, pour ne garder qu'une entree (l'interface accepte les deux
    saisies)."""
    ligature_forms = {form for (form, _), e in entries.items()
                      if ("œ" in form or "æ" in form) and e.sources & {"glaff", "morphalou"}}

    def ligature(word):
        lig = word.replace("oe", "œ").replace("ae", "æ")
        return lig if lig in ligature_forms else word

    merged = 0
    for (form, nature) in [k for k in entries if "oe" in k[0] or "ae" in k[0]]:
        lig = ligature(form)
        if lig == form:
            continue
        src = entries.pop((form, nature))
        dst = entries[(lig, nature)]
        for lemma, weight in src.lemmas.items():
            dst.lemmas[ligature(lemma)] += weight
        dst.genres |= src.genres
        dst.nombres |= src.nombres
        dst.infinitive |= src.infinitive
        add_unique(dst.prons_glaff, src.prons_glaff)
        add_unique(dst.prons_lexique, src.prons_lexique)
        add_unique(dst.prons_morphalou, src.prons_morphalou)
        for attr in ("freq_lexique", "freq_glaff"):
            a, b = getattr(dst, attr), getattr(src, attr)
            setattr(dst, attr, None if a is None and b is None else (a or 0) + (b or 0))
        dst.in_glaff |= src.in_glaff
        dst.sources |= src.sources
        merged += 1
    # lemmes des autres entrees ("coeurs" -> "coeur")
    for e in entries.values():
        if any("oe" in l or "ae" in l for l in e.lemmas):
            e.lemmas = Counter({ligature(l): w for l, w in e.lemmas.items()})
    log(f"Graphies sans ligature fusionnees (coeur -> cœur) : {merged}")
    return ligature


def load_cmudict():
    prons = defaultdict(list)
    with open(CMUDICT_PATH, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) == 2 and not line.startswith(";;;") and "�" not in parts[0]:
                word = norm(re.sub(r"\(\d+\)$", "", parts[0]))
                tokens = parts[1].split()
                if all(t in VOWELS or t in ("bb", "ch", "dd", "ff", "gg", "gn", "jj", "kk", "ll", "mm",
                                            "nn", "pp", "rr", "ss", "tt", "uy", "vv", "ww", "yy", "zz")
                       for t in tokens):
                    add_unique(prons[word], [" ".join(tokens)])
    return prons


# --- fastText --------------------------------------------------------------------------

def load_fasttext_index():
    index = {}
    with open(FASTTEXT_INDEX, encoding="utf-8") as f:
        for rank, line in enumerate(f):
            word, offset = line.rstrip("\n").split("\t")
            if word not in index:
                index[word] = (rank + 1, int(offset))
    return index


def fasttext_candidates(form):
    yield form
    yield form.replace("œ", "oe").replace("æ", "ae")
    yield form.replace("'", "’")
    yield form.replace("oe", "œ")


def fasttext_lookup(form, index):
    for cand in fasttext_candidates(form):
        hit = index.get(cand)
        if hit:
            return hit
    return None


def read_vectors(offsets):
    """offset de ligne -> vecteur, lus dans l'ordre du fichier."""
    vectors = {}
    with open(FASTTEXT_VEC, "rb") as f:
        for offset in sorted(offsets):
            f.seek(offset)
            parts = f.readline().split(b" ")
            vec = np.array(parts[-DIMS:], dtype=np.float32)
            vectors[offset] = vec
    return vectors


# --- Assemblage --------------------------------------------------------------------------

def pick_unique(values):
    """Valeur unique si toutes les lectures concordent, sinon "" (= accepte tous les filtres)."""
    values = set(values)
    return values.pop() if len(values) == 1 else ""


def glaff_scale(candidates):
    ratios = [e.freq_lexique / e.freq_glaff for _, e, _ in candidates
              if e.freq_lexique and e.freq_glaff]
    scale = float(np.median(ratios))
    log(f"Echelle GLAFF -> Lexique : x{scale:.3f} (mediane sur {len(ratios)} entrees)")
    return scale


def fit_rank_frequency(candidates, scale):
    """Ajuste log(freq) = a + b*log(rang fastText) sur les formes de frequence connue."""
    by_form = defaultdict(float)
    rank_of = {}
    for (form, nature), e, rank in candidates:
        if e.freq_lexique is not None:
            by_form[form] += e.freq_lexique
        elif e.freq_glaff:
            by_form[form] += e.freq_glaff * scale
        rank_of[form] = rank
    forms = [f for f, freq in by_form.items() if freq > 0]
    xs = [math.log10(rank_of[f]) for f in forms]
    ys = [math.log10(by_form[f]) for f in forms]
    b, a = np.polyfit(xs, ys, 1)
    log(f"Frequence estimee depuis le rang fastText : log10(f) = {a:.2f} {b:+.2f}*log10(rang) ({len(xs)} formes)")
    return a, b


def estimate_frequency(e, rank, scale, fit):
    """(frequence en occurrences par million, source)"""
    if e.freq_lexique is not None:
        return e.freq_lexique, "Lexique"
    if e.freq_glaff:
        return e.freq_glaff * scale, "GLAFF (corpus)"
    a, b = fit
    freq = 10 ** (a + b * math.log10(rank))
    if e.in_glaff:
        freq = min(freq, GLAFF_ZERO_CAP)
    return freq, "estimee (rang fastText)"


def gruut_prons(forms):
    """Phonetisation automatique (graphemes -> phonemes) des formes restantes.
    Les resultats sont gardes dans GRUUT_CACHE (gruut traite ~20 mots/s)."""
    cache = {}
    if os.path.exists(GRUUT_CACHE):
        with open(GRUUT_CACHE, encoding="utf-8") as f:
            cache = json.load(f)
    todo = [f for f in forms if f not in cache]
    if todo:
        try:
            from gruut import sentences
        except ImportError:
            log("gruut absent (pip install gruut[fr]) : pas de phonetisation automatique")
            todo = []
        for form in todo:
            phonemes = []
            for sentence in sentences(form, lang="fr"):
                for word in sentence:
                    phonemes.extend(word.phonemes or [])
            cache[form] = "".join(phonemes)
        if todo:
            with open(GRUUT_CACHE, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=0)
    result = {}
    for form in forms:
        pron = ipa_to_pron(cache.get(form, ""))
        if pron:
            result[form] = [pron]
    return result


def apply_corrections(rows):
    """corrections.tsv : mot<TAB>nature<TAB>champ=valeur;champ=valeur  ou  mot<TAB>nature<TAB>supprimer
    Champs : prononciation, frequence, lemme, genre, nombre, nature."""
    if not os.path.exists(CORRECTIONS_PATH):
        return rows
    index = {(r["word"], NATURES[r["nature"]]): r for r in rows}
    applied = 0
    with open(CORRECTIONS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or line.startswith("#"):
                continue
            word, nature, action = (line.split("\t") + ["", ""])[:3]
            word = norm(word)
            if nature not in NATURES:
                log(f"  corrections.tsv : nature inconnue '{nature}' pour '{word}'")
                continue
            key = (word, nature)
            if action.strip() == "supprimer":
                if index.pop(key, None):
                    applied += 1
                continue
            row = index.get(key)
            if row is None:
                row = {"word": word, "nature": NATURES.index(nature), "lemme": "", "genre": "",
                       "nombre": "", "freq": 0.0, "prons": "", "syllables": 0}
                index[key] = row
            for assignment in action.split(";"):
                if "=" not in assignment:
                    continue
                field, value = (s.strip() for s in assignment.split("=", 1))
                if field == "prononciation":
                    row["prons"] = value
                    row["syllables"] = count_syllables(value)
                elif field == "frequence":
                    row["freq"] = float(value)
                elif field in ("lemme", "genre", "nombre"):
                    row[field] = value
                elif field == "nature" and value in NATURES:
                    row["nature"] = NATURES.index(value)
                else:
                    log(f"  corrections.tsv : champ inconnu '{field}' pour '{word}'")
            applied += 1
    log(f"Corrections manuelles appliquees : {applied}")
    return list(index.values())


def count_syllables(prons):
    first = prons.split("|")[0]
    return sum(1 for t in first.split() if t in VOWELS)


def build():
    load_glaff()
    load_morphalou()
    load_lexique()
    ligature = merge_ligature_variants()
    log(f"Entrees (forme, nature) avant selection : {len(entries)}")

    ft_index = load_fasttext_index()
    log(f"Index fastText : {len(ft_index)} mots")

    old = None
    if os.path.exists(OLD_META_PATH):
        with open(OLD_META_PATH, encoding="utf-8") as f:
            old = json.load(f)
    old_forms = set(ligature(norm(w)) for w in old["words"]) if old else set()

    candidates = []
    without_vector = 0
    for key, e in entries.items():
        hit = fasttext_lookup(key[0], ft_index)
        if hit:
            candidates.append((key, e, hit[0]))
        else:
            without_vector += 1
    log(f"Entrees avec vecteur fastText : {len(candidates)} (sans vecteur, ecartees : {without_vector})")

    scale = glaff_scale(candidates)
    fit = fit_rank_frequency(candidates, scale)

    # Selection : frequence suffisante, ou forme connue de Lexique ou de l'ancienne
    # base, ou mot grammatical
    selected = []
    by_threshold = Counter()
    thresholds = (0, 0.001, 0.003, 0.01, 0.03, 0.1)
    for key, e, rank in candidates:
        form, nature = key
        freq, freq_src = estimate_frequency(e, rank, scale, fit)
        always = (e.freq_lexique is not None or form in old_forms
                  or (nature == OTHER and e.sources & {"glaff", "lexique"}))
        for t in thresholds:
            if always or freq >= t:
                by_threshold[t] += 1
        if always or freq >= MIN_FREQ:
            selected.append((key, e, freq, freq_src))
    log("Nombre d'entrees selon le seuil de frequence :",
        ", ".join(f"{t} -> {by_threshold[t]}" for t in thresholds))
    log(f"Entrees retenues (seuil {MIN_FREQ}) : {len(selected)}")

    cmudict = load_cmudict()
    old_prons = defaultdict(list)
    if old:
        for w, p in zip(old["words"], old["prons"]):
            if p:
                add_unique(old_prons[norm(w)], p.split("|"))

    # prononciations au niveau de la forme (toutes natures), pour les secours
    form_prons = defaultdict(list)
    for (form, nature), e in entries.items():
        add_unique(form_prons[form], e.prons_glaff or e.prons_lexique or e.prons_morphalou)

    def known_prons(form, e):
        for src, prons in (("GLAFF", e.prons_glaff), ("Lexique", e.prons_lexique),
                           ("Morphalou", e.prons_morphalou), ("meme forme", form_prons.get(form)),
                           ("cmudict", cmudict.get(form))):
            if prons:
                return prons, src
        if form[-1] in "sx" and form_prons.get(form[:-1]):   # pluriel regulier, s/x muet
            return form_prons[form[:-1]], "pluriel"
        return None, None

    missing = [form for (form, nature), e, _, _ in selected if not known_prons(form, e)[0]]
    t0 = time.time()
    guessed = gruut_prons(sorted(set(missing)))
    log(f"Phonetisation automatique (gruut) : {len(guessed)}/{len(set(missing))} formes ({time.time() - t0:.0f}s)")

    pron_source = Counter()
    freq_source = Counter()
    rows = []
    for (form, nature), e, freq, freq_src in selected:
        prons, src = known_prons(form, e)
        if not prons:
            for src, prons in (("gruut", guessed.get(form)), ("ancienne base", old_prons.get(form)),
                               ("aucune", [])):
                if prons:
                    break
        pron_source[src] += 1
        freq_source[freq_src] += 1

        lemma = e.lemmas.most_common(1)[0][0] if e.lemmas else ""
        if nature == VERB and e.infinitive:
            lemma = form
        if nature == VERB and not e.infinitive and lemma == form:
            # forme conjuguee homographe d'un infinitif impossible : lemme inconnu
            lemma = ""
        pron_str = "|".join(prons)
        rows.append({
            "word": form,
            "nature": nature,
            "lemme": "=" if lemma == form else lemma,
            "genre": pick_unique(e.genres) if nature != ADV else "",
            "nombre": pick_unique(e.nombres) if nature != ADV else "",
            "freq": round(freq, 4),
            "prons": pron_str,
            "syllables": count_syllables(pron_str) if pron_str else 0,
        })

    log("Prononciations par source :", dict(pron_source))
    log("Frequences par source :", dict(freq_source))

    rows = apply_corrections(rows)
    rows.sort(key=lambda r: (r["word"], r["nature"]))

    # vecteurs : un par forme, partage par ses homographes
    t0 = time.time()
    row_offsets = []
    for r in rows:
        hit = fasttext_lookup(r["word"], ft_index)
        row_offsets.append(hit[1] if hit else None)
    offsets = sorted(set(o for o in row_offsets if o is not None))
    vectors = read_vectors(offsets)
    slot = {o: k for k, o in enumerate(offsets)}
    for r, o in zip(rows, row_offsets):
        r["vec"] = slot[o] if o is not None else -1
    matrix = np.stack([vectors[o] for o in offsets])
    log(f"Vecteurs lus : {len(offsets)} ({time.time() - t0:.0f}s)")
    return rows, matrix, old


def quantize(matrix, bits):
    """Normalise chaque vecteur puis le quantifie sur `bits` bits signes avec une
    echelle par vecteur : cosinus(a, b) ~= scaleA * scaleB * dot(qA, qB)."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    unit = matrix / np.where(norms == 0, 1, norms)
    levels = 2 ** (bits - 1) - 1
    maxabs = np.abs(unit).max(axis=1, keepdims=True)
    q = np.clip(np.rint(unit / np.where(maxabs == 0, 1, maxabs) * levels), -levels, levels).astype(np.int8)
    scales = (maxabs[:, 0] / levels).astype(np.float32)
    if bits == 4:
        # deux valeurs par octet, decalees de +8 : composante paire en bas, impaire en haut
        u = (q + 8).astype(np.uint8)
        packed = u[:, 0::2] | (u[:, 1::2] << 4)
    else:
        packed = q
    return scales, q, unit, packed


def validate_quantization(rows, scales, quantized, normalized):
    log(f"\n--- Validation de la quantification ({VECTOR_BITS} bits) ---")
    slot_of = {}
    for r in rows:
        if r["vec"] >= 0:
            slot_of.setdefault(r["word"], r["vec"])
    deq = quantized.astype(np.float32) * scales[:, None]
    for test_word in TEST_WORDS:
        i = slot_of.get(test_word)
        if i is None:
            log(f"  '{test_word}' absent ou sans vecteur")
            continue
        exact = normalized @ normalized[i]
        approx = deq @ deq[i]
        for n in (10, TOP_N):
            overlap = len(set(np.argsort(-exact)[:n]) & set(np.argsort(-approx)[:n])) / n
            log(f"  '{test_word}': {n} plus proches voisins identiques a {overlap * 100:.0f}%")


def compare_with_old(meta, old):
    if not old:
        return
    log("\n--- Comparaison avec l'ancienne base (coeur = cœur) ---")

    def key(w):
        return norm(w).replace("œ", "oe").replace("æ", "ae")

    new_forms = set(key(w) for w in meta["words"])
    old_forms = set(key(w) for w in old["words"])
    removed = sorted(old_forms - new_forms)
    added = sorted(new_forms - old_forms)
    log(f"Formes : ancienne {len(old_forms)}, nouvelle {len(new_forms)}, "
        f"communes {len(old_forms & new_forms)}, retirees {len(removed)}, ajoutees {len(added)}")
    log("Exemples de formes retirees :", ", ".join(removed[:: max(1, len(removed) // 60)][:60]))
    log("Exemples de formes ajoutees :", ", ".join(added[:: max(1, len(added) // 60)][:60]))

    old_p = defaultdict(set)
    for w, p in zip(old["words"], old["prons"]):
        old_p[key(w)].update(x for x in p.split("|") if x)
    new_p = defaultdict(set)
    for w, p in zip(meta["words"], meta["prons"]):
        new_p[key(w)].update(x for x in p.split("|") if x)
    common = [w for w in new_p if old_p.get(w)]
    agree = sum(1 for w in common if new_p[w] & old_p[w])
    log(f"Prononciations : {agree}/{len(common)} formes communes ont au moins une variante identique "
        f"({100 * agree / max(1, len(common)):.1f}%)")
    diffs = [w for w in common if not (new_p[w] & old_p[w])]
    log("Exemples de prononciations differentes (ancienne -> nouvelle) :")
    for w in diffs[:: max(1, len(diffs) // 25)][:25]:
        log(f"  {w} : {' / '.join(sorted(old_p[w]))} -> {' / '.join(sorted(new_p[w]))}")


def main():
    t0 = time.time()
    rows, matrix, old = build()
    scales, quantized, normalized, packed = quantize(matrix, VECTOR_BITS)

    meta = {
        "version": 5,
        "count": len(rows),
        "dims": DIMS,
        "vectorBits": VECTOR_BITS,
        "vectorCount": len(matrix),
        "vecIndex": [r["vec"] for r in rows],
        "natureNames": NATURES,
        "words": [r["word"] for r in rows],
        "lemmes": [r["lemme"] for r in rows],
        "natures": [r["nature"] for r in rows],
        "genres": [r["genre"] for r in rows],
        "nombres": [r["nombre"] for r in rows],
        "syllables": [r["syllables"] for r in rows],
        "freqs": [r["freq"] for r in rows],
        "prons": [r["prons"] for r in rows],
    }

    log("\n--- Bilan ---")
    log(f"Entrees : {meta['count']}")
    log("Par nature :", dict(Counter(NATURES[n] for n in meta["natures"])))
    log("Sans prononciation :", sum(1 for p in meta["prons"] if not p))
    log("Sans lemme :", sum(1 for l in meta["lemmes"] if not l))
    nouns_adjs = [i for i, n in enumerate(meta["natures"]) if n in (NOUN, ADJ)]
    log("Noms/adjectifs sans genre unique (epicenes compris) :", sum(1 for i in nouns_adjs if not meta["genres"][i]))
    log("Noms/adjectifs sans nombre unique (invariables compris) :", sum(1 for i in nouns_adjs if not meta["nombres"][i]))

    log("Sans vecteur :", sum(1 for v in meta["vecIndex"] if v < 0))

    with open(VEC_PATH, "wb") as f:
        f.write(scales.astype("<f4").tobytes())
        f.write(packed.tobytes())
    log(f"Ecrit {VEC_PATH} : {len(matrix)} vecteurs ({os.path.getsize(VEC_PATH) / 1e6:.1f} Mo)")
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, separators=(",", ":"))
    log(f"Ecrit {META_PATH} ({os.path.getsize(META_PATH) / 1e6:.1f} Mo)")

    validate_quantization(rows, scales, quantized, normalized)
    compare_with_old(meta, old)
    log(f"\nTermine en {time.time() - t0:.0f}s")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")


if __name__ == "__main__":
    main()
