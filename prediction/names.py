from __future__ import annotations

import re
import unicodedata


ALIASES = {
    "ado den haag": "ADO Den Haag",
    "den haag": "ADO Den Haag",
    "ajax": "Ajax",
    "ajax amsterdam": "Ajax",
    "afc ajax amsterdam": "Ajax",
    "az": "AZ",
    "az alkmaar": "AZ",
    "alkmaar zaanstreek": "AZ",
    "almere city": "Almere City",
    "almere city fc": "Almere City",
    "cambuur": "SC Cambuur",
    "sc cambuur": "SC Cambuur",
    "excelsior": "Excelsior",
    "excelsior rotterdam": "Excelsior",
    "fc dordrecht": "Dordrecht",
    "dordrecht": "Dordrecht",
    "for sittard": "Fortuna Sittard",
    "fortuna sittard": "Fortuna Sittard",
    "fortuna sittardia combinatie": "Fortuna Sittard",
    "go ahead eagles": "Go Ahead Eagles",
    "groningen": "FC Groningen",
    "fc groningen": "FC Groningen",
    "football club groningen": "FC Groningen",
    "heerenveen": "Heerenveen",
    "sc heerenveen": "Heerenveen",
    "nec": "NEC",
    "n e c nijmegen": "NEC",
    "nec nijmegen": "NEC",
    "nijmegen eendracht combinatie": "NEC",
    "nijmegen": "NEC",
    "pec zwolle": "PEC Zwolle",
    "zwolle": "PEC Zwolle",
    "psv": "PSV",
    "psv eindhoven": "PSV",
    "eindhovense voetbalvereniging philips sport vereniging": "PSV",
    "rkc waalwijk": "Waalwijk",
    "waalwijk": "Waalwijk",
    "roda": "Roda JC",
    "roda jc": "Roda JC",
    "roda jc kerkrade": "Roda JC",
    "sparta": "Sparta Rotterdam",
    "sparta rotterdam": "Sparta Rotterdam",
    "sportclub heerenveen": "Heerenveen",
    "telstar": "Telstar",
    "sportclub telstar": "Telstar",
    "twente": "FC Twente",
    "fc twente": "FC Twente",
    "utrecht": "FC Utrecht",
    "fc utrecht": "FC Utrecht",
    "football club utrecht": "FC Utrecht",
    "football club twente": "FC Twente",
    "fc volendam": "Volendam",
    "football club volendam": "Volendam",
    "volendam": "Volendam",
    "de graafschap": "Graafschap",
    "de graafschap doetinchem": "Graafschap",
    "graafschap": "Graafschap",
    "vitesse arnhem": "Vitesse",
    "willem ii": "Willem II",
    "willem ii tilburg": "Willem II",
}


def name_key(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text).strip().lower()
    return re.sub(r"\s+", " ", text)


def canonical_team(value: object) -> str:
    raw = str(value or "").strip()
    return ALIASES.get(name_key(raw), raw)
