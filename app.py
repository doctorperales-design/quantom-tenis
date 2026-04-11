"""
Quantum Tennis Engine v15 — Opus Deep Fix
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Gemini 2.5 Pro  →  Deep Profiling (Mano/Altura/DR5/Tour) + Auditoría H48+Math
Motor Python    →  Monte Carlo NumPy · Tiebreak CORREGIDO · IC 95%
Ajustes v3      →  5 Autónomos: Zurdo·ClayTrap·DR5·TierDrop·Indoors
Bankroll        →  Kelly Fraccionado 25% · Net Adj (no renorm)
Oracle          →  Google Sheets + Liquidador funcional

CHANGELOG v15 vs v14:
  BUG-1  HTML </> → </div> (tag nunca se cerraba)
  BUG-2  st.metric delta vs delta_color (mostraba texto "normal" como delta)
  BUG-3  Fatiga silenciada en lotes >1 partido (act_f = 0 si len(pts)>1)
  BUG-4  net_adj: adj independiente + renorm → ahora net_adj = adj_p1 − adj_p2
  BUG-5  no_vig() llamada 3× por jugador → 1× cacheada
  BUG-6  Dead code: ga==7/gb==7 inalcanzable → eliminado
  BUG-7  Liquidador era pass → implementado
  BUG-8  save_batch sin init en session_state → init explícito
  FEAT-1 3 ajustes v3 faltantes: Clay Trap, DR5, Tier Drop (autónomos via Gemini)
  FEAT-2 Gemini recibe reporte math completo → auditoría real, no solo H48
  FEAT-3 IC 95% del Monte Carlo → mostrado en UI
  FEAT-4 Profiling expandido: last5, tour para ajustes autónomos
"""

import streamlit as st
import json, os, re, random
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
GEMINI_MODEL = "gemini-2.5-pro"
DB_PATH      = "matches_jsonl.jsonl"

METADATA_KW = {
    'local', 'visita', 'empate', 'sencillos', 'dobles', 'vivo', 'apuestas',
    'streaming', 'women', 'men', 'tour', 'challenger', 'atp', 'wta', 'itf',
    'world tennis', 'grand slam', 'futures', 'copa', 'qualifier', 'qualy',
    'hoy', 'mañana', 'lunes', 'martes', 'miércoles', 'miercoles', 'jueves',
    'viernes', 'sábado', 'sabado', 'domingo', 'ene', 'feb', 'mar', 'abr',
    'may', 'jun', 'jul', 'ago', 'sep', 'oct', 'nov', 'dic', 'vs'
}

SURFACE_ADJ = {
    "hard":   {"spw":  0.0, "rpw":  0.0},
    "clay":   {"spw": -3.5, "rpw": +3.0},
    "grass":  {"spw": +4.0, "rpw": -2.5},
    "carpet": {"spw": +3.0, "rpw": -2.0},
}

TIER_ORDER = {"ATP/WTA": 4, "ATP": 4, "WTA": 4, "Challenger": 3, "ITF": 2}

# ─────────────────────────────────────────────────────────────────────────────
# CLIENTE GEMINI
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_gemini_client():
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    return genai.Client(api_key=api_key) if api_key else None

def gemini_available() -> bool:
    return get_gemini_client() is not None

# ─────────────────────────────────────────────────────────────────────────────
# GOOGLE SHEETS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_sheet():
    try:
        import gspread
        if "gcp_service_account" not in st.secrets:
            return None
        gc = gspread.service_account_from_dict(st.secrets["gcp_service_account"])
        return gc.open_by_key("1kciFhxjiVOeScsu_7e6UZvJ36ungHyeQxjIWMBu5CYs").sheet1
    except Exception:
        return None

def log_prediction(match_id, p1, p2, p_mod, p_casa, odd, ev_val, tier, league):
    sheet = get_sheet()
    if not sheet:
        return False
    try:
        ids = sheet.col_values(1)
        if match_id in ids:
            return False
        sheet.append_row([
            match_id, datetime.now().strftime("%Y-%m-%d %H:%M"),
            p1, p2, f"{p_mod*100:.1f}%",
            f"{p_casa*100:.1f}%" if p_casa else "S/C",
            str(odd) if odd else "S/C",
            f"{ev_val*100:+.1f}%" if ev_val is not None else "S/C",
            tier, "", league
        ])
        return True
    except Exception:
        return False

def liquidar_partido(p1: str, p2: str) -> str:
    client = get_gemini_client()
    if not client:
        return ""
    try:
        prompt = (
            f"Busca en Google si ya terminó '{p1}' vs '{p2}'.\n"
            f"Si terminó → responde ÚNICAMENTE el nombre del ganador.\n"
            f"Si no → responde PENDIENTE"
        )
        r = client.models.generate_content(
            model=GEMINI_MODEL, contents=prompt,
            config=types.GenerateContentConfig(
                tools=[{"google_search": {}}], temperature=0.0)
        )
        t = r.text.upper().strip()
        if "PENDIENTE" in t:
            return ""
        p1l = p1.upper().split()[-1]
        p2l = p2.upper().split()[-1]
        if p1.upper() in t or p1l in t:
            return p1
        if p2.upper() in t or p2l in t:
            return p2
        return ""
    except Exception:
        return ""

# ─────────────────────────────────────────────────────────────────────────────
# PARSER
# ─────────────────────────────────────────────────────────────────────────────
def extract_american_odds(text: str):
    m = re.search(r'([+-]\d{3,4})', text)
    if m:
        return text.replace(m.group(1), '').strip(), int(m.group(1))
    return text.strip(), None

def parse_matches(text: str) -> list[tuple]:
    text = text.replace('–', '-').replace('—', '-').replace('−', '-')
    results = []
    cur_league = "ATP/WTA"

    if re.search(r'(?i)\bvs\.?\b', text):
        for line in text.splitlines():
            line = line.strip()
            if not re.search(r'(?i)\bvs\.?\b', line):
                continue
            parts = re.split(r'(?i)\s+vs\.?\s+', line)
            if len(parts) == 2:
                p1, o1 = extract_american_odds(parts[0])
                p2, o2 = extract_american_odds(parts[1])
                if p1 and p2:
                    results.append((p1, o1, p2, o2, cur_league))
        if results:
            return results

    DATE_RE    = re.compile(r'^(\d{1,2}\s+[a-zA-Z]{3}\s+)?\d{2}:\d{2}$')
    COUNTER_RE = re.compile(r'^\+\s*\d{1,2}\s*(Streaming)?$', re.I)
    name_q, odd_q = [], []

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        odds_match = re.findall(r'[+-]\d{3,4}', line)
        if odds_match:
            for o in odds_match:
                odd_q.append(int(o))
            line = re.sub(r'[+-]\d{3,4}.*', '', line).strip()
            if not line:
                continue
        low = line.lower()
        if re.search(r'\b(?:' + '|'.join(map(re.escape, METADATA_KW)) + r')\b', low):
            if "itf" in low or "world tennis" in low:
                cur_league = "ITF"
            elif "challenger" in low:
                cur_league = "Challenger"
            elif "atp" in low:
                cur_league = "ATP"
            elif "wta" in low or "women" in low:
                cur_league = "WTA"
            continue
        if DATE_RE.match(line) or COUNTER_RE.match(line):
            continue
        name_q.append(line)
        while len(name_q) >= 2 and len(odd_q) >= 2:
            results.append((name_q.pop(0), odd_q.pop(0),
                            name_q.pop(0), odd_q.pop(0), cur_league))

    while len(name_q) >= 2 and len(odd_q) >= 2:
        results.append((name_q.pop(0), odd_q.pop(0),
                        name_q.pop(0), odd_q.pop(0), cur_league))
    return results

# ─────────────────────────────────────────────────────────────────────────────
# GEMINI — DEEP AUTONOMOUS CONTEXT (Profiles + Match Envs)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_autonomous_context(match_pairs: list[str]) -> dict:
    """Extrae perfiles de jugadores y condiciones del torneo en un solo call."""
    if not match_pairs:
        return {"players": {}, "matches": {}}
    client = get_gemini_client()
    if not client:
        return {"players": {}, "matches": {}}

    unique_players = []
    for m in match_pairs:
        for p in m.split(' vs '):
            if p.strip() not in unique_players:
                unique_players.append(p.strip())
                
    plist  = "\n".join(f"- {p}" for p in unique_players)
    mlist  = "\n".join(f"- {m}" for m in match_pairs)

    prompt = f"""Devuelve estrictamente un JSON válido, sin explicaciones, con 2 bloques: "players" y "matches".

BLOQUE 1 - Jugadores:
Para cada jugador busca y devuelve:
  hand  → "L" (zurdo) o "R" (diestro)
  ht    → estatura en cm (entero)
  last5 → victorias en sus últimos 5 partidos (0–5, estima si no hay datos exactos)
  tour  → circuito HABITUAL humano: "ATP", "WTA", "Challenger" o "ITF"

{plist}

BLOQUE 2 - Partidos Hoy:
Busca qué torneo están jugando EXACTAMENTE HOY estos emparejamientos y devuelve el entorno de cada partido:
  surface → "hard", "clay", "grass" o "carpet"
  indoor  → true o false
  league  → "ATP", "WTA", "Challenger" o "ITF"
  alt_m   → altitud de esa ciudad en msnm (entero)

{mlist}

Si falta info usa predeterminados: hand="R", ht=183, last5=3, tour="ATP" para jugadores; surface="hard", indoor=false, league="ATP/WTA", alt_m=0 para partidos.
Usa las llaves literales exactas enviadas.

Ejemplo de formato:
{{
  "players": {{
    "Carlos Alcaraz": {{"hand": "R", "ht": 183, "last5": 4, "tour": "ATP"}},
    "Rafael Nadal": {{"hand": "L", "ht": 185, "last5": 2, "tour": "ATP"}}
  }},
  "matches": {{
    "Carlos Alcaraz vs Rafael Nadal": {{"surface": "clay", "indoor": false, "league": "ATP", "alt_m": 600}}
  }}
}}"""

    try:
        r = client.models.generate_content(
            model=GEMINI_MODEL, contents=prompt,
            config=types.GenerateContentConfig(
                tools=[{"google_search": {}}], temperature=0.1)
        )
        raw = r.text.replace("```json", "").replace("```", "").strip()
        s, e = raw.find('{'), raw.rfind('}')
        data = json.loads(raw[s:e+1]) if s != -1 and e != -1 else {"players": {}, "matches": {}}
        
        # Normalize player keys
        prof_norm = {k.lower().strip(): v for k, v in data.get("players", {}).items()}
        # Normalize match keys
        match_norm = {k.lower().strip(): v for k, v in data.get("matches", {}).items()}
        
        return {"players": prof_norm, "matches": match_norm}
    except Exception:
        return {"players": {}, "matches": {}}

def get_player_profile(name: str, master_context: dict) -> dict:
    """Busca jugador en el dict."""
    norm = name.lower().strip()
    last = norm.split()[-1] if " " in norm else norm
    for k, v in master_context.get("players", {}).items():
        if k in norm or norm in k or last in k:
            return {
                "zurdo": v.get("hand", "R") == "L",
                "ht":    int(v.get("ht", 183)),
                "last5": v.get("last5"),
                "tour":  v.get("tour", "ATP"),
            }
    return {"zurdo": False, "ht": 183, "last5": None, "tour": "ATP"}

def get_match_context(mstr: str, default_lg: str, master_context: dict) -> dict:
    norm = mstr.lower().strip()
    for k, v in master_context.get("matches", {}).items():
        if k in norm or norm in k:
            return {
                "surface": v.get("surface", "hard").lower(),
                "indoor":  bool(v.get("indoor", False)),
                "league":  v.get("league", default_lg),
                "alt_m":   int(v.get("alt_m", 0))
            }
    return {"surface": "hard", "indoor": False, "league": default_lg, "alt_m": 0}


# ─────────────────────────────────────────────────────────────────────────────
# CALIBRACIÓN HISTÓRICA
# ─────────────────────────────────────────────────────────────────────────────
CALIBRATION = {
    ("Challenger", "SD"): 0.82, ("Challenger", "D"): 0.80,
    ("Challenger", "FR"): 0.90, ("Challenger", "PY"): 0.88,
    ("ATP/WTA",    "SD"): 0.94,
    ("ATP",        "SD"): 0.96, ("WTA", "SD"): 0.96,
    ("ITF",        "SD"): 1.00,
}

def _liga_key(liga: str) -> str:
    lg = (liga or "").upper()
    if "CHALLENGER" in lg: return "Challenger"
    if "ITF" in lg:        return "ITF"
    if lg in ("ATP","WTA"):return lg
    return "ATP/WTA"

def _tier_key(t: str) -> str:
    t = (t or "").upper()
    if "FRANC" in t:                  return "FR"
    if "SUPER" in t or "SÚPER" in t:  return "SD"
    if "DERECHA" in t:                return "D"
    if "PARLAY" in t or "VALOR" in t: return "PY"
    return "BASURA"

def apply_calibration(pmod: float, liga: str, tier: str) -> float:
    factor = CALIBRATION.get((_liga_key(liga), _tier_key(tier)), 1.0)
    return min(max(0.5 + (pmod - 0.5) * factor, 0.05), 0.95)

# ─────────────────────────────────────────────────────────────────────────────
# AJUSTES v3 — 5 FACTORES AUTÓNOMOS
# ─────────────────────────────────────────────────────────────────────────────
def calcular_sum_adj(
    rival_zurdo: bool,
    indoor: bool,
    alt_cm: int,
    surface: str,
    liga: str,
    rival_tour: str,
    own_tour: str,
    last5: int | None,
) -> tuple[float, list[str]]:
    """Calcula el ajuste total PARA un jugador (positivo = lo ayuda)."""
    tot, notas = 0.0, []

    # 1) Zurdo: si tu rival es zurdo, te cuesta −0.07
    if rival_zurdo:
        tot -= 0.07
        notas.append("🛡️ Rival Zurdo → −0.07")

    # 2) Indoor + Altura: si mides >193cm en indoor, tu saque domina +0.05
    if indoor and alt_cm > 193:
        tot += 0.05
        notas.append(f"🗼 Indoor + {alt_cm}cm → +0.05")

    # 3) Clay Trap: si la superficie es clay en Challenger/ITF y tu RIVAL es
    #    de ese circuito (probable local), tú estás en desventaja −0.06
    if "clay" in surface.lower():
        lg_key  = _liga_key(liga)
        rv_key  = _liga_key(rival_tour)
        if lg_key in ("Challenger", "ITF") and rv_key == lg_key:
            tot -= 0.06
            notas.append(f"🧱 Clay Trap: rival habitual de {lg_key} en clay → −0.06")

    # 4) DR últimos 5: racha caliente/fría ±0.04
    if last5 is not None:
        if last5 >= 4:
            tot += 0.04
            notas.append(f"🔥 Racha caliente: {last5}/5 → +0.04")
        elif last5 <= 1:
            tot -= 0.04
            notas.append(f"❄️ Racha fría: {last5}/5 → −0.04")

    # 5) Tier Drop: si normalmente juegas ATP pero hoy estás en Challenger → −0.04
    t_act = TIER_ORDER.get(_liga_key(liga), 3)
    t_hab = TIER_ORDER.get(_liga_key(own_tour), 3)
    if t_hab > t_act:
        tot -= 0.04
        notas.append(f"📉 Tier Drop: de {own_tour} a {liga} → −0.04")

    return tot, notas

# ─────────────────────────────────────────────────────────────────────────────
# ENTORNO — Shrinkage Bayesiano + Superficie
# ─────────────────────────────────────────────────────────────────────────────
def apply_environment(spw, rpw, n, alt_m, fatigue, surface):
    conf   = min(n / 15.0, 1.0) if n > 0 else 0.0
    shrink = 0.35 * (1.0 - conf)
    a_spw  = spw * (1 - shrink) + 62.0 * shrink
    a_rpw  = rpw * (1 - shrink) + 38.0 * shrink
    sv     = SURFACE_ADJ.get(surface.lower(), SURFACE_ADJ["hard"])
    a_spw += sv["spw"] + (alt_m / 1000.0) * 1.5  - fatigue * 0.5
    a_rpw += sv["rpw"] - (alt_m / 1000.0) * 0.75 - fatigue * 1.5
    return min(max(a_spw, 30.0), 90.0), min(max(a_rpw, 10.0), 70.0)

# ─────────────────────────────────────────────────────────────────────────────
# MATEMÁTICA — Log5 + Markov + Monte Carlo NumPy
# ─────────────────────────────────────────────────────────────────────────────
def american_to_decimal(odd: int) -> float:
    return (odd / 100 + 1) if odd > 0 else (100 / abs(odd) + 1)

def no_vig(o1: int, o2: int):
    i1 = 1 / american_to_decimal(o1)
    i2 = 1 / american_to_decimal(o2)
    t  = i1 + i2
    p1, p2 = i1 / t, i2 / t
    return p1, p2, round(1 / p1, 3), round(1 / p2, 3)

def vig_pct(o1, o2):
    return round((1/american_to_decimal(o1) + 1/american_to_decimal(o2) - 1) * 100, 2)

def ev(p, dec):
    return p * (dec - 1) - (1 - p)

def kelly_fraction(prob, dec, f=0.25):
    b = dec - 1
    return max(0.0, round(((b * prob - (1 - prob)) / b) * f * 100, 2))

def get_tier(p_mod, ev_val, p_casa):
    p = p_mod * 100
    if p_casa is None or ev_val is None:
        if p >= 65:   return "🟢 DERECHA"
        elif p >= 50: return "🟡 VALOR"
        else:         return "🔴 BASURA"
    ev_p, pc_p = ev_val * 100, p_casa * 100
    if p < 45 or (p < pc_p and ev_p <= 0):
        return "🔴 BASURA"
    if p >= 70 and ev_p >= 2 and p > pc_p:
        return "🔥 SÚPER DERECHA"
    if 60 <= p < 70 and ev_p >= 3:
        return "🟢 DERECHA"
    if 50 <= p < 60 and pc_p < 45:
        return "🎯 FRANCOTIRADOR"
    if 45 <= p < 60 and ev_p >= 8:
        return "🟡 VALOR / PARLAY"
    return "🔴 BASURA"

def log5_serve(spw, rpw):
    a, b = spw / 100, rpw / 100
    d = a * (1 - b) + (1 - a) * b
    return (a * (1 - b)) / d if d else 0.5

def _game_wp_np(p: np.ndarray) -> np.ndarray:
    """Prob de ganar un game al servicio — Markov analítico, vectorizado."""
    q   = 1 - p
    pfd = np.where(p**2 + q**2 > 0, p**2 / (p**2 + q**2), 0.5)
    return p**4 + 4*p**4*q + 10*p**4*q**2 + 20*p**3*q**3*pfd

def sim_match(pA_b, pB_b, bo=3, iters=5000, alt=0):
    """Monte Carlo con tiebreak corregido. Retorna (p_hat, margen_IC95)."""
    std = 0.015 * (1.0 + alt / 2000.0)
    nd  = 2 if bo == 3 else 3

    rng  = np.random.default_rng()
    pA_v = np.clip(rng.normal(pA_b, std, iters), 0.35, 0.95)
    pB_v = np.clip(rng.normal(pB_b, std, iters), 0.35, 0.95)
    gA_v = _game_wp_np(pA_v)
    gB_v = _game_wp_np(pB_v)

    wins = 0
    for i in range(iters):
        sA = sB = 0
        srv = True                  # True = A saca este game
        pA, pB = pA_v[i], pB_v[i]  # prob punto al servicio
        gA, gB = gA_v[i], gB_v[i]  # prob game al servicio
        while sA < nd and sB < nd:
            ga = gb = 0
            while True:
                # ── Tiebreak (corregido v15) ──────────────────────
                if ga == 6 and gb == 6:
                    ta = tb = t = 0
                    while True:
                        # Patrón: el que toca saca 1, luego 2-2-2-2...
                        # srv=True → A saca primero en el TB
                        if srv:
                            a_saca = (t % 4 == 0 or t % 4 == 3)
                        else:
                            a_saca = (t % 4 == 1 or t % 4 == 2)
                        # El servidor gana el punto con su prob de servicio
                        server_won = random.random() < (pA if a_saca else pB)
                        if a_saca:
                            if server_won: ta += 1
                            else:          tb += 1
                        else:
                            if server_won: tb += 1
                            else:          ta += 1
                        t += 1
                        if ta >= 7 and ta - tb >= 2:
                            ga += 1; break      # A gana TB
                        if tb >= 7 and tb - ta >= 2:
                            gb += 1; break      # B gana TB
                    # Asignar set
                    if ga > gb: sA += 1
                    else:       sB += 1
                    srv = not srv       # el otro saca primero el próximo set
                    break

                # ── Game regular ──────────────────────────────────
                server_holds = random.random() < (gA if srv else gB)
                if srv:
                    if server_holds: ga += 1
                    else:            gb += 1
                else:
                    if server_holds: gb += 1
                    else:            ga += 1
                srv = not srv
                if ga >= 6 and ga - gb >= 2:
                    sA += 1; break
                if gb >= 6 and gb - ga >= 2:
                    sB += 1; break
                # (ga==7 or gb==7 ya cubiertos por la condición de arriba)

        if sA == nd:
            wins += 1

    p_hat = wins / iters
    se    = np.sqrt(p_hat * (1 - p_hat) / iters)
    return p_hat, 1.96 * se          # media, margen IC 95%

# ─────────────────────────────────────────────────────────────────────────────
# BD LOCAL & FALLBACK
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_stats(name: str) -> dict | None:
    pts = won = gms = r_pts = r_won = r_gms = n = 0
    try:
        with open(DB_PATH, encoding='utf-8') as f:
            for ln in f:
                if name.lower() not in ln.lower():
                    continue
                m  = json.loads(ln)
                aw = name.lower() in m[2].lower()
                al = name.lower() in m[4].lower()
                if not aw and not al:
                    continue
                wp = (m[6],m[8],m[9],m[10],m[11],m[12]) if aw else (m[13],m[15],m[16],m[17],m[18],m[19])
                lp = (m[13],m[15],m[16],m[17],m[18],m[19]) if aw else (m[6],m[8],m[9],m[10],m[11],m[12])
                pts   += wp[0]; won   += wp[1] + wp[2]; gms += wp[3]
                r_pts += lp[0]; r_won += max(0, lp[0] - (lp[1] + lp[2])); r_gms += lp[3]
                n += 1
    except Exception:
        return None
    if n == 0:
        return None
    spw = max(10.1, (won / pts * 100) if pts else 55.0)
    rpw = max(10.1, (r_won / r_pts * 100) if r_pts else 40.0)
    return {"n": n, "spw": round(spw, 1), "rpw": round(rpw, 1), "source": "BD Local"}

def gemini_stats(name: str, surface: str) -> dict:
    c = get_gemini_client()
    if not c:
        return {"n": 1, "spw": 55.0, "rpw": 40.0, "source": "Sin API"}
    try:
        req = (
            f'Estima SPW y RPW de "{name}" en {surface} últimos meses. '
            f'SOLO JSON crudo: {{"spw": 62.5, "rpw": 39.2}}'
        )
        r = c.models.generate_content(
            model=GEMINI_MODEL, contents=req,
            config=types.GenerateContentConfig(
                tools=[{"google_search": {}}], temperature=0.2)
        )
        t = r.text.replace('```json', '').replace('```', '').strip()
        s, e = t.find('{'), t.rfind('}')
        if s != -1 and e != -1:
            d = json.loads(t[s:e+1])
            return {"n": 5,
                    "spw": float(d.get("spw", 55.0)),
                    "rpw": float(d.get("rpw", 40.0)),
                    "source": "Gemini Web"}
    except Exception:
        pass
    return {"n": 1, "spw": 55.0, "rpw": 40.0, "source": "Gemini Fallback"}

# ─────────────────────────────────────────────────────────────────────────────
# GEMINI — AUDITORÍA H48 + MATH (recibe el reporte completo)
# ─────────────────────────────────────────────────────────────────────────────
def gemini_analysis(p1: str, p2: str, report: str) -> str:
    c = get_gemini_client()
    if not c:
        return "❌ Sin API."

    prompt = f"""Quantum Tennis Analyst v15.

PASO 1 — ALERTAS FÍSICAS (busca en Google):
Busca noticias de las últimas 48h sobre {p1} y {p2}.
FILTRO: Solo reporta lesiones CONFIRMADAS, retiros oficiales, o fatiga extrema
demostrable (partido >2h30m ayer, o singles+dobles el mismo día).
Si no hay nada → di "🟢 Sin Alertas Físicas (Camino Libre)".

PASO 2 — AUDITORÍA MATEMÁTICA:
{report}

Evalúa en 2-3 líneas:
- ¿El EV es coherente con la diferencia Pmod vs P_casa?
- ¿Los ajustes v3 son razonables (|net_adj| < 0.15)?
- ¿Alguna señal de sobreajuste?

PASO 3 — VEREDICTO (1 línea con emoji):
✅ VERDE → EV ≥ +5% Y sin alertas
⚠️ AMARILLO → EV 0-5% O fatiga confirmada
🔵 FAVORITO → Pmod >70% pero EV negativo por cuota
🚫 ROJO → EV negativo O lesión confirmada

Formato:
**H48**: [1 línea por jugador]
**AUDITORÍA**: [2-3 líneas]
**VEREDICTO**: [emoji + 1 línea]"""

    try:
        r = c.models.generate_content(
            model=GEMINI_MODEL, contents=prompt,
            config=types.GenerateContentConfig(
                tools=[{"google_search": {}}], temperature=0.1,
                safety_settings=[
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE),
                ])
        )
        if r.text:
            return r.text
        # Fallback sin buscador si Google bloquea
        r2 = c.models.generate_content(
            model=GEMINI_MODEL, contents=f"Analiza solo los números:\n{report}\nDa veredicto.",
            config=types.GenerateContentConfig(temperature=0.1)
        )
        return r2.text if r2.text else "⚠️ Análisis bloqueado."
    except Exception as e:
        return f"Error H48: {e}"

# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Quantum v15", page_icon="🎾", layout="wide")
    st.title("🎾 Quantum Tennis Engine v15")
    st.caption("Deep Autonomous Profiling · 5 Ajustes v3 · NumPy MC + IC 95% · Kelly 25%")

    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        [data-testid="stMetricValue"] { font-size: 1.8rem !important; font-weight: 800 !important; }
        [data-testid="stMetricDelta"] { font-size: 1.1rem !important; font-weight: 600 !important;
            padding: 2px 6px; border-radius: 4px; background: rgba(255,255,255,0.05); }
        .tier-badge { display: inline-block; padding: 8px 16px; border-radius: 8px;
            font-weight: 800; font-size: 1.05rem; text-align: center; width: 100%;
            margin-top: 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.4);
            text-transform: uppercase; letter-spacing: 0.5px;
            border: 1px solid rgba(255,255,255,0.1); }
        .tr { background:linear-gradient(135deg,#4a0000,#220000); color:#ff6666; border-color:#ff3333; }
        .ty { background:linear-gradient(135deg,#4a3800,#221a00); color:#ffcc00; border-color:#ffaa00; }
        .tg { background:linear-gradient(135deg,#004a11,#002208); color:#44ff66; border-color:#00ff33;
               box-shadow:0 0 15px rgba(0,255,51,0.2); }
        .tf { background:linear-gradient(135deg,#002b4a,#001222); color:#44aaff; border-color:#0088ff; }
        .ts { background:linear-gradient(135deg,#4a004a,#220022); color:#ff44ff; border-color:#aa00ff;
               box-shadow:0 0 20px rgba(170,0,255,0.5); }
        button[kind="primary"] { background:linear-gradient(135deg,#0d965e,#086d42) !important;
            box-shadow:0 4px 15px rgba(13,150,94,0.4) !important; border:none !important;
            font-weight:800 !important; letter-spacing:1px; border-radius:8px !important; }
        .stTextArea textarea { background-color:#111 !important; border:1px solid #333 !important;
            color:#00ff9d !important; font-family:monospace; font-size:15px !important;
            border-radius:8px !important; }
        .kelly { background:rgba(13,150,94,0.15); border:1px solid #0d965e;
                 border-radius:8px; padding:8px 14px; font-weight:700; font-size:1rem; }
        </style>
    """, unsafe_allow_html=True)

    if not gemini_available():
        st.error("⚠️ Falta GOOGLE_API_KEY")
        st.stop()

    with st.sidebar:
        st.header("🌍 Ecosistema")
        st.info("🤖 Autonomía Nivel Dios: Gemini parametriza Torneo, Superficie, Altitud y Jugadores dinámicamente con una lectura en vivo.")
        
        st.header("🤕 Fatiga (0–5)")
        fat_p1 = st.slider("Fatiga P1", 0, 5, 0)
        fat_p2 = st.slider("Fatiga P2", 0, 5, 0)
        st.divider()
        iters = st.select_slider("MC Iters", [1000, 3000, 5000, 10000], 5000)

    # ── Estado de sesión ──────────────────────────────────────────────────────
    if "txt"        not in st.session_state: st.session_state.txt = ""
    if "save_batch" not in st.session_state: st.session_state.save_batch = []

    tab1, tab2 = st.tabs(["🎾 Motor Analítico", "🔮 El Oráculo"])

    # ══════════════════════════════════════════════════════════════════════════
    with tab1:
        c1, _, c3 = st.columns([2, 1, 1])
        with c1:
            bo = st.radio("Formato", [3, 5], horizontal=True,
                          format_func=lambda x: f"Bo{x}")
        with c3:
            if st.button("🗑️ Limpiar", use_container_width=True):
                st.session_state.txt = ""
                st.rerun()

        txt = st.text_area("📋 Pega tus partidos:", key="txt", height=160,
                           placeholder="Sinner J\nAlcaraz C\n-140\n+110")

        if st.button("🚀 Ignición Autónoma", type="primary", use_container_width=True):
            pts = parse_matches(txt)
            if not pts:
                st.error("No se detectaron emparejamientos.")
                st.warning("Asegúrate de copiar el texto completo incluyendo la última cuota.")
            else:
                # Extraemos emparejamientos
                match_pairs = [f"{p[0]} vs {p[2]}" for p in pts]
                with st.spinner(f"🌐 Extrayendo Contexto Global (Torneos y {len(match_pairs)*2} jugadores)…"):
                    master_context = get_autonomous_context(match_pairs)

                sv = []
                for p1_r, o1, p2_r, o2, lg_parsed in pts:
                    p1, p2 = p1_r or "P1", p2_r or "P2"
                    if "utr" in p1.lower() or "utr" in p2.lower():
                        st.warning(f"🚫 UTR excluido: {p1} vs {p2}")
                        continue

                    # Contexto del torneo
                    mkey = f"{p1} vs {p2}"
                    m_env = get_match_context(mkey, lg_parsed, master_context)
                    surf  = m_env["surface"]
                    ind   = m_env["indoor"]
                    lg    = m_env["league"]
                    alt_m = m_env["alt_m"]

                    st.divider()
                    st.subheader(f"🧠 {p1} vs {p2}")
                    vig = vig_pct(o1, o2) if o1 and o2 else None
                    
                    st.caption(
                        f"Liga: {lg} | {surf.title()} "
                        f"{'(Indoor)' if ind else ''} | Altitud: {alt_m}m "
                        + (f"| Vig: {vig}%" if vig else "")
                    )

                    # ── Perfiles autónomos ────────────────────────────
                    pf1 = get_player_profile(p1, master_context)
                    pf2 = get_player_profile(p2, master_context)

                    # ── Stats BD / Gemini ─────────────────────────────
                    s1 = get_stats(p1) or gemini_stats(p1, surf)
                    s2 = get_stats(p2) or gemini_stats(p2, surf)

                    # ── Ajustes v3 autónomos para CADA jugador ────────
                    adj_p1, not_p1 = calcular_sum_adj(
                        rival_zurdo=pf2["zurdo"], indoor=ind, alt_cm=pf1["ht"],
                        surface=surf, liga=lg,
                        rival_tour=pf2["tour"], own_tour=pf1["tour"],
                        last5=pf1["last5"],
                    )
                    adj_p2, not_p2 = calcular_sum_adj(
                        rival_zurdo=pf1["zurdo"], indoor=ind, alt_cm=pf2["ht"],
                        surface=surf, liga=lg,
                        rival_tour=pf1["tour"], own_tour=pf2["tour"],
                        last5=pf2["last5"],
                    )

                    # ── Entorno (fatiga SIEMPRE aplicada) ─────────────
                    a1s, a1r = apply_environment(s1["spw"], s1["rpw"], s1["n"], alt_m, fat_p1, surf)
                    a2s, a2r = apply_environment(s2["spw"], s2["rpw"], s2["n"], alt_m, fat_p2, surf)

                    # ── Log5 + Monte Carlo ────────────────────────────
                    pA_b = log5_serve(a1s, a2r)
                    pB_b = log5_serve(a2s, a1r)

                    with st.spinner(f"NumPy MC ({iters:,} iter)…"):
                        mc_raw, mc_ic = sim_match(pA_b, pB_b, bo, iters, alt_m)

                    # ── Net Adj (FIX: no renormalizamos independientemente)
                    # adj_p1 = lo que afecta a P1 (+ ayuda, − daña)
                    # adj_p2 = lo que afecta a P2 (+ ayuda, − daña)
                    # Net para P1: sus ventajas − las ventajas de P2
                    net_adj = adj_p1 - adj_p2
                    mc_A = min(max(mc_raw + net_adj, 0.05), 0.95)
                    mc_B = 1 - mc_A

                    # ── Calibración ───────────────────────────────────
                    ev1_pre  = ev(mc_A, american_to_decimal(o1)) if o1 and o2 else None
                    nv1_pre  = no_vig(o1, o2)[0] if o1 and o2 else None
                    tier_pre = get_tier(mc_A, ev1_pre, nv1_pre)

                    mc_A_cal = apply_calibration(mc_A, lg, tier_pre)
                    mc_B_cal = 1 - mc_A_cal

                    # ── Alerta de sobreajuste ─────────────────────────
                    if abs(net_adj) > 0.15:
                        st.warning(
                            f"⚠️ Sobreajuste: |net_adj| = {abs(net_adj):.2f} > 0.15. "
                            f"Los ajustes v3 acumulan demasiada corrección."
                        )

                    # ── Display ───────────────────────────────────────
                    m1c, m2c = st.columns(2)
                    e1 = e2 = nv1 = nv2 = t1c = t2c = None

                    # Cachear no_vig una sola vez
                    nv_data = no_vig(o1, o2) if o1 and o2 else None

                    for cl, nm, mc, o, s, asp, arp, pf, adj, nts, is_p1 in [
                        (m1c, p1, mc_A_cal, o1, s1, a1s, a1r, pf1, adj_p1, not_p1, True),
                        (m2c, p2, mc_B_cal, o2, s2, a2s, a2r, pf2, adj_p2, not_p2, False),
                    ]:
                        hand_str = "Zurdo" if pf["zurdo"] else "Diestro"
                        tour_str = pf["tour"]
                        l5_str   = f"DR:{pf['last5']}/5" if pf["last5"] is not None else "DR:?"
                        cl.markdown(f"**{nm}** · `{hand_str} {pf['ht']}cm` · `{tour_str}` · `{l5_str}`")
                        cl.caption(f"{s['source']} n={s['n']} | SPW {asp:.1f}% RPW {arp:.1f}%")
                        cl.metric(
                            "P(Win) Calibrada", f"{mc*100:.1f}%",
                            delta=f"IC 95%: ±{mc_ic*100:.1f}%", delta_color="off"
                        )

                        if nv_data:
                            nvv  = nv_data[0] if is_p1 else nv_data[1]
                            fair = nv_data[2] if is_p1 else nv_data[3]
                            dec  = american_to_decimal(o)
                            evv  = ev(mc, dec)
                            kf   = kelly_fraction(mc, dec)
                            tr   = get_tier(mc, evv, nvv)

                            if is_p1: e1, nv1, t1c = evv, nvv, tr
                            else:     e2, nv2, t2c = evv, nvv, tr

                            cl.metric("No-Vig", f"{nvv*100:.1f}%",
                                      delta=f"Fair: {fair}", delta_color="off")
                            cl.metric("EV Return", f"{evv*100:+.1f}%",
                                      delta=f"{evv*100:+.1f}%",
                                      delta_color="normal" if evv > 0 else "inverse")
                            cl.metric("Edge", f"{(mc-nvv)*100:+.1f}%")

                            if kf > 0:
                                cl.markdown(
                                    f'<div class="kelly">🏦 Kelly 25%: {kf:.2f}% del bankroll</div>',
                                    unsafe_allow_html=True
                                )

                            hc = ('ts' if 'SUPER' in tr or 'SÚPER' in tr
                                  else 'tg' if 'DERE' in tr or 'FRAN' in tr
                                  else 'tf' if 'FAVO' in tr
                                  else 'ty' if 'VALOR' in tr else 'tr')
                            cl.markdown(
                                f'<div class="tier-badge {hc}">{tr}</div>',
                                unsafe_allow_html=True
                            )

                        if nts:
                            with cl.expander(f"👁️ Ajustes v3 ({len(nts)})"):
                                for x in nts:
                                    st.caption(x)

                    # Mostrar net_adj + calibración
                    cal_f = CALIBRATION.get((_liga_key(lg), _tier_key(tier_pre)), 1.0)
                    st.caption(
                        f"net_adj = {adj_p1:+.3f} − ({adj_p2:+.3f}) = {net_adj:+.3f} | "
                        f"Cal: {_liga_key(lg)}/{_tier_key(tier_pre)} → ×{cal_f}"
                    )

                    # ── Reporte para Gemini ───────────────────────────
                    report = (
                        f"{p1}: SPW={a1s:.1f}% RPW={a1r:.1f}% n={s1['n']} [{s1['source']}] "
                        f"{'Zurdo' if pf1['zurdo'] else 'Diestro'} {pf1['ht']}cm Tour:{pf1['tour']} DR:{pf1.get('last5','?')}/5\n"
                        f"{p2}: SPW={a2s:.1f}% RPW={a2r:.1f}% n={s2['n']} [{s2['source']}] "
                        f"{'Zurdo' if pf2['zurdo'] else 'Diestro'} {pf2['ht']}cm Tour:{pf2['tour']} DR:{pf2.get('last5','?')}/5\n"
                        f"Superficie: {surf} | Liga: {lg} | {'Indoor' if ind else 'Outdoor'}\n"
                        f"Log5: {p1}={pA_b:.3f} | {p2}={pB_b:.3f}\n"
                        f"MC raw={mc_raw*100:.1f}% IC±{mc_ic*100:.1f}% → net_adj={net_adj:+.3f} → adj={mc_A*100:.1f}% → cal={mc_A_cal*100:.1f}%\n"
                        f"Ajustes P1: {adj_p1:+.3f} ({', '.join(not_p1) or 'ninguno'})\n"
                        f"Ajustes P2: {adj_p2:+.3f} ({', '.join(not_p2) or 'ninguno'})\n"
                        + (f"EV: {p1}={e1*100:+.2f}% {t1c} | {p2}={e2*100:+.2f}% {t2c}"
                           if e1 is not None else f"Tier: {t1c}")
                    )

                    # ── Gemini H48 + Math Audit ───────────────────────
                    st.markdown("#### 📡 Gemini — H48 + Auditoría Matemática")
                    with st.spinner("Consultando Google…"):
                        analysis = gemini_analysis(p1, p2, report)
                    st.info(analysis)

                    # ── Batch ─────────────────────────────────────────
                    if e1 is not None:
                        mid = f"{p1}_{p2}_{datetime.now().strftime('%Y%m%d')}"
                        sv.extend([
                            (mid,         p1, p2, mc_A_cal, nv1, o1, e1, t1c, lg),
                            (mid + "_inv", p2, p1, mc_B_cal, nv2, o2, e2, t2c, lg),
                        ])

                st.session_state.save_batch = sv

        # ── Guardar ───────────────────────────────────────────────────────
        if st.session_state.save_batch:
            sb = st.session_state.save_batch
            st.divider()
            if st.button(f"💾 Guardar en Oráculo ({len(sb)//2} partidos)",
                         type="primary", use_container_width=True):
                ok = 0
                for d in sb:
                    try:
                        if log_prediction(*d):
                            ok += 1
                    except Exception:
                        pass
                if ok:
                    st.success(f"✅ {ok//2} partidos guardados.")
                else:
                    st.info("Sin duplicados nuevos.")
                st.session_state.save_batch = []

    # ══════════════════════════════════════════════════════════════════════════
    with tab2:
        st.header("📊 El Oráculo")
        sh = get_sheet()
        if not sh:
            st.warning("⚠️ Google Sheets no conectado.")
        else:
            data = sh.get_all_records()
            if not data:
                st.info("Sin predicciones guardadas aún.")
            else:
                df = pd.DataFrame(data)
                st.dataframe(df.tail(20), use_container_width=True)

                if st.button("🔎 Ejecutar Liquidador"):
                    liquidados = 0
                    for i, row in df.iterrows():
                        if not row.get("Winner", ""):
                            p1n = row.get("P1_Name", "")
                            p2n = row.get("P2_Name", "")
                            with st.spinner(f"Liquidando {p1n} vs {p2n}…"):
                                w = liquidar_partido(p1n, p2n)
                            if w:
                                sh.update_cell(i + 2, 10, w)
                                st.success(f"✅ {w}")
                                liquidados += 1
                    if liquidados:
                        st.success(f"Auditoría: {liquidados} partidos liquidados.")
                        st.rerun()
                    else:
                        st.info("Ningún partido pendiente por liquidar.")

if __name__ == "__main__":
    main()
