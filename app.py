"""
Quantum Tennis Engine v6.5 — Gemini Quant Edition
Markov Chains Analíticas · Shrinkage Bayesiano · Oráculo Google Sheets
"""

import streamlit as st
import json
import os
import re
import random
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
# GOOGLE SHEETS — ORÁCULO
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_sheet():
    try:
        import gspread
        if "gcp_service_account" not in st.secrets:
            return None
        gc = gspread.service_account_from_dict(st.secrets["gcp_service_account"])
        return gc.open_by_key("1kciFhxjiVOeScsu_7e6UZvJ36ungHyeQxjIWMBu5CYs").sheet1
    except Exception as e:
        st.error(f"Error de Conexión o Permisos: {e}")
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
            match_id,
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            p1, p2,
            f"{p_mod*100:.1f}%",
            f"{p_casa*100:.1f}%" if p_casa else "S/C",
            str(odd) if odd else "S/C",
            f"{ev_val*100:+.1f}%" if ev_val is not None else "S/C",
            tier, "", league
        ])
        return True
    except Exception as e:
        st.error(f"Error escribiendo en Excel: {e}")
        return False

def liquidar_partido(p1: str, p2: str) -> str:
    client = get_gemini_client()
    if not client:
        return ""
    try:
        prompt = (
            f"Busca en Google usando Flashscore o webs de tenis el resultado final del partido entre '{p1}' y '{p2}' "
            f"jugado en las últimas horas.\n"
            f"REGLA 1: Si el partido ya terminó, dime ÚNICAMENTE el NOMBRE COMPLETO del ganador (sin explicaciones).\n"
            f"REGLA 2: Si el partido no se ha jugado, sigue en vivo o está cancelado, responde exactamente la palabra 'PENDIENTE'."
        )
        r = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[{"google_search": {}}],
                temperature=0.0,
            )
        )
        t = r.text.upper().strip()
        if "PENDIENTE" in t: return ""
        
        p1_upper = p1.upper()
        p2_upper = p2.upper()
        p1_last = p1_upper.split()[-1] if " " in p1_upper else p1_upper
        p2_last = p2_upper.split()[-1] if " " in p2_upper else p2_upper
        
        if p1_upper in t or p1_last in t: return p1
        if p2_upper in t or p2_last in t: return p2
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
    results      = []
    cur_league   = "ATP/WTA"

    # Modo A: línea con 'vs'
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

    # Modo B: apilado (Caliente móvil)
    DATE_RE    = re.compile(r'^(\d{1,2}\s+[a-zA-Z]{3}\s+)?\d{2}:\d{2}$')
    COUNTER_RE = re.compile(r'^\+\s*\d{1,2}\s*(Streaming)?$', re.I)
    ODD_LONE   = re.compile(r'^[+-]\d{3,4}$')

    name_q, odd_q = [], []

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        # Extraer cuotas primerísimamente para que LOCAL +260 no se borre junto
        odds_match = re.findall(r'[+-]\d{3,4}', line)
        if odds_match:
            for o in odds_match:
                odd_q.append(int(o))
            line = re.sub(r'[+-]\d{3,4}.*', '', line).strip()
            if not line:
                continue

        low = line.lower()
        if any(kw in low for kw in METADATA_KW):
            if "itf" in low or "world tennis" in low: cur_league = "ITF"
            elif "challenger" in low:                  cur_league = "Challenger"
            elif "atp" in low:                         cur_league = "ATP"
            elif "wta" in low or "women" in low:       cur_league = "WTA"
            continue

        if DATE_RE.match(line) or COUNTER_RE.match(line):
            continue

        name_q.append(line)

        while len(name_q) >= 2 and len(odd_q) >= 2:
            results.append((name_q.pop(0), odd_q.pop(0),
                            name_q.pop(0), odd_q.pop(0), cur_league))

    return results

# ─────────────────────────────────────────────────────────────────────────────
# FÍSICA Y ENTORNO
# ─────────────────────────────────────────────────────────────────────────────
def apply_environment(spw: float, rpw: float, n: int,
                      altitude_m: int, fatigue: int) -> tuple[float, float]:
    """Shrinkage Bayesiano + altitud + fatiga."""
    AVG_SPW, AVG_RPW = 62.0, 38.0
    confidence    = min(n / 15.0, 1.0) if n > 0 else 0.0
    shrink        = 0.35 * (1.0 - confidence)
    adj_spw       = spw * (1 - shrink) + AVG_SPW * shrink
    adj_rpw       = rpw * (1 - shrink) + AVG_RPW * shrink
    alt_bonus     = (altitude_m / 1000.0) * 1.5
    adj_spw      += alt_bonus
    adj_rpw      -= alt_bonus * 0.5
    adj_spw      -= fatigue * 0.5
    adj_rpw      -= fatigue * 1.5
    return min(max(adj_spw, 30.0), 90.0), min(max(adj_rpw, 10.0), 70.0)

# ─────────────────────────────────────────────────────────────────────────────
# MATEMÁTICA — MARKOV + MONTE CARLO
# ─────────────────────────────────────────────────────────────────────────────
def american_to_decimal(odd: int) -> float:
    return (odd / 100 + 1) if odd > 0 else (100 / abs(odd) + 1)

def no_vig(odd1: int, odd2: int) -> tuple:
    i1 = 1 / american_to_decimal(odd1)
    i2 = 1 / american_to_decimal(odd2)
    t  = i1 + i2
    p1, p2 = i1 / t, i2 / t
    return p1, p2, round(1 / p1, 3), round(1 / p2, 3)

def ev(prob: float, dec_odd: float) -> float:
    return prob * (dec_odd - 1) - (1 - prob)

def get_tier(p_mod: float, ev_val: float | None, p_casa: float | None) -> str:
    p   = p_mod * 100
    if p_casa is None or ev_val is None:
        if p >= 65:  return "🟢 DERECHA (Sin Cuota)"
        elif p >= 50: return "🟡 VALOR (Sin Cuota)"
        else:         return "🔴 BASURA (Sin Cuota)"
    ev_p = ev_val * 100
    pc_p = p_casa * 100
    if p < 45:                              return "🔴 BASURA (Underdog Tóxico)"
    if p < pc_p and ev_p <= 0:             return "🔴 BASURA (A favor de la Casa)"
    if p >= 70 and ev_p >= 2 and p > pc_p: return "🔥 SÚPER DERECHA"
    if 60 <= p < 70 and ev_p >= 3:         return "🟢 DERECHA"
    if 50 <= p < 60 and pc_p < 45:         return "🎯 FRANCOTIRADOR"
    if 45 <= p < 60 and ev_p >= 8:         return "🟡 VALOR / PARLAY"
    return "🔴 BASURA (No cumple umbrales)"

def log5_serve(spw: float, rpw: float) -> float:
    A, B = spw / 100, rpw / 100
    d = A * (1 - B) + (1 - A) * B
    return (A * (1 - B)) / d if d else 0.5

def calc_game_win_prob(p: float) -> float:
    """Solución analítica O(1) — Cadenas de Markov para ganar 1 game."""
    q = 1 - p
    p4  = p**4
    p5  = 4  * p**4 * q
    p6  = 10 * p**4 * q**2
    pd  = 20 * p**3 * q**3
    pfd = p**2 / (p**2 + q**2) if (p**2 + q**2) > 0 else 0
    return p4 + p5 + p6 + pd * pfd

def sim_tiebreak(pA: float, pB: float) -> int:
    a = b = turn = 0
    while True:
        serves_A = (turn % 4) in [0, 3]
        if random.random() < (pA if serves_A else pB): a += 1
        else: b += 1
        turn += 1
        if a >= 7 and a - b >= 2: return 1
        if b >= 7 and b - a >= 2: return 0

def sim_set(pGameA: float, pGameB: float,
            pPtA: float, pPtB: float,
            a_serves_first: bool = True) -> tuple[int, bool]:
    """
    Simula un set. Devuelve (ganador 1=A/0=B, quién saca primero en el siguiente set).
    FIX: rastrea correctamente el turno de saque en cada game.
    """
    gA = gB = 0
    a_srv = a_serves_first

    while True:
        if gA == 6 and gB == 6:
            tb_first = pPtA if a_srv else pPtB
            tb_sec   = pPtB if a_srv else pPtA
            w = sim_tiebreak(tb_first, tb_sec)
            if w: gA += 1
            else: gB += 1
            return (1 if gA > gB else 0), not a_srv   # el que no sirvió el TB saca primero

        if a_srv:
            if random.random() < pGameA: gA += 1
            else: gB += 1
        else:
            if random.random() < pGameB: gB += 1
            else: gA += 1

        a_srv = not a_srv   # alternar cada game

        if gA >= 6 and gA - gB >= 2: return 1, a_srv
        if gB >= 6 and gB - gA >= 2: return 0, a_srv
        if gA == 7: return 1, a_srv
        if gB == 7: return 0, a_srv

def sim_match(pA_base: float, pB_base: float,
              best_of: int = 3, iterations: int = 5000,
              altitude_m: int = 0) -> float:
    needed   = 2 if best_of == 3 else 3
    var_mult = 1.0 + altitude_m / 2000.0
    std_dev  = 0.015 * var_mult
    wins     = 0

    for _ in range(iterations):
        pA = max(0.35, min(0.95, random.gauss(pA_base, std_dev)))
        pB = max(0.35, min(0.95, random.gauss(pB_base, std_dev)))
        gA_p = calc_game_win_prob(pA)
        gB_p = calc_game_win_prob(pB)

        sA = sB = 0
        a_srv = True
        while sA < needed and sB < needed:
            w, a_srv = sim_set(gA_p, gB_p, pA, pB, a_srv)   # ✅ FIX: se pasa a_srv entre sets
            if w: sA += 1
            else: sB += 1
        if sA == needed: wins += 1

    return wins / iterations

# ─────────────────────────────────────────────────────────────────────────────
# BASE DE DATOS LOCAL
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_stats(name: str, path: str = DB_PATH) -> dict | None:
    sv_pts = sv_won = sv_gms = sv_held = 0
    rt_pts = rt_won = rt_gms = rt_brk  = 0
    n = 0
    try:
        with open(path, encoding='utf-8') as f:
            for line in f:
                if name.lower() not in line.lower(): continue
                m = json.loads(line)
                as_w = name.lower() in m[2].lower()
                as_l = name.lower() in m[4].lower()
                if not as_w and not as_l: continue
                wp = (m[6],m[8],m[9],m[10],m[11],m[12]) if as_w else (m[13],m[15],m[16],m[17],m[18],m[19])
                lp = (m[13],m[15],m[16],m[17],m[18],m[19]) if as_w else (m[6],m[8],m[9],m[10],m[11],m[12])
                sv_pts  += wp[0]; sv_won  += wp[1]+wp[2]; sv_gms  += wp[3]
                sv_held += max(0, wp[3]-(wp[5]-wp[4]))
                rt_pts  += lp[0]; rt_won  += max(0, lp[0]-(lp[1]+lp[2]))
                rt_gms  += lp[3]; rt_brk  += max(0, lp[5]-lp[4])
                n += 1
    except FileNotFoundError: return None
    except Exception as e: st.error(f"BD error: {e}"); return None
    if n == 0: return None
    spw = max(10.1, sv_won/sv_pts*100 if sv_pts else 55.0)
    rpw = max(10.1, rt_won/rt_pts*100 if rt_pts else 40.0)
    return {"n": n, "hold": round(sv_held/sv_gms*100,1) if sv_gms else 0,
            "brk": round(rt_brk/rt_gms*100,1) if rt_gms else 0,
            "spw": round(spw,1), "rpw": round(rpw,1), "source": "BD Local"}

# ─────────────────────────────────────────────────────────────────────────────
# GEMINI — STATS FALLBACK
# ─────────────────────────────────────────────────────────────────────────────
def gemini_stats(name: str) -> dict:
    """Siempre devuelve un dict (nunca None) para no romper el flujo."""
    client = get_gemini_client()
    if not client:
        return {"n":1,"hold":0.0,"brk":0.0,"spw":55.0,"rpw":40.0,"source":"Sin API"}
    try:
        prompt = (
            f'Busca estadísticas de tenis recientes (últimos 12 meses) de "{name}". '
            f'Responde SOLO JSON crudo (sin backticks):\n{{"spw_pct":64.5,"rpw_pct":38.2}}'
        )
        r = client.models.generate_content(
            model=GEMINI_MODEL, contents=prompt,
            config=types.GenerateContentConfig(
                tools=[{"google_search": {}}], temperature=0.1
            )
        )
        raw = r.text.replace("```json","").replace("```","").strip()
        s, e = raw.find('{'), raw.rfind('}')
        data = json.loads(raw[s:e+1]) if s != -1 and e != -1 else {}
        spw  = max(10.1, float(data.get("spw_pct", 55.0)))
        rpw  = max(10.1, float(data.get("rpw_pct", 40.0)))
        return {"n":5,"hold":0.0,"brk":0.0,"spw":spw,"rpw":rpw,"source":"Gemini Web"}
    except Exception:
        return {"n":1,"hold":0.0,"brk":0.0,"spw":55.0,"rpw":40.0,"source":"Gemini (Fallback)"}

# ─────────────────────────────────────────────────────────────────────────────
# GEMINI — ANÁLISIS H48 + VEREDICTO
# ─────────────────────────────────────────────────────────────────────────────
def gemini_analysis(p1: str, p2: str, report: str) -> str:
    client = get_gemini_client()
    if not client: return "❌ Sin API Key."
    try:
        prompt = f"""Eres un analista cuantitativo de apuestas de tenis.

PASO 1 — CONTEXTO H48 (busca en Google):
Hechos objetivos de las últimas 48h sobre {p1} y {p2}:
- Partidos recientes (duración, esfuerzo físico)
- Lesiones, retiros o problemas físicos reportados
- Cambios de superficie o viajes largos

PASO 2 — AUDITORÍA MATEMÁTICA:
Analiza este reporte (Markov + Gaussiana + Monte Carlo):
{report}

PASO 3 — VEREDICTO (reglas estrictas):
- EV > 5% Y sin alertas físicas → ✅ VERDE — Apostar
- 0% < EV ≤ 5% O fatiga menor → ⚠️ AMARILLO — Reducir stake
- EV negativo O lesión confirmada → 🚫 ROJO — Evitar

Formato:
**CONTEXTO H48**
[viñetas con hechos o "Sin alertas detectadas"]

**AUDITORÍA**
[1-2 líneas sobre coherencia del EV]

**VEREDICTO**
[emoji + dictamen en 1 línea]"""

        r = client.models.generate_content(
            model=GEMINI_MODEL, contents=prompt,
            config=types.GenerateContentConfig(
                tools=[{"google_search": {}}], temperature=0.2
            )
        )
        return r.text
    except Exception as e:
        return f"Error análisis Gemini: {e}"

# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Quantum Tennis v6.5",
        page_icon="🎾", layout="wide"
    )
    st.title("🎾 Quantum Tennis Engine v6.5 — Gemini Quant")
    st.caption("Markov Chains · Shrinkage Bayesiano · Gaussiana Diaria · Oráculo Google Sheets")

    if not gemini_available():
        st.error("⚠️ Falta GOOGLE_API_KEY en tu archivo .env")
        st.stop()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("🌍 Variables Físicas")
        altitude_m = st.number_input("Altitud (m snm)", 0, 4000, 0, 100)
        st.divider()
        st.header("🤕 Fatiga (0–5)")
        fat_p1 = st.slider("Jugador 1", 0, 5, 0)
        fat_p2 = st.slider("Jugador 2", 0, 5, 0)
        st.divider()
        iterations = st.select_slider(
            "Iteraciones MC",
            options=[1000, 3000, 5000, 10000], value=5000
        )

    # ── Estado de sesión ──────────────────────────────────────────────────────
    if "txt"          not in st.session_state: st.session_state.txt = ""
    if "save_batch"   not in st.session_state: st.session_state.save_batch = []  # ✅ FIX: init correcto

    tab_play, tab_oracle = st.tabs(["🎾 Analizador Quant", "🔮 El Oráculo"])

    # ══════════════════════════════════════════════════════════════════════════
    with tab_play:
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1: best_of = st.radio("Formato:", [3, 5], horizontal=True, format_func=lambda x: f"Mejor de {x}")
        with c2: db_path = st.text_input("BD", value=DB_PATH, label_visibility="collapsed")
        with c3:
            if st.button("🗑️ Limpiar", use_container_width=True):
                st.session_state.txt = ""
                st.rerun()

        txt = st.text_area(
            "📋 Pega los partidos:", key="txt", height=160,
            placeholder="Alcaraz C\\nSinner J\\n-140\\n+110"
        )

        analizar = st.button("🚀 Analizar", type="primary", use_container_width=True)

        if analizar:  # ✅ FIX: sin return — el código sigue hacia tab_oracle
            if not txt.strip():
                st.warning("Pega al menos un partido.")
            else:
                partidos = parse_matches(txt)
                if not partidos:
                    st.error("No se encontraron pares jugador/cuota.")
                    with st.expander("Debug"):
                        st.code(txt)
                else:
                    st.session_state.save_batch = []

                    for p1_raw, odd1, p2_raw, odd2, league in partidos:
                        p1 = p1_raw or "Jugador 1"
                        p2 = p2_raw or "Jugador 2"

                        if "utr" in p1.lower() or "utr" in p2.lower():
                            st.warning(f"🚫 UTR excluido: {p1} vs {p2}"); continue
                        if odd1 and odd2 and (odd1 <= -500 or odd2 <= -500):
                            st.warning(f"🚫 Cuota extrema: {p1} ({odd1}) vs {p2} ({odd2})"); continue

                        st.divider()
                        st.subheader(f"⚡ {p1}  vs  {p2}")
                        st.caption(f"Liga detectada: {league}")

                        # Stats
                        col_s1, col_s2 = st.columns(2)
                        with col_s1:
                            with st.spinner(f"{p1}…"):
                                s1 = get_stats(p1, db_path) or gemini_stats(p1)
                        with col_s2:
                            with st.spinner(f"{p2}…"):
                                s2 = get_stats(p2, db_path) or gemini_stats(p2)

                        # Ajustes entorno
                        asp1, arp1 = apply_environment(s1["spw"], s1["rpw"], s1["n"], altitude_m, fat_p1)
                        asp2, arp2 = apply_environment(s2["spw"], s2["rpw"], s2["n"], altitude_m, fat_p2)

                        pA_base = log5_serve(asp1, arp2)
                        pB_base = log5_serve(asp2, arp1)

                        with st.spinner(f"Corriendo {iterations:,} simulaciones Gaussianas…"):
                            mc_A = sim_match(pA_base, pB_base, best_of, iterations, altitude_m)
                            mc_B = 1 - mc_A

                        # Resultados
                        st.markdown("#### 🧮 Motor Matemático")
                        m1, m2 = st.columns(2)

                        ev1 = ev2 = nv1 = nv2 = tier1 = tier2 = None

                        for col, name, p_srv, mc_w, odd, stats, as_spw, as_rpw, fat in [
                            (m1,p1,pA_base,mc_A,odd1,s1,asp1,arp1,fat_p1),
                            (m2,p2,pB_base,mc_B,odd2,s2,asp2,arp2,fat_p2),
                        ]:
                            col.markdown(f"**{name}**")
                            col.caption(
                                f"Fuente: {stats['source']} · n={stats['n']} · "
                                f"SPW raw {stats['spw']}% → adj {as_spw:.1f}% · "
                                f"RPW raw {stats['rpw']}% → adj {as_rpw:.1f}% · "
                                f"Fatiga: {fat}/5"
                            )
                            col.metric("P(match) Gaussiana", f"{mc_w*100:.1f}%")

                            if odd1 and odd2:
                                _nv1, _nv2, _f1, _f2 = no_vig(odd1, odd2)
                                nv_this = _nv1 if col is m1 else _nv2
                                fair    = _f1  if col is m1 else _f2
                                ev_val  = ev(mc_w, american_to_decimal(odd))
                                tier    = get_tier(mc_w, ev_val, nv_this)

                                if col is m1: ev1,nv1,tier1 = ev_val, nv_this, tier
                                else:         ev2,nv2,tier2 = ev_val, nv_this, tier

                                col.metric("P(Casa) No-Vig", f"{nv_this*100:.1f}%",
                                           f"Fair odd {fair}", delta_color="off")
                                col.metric("💰 EV Return", f"{ev_val*100:+.1f}%",
                                           delta_color="normal" if ev_val > 0 else "inverse")
                                col.metric("Edge vs Casa", f"{(mc_w-nv_this)*100:+.1f}%")
                                col.markdown(f"### {tier}")
                            else:
                                tier = get_tier(mc_w, None, None)
                                if col is m1: tier1 = tier
                                else:         tier2 = tier
                                col.markdown(f"### {tier}")

                        # Reporte
                        report = (
                            f"{p1}: SPW adj={asp1:.1f}% RPW adj={arp1:.1f}% n={s1['n']} [{s1['source']}]\n"
                            f"{p2}: SPW adj={asp2:.1f}% RPW adj={arp2:.1f}% n={s2['n']} [{s2['source']}]\n"
                            f"P(srv/punto) Log5: {p1}={pA_base:.3f} | {p2}={pB_base:.3f}\n"
                            f"Monte Carlo ({iterations} iter, Bo{best_of}): {p1}={mc_A*100:.1f}% | {p2}={mc_B*100:.1f}%\n"
                            f"EV: {p1}={ev1*100:+.2f}% | {p2}={ev2*100:+.2f}%" if ev1 is not None else ""
                        )

                        # Gemini análisis
                        st.markdown("#### 🤖 Gemini — Análisis H48 + Veredicto")
                        with st.spinner("Consultando Gemini Search…") :
                            st.markdown(gemini_analysis(p1, p2, report))

                        # Guardar batch
                        mid = f"{p1}_vs_{p2}_{datetime.now().strftime('%Y%m%d')}"
                        if ev1 is not None:
                            st.session_state.save_batch.extend([
                                (mid,         p1, p2, mc_A, nv1, odd1, ev1, tier1, league),
                                (mid+"_inv",  p2, p1, mc_B, nv2, odd2, ev2, tier2, league),
                            ])

        # Botón guardar (FUERA del if analizar)
        if st.session_state.save_batch:
            st.divider()
            n_partidos = len(st.session_state.save_batch) // 2
            if st.button(f"💾 Guardar en Oráculo ({n_partidos} partidos)",
                         type="primary", use_container_width=True):
                ok = 0
                with st.spinner("Guardando en Google Sheets…"):
                    for d in st.session_state.save_batch:
                        try:
                            res = log_prediction(*d)
                            if res: ok += 1
                            else: st.warning(f"Rechazado (retornó False): {d[0]}")
                        except Exception as ex:
                            st.error(f"Falla fatal en iteración: {ex}")
                
                if ok > 0: st.success(f"✅ {ok//2} partidos guardados en el Oráculo.")
                else:      st.info("Ya estaban guardados (sin duplicados).")
                st.session_state.save_batch = []

    # ══════════════════════════════════════════════════════════════════════════
    with tab_oracle:
        st.header("📊 El Oráculo — Dashboard de Rentabilidad")
        sheet = get_sheet()

        if not sheet:
            st.warning("⚠️ Google Sheets no conectado.")
            st.info("Agrega tu Service Account JSON a los Streamlit Secrets para activar el Oráculo.")
        else:
            data = sheet.get_all_records()
            if not data:
                st.info("Aún no hay predicciones guardadas.")
            else:
                df = pd.DataFrame(data)
                st.dataframe(df.tail(20), use_container_width=True)

                if st.button("🔎 Ejecutar Agente Liquidador"):
                    for i, row in df.iterrows():
                        if not row.get("Winner", ""):
                            with st.spinner(f"Liquidando {row.get('P1_Name', 'P1')} vs {row.get('P2_Name', 'P2')}..."):
                                w = liquidar_partido(row.get("P1_Name", ""), row.get("P2_Name", ""))
                            if w:
                                sheet.update_cell(i + 2, 10, w)
                                st.success(f"Ganador: {w}")
                    st.success("Auditoría finalizada.")
                    st.rerun()

if __name__ == "__main__":
    main()
