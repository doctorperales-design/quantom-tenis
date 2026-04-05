"""
Quantum Tennis Engine v6.0 — Gemini Quant Edition
Solo Gemini + Motor Matemático Analítico (Cadenas de Markov O(1) + Encogimiento Bayesiano + Física)
"""

import streamlit as st
import json
import os
import re
import random
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
GEMINI_MODEL  = "gemini-2.5-pro"
DB_PATH       = "matches_jsonl.jsonl"
MC_ITERATIONS = 10000

METADATA_KW = {
    'local', 'visita', 'empate', 'sencillos', 'dobles', 'vivo', 'apuestas',
    'streaming', 'women', 'men', 'tour', 'challenger', 'atp', 'wta', 'itf',
    'world tennis', 'grand slam', 'futures', 'copa', 'qualifier', 'qualy',
    'hoy', 'mañana', 'lunes', 'martes', 'miércoles', 'miercoles', 'jueves', 
    'viernes', 'sábado', 'sabado', 'domingo', 'ene', 'feb', 'mar', 'abr', 
    'may', 'jun', 'jul', 'ago', 'sep', 'oct', 'nov', 'dic', 'vs'
}

# ─────────────────────────────────────────────────────────────────────────────
# CLIENTE GEMINI (singleton)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_gemini_client():
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        return None
    return genai.Client(api_key=api_key)

def gemini_available() -> bool:
    return get_gemini_client() is not None

# ─────────────────────────────────────────────────────────────────────────────
# PARSER DE LÍNEAS DE SPORTSBOOK
# ─────────────────────────────────────────────────────────────────────────────
def extract_american_odds(text: str):
    m = re.search(r'([+-]\d{3,4})', text)
    if m:
        return text.replace(m.group(1), '').strip(), int(m.group(1))
    return text.strip(), None

def parse_matches(text: str) -> list[tuple]:
    text = text.replace('–', '-').replace('—', '-').replace('−', '-')
    results = []
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
                    results.append((p1, o1, p2, o2))
        if results:
            return results

    DATE_TIME_RE = re.compile(r'^(\d{1,2}\s+[a-zA-Z]{3}\s+)?\d{2}:\d{2}$')
    COUNTER_RE = re.compile(r'^\+\s*\d{1,2}\s*(Streaming)?$', re.I)
    ODD_LONE   = re.compile(r'^[+-]\d{3,4}$')

    name_q, odd_q = [], []

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        
        # Búsqueda de palabra exacta para no ignorar nombres verdaderos (ej. "Mendez" contiene "men")
        if any(re.search(rf'\b{re.escape(kw)}\b', line.lower()) for kw in METADATA_KW):
            continue
        if DATE_TIME_RE.match(line) or COUNTER_RE.match(line):
            continue

        odds_in = re.findall(r'[+-]\d{3,4}', line)

        if ODD_LONE.match(line):
            odd_q.append(int(line))
        elif odds_in:
            for o in odds_in:
                odd_q.append(int(o))
            name_part = re.sub(r'[+-]\d{3,4}.*', '', line).strip()
            if name_part:
                name_q.append(name_part)
        else:
            name_q.append(line)

        while len(name_q) >= 2 and len(odd_q) >= 2:
            results.append((name_q.pop(0), odd_q.pop(0),
                            name_q.pop(0), odd_q.pop(0)))

    return results

# ─────────────────────────────────────────────────────────────────────────────
# FÍSICA Y ENTORNO
# ─────────────────────────────────────────────────────────────────────────────
def apply_realism_and_environment(spw, rpw, matches_found, altitude_m, fatigue_idx):
    """
    1. Shrinkage Bayesiano para enfriar sesgo de optimismo (menos datos = más regresión al promedio)
    2. Modificadores físicos elementales.
    """
    ATP_WTA_AVG_SPW = 62.0 
    ATP_WTA_AVG_RPW = 38.0
    
    # Regresión Bayesiana (Afinando sin ahogar torneos ITF o Challengers):
    # La máxima confianza se logra desde los 15 partidos (justo para no castigar circuitos menores).
    # El 'castigo' MÁXIMO contra jugadores fantasma será del 35%, protegiendo SIEMPRE
    # al menos el 65% de la identidad del jugador para permitir ROI alto (70%-80%).
    confidence = min(matches_found / 15.0, 1.0) if matches_found > 0 else 0.0
    shrink_factor = 0.35 * (1.0 - confidence)
    
    adj_spw = (spw * (1 - shrink_factor)) + (ATP_WTA_AVG_SPW * shrink_factor)
    adj_rpw = (rpw * (1 - shrink_factor)) + (ATP_WTA_AVG_RPW * shrink_factor)
    
    # Altitud: Favorece fuertemente al saque.
    alt_bonus = (altitude_m / 1000.0) * 1.5
    adj_spw += alt_bonus
    adj_rpw -= alt_bonus * 0.5 
    
    # Fatiga (Escala 0-5)
    adj_spw -= fatigue_idx * 0.5
    adj_rpw -= fatigue_idx * 1.5
    
    return min(max(adj_spw, 30.0), 90.0), min(max(adj_rpw, 10.0), 70.0)

# ─────────────────────────────────────────────────────────────────────────────
# MATEMÁTICA ANALÍTICA Y MARKOV CHAINS
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

def log5_serve(spw: float, rpw: float) -> float:
    A, B = spw / 100.0, rpw / 100.0
    d = A * (1 - B) + (1 - A) * B
    return (A * (1 - B)) / d if d else 0.5

def calc_game_win_prob(p: float) -> float:
    """ Solución Analítica O(1) vía Cadenas de Markov para ganar 1 Game. """
    q = 1.0 - p
    p_win_in_4 = p**4
    p_win_in_5 = 4 * (p**4) * q
    p_win_in_6 = 10 * (p**4) * (q**2)
    p_reach_deuce = 20 * (p**3) * (q**3)
    p_win_from_deuce = (p**2) / (p**2 + q**2) if (p**2 + q**2) > 0 else 0
    return p_win_in_4 + p_win_in_5 + p_win_in_6 + (p_reach_deuce * p_win_from_deuce)

def simulate_tiebreak_fast(pA: float, pB: float) -> int:
    ptsA, ptsB = 0, 0
    turn = 0
    while True:
        if turn % 4 in [0, 3]:
            if random.random() < pA: ptsA += 1
            else: ptsB += 1
        else:
            if random.random() < pB: ptsB += 1
            else: ptsA += 1
        if ptsA >= 7 and ptsA - ptsB >= 2: return 1
        if ptsB >= 7 and ptsB - ptsA >= 2: return 0
        turn += 1

def simulate_set_markov(pGameA, pGameB, pPointA, pPointB):
    gamesA, gamesB = 0, 0
    serve_turn = 0 # 0=A saca, 1=B saca
    while True:
        if gamesA == 6 and gamesB == 6:
            return simulate_tiebreak_fast(pPointA, pPointB)
        
        if serve_turn == 0:
            if random.random() < pGameA: gamesA += 1
            else: gamesB += 1
            serve_turn = 1
        else:
            if random.random() < pGameB: gamesB += 1
            else: gamesA += 1
            serve_turn = 0
            
        if gamesA >= 6 and gamesA - gamesB >= 2: return 1
        if gamesB >= 6 and gamesB - gamesA >= 2: return 0
        if gamesA == 7: return 1
        if gamesB == 7: return 0

def simulate_match_quantum(p_serve_A_base, p_serve_B_base, best_of=3, iterations=MC_ITERATIONS, altitude_m=0):
    sets_to_win = 2 if best_of == 3 else 3
    A_wins = 0
    
    # Mayor altitud = Mayor varianza estocástica
    variance_mult = 1.0 + (altitude_m / 2000.0)
    base_std_dev = 0.015 * variance_mult
    
    for _ in range(iterations):
        # Gaussiana Diaria (Mitiga "Optimismo Robotico")
        pA_diario = max(0.35, min(0.95, random.gauss(p_serve_A_base, base_std_dev)))
        pB_diario = max(0.35, min(0.95, random.gauss(p_serve_B_base, base_std_dev)))
        
        pGameA = calc_game_win_prob(pA_diario)
        pGameB = calc_game_win_prob(pB_diario)
        
        setsA, setsB = 0, 0
        while setsA < sets_to_win and setsB < sets_to_win:
            if simulate_set_markov(pGameA, pGameB, pA_diario, pB_diario): setsA += 1
            else: setsB += 1
            
        if setsA == sets_to_win: A_wins += 1
        
    return A_wins / iterations

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
                if name.lower() not in line.lower():
                    continue
                m = json.loads(line)
                wn, ln = m[2], m[4]

                as_w = name.lower() in wn.lower()
                as_l = name.lower() in ln.lower()
                if not as_w and not as_l: continue

                if as_w:
                    wp = (m[6], m[8], m[9], m[10], m[11], m[12])
                    lp = (m[13], m[15], m[16], m[17], m[18], m[19])
                else:
                    wp = (m[13], m[15], m[16], m[17], m[18], m[19])
                    lp = (m[6], m[8], m[9], m[10], m[11], m[12])

                sv_pts  += wp[0]
                sv_won  += wp[1] + wp[2]
                sv_gms  += wp[3]
                sv_held += max(0, wp[3] - (wp[5] - wp[4]))
                rt_pts  += lp[0]
                rt_won  += max(0, lp[0] - (lp[1] + lp[2]))
                rt_gms  += lp[3]
                rt_brk  += max(0, lp[5] - lp[4])
                n += 1

    except FileNotFoundError: return None
    except Exception as e:
        st.error(f"BD error: {e}")
        return None

    if n == 0: return None

    spw = sv_won / sv_pts * 100 if sv_pts else 55.0
    rpw = rt_won / rt_pts * 100 if rt_pts else 40.0
    if spw <= 10: spw = 55.0
    if rpw <= 10: rpw = 40.0

    return {
        "n":       n,
        "hold":    round(sv_held / sv_gms * 100, 1) if sv_gms else 0,
        "brk":     round(rt_brk  / rt_gms * 100, 1) if rt_gms else 0,
        "spw":     round(spw, 1),
        "rpw":     round(rpw, 1),
        "source":  "BD Local",
    }

# ─────────────────────────────────────────────────────────────────────────────
# GEMINI — STATS FALLBACK
# ─────────────────────────────────────────────────────────────────────────────
def gemini_stats(name: str) -> dict | None:
    client = get_gemini_client()
    if not client: return None
    try:
        prompt = f"""Busca estadísticas de tenis recientes (últimos 12 meses) de este jugador: "{name}"
Manda ÚNICAMENTE un JSON crudo (sin backticks) con:
{{"spw_pct": 65.5, "rpw_pct": 39.2}}"""

        r = client.models.generate_content(
            model=GEMINI_MODEL, contents=prompt,
            config=types.GenerateContentConfig(tools=[{"google_search": {}}], temperature=0.1)
        )
        raw  = r.text.replace("```json", "").replace("```", "").strip()
        
        # Extracción segura del JSON si Gemini decide agregar texto extra o alucinar
        start = raw.find('{')
        end = raw.rfind('}')
        if start != -1 and end != -1:
            raw = raw[start:end+1]
            
        data = json.loads(raw)
        spw  = max(10.1, float(data.get("spw_pct", 55.0)))
        rpw  = max(10.1, float(data.get("rpw_pct", 40.0)))
        return {"n": 5, "hold": 0.0, "brk": 0.0,
                "spw": spw, "rpw": rpw, "source": "Gemini Web"} 
    except Exception:
        # Blindaje anti-crashes. Si Gemini falla (ej. sin resultados o desconectado),
        # inyectamos el perfil 'fantasma' asumiendo N=1 para que el Shrinkage Bayesiano lo castigue al promedio ATP.
        return {"n": 1, "hold": 0.0, "brk": 0.0, "spw": 55.0, "rpw": 40.0, "source": "Gemini (Error Genérico)"}

# ─────────────────────────────────────────────────────────────────────────────
# GEMINI — ANÁLISIS H48 UNIFICADO
# ─────────────────────────────────────────────────────────────────────────────
def gemini_full_analysis(p1: str, p2: str, report: str, alt_m: int) -> str:
    client = get_gemini_client()
    if not client: return "❌ GOOGLE_API_KEY no configurada."
    try:
        prompt = f"""ERES UN MODELO EXCLUSIVO DE GOOGLE SEARCH.

CONTEXTO: Juegan {p1} vs {p2} (Altitud: {alt_m} metros).
MATEMÁTICA INTERNA YA PROCESADA:
{report}

PASO 1 BÚSQUEDA WEB: Busca hechos objetivos de las últimas 48h (Fatiga por maratones 3sets, Medical TimeOuts recientes, declaraciones).
PASO 2 VEREDICTO DE APUESTAS (ESTRICTO): 
- Si Expected Value es mayor a 5% Y no hay ninguna lesión grave: ✅ VERDE APOSTAR.
- Si Expected Value es marginal (0 a 5%) O hay mucha fatiga: ⚠️ AMARILLO PRECAUCIÓN.
- Si EV es negativo: 🚫 ROJO EVITAR.

Responde únicamente con "Contexto Encontrado" (viñetas cortas) y tu "Veredicto Binario"."""
        r = client.models.generate_content(
            model=GEMINI_MODEL, contents=prompt,
            config=types.GenerateContentConfig(tools=[{"google_search": {}}], temperature=0.2)
        )
        return r.text
    except Exception as e: return f"Error Gemini: {e}"

# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Quantum Tennis v6", page_icon="🎾", layout="wide")

    st.title("🎾 Quantum Tennis Engine v6.0")
    st.caption("Markov Chains Analíticas · Corrección Bayesiana Gaussiana · Gemini 2.5 Pro")

    if not gemini_available():
        st.error("⚠️ Falta GOOGLE_API_KEY en tu archivo .env")
        st.stop()

    with st.sidebar:
        st.header("🌍 Variables Físicas")
        altitude_m = st.number_input("Altitud (m snm)", min_value=0, max_value=4000, value=0, step=100, help="A mayor altitud, menos densidad aerodinámica (+% Saque, -% Resto). Genera escenarios de Tie-break impredecibles.")
        st.divider()
        st.header("🤕 Índice Físico (Fatiga)")
        st.caption("Penaliza severamente los reflejos al Resto")
        fat_p1 = st.slider("Fatiga Jugador 1 (0=Fresco, 5=Exhausto)", 0, 5, 0)
        fat_p2 = st.slider("Fatiga Jugador 2 (0=Fresco, 5=Exhausto)", 0, 5, 0)

    if "txt" not in st.session_state: st.session_state.txt = ""

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1: best_of = st.radio("Formato:", [3, 5], horizontal=True, format_func=lambda x: f"Mejor de {x}")
    with c2: db_path = st.text_input("BD", value=DB_PATH, label_visibility="collapsed")
    with c3:
        if st.button("🗑️ Limpiar", use_container_width=True):
            st.session_state.txt = ""
            st.rerun()

    txt = st.text_area("📋 Pega los partidos:", key="txt", height=160)

    if not st.button("🚀 Analizar M5 (Markov)", type="primary", use_container_width=True): return
    if not txt.strip(): return

    partidos = parse_matches(txt)
    if not partidos:
        st.error("Error de Parsing. Verifica el texto.")
        return

    for p1_raw, odd1, p2_raw, odd2 in partidos:
        p1 = p1_raw or "Jugador 1"
        p2 = p2_raw or "Jugador 2"

        if "utr" in p1.lower() or "utr" in p2.lower(): continue
        if odd1 and odd2 and (odd1 <= -500 or odd2 <= -500): continue

        st.divider()
        st.subheader(f"⚡ {p1}  vs  {p2}")

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            s1 = get_stats(p1, db_path) or gemini_stats(p1)
        with col_s2:
            s2 = get_stats(p2, db_path) or gemini_stats(p2)

        if not s1 or not s2:
            st.error("Stats insuficientes.")
            continue

        # 1. Enfriamiento Bayesiano y Ajuste Físico
        adj_spw1, adj_rpw1 = apply_realism_and_environment(s1["spw"], s1["rpw"], s1["n"], altitude_m, fat_p1)
        adj_spw2, adj_rpw2 = apply_realism_and_environment(s2["spw"], s2["rpw"], s2["n"], altitude_m, fat_p2)

        # 2. Base Log5 Serve (antes de ruido Gaussiano)
        pA_base = log5_serve(adj_spw1, adj_rpw2)
        pB_base = log5_serve(adj_spw2, adj_rpw1)

        with st.spinner(f"Compilando {MC_ITERATIONS:,} Redes de Markov estocásticas…"):
            mc_A = simulate_match_quantum(pA_base, pB_base, best_of, altitude_m=altitude_m)
            mc_B = 1.0 - mc_A

        st.markdown("#### 🧮 Motor Matemático (Markov + Bayes)")
        m1, m2 = st.columns(2)
        pairs = [(m1, p1, pA_base, mc_A, odd1, s1, adj_spw1, adj_rpw1), (m2, p2, pB_base, mc_B, odd2, s2, adj_spw2, adj_rpw2)]

        ev_lines = []
        for col, name, p_srv, mc_w, odd, stats, as_spw, as_rpw in pairs:
            col.markdown(f"**{name}**")
            col.caption(f"Stats Crudas ({stats['source']}, n={stats['n']}): SPW {stats['spw']}% / RPW {stats['rpw']}%")
            col.caption(f"🧬 Física Corregida: SPW {as_spw:.1f}% / RPW {as_rpw:.1f}%")
            col.metric("P(srv) Markov Base", f"{p_srv*100:.1f}%")
            col.metric("P(match) Gaussiana", f"{mc_w*100:.1f}%")

            if odd:
                nv1, nv2, f1, f2 = no_vig(odd1 or 100, odd2 or 100)
                nv_this = nv1 if col is m1 else nv2
                fair    = f1  if col is m1 else f2
                ev_val  = ev(mc_w, american_to_decimal(odd))
                edge    = (mc_w - nv_this) * 100
                col.metric("P(Casa) No-Vig", f"{nv_this*100:.1f}%", f"Fair: {fair}", delta_color="off")
                col.metric("💰 EV Return", f"{ev_val*100:+.1f}%", delta_color="normal" if ev_val > 0 else "inverse")
                ev_lines.append(f"{name}: EV={ev_val*100:+.2f}%")

        report = f"""
{p1} Corregido: SPW={adj_spw1:.1f}% RPW={adj_rpw1:.1f}% | P(Srv)={pA_base:.3f}
{p2} Corregido: SPW={adj_spw2:.1f}% RPW={adj_rpw2:.1f}% | P(Srv)={pB_base:.3f}
Gaussiana Fin: {mc_A*100:.1f}% vs {mc_B*100:.1f}%. Altitud={altitude_m}m.
{chr(10).join(ev_lines) if ev_lines else 'Sin cuotas.'}
"""

        st.markdown("#### 🤖 Gemini 2.5 Pro — Juez de Entorno H48")
        with st.spinner("Conectando Search Grounding con Evaluador Final…"):
            analisis = gemini_full_analysis(p1, p2, report, altitude_m)
            st.info(analisis)

if __name__ == "__main__":
    main()
