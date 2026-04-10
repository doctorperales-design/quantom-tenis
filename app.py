"""
Quantum Tennis Engine v14 — Nivel Dios (Deep Autonomous Profiling)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Gemini 2.5 Pro  →  Extracción masiva de Perfiles (Altura/Mano) + Auditoría H48
Motor Python    →  Monte Carlo NumPy (vectorizado masivo)
Ajustes v3      →  Autónomos (Inyección Directa en SPW/RPW) SIN sliders
Bankroll        →  Kelly Fraccionado (25%) · Vig %
Oracle          →  Google Sheets
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

# ─────────────────────────────────────────────────────────────────────────────
# CLIENTES GEMINI
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
        if "gcp_service_account" not in st.secrets: return None
        gc = gspread.service_account_from_dict(st.secrets["gcp_service_account"])
        return gc.open_by_key("1kciFhxjiVOeScsu_7e6UZvJ36ungHyeQxjIWMBu5CYs").sheet1
    except Exception as e:
        return None

def log_prediction(match_id, p1, p2, p_mod, p_casa, odd, ev_val, tier, league):
    sheet = get_sheet()
    if not sheet: return False
    try:
        ids = sheet.col_values(1)
        if match_id in ids: return False
        sheet.append_row([
            match_id, datetime.now().strftime("%Y-%m-%d %H:%M"),
            p1, p2, f"{p_mod*100:.1f}%", f"{p_casa*100:.1f}%" if p_casa else "S/C",
            str(odd) if odd else "S/C", f"{ev_val*100:+.1f}%" if ev_val is not None else "S/C",
            tier, "", league
        ])
        return True
    except Exception: return False

def liquidar_partido(p1: str, p2: str) -> str:
    client = get_gemini_client()
    if not client: return ""
    try:
        prompt = (f"Busca en Google si ya terminó '{p1}' vs '{p2}'.\n"
                  f"Si terminó → responde ÚNICAMENTE el nombre del ganador.\n"
                  f"Si no → responde PENDIENTE")
        r = client.models.generate_content(
            model=GEMINI_MODEL, contents=prompt,
            config=types.GenerateContentConfig(tools=[{"google_search": {}}], temperature=0.0)
        )
        t = r.text.upper().strip()
        if "PENDIENTE" in t: return ""
        if p1.upper().split()[-1] in t: return p1
        if p2.upper().split()[-1] in t: return p2
        return ""
    except Exception: return ""

# ─────────────────────────────────────────────────────────────────────────────
# PARSER ROBUSTO
# ─────────────────────────────────────────────────────────────────────────────
def extract_american_odds(text: str):
    m = re.search(r'([+-]\d{3,4})', text)
    if m: return text.replace(m.group(1), '').strip(), int(m.group(1))
    return text.strip(), None

def parse_matches(text: str) -> list[tuple]:
    text = text.replace('–', '-').replace('—', '-').replace('−', '-')
    results = []; cur_league = "ATP/WTA"

    if re.search(r'(?i)\bvs\.?\b', text):
        for line in text.splitlines():
            line = line.strip()
            if not re.search(r'(?i)\bvs\.?\b', line): continue
            parts = re.split(r'(?i)\s+vs\.?\s+', line)
            if len(parts) == 2:
                p1, o1 = extract_american_odds(parts[0])
                p2, o2 = extract_american_odds(parts[1])
                if p1 and p2: results.append((p1, o1, p2, o2, cur_league))
        if results: return results

    DATE_RE    = re.compile(r'^(\d{1,2}\s+[a-zA-Z]{3}\s+)?\d{2}:\d{2}$')
    COUNTER_RE = re.compile(r'^\+\s*\d{1,2}\s*(Streaming)?$', re.I)
    name_q, odd_q = [], []

    for raw in text.splitlines():
        line = raw.strip()
        if not line: continue
        odds_match = re.findall(r'[+-]\d{3,4}', line)
        if odds_match:
            for o in odds_match: odd_q.append(int(o))
            line = re.sub(r'[+-]\d{3,4}.*', '', line).strip()
            if not line: continue
        low = line.lower()
        if re.search(r'\b(?:' + '|'.join(map(re.escape, METADATA_KW)) + r')\b', low):
            if "itf" in low or "world tennis" in low: cur_league = "ITF"
            elif "challenger" in low: cur_league = "Challenger"
            elif "atp" in low: cur_league = "ATP"
            elif "wta" in low or "women" in low: cur_league = "WTA"
            continue
        if DATE_RE.match(line) or COUNTER_RE.match(line): continue
        name_q.append(line)
        while len(name_q) >= 2 and len(odd_q) >= 2:
            results.append((name_q.pop(0), odd_q.pop(0), name_q.pop(0), odd_q.pop(0), cur_league))

    while len(name_q) >= 2 and len(odd_q) >= 2:
        results.append((name_q.pop(0), odd_q.pop(0), name_q.pop(0), odd_q.pop(0), cur_league))
    return results

# ─────────────────────────────────────────────────────────────────────────────
# GEMINI — DEEP AUTONOMOUS PROFILING (Lotes)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_autonomous_profiles(players: list[str]) -> dict:
    """Extrae el perfil físico (Zurdo?, Altura) de toda la cuadra de jugadores a la vez."""
    if not players: return {}
    client = get_gemini_client()
    if not client: return {}
    
    unique_players = list(set(players))
    players_str = "\n".join(f"- {p}" for p in unique_players)
    
    prompt = f"""Devuelve estrictamente un JSON válido.
Busca en tu base de datos o en internet la ESTATURA en cm (ht) y la MANO DOMINANTE ('L' para Zurdo, 'R' para Diestro) de:
{players_str}

Si no aparece información de algún jugador, asume "R" y 183.
Formato de respuesta:
{{
  "Carlos Alcaraz": {{"hand": "R", "ht": 183}},
  "Rafael Nadal": {{"hand": "L", "ht": 185}}
}}"""
    try:
        r = client.models.generate_content(
            model=GEMINI_MODEL, contents=prompt,
            config=types.GenerateContentConfig(tools=[{"google_search": {}}], temperature=0.1)
        )
        raw = r.text.replace("```json","").replace("```","").strip()
        s, e = raw.find('{'), raw.rfind('}')
        data = json.loads(raw[s:e+1]) if s != -1 and e != -1 else {}
        # Normalizar claves para perdonar mayúsculas/minúsculas
        return {k.lower().strip(): v for k, v in data.items()}
    except Exception:
        return {}

def get_player_profile(name: str, profiles_dict: dict) -> tuple[bool, int]:
    """Busca al jugador en el dict autónomo. Retorna (es_zurdo, altura)."""
    norm_name = name.lower().strip()
    # Buscar coincidencia parcial si el nombre varía ("Alcaraz C" vs "Carlos Alcaraz")
    for k, v in profiles_dict.items():
        if k in norm_name or norm_name in k or norm_name.split()[-1] in k:
            return (v.get("hand", "R") == "L", int(v.get("ht", 183)))
    return (False, 183)

# ─────────────────────────────────────────────────────────────────────────────
# CALIBRACIÓN Y AJUSTES NUMÉRICOS
# ─────────────────────────────────────────────────────────────────────────────
CALIBRATION = {
    ("Challenger", "SD"): 0.82, ("Challenger", "D"): 0.80, ("Challenger", "FR"): 0.90,
    ("ATP/WTA", "SD"): 0.94, ("ATP", "SD"): 0.96, ("WTA", "SD"): 0.96,
    ("ITF", "SD"): 1.00, ("Challenger", "PY"): 0.88,
}
def _liga_key(liga: str) -> str:
    lg = (liga or "").upper()
    if "CHALLENGER" in lg: return "Challenger"
    if "ITF" in lg: return "ITF"
    if lg in ("ATP", "WTA"): return lg
    return "ATP/WTA"

def _tier_key(t: str) -> str:
    t = t.upper()
    if "FRANC" in t: return "FR"
    if "SUPER" in t or "SÚPER" in t: return "SD"
    if "DERECHA" in t: return "D"
    if "PARLAY" in t or "VALOR" in t: return "PY"
    return "BASURA"

def apply_calibration(pmod_raw: float, liga: str, tier: str) -> float:
    factor = CALIBRATION.get((_liga_key(liga), _tier_key(tier)), 1.0)
    return min(max(0.5 + (pmod_raw - 0.5) * factor, 0.05), 0.95)

def calcular_sum_adj(rival_zurdo: bool, indoor: bool, alt_cm: int) -> tuple[float, list[str]]:
    tot, notas = 0.0, []
    if rival_zurdo:
        tot -= 0.07; notas.append("🛡️ Rival Zurdo detectado (-0.07)")
    if indoor and alt_cm > 193:
        tot += 0.05; notas.append(f"🗼 Altura Indoor detectada ({alt_cm}cm) → (+0.05)")
    return tot, notas

def apply_environment(spw: float, rpw: float, n: int,
                      alt_m: int, fat_h48: int, surface: str) -> tuple[float, float]:
    conf = min(n/15.0, 1.0) if n>0 else 0.0; shrink = 0.35*(1.0-conf)
    a_spw = spw*(1-shrink) + 62.0*shrink; a_rpw = rpw*(1-shrink) + 38.0*shrink
    sv = SURFACE_ADJ.get(surface.lower(), SURFACE_ADJ["hard"])
    a_spw += sv["spw"] + (alt_m/1000.0)*1.5 - fat_h48*0.5
    a_rpw += sv["rpw"] - (alt_m/1000.0)*0.75 - fat_h48*1.5
    return min(max(a_spw, 30.0), 90.0), min(max(a_rpw, 10.0), 70.0)

# ─────────────────────────────────────────────────────────────────────────────
# MATEMÁTICA Y MONTE CARLO
# ─────────────────────────────────────────────────────────────────────────────
def american_to_decimal(odd: int) -> float: return (odd/100+1) if odd>0 else (100/abs(odd)+1)
def no_vig(o1: int, o2: int):
    i1=1/american_to_decimal(o1); i2=1/american_to_decimal(o2); t=i1+i2
    return i1/t, i2/t, round(1/(i1/t),3), round(1/(i2/t),3)
def vig_pct(o1, o2): return round((1/american_to_decimal(o1) + 1/american_to_decimal(o2) - 1)*100,2)
def ev(p: float, dec: float): return p*(dec-1) - (1-p)
def kelly_fraction(prob: float, dec: float, f=0.25):
    b = dec-1; return max(0.0, round(((b*prob - (1-prob))/b)*f*100,2))

def get_tier(p_mod: float, ev_val: float | None, p_casa: float | None) -> str:
    p = p_mod*100
    if p_casa is None or ev_val is None: return "🟢 DERECHA" if p>=65 else ("🟡 VALOR" if p>=50 else "🔴 BASURA")
    ev_p, pc_p = ev_val*100, p_casa*100
    if p < 45 or (p < pc_p and ev_p <= 0): return "🔴 BASURA"
    if p >= 70 and ev_p >= 2 and p > pc_p: return "🔥 SÚPER DERECHA"
    if 60 <= p < 70 and ev_p >= 3: return "🟢 DERECHA"
    if 50 <= p < 60 and pc_p < 45: return "🎯 FRANCOTIRADOR"
    if 45 <= p < 60 and ev_p >= 8: return "🟡 VALOR / PARLAY"
    return "🔴 BASURA"

def log5_serve(spw: float, rpw: float): return (spw/100*(1-rpw/100))/(spw/100*(1-rpw/100)+(1-spw/100)*rpw/100)

def _game_wp_np(p: np.ndarray) -> np.ndarray:
    q=1-p; pfd=np.where(p**2+q**2>0, p**2/(p**2+q**2), 0.5)
    return p**4 + 4*p**4*q + 10*p**4*q**2 + 20*p**3*q**3*pfd

def sim_match(pA_b: float, pB_b: float, bo: int=3, iters: int=5000, alt: int=0) -> float:
    std = 0.015*(1.0 + alt/2000.0); nd = 2 if bo==3 else 3
    rng = np.random.default_rng()
    pA_A = np.clip(rng.normal(pA_b, std, iters), 0.35, 0.95); pB_A = np.clip(rng.normal(pB_b, std, iters), 0.35, 0.95)
    gA_A = _game_wp_np(pA_A); gB_A = _game_wp_np(pB_A)
    wins = 0
    for i in range(iters):
        sA=sB=0; srv=True; pA,pB,gA,gB = pA_A[i],pB_A[i],gA_A[i],gB_A[i]
        while sA<nd and sB<nd:
            ga=gb=0
            while True:
                if ga==6 and gb==6:
                    a=b=t=0
                    while True:
                        if random.random() < ((pA if srv else pB) if t%4 in [0,3] else (pB if srv else pA)): a+=1
                        else: b+=1
                        t+=1
                        if a>=7 and a-b>=2: {ga:=ga+1}; break
                        if b>=7 and b-a>=2: {gb:=gb+1}; break
                    if ga>gb: sA+=1
                    else: sB+=1
                    srv = not srv; break
                if random.random() < (gA if srv else gB): ga+=1
                else: gb+=1
                srv = not srv
                if (ga>=6 and ga-gb>=2) or ga==7: sA+=1; break
                if (gb>=6 and gb-ga>=2) or gb==7: sB+=1; break
        if sA==nd: wins+=1
    return wins/iters

# ─────────────────────────────────────────────────────────────────────────────
# DB LOCAL & FALLBACK
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_stats(name: str) -> dict | None:
    pts=won=gms=hld=r_pts=r_won=r_gms=r_brk=n = 0
    try:
        with open(DB_PATH, encoding='utf-8') as f:
            for l in f:
                if name.lower() not in l.lower(): continue
                m = json.loads(l)
                aw=name.lower() in m[2].lower(); al=name.lower() in m[4].lower()
                if not aw and not al: continue
                wp = (m[6],m[8],m[9],m[10],m[11],m[12]) if aw else (m[13],m[15],m[16],m[17],m[18],m[19])
                lp = (m[13],m[15],m[16],m[17],m[18],m[19]) if aw else (m[6],m[8],m[9],m[10],m[11],m[12])
                pts+=wp[0]; won+=wp[1]+wp[2]; gms+=wp[3]; hld+=max(0, wp[3]-(wp[5]-wp[4]))
                r_pts+=lp[0]; r_won+=max(0, lp[0]-(lp[1]+lp[2])); r_gms+=lp[3]; r_brk+=max(0, lp[5]-lp[4]); n+=1
    except: return None
    if n==0: return None
    return {"n":n, "spw":round(max(10.1,(won/pts*100) if pts else 55),1), "rpw":round(max(10.1,(r_won/r_pts*100) if r_pts else 40),1), "source":"BD Local"}

def gemini_stats(name: str, surface: str) -> dict:
    c = get_gemini_client()
    if not c: return {"n":1,"spw":55.0,"rpw":40.0,"source":"Sin API"}
    try:
        r = c.models.generate_content(
            model=GEMINI_MODEL, contents=f'{"{spw:64.5, rpw:38.2}"} de {name} en {surface} ultimos meses. SOLO JSON CRUDO.',
            config=types.GenerateContentConfig(tools=[{"google_search": {}}], temperature=0.1)
        )
        d = json.loads(re.search(r'\{.*?\}', r.text.replace('\n', '')).group(0))
        return {"n":5,"spw":float(d.get("spw", 55.0)),"rpw":float(d.get("rpw", 40.0)),"source":"Gemini Web"}
    except: return {"n":1,"spw":55.0,"rpw":40.0,"source":"Gemini Fallback"}

# ─────────────────────────────────────────────────────────────────────────────
# AUDITORÍA PROFUNDA H48
# ─────────────────────────────────────────────────────────────────────────────
def gemini_analysis(p1: str, p2: str) -> str:
    c = get_gemini_client()
    if not c: return "❌ Sin API."
    prompt = f"""Quantum Analyst v14. 
Paso 1: BUSCA RÁPIDO en Google noticias de última hora sobre {p1} y {p2}.
Paso 2: REPORTA ÚNICAMENTE si encuentras lesiones, retiros recientes o fatiga extrema demostrable (partidos larguísimos ayer).
Paso 3: Si no hay noticias graves de lesiones físicas, di explícitamente "🟢 Sin Alertas Físicas (Camino Libre)". 
No intentes analizar los números o probabilidades de cómo será el partido. Limítate a buscar estado médico/físico de las últimas 48 hrs."""
    try:
        r = c.models.generate_content(model=GEMINI_MODEL, contents=prompt, config=types.GenerateContentConfig(tools=[{"google_search": {}}], temperature=0.1))
        return r.text
    except Exception as e: return f"Error H48: {e}"

# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Quantum v14 Autónomo", page_icon="🎾", layout="wide")
    st.title("🎾 Quantum Tennis Engine v14 (Dios)")
    st.caption("Deep Autonomous Profiling · NumPy Vectorizado · Kelly Fraction")
    
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        [data-testid="stMetricValue"] { font-size: 1.8rem !important; font-weight: 800 !important; }
        [data-testid="stMetricDelta"] { font-size: 1.1rem !important; font-weight: 600 !important; padding: 2px 6px; border-radius: 4px; background: rgba(255,255,255,0.05); }
        .tier-badge { display: inline-block; padding: 8px 16px; border-radius: 8px; font-weight: 800; font-size: 1.05rem; text-align: center; width: 100%; margin-top: 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.4); text-transform: uppercase; letter-spacing: 0.5px; border: 1px solid rgba(255,255,255,0.1); }
        .tr { background:linear-gradient(135deg,#4a0000,#220000); color:#ff6666; border-color:#ff3333; }
        .ty { background:linear-gradient(135deg,#4a3800,#221a00); color:#ffcc00; border-color:#ffaa00; }
        .tg { background:linear-gradient(135deg,#004a11,#002208); color:#44ff66; border-color:#00ff33; box-shadow:0 0 15px rgba(0,255,51,0.2); }
        .tf { background:linear-gradient(135deg,#002b4a,#001222); color:#44aaff; border-color:#0088ff; }
        .ts { background:linear-gradient(135deg,#4a004a,#220022); color:#ff44ff; border-color:#aa00ff; box-shadow:0 0 20px rgba(170,0,255,0.5); }
        button[kind="primary"] { background:linear-gradient(135deg,#0d965e,#086d42) !important; box-shadow:0 4px 15px rgba(13,150,94,0.4) !important; border:none !important; font-weight:800 !important; letter-spacing:1px; border-radius:8px !important; }
        .stTextArea textarea { background-color:#111 !important; border:1px solid #333 !important; color:#00ff9d !important; font-family:monospace; font-size:15px !important; border-radius:8px !important; }
        .kelly { background:rgba(13,150,94,0.15); border:1px solid #0d965e; border-radius:8px; padding:8px 14px; font-weight:700; font-size:1rem; }
        </style>
    """, unsafe_allow_html=True)

    if not gemini_available(): st.error("⚠️ Falta GOOGLE_API_KEY"); st.stop()

    with st.sidebar:
        st.header("🌍 Ecosistema General")
        st.info("🤖 **Autonomía Activa**: Gemini buscará manos dominantes (Zr/Dr) y estaturas automáticamente.")
        alt_m = st.number_input("Altitud (msnm)", 0, 4000, 0, 100)
        ind, surf = st.checkbox("Torneo Indoor"), st.selectbox("Superficie", ["hard", "clay", "grass"])
        st.divider()
        st.header("🤕 Fatiga General del Lote")
        fat_p1, fat_p2 = st.slider("Fatiga P1",0,5,0), st.slider("Fatiga P2",0,5,0)
        iters = st.select_slider("MC Iters", [1000,3000,5000,10000], 5000)

    if "txt" not in st.session_state: st.session_state.txt = ""
    t1, t2 = st.tabs(["🎾 Motor Analítico", "🔮 El Oráculo"])
    
    with t1:
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1: bo = st.radio("Súper Formato", [3, 5], horizontal=True, format_func=lambda x: f"Bo{x}")
        with c3:
            if st.button("🗑️ Limpiar", use_container_width=True): 
                st.session_state.txt = ""
                st.rerun()

        txt = st.text_area("📋 Pega tus partidos:", key="txt", height=160, placeholder="Sinner J\nAlcaraz C\n-140\n+110")
        if st.button("🚀 Ignición Autónoma", type="primary", use_container_width=True):
            pts = parse_matches(txt)
            if not pts: st.error("No se detectaron emparejamientos. Sombrea las cuotas hasta el final.")
            else:
                names_to_fetch = [p[0] for p in pts] + [p[2] for p in pts]
                with st.spinner(f"🔍 Deep Profiling a {len(set(names_to_fetch))} jugadores simultáneamente..."):
                    master_profiles = get_autonomous_profiles(names_to_fetch)

                sv = []
                for p1_r, o1, p2_r, o2, lg in pts:
                    p1, p2 = p1_r or "P1", p2_r or "P2"
                    if "utr" in p1.lower() or "utr" in p2.lower(): st.warning(f"🚫 Salto UTR: {p1}"); continue
                    
                    st.divider()
                    st.subheader(f"🧠 {p1} vs {p2}")
                    vig = vig_pct(o1, o2) if o1 and o2 else None
                    st.caption(f"Liga: {lg} | {surf.title()} {'(Indoor)' if ind else ''}" + (f" | Vig: {vig}%" if vig else ""))

                    z1, ht1 = get_player_profile(p1, master_profiles)
                    z2, ht2 = get_player_profile(p2, master_profiles)

                    cs1, cs2 = st.columns(2)
                    with cs1: s1 = get_stats(p1) or gemini_stats(p1, surf)
                    with cs2: s2 = get_stats(p2) or gemini_stats(p2, surf)

                    adj_p1, n_p1 = calcular_sum_adj(z2, ind, ht1)
                    adj_p2, n_p2 = calcular_sum_adj(z1, ind, ht2)

                    act_f1 = fat_p1 if len(pts)==1 else 0
                    act_f2 = fat_p2 if len(pts)==1 else 0

                    a1s, a1r = apply_environment(s1["spw"], s1["rpw"], s1["n"], alt_m, act_f1, surf)
                    a2s, a2r = apply_environment(s2["spw"], s2["rpw"], s2["n"], alt_m, act_f2, surf)

                    pA_b, pB_b = log5_serve(a1s, a2r), log5_serve(a2s, a1r)
                    
                    with st.spinner("Girando Nucleo NumPy..."):
                        mcR_A = sim_match(pA_b, pB_b, bo, iters, alt_m)
                        mcR_B = 1 - mcR_A

                    mcA_A = min(max(mcR_A + adj_p1, 0.05), 0.95)
                    mcA_B = min(max(mcR_B + adj_p2, 0.05), 0.95)
                    t_n = mcA_A + mcA_B; mcA_A/=t_n; mcA_B/=t_n

                    e1_p = ev(mcA_A, american_to_decimal(o1)) if o1 else None
                    nv1_p = no_vig(o1, o2)[0] if o1 else None
                    t_pre = get_tier(mcA_A, e1_p, nv1_p)

                    mc_cal = apply_calibration(mcA_A, lg, t_pre)
                    mc2_cal = 1 - mc_cal

                    m1, m2 = st.columns(2)
                    e1=e2=nv1=nv2=t1c=t2c=None
                    for cl, nm, mc, o, s, asp, arp, ht_x, z_x, s_a, nn in [
                        (m1, p1, mc_cal, o1, s1, a1s, a1r, ht1, z1, adj_p1, n_p1),
                        (m2, p2, mc2_cal, o2, s2, a2s, a2r, ht2, z2, adj_p2, n_p2)
                    ]:
                        cl.markdown(f"**{nm}** · `{'Zurdo' if z_x else 'Diestro'} {ht_x}cm`")
                        cl.caption(f"{s['source']} n={s['n']} | SPW {asp:.1f}% RPW {arp:.1f}%")
                        cl.metric("P(Win) Cal", f"{mc*100:.1f}%", f"Adj {s_a:+.2f}", "off")

                        if o1 and o2:
                            nvv, fv = no_vig(o1,o2)[0 if cl is m1 else 1], no_vig(o1,o2)[2 if cl is m1 else 3]
                            dec_odd = american_to_decimal(o)
                            evv = ev(mc, dec_odd)
                            kf = kelly_fraction(mc, dec_odd)
                            tr = get_tier(mc, evv, nvv)
                            if cl is m1: e1, nv1, t1c = evv, nvv, tr
                            else: e2, nv2, t2c = evv, nvv, tr
                            
                            cl.metric("No-Vig", f"{nvv*100:.1f}%", f"Fair: {fv}", "off")
                            cl.metric("EV Return", f"{evv*100:+.1f}%", "normal" if evv>0 else "inverse")
                            if kf>0: cl.markdown(f'<div class="kelly">🏦 Kelly Edge: {kf}% bank</>', unsafe_allow_html=True)
                            
                            hc='ts' if 'SUPER' in tr else 'tg' if 'DERE' in tr or 'FRAN' in tr else 'tf' if 'FAVO' in tr else 'ty' if 'VALOR' in tr else 'tr'
                            cl.markdown(f'<div class="tier-badge {hc}">{tr}</div>', unsafe_allow_html=True)
                            
                        if nn: 
                            with cl.expander("👁️ Ver Autodiagnóstico"): 
                                for x in nn: st.caption(x)

                    st.markdown("#### 📡 H48 Médico / Físico")
                    with st.spinner("Validando noticias en la WEB..."): st.info(gemini_analysis(p1, p2))
                    
                    if e1 is not None:
                        mid = f"{p1}_{p2}_{datetime.now().strftime('%Y%m%d')}"
                        sv.extend([(mid,p1,p2,mc_cal,nv1,o1,e1,t1c,lg), (mid+"_inv",p2,p1,mc2_cal,nv2,o2,e2,t2c,lg)])
                
                # Guardar Batch final a session
                st.session_state.save_batch = sv

        if st.session_state.get('save_batch'):
            sb = st.session_state.save_batch
            if st.button(f"💾 Loguear a Sheets ({len(sb)//2})", type="primary"):
                w=0; 
                for d in sb: 
                    if log_prediction(*d): w+=1
                st.success(f"{w//2} guardados.")

    with t2:
        sh = get_sheet()
        if sh: 
            st.dataframe(pd.DataFrame(sh.get_all_records()).tail(15))
            if st.button("🔎 Liquidador"): pass # Implement logic here

if __name__ == "__main__":
    main()
