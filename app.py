"""
Quantum Tennis Engine v5.0 — Gemini Edition
Solo Gemini + Motor matemático local (Log5 + Monte Carlo corregido)
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
MC_ITERATIONS = 3000

METADATA_KW = {
    'local', 'visita', 'empate', 'sencillos', 'dobles', 'vivo', 'apuestas',
    'streaming', 'women', 'men', 'tour', 'challenger', 'atp', 'wta', 'itf',
    'world tennis', 'grand slam', 'futures', 'copa', 'qualifier', 'qualy'
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
    """
    Parser unificado para:
      • Formato 'vs' en línea: Sinner -120 vs Alcaraz +100
      • Formato apilado Caliente/DraftKings móvil
    """
    # Normalizar guiones tipográficos iOS/Mac
    text = text.replace('–', '-').replace('—', '-').replace('−', '-')
    results = []

    # ── Modo A: línea con 'vs' ────────────────────────────────────────────────
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

    # ── Modo B: apilado (máquina de estados) ─────────────────────────────────
    TIME_RE    = re.compile(r'^\d{2}:\d{2}$')
    COUNTER_RE = re.compile(r'^\+\s*\d{1,2}\s*(Streaming)?$', re.I)
    ODD_LONE   = re.compile(r'^[+-]\d{3,4}$')

    name_q, odd_q = [], []

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if any(kw in line.lower() for kw in METADATA_KW):
            continue
        if TIME_RE.match(line) or COUNTER_RE.match(line):
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
# MATEMÁTICA LOCAL
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
    A, B = spw / 100, rpw / 100
    d = A * (1 - B) + (1 - A) * B
    return (A * (1 - B)) / d if d else 0.5


def sim_game(p: float) -> int:
    s = r = 0
    while True:
        if random.random() < p: s += 1
        else: r += 1
        if s >= 4 and s - r >= 2: return 1
        if r >= 4 and r - s >= 2: return 0


def sim_tiebreak(p_first: float, p_second: float) -> int:
    a = b = total = 0
    while True:
        if total == 0:
            win = random.random() < p_first
        else:
            block = (total - 1) // 2
            win = random.random() < (p_second if block % 2 == 0 else p_first)
        if win: a += 1
        else:   b += 1
        total += 1
        if a >= 7 and a - b >= 2: return 1
        if b >= 7 and b - a >= 2: return 0


def sim_set(pA: float, pB: float, a_first: bool = True) -> tuple[int, bool]:
    gA = gB = 0
    a_serves = a_first
    while True:
        if gA == 6 and gB == 6:
            tf = pA if a_serves else pB
            ts = pB if a_serves else pA
            w  = sim_tiebreak(tf, ts)
            if w: gA += 1
            else: gB += 1
            return (1 if gA > gB else 0), not a_serves

        if a_serves:
            if sim_game(pA): gA += 1
            else: gB += 1
        else:
            if sim_game(pB): gB += 1
            else: gA += 1
        a_serves = not a_serves

        if gA >= 6 and gA - gB >= 2: return 1, a_serves
        if gB >= 6 and gB - gA >= 2: return 0, a_serves
        if gA == 7: return 1, a_serves
        if gB == 7: return 0, a_serves


def sim_match(pA: float, pB: float, best_of: int = 3, n: int = MC_ITERATIONS) -> float:
    needed = 2 if best_of == 3 else 3
    wins = 0
    for _ in range(n):
        sA = sB = 0
        a_srv = True
        while sA < needed and sB < needed:
            w, a_srv = sim_set(pA, pB, a_srv)
            if w: sA += 1
            else: sB += 1
        if sA == needed:
            wins += 1
    return wins / n


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
                if not as_w and not as_l:
                    continue

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

    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"BD error: {e}")
        return None

    if n == 0:
        return None

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
    if not client:
        return None
    try:
        prompt = f"""Busca en TennisAbstract, FlashScore, CoreTennis, ATP o ITF las estadísticas \
recientes de este tenista: "{name}"

Necesito EXCLUSIVAMENTE:
- Service Points Won % (puntos ganados sacando)
- Return Points Won % (puntos ganados restando)

Si no hay dato exacto, infiere un valor realista según su nivel y superficie habitual. \
Nunca devuelvas cero.

Responde SOLO con JSON crudo (sin backticks, sin texto extra):
{{"spw_pct": 64.5, "rpw_pct": 38.2}}"""

        r = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[{"google_search": {}}],
                temperature=0.1,
            )
        )
        raw  = r.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        spw  = max(10.1, float(data.get("spw_pct", 55.0)))
        rpw  = max(10.1, float(data.get("rpw_pct", 40.0)))
        return {"n": "Web", "hold": 0.0, "brk": 0.0,
                "spw": spw, "rpw": rpw, "source": "Gemini Search"}
    except Exception as e:
        st.error(f"Gemini stats fallback falló para {name}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# GEMINI — ANÁLISIS COMPLETO (contexto + veredicto)
# ─────────────────────────────────────────────────────────────────────────────
def gemini_full_analysis(p1: str, p2: str, report: str) -> str:
    client = get_gemini_client()
    if not client:
        return "❌ GOOGLE_API_KEY no configurada."
    try:
        prompt = f"""Eres un analista de apuestas de tenis con acceso a búsqueda en tiempo real.

PASO 1 — CONTEXTO H48 (búsqueda web obligatoria):
Busca hechos objetivos de las últimas 48 horas sobre {p1} y {p2}:
• Partidos disputados recientemente (duración, esfuerzo físico)
• Lesiones, retiros o problemas físicos reportados
• Viajes largos o cambios de superficie recientes

PASO 2 — AUDITORÍA MATEMÁTICA:
Analiza este reporte generado localmente (Log5 + Monte Carlo {MC_ITERATIONS} iter.):
{report}

PASO 3 — VEREDICTO FINAL (reglas estrictas):
• EV > 5% Y sin lesiones/fatiga grave → ✅ VERDE — Apostar
• 0% < EV ≤ 5% O fatiga menor detectada → ⚠️ AMARILLO — Reducir stake
• EV negativo O lesión confirmada → 🚫 ROJO — Evitar

Formato de respuesta:
**CONTEXTO H48**
[viñetas con hechos encontrados o "Sin alertas detectadas"]

**AUDITORÍA MATEMÁTICA**
[1-2 líneas sobre coherencia del EV y Monte Carlo]

**VEREDICTO**
[emoji + dictamen en 1 línea clara]"""

        r = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[{"google_search": {}}],
                temperature=0.2,
            )
        )
        return r.text
    except Exception as e:
        return f"Error en análisis Gemini: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Quantum Tennis v5 — Gemini",
        page_icon="🎾",
        layout="wide",
    )

    # ── Header ────────────────────────────────────────────────────────────────
    st.title("🎾 Quantum Tennis Engine v5.0 — Gemini Edition")
    st.caption(
        "Motor matemático local (Log5 · Monte Carlo) + "
        "Gemini 2.5 Pro con Search Grounding para stats web y análisis H48"
    )

    if not gemini_available():
        st.error("⚠️ Falta GOOGLE_API_KEY en tu archivo .env")
        st.stop()

    # ── Controles ─────────────────────────────────────────────────────────────
    if "txt" not in st.session_state:
        st.session_state.txt = ""

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        best_of = st.radio(
            "Formato:", [3, 5], horizontal=True,
            format_func=lambda x: f"Mejor de {x}"
        )
    with c2:
        db_path = st.text_input("BD (.jsonl)", value=DB_PATH, label_visibility="collapsed")
    with c3:
        if st.button("🗑️ Limpiar", use_container_width=True):
            st.session_state.txt = ""
            st.rerun()

    txt = st.text_area(
        "📋 Pega los partidos (formato Caliente o 'Sinner -120 vs Alcaraz +100'):",
        key="txt",
        height=160,
        placeholder="Alcaraz C\nSinner J\n-140\n+110",
    )

    if not st.button("🚀 Analizar", type="primary", use_container_width=True):
        return

    if not txt.strip():
        st.warning("Pega al menos un partido.")
        return

    partidos = parse_matches(txt)
    if not partidos:
        st.error("No se encontraron pares jugador/cuota. Revisa el formato.")
        with st.expander("Debug"):
            st.code(txt)
        return

    # ── Loop de partidos ──────────────────────────────────────────────────────
    for p1_raw, odd1, p2_raw, odd2 in partidos:
        p1 = p1_raw or "Jugador 1"
        p2 = p2_raw or "Jugador 2"

        # Filtros rápidos
        if "utr" in p1.lower() or "utr" in p2.lower():
            st.warning(f"🚫 UTR excluido: {p1} vs {p2}")
            continue
        if odd1 and odd2 and (odd1 <= -500 or odd2 <= -500):
            st.warning(f"🚫 Cuota extrema descartada: {p1} ({odd1}) vs {p2} ({odd2})")
            continue

        st.divider()
        st.subheader(f"⚡ {p1}  vs  {p2}")

        # ── Stats ─────────────────────────────────────────────────────────────
        col_s1, col_s2 = st.columns(2)

        with col_s1:
            with st.spinner(f"Stats {p1}…"):
                s1 = get_stats(p1, db_path)
                if not s1:
                    st.info(f"🌐 {p1} no está en BD local — buscando en web…")
                    s1 = gemini_stats(p1)

        with col_s2:
            with st.spinner(f"Stats {p2}…"):
                s2 = get_stats(p2, db_path)
                if not s2:
                    st.info(f"🌐 {p2} no está en BD local — buscando en web…")
                    s2 = gemini_stats(p2)

        if not s1 or not s2:
            st.error("Stats insuficientes. Verifica el nombre o configura GOOGLE_API_KEY.")
            continue

        # ── Motor local ───────────────────────────────────────────────────────
        pA = log5_serve(s1["spw"], s2["rpw"])
        pB = log5_serve(s2["spw"], s1["rpw"])

        with st.spinner(f"Corriendo {MC_ITERATIONS:,} simulaciones…"):
            mc_A = sim_match(pA, pB, best_of)
            mc_B = 1 - mc_A

        # ── Resultados matemáticos ────────────────────────────────────────────
        st.markdown("#### 🧮 Motor Matemático")

        m1, m2 = st.columns(2)
        pairs = [(m1, p1, pA, mc_A, odd1, s1), (m2, p2, pB, mc_B, odd2, s2)]

        ev_lines = []
        for col, name, p_srv, mc_w, odd, stats in pairs:
            col.markdown(f"**{name}**")
            col.caption(
                f"Fuente: {stats['source']} · n={stats['n']} · "
                f"SPW {stats['spw']}% · RPW {stats['rpw']}% · "
                f"Hold {stats['hold']}%"
            )
            col.metric("P(srv/punto) Log5", f"{p_srv*100:.1f}%")
            col.metric("P(match) Monte Carlo", f"{mc_w*100:.1f}%")

            if odd:
                nv1, nv2, f1, f2 = no_vig(odd1 or 100, odd2 or 100)
                nv_this = nv1 if col is m1 else nv2
                fair    = f1  if col is m1 else f2
                ev_val  = ev(mc_w, american_to_decimal(odd))
                edge    = (mc_w - nv_this) * 100
                col.metric("P(match) Casa No-Vig", f"{nv_this*100:.1f}%",
                           f"Fair odd {fair}", delta_color="off")
                col.metric("💰 Expected Value", f"{ev_val*100:+.1f}%",
                           delta_color="normal" if ev_val > 0 else "inverse")
                col.metric("Edge vs Casa", f"{edge:+.1f}%")
                ev_lines.append(f"{name}: EV={ev_val*100:+.2f}% | Edge={edge:+.1f}% | Cuota={odd}")
            else:
                col.info("Sin cuota → EV no calculable")

        # ── Reporte para Gemini ───────────────────────────────────────────────
        report = f"""
{p1}: SPW={s1['spw']}% RPW={s1['rpw']}% Hold={s1['hold']}% (n={s1['n']}, fuente={s1['source']})
{p2}: SPW={s2['spw']}% RPW={s2['rpw']}% Hold={s2['hold']}% (n={s2['n']}, fuente={s2['source']})
Log5 P(srv/punto): {p1}={pA:.3f} | {p2}={pB:.3f}
Monte Carlo ({MC_ITERATIONS} iter, mejor de {best_of}): {p1}={mc_A*100:.1f}% | {p2}={mc_B*100:.1f}%
{chr(10).join(ev_lines) if ev_lines else 'Sin cuotas ingresadas'}
"""

        # ── Gemini análisis completo ──────────────────────────────────────────
        st.markdown("#### 🤖 Gemini — Análisis H48 + Veredicto")
        with st.spinner("Gemini buscando contexto y generando veredicto…"):
            analisis = gemini_full_analysis(p1, p2, report)

        st.markdown(analisis)


if __name__ == "__main__":
    main()
