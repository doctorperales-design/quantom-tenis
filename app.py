"""
Quantum Tennis Engine v9.4 — Quant Risk Management Edition
Markov Chains · Shrinkage Bayesiano · Oráculo Enriquecido 2022-2025

CHANGELOG v9.4 vs v9.3:
──────────────────────────────────────────────────────────────────────
BUG FIXES (B1-B21):
B1  _get_stats_all_surfaces: `mixed_sample` era código muerto. Ahora
    controla la etiqueta MUESTRA MIXTA vs SIN SUPERFICIE correctamente.
B2  Eliminadas variables muertas p1_top100 / p2_top100 en main().
B3  _name_match simétrico: "Rublev A" ↔ "Rublev" matchea en ambos sentidos.
B4  classify: guard "EV > 20% en favoritos" aplica fuera del Valle Muerte.
B5  calibrate_probability: penalty_factor aplica a todo el rango DV.
B6  kelly_fraction: guard explícito para EV negativo → stake = 0.
B7  sim_match usa rng = random.Random(seed) — sin afectar estado global.
B8  log_prediction: retry con backoff exponencial (3 intentos).
B9  Google Sheets ID desde st.secrets["SHEETS_KEY"] (no hardcoded).
B10 Excepts silenciosos loguean a st.session_state["debug_errors"].
B11 Oráculo indexado por jugador: ~10x speedup en get_stats/h2h/tanking.
B12 parse_matches: warning si hay nombres/cuotas huérfanas.
B13 extract_match_context con TTL=3600s (evita caché indefinido).
B14 Versión unificada a v9.4 en docstring y UI.
B15 UTR exclusion con word boundary (no substring → no falsos positivos).
B16 FAST/SLOW_COURTS indexados en dict para lookup O(1).
B17 Shadow Bet stake como constante SHADOW_BET_STAKE.
B18 load_oracle: contador de registros corruptos con aviso al usuario.
B19 log_prediction usa dataclass Prediction.
B20 SPW_FLOOR_MIN (25.1) documentado como mínimo fisiológico.
B21 make_seed usa SHA256 (MD5 deprecated).

AJUSTES CUANTITATIVOS (Q1-Q6 — framework Quant Risk):
Q1  Kill-switch EV > 35% como "Alucinación Persistente".
Q2  Valle de la Muerte: ventana EV 2%-10% estricta; fuera → rechazo.
Q3  Shadow Bot "Back to Basura": sweet spot +100 a +150 + n_total < 40.
Q4  Lista negra estricta > +200 (Cisne Negro) con rationale documentado.
Q5  Bomba Nuclear refinada: +150 a +199 con pmod ≥ 0.50.
Q6  Kelly con probabilidad empírica por tier (TIER_EMPIRICAL_P).
──────────────────────────────────────────────────────────────────────
"""

import streamlit as st
import streamlit.components.v1 as components
import json
import os
import re
import random
import hashlib
import csv
import time
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("quantum_tennis")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
APP_VERSION   = "v9.4"
GEMINI_MODEL  = "gemini-2.5-pro"
DB_PATH       = "matches_comprimidos.csv"
MC_ITERATIONS = 10000

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES DE AJUSTE
# ─────────────────────────────────────────────────────────────────────────────
ADJ_LEFTY           = 0.05
ADJ_HEIGHT_FAST     = 0.03
ADJ_CLAY_TRAP       = 0.06
ADJ_H2H_DOMINANT    = 0.03
ADJ_DR_HOT          = 0.04
ADJ_DR_COLD         = 0.04
ADJ_CLUTCH          = 0.05
ADJ_WEB_SOURCE      = 0.03
IC_PENALTY_MIXED    = 0.05
IC_PENALTY_FALLBACK = 0.15
COURT_PACE_PP       = 1.5
SHRINKAGE_BASE      = 0.35
SHRINKAGE_THRESHOLD = 20
SPW_FLOOR           = 30.0
SPW_CEIL            = 85.0
RPW_FLOOR           = 25.0
RPW_CEIL            = 60.0
SPW_FLOOR_MIN       = 25.1   # B20: mínimo fisiológico antes de aplicar fallback
HEIGHT_THRESHOLD    = 193

# B17 + Q-framework: stakes y umbrales cuantitativos
SHADOW_BET_STAKE        = 1.5     # % bankroll para Shadow Bot
KILL_SWITCH_EV          = 0.35    # Q1: EV > 35% → alucinación
DEATH_VALLEY_EV_MIN     = 0.02    # Q2: límite inferior de ventana aceptable
DEATH_VALLEY_EV_MAX     = 0.10    # Q2: límite superior
SHADOW_SPARSE_THRESHOLD = 40      # Q3: si n_total ≥ 40, NO shadow (datos confirman)
SHADOW_DEC_MIN          = 2.00    # Q3: cuota americana +100
SHADOW_DEC_MAX          = 2.50    # Q3: cuota americana +150
BOMBA_DEC_MIN           = 2.50    # Q5: +150
BOMBA_DEC_MAX           = 2.99    # Q5: +199
BOMBA_PMOD_MIN          = 0.50    # Q5: pmod umbral para Bomba Nuclear

# Q6: probabilidad empírica por tier (backtest del usuario)
TIER_EMPIRICAL_P = {
    "BOMBA NUCLEAR":        0.50,
    "BASURA PRO (Shadow Bet)": 0.567,
    "SUPER DERECHA":        0.65,
    "DERECHA":              0.58,
    "PARLAY":               0.55,
}

# ─────────────────────────────────────────────────────────────────────────────
# FALLBACKS EMPÍRICOS — 40,225 registros (2022-2024)
# ─────────────────────────────────────────────────────────────────────────────
SURFACE_FALLBACKS = {
    "Hard":   {"hold": 74.0, "spw": 68.9, "rpw": 42.8},
    "Clay":   {"hold": 70.5, "spw": 66.1, "rpw": 45.9},
    "Grass":  {"hold": 77.6, "spw": 70.9, "rpw": 39.5},
    "Carpet": {"hold": 75.8, "spw": 69.5, "rpw": 41.1},
}

# B16: dict indexado
COURT_PACE = {
    # Pistas rápidas
    "dubai": "FAST", "cincinnati": "FAST", "brisbane": "FAST",
    "stuttgart": "FAST", "s hertogenbosch": "FAST", "queens": "FAST",
    "halle": "FAST", "wimbledon": "FAST", "us open": "FAST",
    "tokyo": "FAST", "vienna": "FAST", "basel": "FAST",
    "paris masters": "FAST", "atp finals": "FAST",
    # Pistas lentas
    "miami": "SLOW", "indian wells": "SLOW", "rome": "SLOW",
    "madrid": "SLOW", "roland garros": "SLOW", "monte carlo": "SLOW",
    "barcelona": "SLOW", "canadian open": "SLOW", "montreal": "SLOW",
    "buenos aires": "SLOW", "rio": "SLOW", "houston": "SLOW", "umag": "SLOW",
}

SRF_MAP = {1: "Hard", 2: "Clay", 3: "Grass", 4: "Carpet", 0: "Unknown"}
LVL_MAP = {1: "ATP", 2: "Masters", 3: "GrandSlam", 4: "Challenger",
           5: "Finals", 6: "DavisCup", 0: "ITF"}

LEVEL_GROUPS = {
    "ATP":       {1, 2, 3},
    "Masters":   {1, 2, 3},
    "GrandSlam": {1, 2, 3},
    "Challenger": {3, 4},
    "Finals":    {1, 2, 3, 5},
    "DavisCup":  {1, 2, 3, 6},
    "ITF":       {0, 1, 2, 3, 4},
}

METADATA_KW = {
    'local', 'visita', 'empate', 'sencillos', 'dobles', 'vivo', 'apuestas',
    'streaming', 'women', 'men', 'tour', 'challenger', 'atp', 'wta', 'itf',
    'world tennis', 'grand slam', 'futures', 'copa', 'qualifier', 'qualy',
    'hoy', 'mañana', 'lunes', 'martes', 'miércoles', 'miercoles', 'jueves',
    'viernes', 'sábado', 'sabado', 'domingo',
}

# B15: regex para UTR con word boundary
UTR_RE = re.compile(r'\butr\b', re.IGNORECASE)


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING DE DEBUG — B10
# ─────────────────────────────────────────────────────────────────────────────
def _log_debug(context: str, err: Exception) -> None:
    """Registra errores en session_state para visibilidad sin romper UX."""
    msg = f"[{context}] {type(err).__name__}: {err}"
    logger.warning(msg)
    try:
        st.session_state.setdefault("debug_errors", []).append(msg)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# CLIENTE GEMINI — B9: key desde env o secrets
# ─────────────────────────────────────────────────────────────────────────────
def _get_api_key() -> str:
    key = os.environ.get("GOOGLE_API_KEY", "")
    if not key:
        try:
            key = st.secrets.get("GOOGLE_API_KEY", "")
        except Exception:
            key = ""
    return key


@st.cache_resource
def get_gemini_client():
    api_key = _get_api_key()
    return genai.Client(api_key=api_key) if api_key else None


def gemini_available() -> bool:
    return get_gemini_client() is not None


# ─────────────────────────────────────────────────────────────────────────────
# ORÁCULO EN RAM + ÍNDICE — B11, B18
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_oracle(path: str) -> tuple[list[list], dict[str, list[int]], int]:
    """
    Carga el oráculo completo + construye índice por apellido → list[idx].
    Retorna (records, index_by_surname, n_skipped).
    """
    records = []
    n_skipped = 0
    try:
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parsed = None
                if line.startswith('['):
                    try:
                        rec = json.loads(line)
                        if len(rec) >= 20:
                            parsed = rec
                    except Exception as e:
                        n_skipped += 1
                        continue
                else:
                    try:
                        parts = next(csv.reader([line]))
                        tmp = []
                        for p in parts:
                            p = p.strip()
                            if p.lstrip('-').isdigit():
                                tmp.append(int(p))
                            else:
                                try:
                                    tmp.append(float(p))
                                except ValueError:
                                    tmp.append(p)
                        if len(tmp) >= 20:
                            parsed = tmp
                    except Exception:
                        n_skipped += 1
                        continue
                if parsed is not None:
                    records.append(parsed)
    except FileNotFoundError:
        st.warning(f"⚠️ Archivo no encontrado: {path}")
        return [], {}, 0

    # Índice por apellido (primer token del nombre)
    index: dict[str, list[int]] = {}
    for i, rec in enumerate(records):
        for field in (2, 4):
            name = str(rec[field]).strip().lower()
            if not name:
                continue
            surname = name.split()[0] if ' ' in name else name
            index.setdefault(surname, []).append(i)
    return records, index, n_skipped


def _candidates_for(name: str, oracle: list[list], index: dict[str, list[int]]) -> list[list]:
    """Devuelve records que posiblemente contengan al jugador (vía índice)."""
    q = name.lower().strip()
    surname = q.split()[0] if q else q
    idxs = index.get(surname, [])
    return [oracle[i] for i in idxs]


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACCIÓN AUTÓNOMA (GEMINI) — B10, B13
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=3600)
def extract_match_context(p1: str, p2: str) -> dict:
    client = get_gemini_client()
    default = {
        "surface": "Hard", "tourney": "Unknown", "altitude": 0, "level": "ATP",
        "p1_hand": "R", "p1_height": 185, "p1_local": False,
        "p2_hand": "R", "p2_height": 185, "p2_local": False,
    }
    if not client:
        return default
    try:
        prompt = (
            f'Investiga el partido de tenis entre "{p1}" (P1) y "{p2}" (P2).\n'
            f'Detecta 4 cosas crudas:\n'
            f'1. Torneo, superficie (Hard, Clay, Grass, Carpet), nivel '
            f'(ATP/Masters/GrandSlam/Challenger/ITF), altitud en metros.\n'
            f'2. Mano Dominante de ambos (L para Zurdo, R para Diestro).\n'
            f'3. Estatura (altura) de ambos en cm (entero).\n'
            f'4. Localía: ¿Algún jugador es de la misma nacionalidad/país sede del torneo? (True/False).\n'
            f'Responde SOLO con JSON crudo:\n'
            f'{{"surface": "Hard", "tourney": "Monte Carlo", "altitude": 35, '
            f'"level": "Masters", "p1_hand": "R", "p1_height": 185, "p1_local": false, '
            f'"p2_hand": "L", "p2_height": 198, "p2_local": true}}'
        )
        r = client.models.generate_content(
            model=GEMINI_MODEL, contents=prompt,
            config=types.GenerateContentConfig(
                tools=[{"google_search": {}}], temperature=0.1
            )
        )
        raw = r.text.replace("```json", "").replace("```", "").strip()
        s, e = raw.find('{'), raw.rfind('}')
        if s != -1 and e != -1:
            data = json.loads(raw[s:e + 1])
            lvl = data.get("level", "ATP")
            if lvl not in LEVEL_GROUPS:
                lvl = "ATP"
            data["level"] = lvl
            for k, v in default.items():
                if k not in data:
                    data[k] = v
            return data
    except Exception as err:
        _log_debug("extract_match_context", err)
    return default


def batch_guess_winners_gemini(matches: list[dict]) -> dict:
    if not matches:
        return {}
    client = get_gemini_client()
    if not client:
        return {m['match_id']: "Ninguno" for m in matches}

    resultados_finales = {}
    chunk_size = 10
    for i in range(0, len(matches), chunk_size):
        chunk = matches[i:i + chunk_size]
        prompt = (
            "Eres un analista de resultados de tenis. Ve y busca en internet "
            "(usa Flashscore, ATP Tour, WTA Tour u otros sitios de tenis en vivo) "
            "el resultado de los siguientes partidos recientes:\n"
        )
        for m in chunk:
            prompt += f"- ID '{m['match_id']}': {m['p1']} vs {m['p2']}\n"
        prompt += (
            '\nDevuelve ÚNICAMENTE un JSON crudo con las respuestas.\n'
            'Usa la estructura: {"match_id": "Nombre Exacto del Ganador"}.\n'
            'Usa estrictamente los nombres de los jugadores tal y como te los envié.\n'
            'Si el partido no se ha jugado, se canceló, o no encuentras resultado oficial, pon "Ninguno".\n'
            'Ejemplo: {"id_1": "Carlos Alcaraz", "id_2": "Ninguno"}\n'
        )
        try:
            r = client.models.generate_content(
                model=GEMINI_MODEL, contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[{"google_search": {}}], temperature=0.1
                )
            )
            raw = r.text.replace("```json", "").replace("```", "").strip()
            s, e = raw.find('{'), raw.rfind('}')
            if s != -1 and e != -1:
                batch_res = json.loads(raw[s:e + 1])
                resultados_finales.update(batch_res)
        except Exception as err:
            _log_debug("batch_guess_winners", err)
    return resultados_finales


# ─────────────────────────────────────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────────────────────────────────────
def get_fallback(surface: str) -> dict:
    s = surface.lower()
    if "clay" in s or "arcilla" in s or "tierra" in s:
        return SURFACE_FALLBACKS["Clay"]
    if "grass" in s or "hierba" in s or "pasto" in s:
        return SURFACE_FALLBACKS["Grass"]
    if "carpet" in s or "moqueta" in s:
        return SURFACE_FALLBACKS["Carpet"]
    return SURFACE_FALLBACKS["Hard"]


def _name_match(query: str, record_name: str) -> bool:
    """
    B3 FIX: Match simétrico por apellido.
    'Sinner'   ↔ 'Sinner J'  (match)
    'Rublev A' ↔ 'Rublev'    (match — ahora simétrico)
    'Sin'      ✗ 'Sinner J'  (partial rechazado)
    """
    q = query.lower().strip()
    r = record_name.lower().strip()
    if q == r:
        return True
    q_parts = q.split()
    r_parts = r.split()

    # Query apellido solo
    if len(q_parts) == 1:
        return r.startswith(q + ' ') or r == q

    # Record apellido solo y query con apellido + inicial → match por apellido
    if len(q_parts) >= 2 and len(r_parts) == 1:
        return q_parts[0] == r_parts[0]

    # Ambos tienen >= 2 partes → apellido + primera inicial
    if len(q_parts) >= 2 and len(r_parts) >= 2:
        return q_parts[0] == r_parts[0] and q_parts[1][0] == r_parts[1][0]
    return False


def _date_int_to_dt(d: int) -> datetime:
    try:
        return datetime.strptime(str(d), "%Y%m%d")
    except Exception:
        return datetime(2020, 1, 1)


def _safe_int(val) -> int:
    try:
        return int(val)
    except (ValueError, TypeError):
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# ORÁCULO — STATS CON FILTROS (ahora sobre candidatos indexados)
# ─────────────────────────────────────────────────────────────────────────────
def get_stats(name: str, surface: str, level: str,
              oracle: list[list], index: dict[str, list[int]]) -> dict | None:
    allowed_levels = LEVEL_GROUPS.get(level, {1, 2, 3, 4})
    srf_code = {"Hard": 1, "Clay": 2, "Grass": 3, "Carpet": 4}.get(surface, 1)

    sv_pts = sv_won = sv_gms = sv_held = 0
    rt_pts = rt_won = rt_gms = rt_brk = 0
    n_total = n_surface = 0
    bp_saved_total = bp_faced_total = bp_conv_won = bp_conv_total = 0
    recent_minutes = []
    recent_results = []

    for rec in _candidates_for(name, oracle, index):
        wn, ln = str(rec[2]), str(rec[4])
        is_w = _name_match(name, wn)
        is_l = _name_match(name, ln)
        if not is_w and not is_l:
            continue

        rec_lvl = _safe_int(rec[1])
        if rec_lvl not in allowed_levels:
            continue

        rec_srf = _safe_int(rec[0])
        on_surface = (rec_srf == srf_code)
        rec_date = _safe_int(rec[20]) if len(rec) > 20 else 0
        rec_mins = _safe_int(rec[21]) if len(rec) > 21 else 0

        if is_w:
            wp = (_safe_int(rec[6]), _safe_int(rec[8]), _safe_int(rec[9]),
                  _safe_int(rec[10]), _safe_int(rec[11]), _safe_int(rec[12]))
            lp = (_safe_int(rec[13]), _safe_int(rec[15]), _safe_int(rec[16]),
                  _safe_int(rec[17]), _safe_int(rec[18]), _safe_int(rec[19]))
        else:
            wp = (_safe_int(rec[13]), _safe_int(rec[15]), _safe_int(rec[16]),
                  _safe_int(rec[17]), _safe_int(rec[18]), _safe_int(rec[19]))
            lp = (_safe_int(rec[6]), _safe_int(rec[8]), _safe_int(rec[9]),
                  _safe_int(rec[10]), _safe_int(rec[11]), _safe_int(rec[12]))

        if on_surface:
            sv_pts += wp[0]
            sv_won += wp[1] + wp[2]
            sv_gms += wp[3]
            sv_held += max(0, wp[3] - (wp[5] - wp[4]))
            rt_pts += lp[0]
            rt_won += max(0, lp[0] - (lp[1] + lp[2]))
            rt_gms += lp[3]
            rt_brk += max(0, lp[5] - lp[4])
            n_surface += 1
            bp_saved_total += wp[4]
            bp_faced_total += wp[5]
            bp_conv_won += max(0, lp[5] - lp[4])
            bp_conv_total += lp[5]

        n_total += 1

        if rec_mins > 0 and rec_date > 0:
            recent_minutes.append((rec_date, rec_mins))

        if on_surface and wp[0] > 0 and lp[0] > 0:
            my_spw = (wp[1] + wp[2]) / wp[0]
            opp_spw = (lp[1] + lp[2]) / lp[0]
            if opp_spw > 0:
                recent_results.append((rec_date, my_spw / opp_spw, is_w))

    if n_surface < 10 and n_total >= 10:
        return _get_stats_all_surfaces(name, level, oracle, index, mixed_sample=True)

    if n_surface == 0:
        if level != "ITF":
            return get_stats(name, surface, "ITF", oracle, index)
        return None

    spw = sv_won / sv_pts * 100 if sv_pts else 0
    rpw = rt_won / rt_pts * 100 if rt_pts else 0
    hold = sv_held / sv_gms * 100 if sv_gms else 0
    brk = rt_brk / rt_gms * 100 if rt_gms else 0

    fb = get_fallback(surface)
    if spw <= SPW_FLOOR_MIN:
        spw = fb["spw"]
    if rpw <= SPW_FLOOR_MIN:
        rpw = fb["rpw"]

    bp_saved_pct = (bp_saved_total / bp_faced_total * 100) if bp_faced_total > 0 else 50.0
    bp_conv_pct = (bp_conv_won / bp_conv_total * 100) if bp_conv_total > 0 else 40.0
    clutch = bp_saved_pct + bp_conv_pct

    recent_results.sort(key=lambda x: x[0], reverse=True)
    last5_dr = [r[1] for r in recent_results[:5]]
    avg_dr = sum(last5_dr) / len(last5_dr) if last5_dr else 1.0
    last5_wins = sum(1 for r in recent_results[:5] if r[2])

    recent_minutes.sort(key=lambda x: x[0], reverse=True)
    fatigue_mins = sum(m[1] for m in recent_minutes[:2]) if len(recent_minutes) >= 2 else 0

    tags = []
    if n_surface < 10:
        tags.append("FALLBACK")

    return {
        "n": n_surface, "n_total": n_total,
        "hold": round(hold, 1), "brk": round(brk, 1),
        "spw": round(spw, 1), "rpw": round(rpw, 1),
        "clutch": round(clutch, 1), "dr_last5": round(avg_dr, 3),
        "last5_wins": last5_wins, "fatigue_mins": fatigue_mins,
        "tags": tags, "source": "Oráculo Local",
    }


def _get_stats_all_surfaces(name: str, level: str, oracle: list[list],
                            index: dict[str, list[int]],
                            mixed_sample: bool = False) -> dict | None:
    """B1 FIX: `mixed_sample` ahora controla la etiqueta correctamente."""
    allowed_levels = LEVEL_GROUPS.get(level, {1, 2, 3, 4})
    sv_pts = sv_won = sv_gms = sv_held = 0
    rt_pts = rt_won = rt_gms = rt_brk = 0
    bp_saved_total = bp_faced_total = bp_conv_won = bp_conv_total = 0
    recent_minutes = []
    recent_results = []
    n = 0

    for rec in _candidates_for(name, oracle, index):
        wn, ln = str(rec[2]), str(rec[4])
        is_w = _name_match(name, wn)
        is_l = _name_match(name, ln)
        if not is_w and not is_l:
            continue
        if _safe_int(rec[1]) not in allowed_levels:
            continue

        rec_date = _safe_int(rec[20]) if len(rec) > 20 else 0
        rec_mins = _safe_int(rec[21]) if len(rec) > 21 else 0

        if is_w:
            wp = (_safe_int(rec[6]), _safe_int(rec[8]), _safe_int(rec[9]),
                  _safe_int(rec[10]), _safe_int(rec[11]), _safe_int(rec[12]))
            lp = (_safe_int(rec[13]), _safe_int(rec[15]), _safe_int(rec[16]),
                  _safe_int(rec[17]), _safe_int(rec[18]), _safe_int(rec[19]))
        else:
            wp = (_safe_int(rec[13]), _safe_int(rec[15]), _safe_int(rec[16]),
                  _safe_int(rec[17]), _safe_int(rec[18]), _safe_int(rec[19]))
            lp = (_safe_int(rec[6]), _safe_int(rec[8]), _safe_int(rec[9]),
                  _safe_int(rec[10]), _safe_int(rec[11]), _safe_int(rec[12]))

        sv_pts += wp[0]
        sv_won += wp[1] + wp[2]
        sv_gms += wp[3]
        sv_held += max(0, wp[3] - (wp[5] - wp[4]))
        rt_pts += lp[0]
        rt_won += max(0, lp[0] - (lp[1] + lp[2]))
        rt_gms += lp[3]
        rt_brk += max(0, lp[5] - lp[4])
        bp_saved_total += wp[4]
        bp_faced_total += wp[5]
        bp_conv_won += max(0, lp[5] - lp[4])
        bp_conv_total += lp[5]
        n += 1

        if rec_mins > 0 and rec_date > 0:
            recent_minutes.append((rec_date, rec_mins))
        if wp[0] > 0 and lp[0] > 0:
            my_spw = (wp[1] + wp[2]) / wp[0]
            opp_spw = (lp[1] + lp[2]) / lp[0]
            if opp_spw > 0:
                recent_results.append((rec_date, my_spw / opp_spw, is_w))

    if n == 0:
        if level != "ITF":
            return _get_stats_all_surfaces(name, "ITF", oracle, index, mixed_sample)
        return None

    spw = max(SPW_FLOOR_MIN, sv_won / sv_pts * 100 if sv_pts else 55.0)
    rpw = max(SPW_FLOOR_MIN, rt_won / rt_pts * 100 if rt_pts else 40.0)
    hold = sv_held / sv_gms * 100 if sv_gms else 0
    brk = rt_brk / rt_gms * 100 if rt_gms else 0

    bp_saved_pct = (bp_saved_total / bp_faced_total * 100) if bp_faced_total > 0 else 50.0
    bp_conv_pct = (bp_conv_won / bp_conv_total * 100) if bp_conv_total > 0 else 40.0
    clutch = bp_saved_pct + bp_conv_pct

    recent_results.sort(key=lambda x: x[0], reverse=True)
    last5_dr = [r[1] for r in recent_results[:5]]
    avg_dr = sum(last5_dr) / len(last5_dr) if last5_dr else 1.0
    last5_wins = sum(1 for r in recent_results[:5] if r[2])

    recent_minutes.sort(key=lambda x: x[0], reverse=True)
    fatigue_mins = sum(m[1] for m in recent_minutes[:2]) if len(recent_minutes) >= 2 else 0

    # B1 FIX: tag depende de mixed_sample
    tags = ["MUESTRA MIXTA"] if mixed_sample else ["SIN SUPERFICIE"]
    if n < 10:
        tags.append("FALLBACK")

    return {
        "n": n, "n_total": n,
        "hold": round(hold, 1), "brk": round(brk, 1),
        "spw": round(spw, 1), "rpw": round(rpw, 1),
        "clutch": round(clutch, 1), "dr_last5": round(avg_dr, 3),
        "last5_wins": last5_wins, "fatigue_mins": fatigue_mins,
        "tags": tags,
        "source": "Oráculo Local (mixta)" if mixed_sample else "Oráculo Local (todas sup.)",
    }


# ─────────────────────────────────────────────────────────────────────────────
# H2H (≤36 MESES)
# ─────────────────────────────────────────────────────────────────────────────
def get_h2h(p1: str, p2: str, oracle: list[list], index: dict[str, list[int]],
            months_limit: int = 36) -> dict:
    wins_p1 = wins_p2 = 0
    matches = []
    cutoff = datetime.now() - timedelta(days=months_limit * 30)

    # Unión de candidatos de p1 y p2 (más pequeño)
    cand = _candidates_for(p1, oracle, index)

    for rec in cand:
        wn, ln = str(rec[2]), str(rec[4])
        p1_is_w = _name_match(p1, wn) and _name_match(p2, ln)
        p1_is_l = _name_match(p1, ln) and _name_match(p2, wn)
        if not p1_is_w and not p1_is_l:
            continue

        rec_date = _safe_int(rec[20]) if len(rec) > 20 else 0
        if rec_date > 0:
            if _date_int_to_dt(rec_date) < cutoff:
                continue

        surface = SRF_MAP.get(_safe_int(rec[0]), "?")
        tourney = str(rec[24]) if len(rec) > 24 else "?"

        if p1_is_w:
            wins_p1 += 1
            matches.append({"date": rec_date, "winner": p1, "surface": surface, "tourney": tourney})
        else:
            wins_p2 += 1
            matches.append({"date": rec_date, "winner": p2, "surface": surface, "tourney": tourney})

    return {
        "p1_wins": wins_p1, "p2_wins": wins_p2,
        "total": wins_p1 + wins_p2,
        "matches": sorted(matches, key=lambda x: x["date"], reverse=True),
    }


# ─────────────────────────────────────────────────────────────────────────────
# TANKING & TIER DROP
# ─────────────────────────────────────────────────────────────────────────────
def check_tanking(name: str, oracle: list[list], index: dict[str, list[int]],
                  last_n: int = 5) -> bool:
    losses = []
    for rec in _candidates_for(name, oracle, index):
        if len(rec) < 22:
            continue
        if _name_match(name, str(rec[4])):
            d = _safe_int(rec[20])
            m = _safe_int(rec[21])
            if d > 0:
                losses.append((d, m))
    losses.sort(key=lambda x: x[0], reverse=True)
    recent = losses[:last_n]
    return sum(1 for _, mins in recent if 0 < mins < 60) >= 2


def check_tier_drop(name: str, level: str, oracle: list[list],
                    index: dict[str, list[int]]) -> bool:
    if level not in ("Challenger", "ITF"):
        return False

    ranks = []
    for rec in _candidates_for(name, oracle, index):
        if len(rec) < 24:
            continue
        wn, ln = str(rec[2]), str(rec[4])
        d = _safe_int(rec[20])
        if _name_match(name, wn):
            wr = _safe_int(rec[22])
            if wr > 0:
                ranks.append((d, wr))
        elif _name_match(name, ln):
            lr = _safe_int(rec[23])
            if lr > 0:
                ranks.append((d, lr))

    if not ranks:
        return False
    ranks.sort(key=lambda x: x[0], reverse=True)
    best_recent = min(r[1] for r in ranks[:10])
    return best_recent <= 100


# ─────────────────────────────────────────────────────────────────────────────
# FÍSICA Y ENTORNO
# ─────────────────────────────────────────────────────────────────────────────
def _clip_spw(val: float) -> float:
    return max(SPW_FLOOR, min(SPW_CEIL, val))


def _clip_rpw(val: float) -> float:
    return max(RPW_FLOOR, min(RPW_CEIL, val))


def apply_environment(spw: float, rpw: float, n: int,
                      altitude_m: int, fatigue_mins: int,
                      surface: str, tourney: str = "") -> tuple[float, float, list]:
    fb = get_fallback(surface)
    AVG_SPW, AVG_RPW = fb["spw"], fb["rpw"]
    adjustments = []

    # Shrinkage Bayesiano independiente
    confidence = min(n / float(SHRINKAGE_THRESHOLD), 1.0) if n > 0 else 0.0
    shrink = SHRINKAGE_BASE * (1.0 - confidence)

    adj_spw = spw * (1 - shrink) + AVG_SPW * shrink
    adj_rpw = rpw * (1 - shrink) + AVG_RPW * shrink

    if shrink > 0.01:
        adjustments.append(f"Shrinkage {shrink:.0%} (n={n})")

    # Altitud
    if altitude_m > 200:
        alt_bonus = (altitude_m / 1000.0) * 1.5
        adj_spw += alt_bonus
        adj_rpw -= alt_bonus * 0.5
        adjustments.append(f"Altitud +{alt_bonus:.2f} SPW ({altitude_m}m)")

    # B16: Court Pace via dict O(1)
    tourney_low = tourney.lower() if tourney else ""
    pace = None
    for key, tag in COURT_PACE.items():
        if key in tourney_low:
            pace = tag
            break
    if pace == "FAST":
        adj_spw += COURT_PACE_PP
        adj_rpw -= COURT_PACE_PP
        adjustments.append(f"Court Pace RÁPIDA ({tourney})")
    elif pace == "SLOW":
        adj_spw -= COURT_PACE_PP
        adj_rpw += COURT_PACE_PP
        adjustments.append(f"Court Pace LENTA ({tourney})")

    # Fatiga (única ubicación)
    if fatigue_mins > 150:
        penalty = min((fatigue_mins - 150) / 100.0 * 0.5, 2.0)
        adj_spw -= penalty
        adj_rpw -= penalty * 0.3
        adjustments.append(f"Fatiga -{penalty:.2f} SPW ({fatigue_mins} min recientes)")

    adj_spw = _clip_spw(adj_spw)
    adj_rpw = _clip_rpw(adj_rpw)
    return adj_spw, adj_rpw, adjustments


# ─────────────────────────────────────────────────────────────────────────────
# FASE CERO: AJUSTES DINÁMICOS
# ─────────────────────────────────────────────────────────────────────────────
def compute_adjustments(s1: dict, s2: dict, h2h: dict,
                        ctx: dict | None = None) -> tuple[float, float, list]:
    sum_adj = 0.0
    ic = 1.00
    notes = []

    if ctx:
        p1_h = ctx.get("p1_hand", "R")
        p2_h = ctx.get("p2_hand", "R")
        p1_ht = ctx.get("p1_height", 0)
        p2_ht = ctx.get("p2_height", 0)
        p1_l = ctx.get("p1_local", False)
        p2_l = ctx.get("p2_local", False)
        surf = ctx.get("surface", "Hard")
        lvl = ctx.get("level", "ATP")

        if p2_h == "L" and p1_h != "L":
            sum_adj -= ADJ_LEFTY
            notes.append(f"Factor V3: Oponente (P2) es Zurdo → -{ADJ_LEFTY}")
        if p1_h == "L" and p2_h != "L":
            sum_adj += ADJ_LEFTY
            notes.append(f"Factor V3: P1 es Zurdo → +{ADJ_LEFTY}")

        if surf in ["Hard", "Grass", "Carpet"]:
            if isinstance(p1_ht, (int, float)) and p1_ht > HEIGHT_THRESHOLD:
                sum_adj += ADJ_HEIGHT_FAST
                notes.append(f"Ajuste Físico: P1 alto ({p1_ht}cm) en {surf} → +{ADJ_HEIGHT_FAST}")
            if isinstance(p2_ht, (int, float)) and p2_ht > HEIGHT_THRESHOLD:
                sum_adj -= ADJ_HEIGHT_FAST
                notes.append(f"Ajuste Físico: P2 alto ({p2_ht}cm) en {surf} → -{ADJ_HEIGHT_FAST}")

        if surf == "Clay" and lvl in ["Challenger", "ITF"]:
            if p2_l:
                sum_adj -= ADJ_CLAY_TRAP
                notes.append(f"Clay Trap: P2 es Local en Challenger arcilla → -{ADJ_CLAY_TRAP}")
            if p1_l:
                sum_adj += ADJ_CLAY_TRAP
                notes.append(f"Clay Trap: P1 es Local en Challenger arcilla → +{ADJ_CLAY_TRAP}")

    for tag in s1.get("tags", []) + s2.get("tags", []):
        if "MIXTA" in tag and ic > 0.85:
            ic -= IC_PENALTY_MIXED
            notes.append(f"[MUESTRA MIXTA] IC -{IC_PENALTY_MIXED}")
        if "FALLBACK" in tag:
            ic -= IC_PENALTY_FALLBACK
            notes.append(f"[FALLBACK] IC -{IC_PENALTY_FALLBACK}")
        if "WEB" in tag:
            sum_adj -= ADJ_WEB_SOURCE
            notes.append(f"[FUENTE WEB] sum_adj -{ADJ_WEB_SOURCE}")

    if s1.get("n", 0) < 10 or s2.get("n", 0) < 10:
        ic = min(ic, 0.85)
        notes.append("Muestra <10 en superficie → IC ≤ 0.85")

    if h2h["total"] == 0:
        ic = min(ic, 0.90)
        notes.append("H2H = 0 partidos → IC ≤ 0.90")
    elif h2h["total"] >= 3:
        ratio = h2h["p1_wins"] / h2h["total"]
        if ratio >= 0.7:
            sum_adj += ADJ_H2H_DOMINANT
            notes.append(f"H2H dominante ({h2h['p1_wins']}-{h2h['p2_wins']}) → +{ADJ_H2H_DOMINANT}")
        elif ratio <= 0.3:
            sum_adj -= ADJ_H2H_DOMINANT
            notes.append(f"H2H desfavorable ({h2h['p1_wins']}-{h2h['p2_wins']}) → -{ADJ_H2H_DOMINANT}")

    dr1 = s1.get("dr_last5", 1.0)
    wins1 = s1.get("last5_wins", 0)
    if dr1 > 1.25:
        sum_adj += ADJ_DR_HOT
        notes.append(f"DR últimos 5 = {dr1:.3f} → +{ADJ_DR_HOT}")
    if wins1 <= 1:
        sum_adj -= ADJ_DR_COLD
        notes.append(f"Solo {wins1}/5 victorias recientes → -{ADJ_DR_COLD}")

    c1 = s1.get("clutch", 100.0)
    c2 = s2.get("clutch", 100.0)
    if c1 > 110 and c2 < 90:
        sum_adj += ADJ_CLUTCH
        notes.append(f"Clutch ventaja ({c1:.0f} vs {c2:.0f}) → +{ADJ_CLUTCH}")
    elif c1 < 90 and c2 > 110:
        sum_adj -= ADJ_CLUTCH
        notes.append(f"Clutch desventaja ({c1:.0f} vs {c2:.0f}) → -{ADJ_CLUTCH}")

    if abs(sum_adj) > 0.15:
        damping_factor = 0.10 / abs(sum_adj)
        sum_adj *= damping_factor
        notes.append(f"Damping aplicado por colinealidad (Factor: {damping_factor:.2f})")

    return sum_adj, max(ic, 0.50), notes


# ─────────────────────────────────────────────────────────────────────────────
# CLASIFICACIÓN (SEMÁFORO) — B4, Q1-Q5
# ─────────────────────────────────────────────────────────────────────────────
def classify(pmod: float, ev_pct: float, odd_dec: float,
             am_odd: int, level: str, n_total: int = 999) -> tuple[str, str, float]:
    """
    v9.4: Clasificación blindada con Kill-Switches Cuantitativos.
    - Q1: EV > 35% → Alucinación Persistente (bloqueo duro)
    - Q2: Valle de la Muerte (-200..-110) → solo EV 2%-10%
    - Q3: Shadow Bot "Back to Basura" → +100..+150 + n_total < 40
    - Q4: Cisne Negro > +200 → exclusión (rationale: -24.7% ROI empírico)
    - Q5: Bomba Nuclear refinada → +150..+199 con pmod ≥ 0.50
    - B4: Guard "EV > 20% en favoritos" aplica fuera del Valle Muerte.
    """
    ev_dec = ev_pct / 100.0
    implied = 1.0 / odd_dec if odd_dec > 0 else 1.0

    # Q1 — Kill-switch universal
    if ev_dec > KILL_SWITCH_EV:
        return "BLOQUEO - Alucinación Persistente (EV>35%)", "🛑", 0.0

    # B4 + Q2 — Favoritos
    if am_odd and am_odd < 0:
        in_death_valley = (-200 <= am_odd <= -110)
        if in_death_valley:
            # Q2: ventana EV estricta 2%-10%
            if not (DEATH_VALLEY_EV_MIN <= ev_dec <= DEATH_VALLEY_EV_MAX):
                if ev_dec < DEATH_VALLEY_EV_MIN:
                    return "RECHAZO - Micro-Edge en Valle de la Muerte", "🚫", 0.0
                else:
                    return "BLOQUEO - Trampa de Vig en Valle de la Muerte", "🛑", 0.0
        else:
            # B4: favoritos fuera del Valle Muerte → no permitir EV > 20%
            if ev_dec > 0.20:
                return "NO APUESTA - Trampa de Favorito (fuera DV)", "🛑", 0.0

    # Q4 — Cisne Negro (lista negra empírica: -24.7% ROI)
    if am_odd and am_odd > 200:
        return "EXCLUIDO - Riesgo Cisne Negro (>+200)", "🚫", 0.0

    # Q3 — Shadow Bot "Back to Basura"
    if level in ["Challenger", "ITF"] and pmod < 0.30:
        sparse = (n_total < SHADOW_SPARSE_THRESHOLD)
        in_sweet = (SHADOW_DEC_MIN <= odd_dec <= SHADOW_DEC_MAX)
        if sparse and in_sweet:
            return "BASURA PRO (Shadow Bet)", "💀", SHADOW_BET_STAKE

    if pmod < 0.45:
        return "BASURA", "🗑️", 0.0

    is_death_valley = (am_odd and am_odd < 0 and -200 <= am_odd <= -110)
    base_stake = kelly_fraction(pmod, odd_dec, is_death_valley) * 100.0

    # Q5 — Bomba Nuclear refinada
    if BOMBA_DEC_MIN <= odd_dec <= BOMBA_DEC_MAX and pmod >= BOMBA_PMOD_MIN:
        return "BOMBA NUCLEAR", "💣", max(3.0, base_stake)

    if odd_dec >= 2.0:
        if pmod >= 0.65: return "BOMBA NUCLEAR", "💣", max(3.0, base_stake)
        if ev_pct >= 6.5: return "DERECHA", "✅", base_stake
        if ev_pct >= 4.0: return "PARLAY", "🟡", base_stake
        return "BASURA", "🗑️", 0.0

    if pmod >= 0.70:
        if ev_pct >= 3.0: return "SUPER DERECHA", "🟢", base_stake
        if ev_pct >= 0.0: return "DERECHA", "✅", base_stake
        return "FAVORITO SOBREVENDIDO", "🔵", 0.0

    if ev_pct >= 6.5: return "DERECHA", "✅", base_stake
    if ev_pct >= 4.0: return "PARLAY", "🟡", base_stake
    if ev_pct >= 0.0 and pmod >= 0.55: return "VALOR MARGINAL", "⚠️", 0.0

    return "BASURA", "🗑️", 0.0


# ─────────────────────────────────────────────────────────────────────────────
# PARSER DE SPORTSBOOK — B12
# ─────────────────────────────────────────────────────────────────────────────
def extract_american_odds(text: str):
    m = re.search(r'([+-]\d{3,4})', text)
    if m:
        return text.replace(m.group(1), '').strip(), int(m.group(1))
    return text.strip(), None


def parse_matches(text: str) -> list[tuple]:
    text = text.replace('–', '-').replace('—', '-').replace('−', '-')
    results = []
    cur_league = "ATP"

    if re.search(r'(?i)\bvs\.?\b', text):
        for line in text.splitlines():
            line = line.strip()
            if not re.search(r'(?i)\bvs\.?\b', line):
                low = line.lower()
                if "itf" in low or "world tennis" in low:
                    cur_league = "ITF"
                elif "challenger" in low:
                    cur_league = "Challenger"
                elif "masters" in low or "1000" in low:
                    cur_league = "Masters"
                elif "grand slam" in low:
                    cur_league = "GrandSlam"
                elif "atp" in low:
                    cur_league = "ATP"
                continue
            parts = re.split(r'(?i)\s+vs\.?\s+', line)
            if len(parts) == 2:
                p1, o1 = extract_american_odds(parts[0])
                p2, o2 = extract_american_odds(parts[1])
                if p1 and p2:
                    results.append((p1, o1, p2, o2, cur_league))
        if results:
            return results

    DATE_RE = re.compile(r'^(\d{1,2}\s+[a-zA-Z]{3}\s+)?\d{2}:\d{2}$')
    COUNTER_RE = re.compile(r'^\+\s*\d{1,2}\s*(Streaming)?$', re.I)
    ODD_LONE = re.compile(r'^[+-]\d{3,4}$')
    name_q, odd_q = [], []

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        low = line.lower()
        if any(kw in low for kw in METADATA_KW):
            if "itf" in low or "world tennis" in low:
                cur_league = "ITF"
            elif "challenger" in low:
                cur_league = "Challenger"
            elif "masters" in low or "1000" in low:
                cur_league = "Masters"
            elif "atp" in low:
                cur_league = "ATP"
            continue
        if DATE_RE.match(line) or COUNTER_RE.match(line):
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
                            name_q.pop(0), odd_q.pop(0), cur_league))

    # B12: warning si quedan huérfanos
    if name_q or odd_q:
        msg = []
        if name_q:
            msg.append(f"{len(name_q)} nombre(s) sin cuota: {name_q}")
        if odd_q:
            msg.append(f"{len(odd_q)} cuota(s) sin nombre: {odd_q}")
        try:
            st.warning("⚠️ Parser: " + " | ".join(msg))
        except Exception:
            logger.warning("Parser orphans: %s", msg)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# MATEMÁTICA LOCAL — B6, B7, B21
# ─────────────────────────────────────────────────────────────────────────────
def american_to_decimal(odd: int) -> float:
    if odd > 0:
        return (odd / 100.0) + 1.0
    elif odd < 0:
        return (100.0 / abs(odd)) + 1.0
    return 1.0


def no_vig(odd1: int, odd2: int) -> tuple:
    i1 = 1 / american_to_decimal(odd1)
    i2 = 1 / american_to_decimal(odd2)
    t = i1 + i2
    p1, p2 = i1 / t, i2 / t
    return p1, p2, round(1 / p1, 3), round(1 / p2, 3)


def calc_ev(prob: float, dec_odd: float) -> float:
    return prob * (dec_odd - 1) - (1 - prob)


def calibrate_probability(pmod: float, american_odds: int) -> float:
    """
    B5 FIX: penalty_factor aplica a todo el rango Death Valley.
    """
    if not american_odds:
        return pmod
    p_casa = 1 / american_to_decimal(american_odds)

    pmod_capped = min(pmod, 0.88)
    T = 0.65
    pmod_suavizado = (pmod_capped ** T) / ((pmod_capped ** T) + ((1 - pmod_capped) ** T))
    ev_inicial = (pmod_suavizado / p_casa) - 1

    if ev_inicial > 0.20:
        peso_casa = min(0.85, ev_inicial * 2)
        pmod_final = (pmod_suavizado * (1 - peso_casa)) + (p_casa * peso_casa)
    else:
        pmod_final = pmod_suavizado

    # B5: penalty_factor aplica SIEMPRE dentro del Valle Muerte
    if american_odds < 0 and -200 <= american_odds <= -110:
        penalty_factor = 1.0 - (abs(american_odds + 150) / 100.0) * 0.6
        if ev_inicial > 0.4:
            lambda_shrink = min(0.85, ev_inicial * 0.8)
            pmod_calibrada = p_casa * lambda_shrink + pmod_final * (1 - lambda_shrink)
            return min(pmod_calibrada, 0.82) * penalty_factor
        # B5: fuera del if, dentro del Valle, también penalizar
        return pmod_final * penalty_factor

    return pmod_final


def kelly_fraction(pmod: float, dec_odd: float, is_death_valley: bool) -> float:
    """B6 FIX: guard explícito si EV ≤ 0."""
    # B6: rechazo explícito de EV negativo
    if pmod * dec_odd <= 1.0:
        return 0.0
    b = dec_odd - 1.0
    p = pmod
    q = 1.0 - p
    kelly = (b * p - q) / b if b > 0 else 0.0
    if kelly <= 0:
        return 0.0
    if is_death_valley and pmod > 0.75:
        return max(0.0, kelly * 0.10)
    return max(0.0, kelly * 0.25)


def log5_serve(spw: float, rpw: float) -> float:
    A, B = spw / 100, rpw / 100
    d = A * (1 - B) + (1 - A) * B
    return (A * (1 - B)) / d if d else 0.5


def game_prob(p: float) -> float:
    """Markov analítico: prob de ganar un game al saque con prob p de ganar punto."""
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    q = 1 - p
    p_deuce = (p ** 2) / (p ** 2 + q ** 2)
    return (p ** 4) * (1 + 4 * q + 10 * q ** 2) + 20 * (p ** 3) * (q ** 3) * p_deuce


def sim_tiebreak(p_first: float, p_second: float, rng: random.Random) -> int:
    a = b = total = 0
    while True:
        if total == 0:
            win = rng.random() < p_first
        else:
            block = (total - 1) // 2
            win = rng.random() < (p_second if block % 2 == 0 else p_first)
        if win:
            a += 1
        else:
            b += 1
        total += 1
        if a >= 7 and a - b >= 2:
            return 1
        if b >= 7 and b - a >= 2:
            return 0


def sim_set(pg_A: float, pg_B: float, pA: float, pB: float,
            rng: random.Random, a_first: bool = True) -> tuple[int, bool]:
    gA = gB = 0
    a_serves = a_first
    while True:
        if gA == 6 and gB == 6:
            tf = pA if a_serves else pB
            ts = pB if a_serves else pA
            w = sim_tiebreak(tf, ts, rng)
            if w:
                gA += 1
            else:
                gB += 1
            return (1 if gA > gB else 0), not a_serves

        if a_serves:
            if rng.random() < pg_A:
                gA += 1
            else:
                gB += 1
        else:
            if rng.random() < pg_B:
                gB += 1
            else:
                gA += 1
        a_serves = not a_serves

        if gA >= 6 and gA - gB >= 2:
            return 1, a_serves
        if gB >= 6 and gB - gA >= 2:
            return 0, a_serves


def sim_match(pA: float, pB: float, best_of: int = 3,
              n: int = MC_ITERATIONS, seed: int = 0) -> float:
    """B7 FIX: RNG local (random.Random) en vez de random.seed() global."""
    rng = random.Random(seed)
    needed = 2 if best_of == 3 else 3
    wins = 0
    pg_A = game_prob(pA)
    pg_B = game_prob(pB)
    for _ in range(n):
        sA = sB = 0
        a_srv = True
        while sA < needed and sB < needed:
            w, a_srv = sim_set(pg_A, pg_B, pA, pB, rng, a_srv)
            if w:
                sA += 1
            else:
                sB += 1
        if sA == needed:
            wins += 1
    return wins / n


def make_seed(p1: str, p2: str) -> int:
    """B21 FIX: SHA256 en vez de MD5."""
    key = f"{p1.lower().strip()}|{p2.lower().strip()}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


# ─────────────────────────────────────────────────────────────────────────────
# GEMINI — FALLBACK WEB + ANÁLISIS H48
# ─────────────────────────────────────────────────────────────────────────────
def gemini_stats(name: str) -> dict:
    client = get_gemini_client()
    synthetic_fallback = {
        "n": 1, "n_total": 1, "hold": 0.0, "brk": 0.0,
        "spw": 62.0, "rpw": 38.0,
        "clutch": 100.0, "dr_last5": 1.0, "last5_wins": 0,
        "fatigue_mins": 0, "tags": ["SYNTHETIC FALLBACK"], "source": "Gemini Fallback",
    }
    if not client:
        return synthetic_fallback
    try:
        prompt = (
            f'Busca estadísticas de tenis recientes de "{name}". '
            f'Necesito SOLO: Service Points Won % y Return Points Won %. '
            f'Responde SOLO con JSON crudo: {{"spw_pct": 64.5, "rpw_pct": 38.2}}'
        )
        r = client.models.generate_content(
            model=GEMINI_MODEL, contents=prompt,
            config=types.GenerateContentConfig(
                tools=[{"google_search": {}}], temperature=0.1
            )
        )
        raw = r.text.replace("```json", "").replace("```", "").strip()
        s, e = raw.find('{'), raw.rfind('}')
        if s != -1 and e != -1:
            data = json.loads(raw[s:e + 1])
        else:
            data = json.loads(raw)
        return {
            "n": 0, "n_total": 0, "hold": 0.0, "brk": 0.0,
            "spw": round(max(SPW_FLOOR_MIN, float(data.get("spw_pct", 62.0))), 1),
            "rpw": round(max(SPW_FLOOR_MIN, float(data.get("rpw_pct", 38.0))), 1),
            "clutch": 100.0, "dr_last5": 1.0, "last5_wins": 0,
            "fatigue_mins": 0, "tags": ["FUENTE WEB"], "source": "Gemini Search",
        }
    except Exception as err:
        _log_debug("gemini_stats", err)
        return synthetic_fallback


def gemini_full_analysis(p1: str, p2: str, report: str) -> str:
    client = get_gemini_client()
    if not client:
        return "❌ Sin API Key."
    try:
        prompt = (
            f"Eres analista de apuestas de tenis. "
            f"Busca hechos de {p1} y {p2} en últimas 48h (fatiga, lesiones, retiros).\n"
            f"AUDITORÍA del reporte:\n{report}\n"
            f"VEREDICTO: EV>5%→✅VERDE | 0<EV≤5%→⚠️AMARILLO | EV<0 o lesión→🚫ROJO\n"
            f"Responde con: CONTEXTO H48, AUDITORÍA, VEREDICTO (corto)."
        )
        r = client.models.generate_content(
            model=GEMINI_MODEL, contents=prompt,
            config=types.GenerateContentConfig(
                tools=[{"google_search": {}}], temperature=0.2
            )
        )
        return r.text
    except Exception as err:
        _log_debug("gemini_full_analysis", err)
        return f"Error Gemini: {err}"


# ─────────────────────────────────────────────────────────────────────────────
# GOOGLE SHEETS — B8, B9, B19
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Prediction:
    match_id: str
    p1: str
    p2: str
    p_mod: float
    p_casa: float
    odd: int | None
    ev_val: float | None
    tier: str
    league: str
    ic: float
    real_winner: str = ""


def _sheets_key() -> str:
    try:
        return st.secrets.get("SHEETS_KEY", "") or os.environ.get("SHEETS_KEY", "")
    except Exception:
        return os.environ.get("SHEETS_KEY", "")


@st.cache_resource
def get_sheet():
    try:
        import gspread
        if "gcp_service_account" not in st.secrets:
            return None
        key = _sheets_key()
        if not key:
            return None
        gc = gspread.service_account_from_dict(st.secrets["gcp_service_account"])
        return gc.open_by_key(key).sheet1
    except Exception as err:
        _log_debug("get_sheet", err)
        return None


def log_prediction(pred: Prediction, max_retries: int = 3) -> bool:
    """B8 FIX: retry con backoff exponencial para mitigar race conditions."""
    sheet = get_sheet()
    if not sheet:
        return False
    for attempt in range(max_retries):
        try:
            col_ids = sheet.col_values(1)
            if pred.match_id in col_ids:
                if pred.real_winner:
                    idx = col_ids.index(pred.match_id) + 1
                    sheet.update_cell(idx, 10, pred.real_winner)
                return True
            sheet.append_row([
                pred.match_id, datetime.now().strftime("%Y-%m-%d %H:%M"),
                pred.p1, pred.p2, f"{pred.p_mod * 100:.1f}%",
                f"{pred.p_casa * 100:.1f}%" if pred.p_casa else "S/C",
                str(pred.odd) if pred.odd else "S/C",
                f"{pred.ev_val * 100:+.1f}%" if pred.ev_val is not None else "S/C",
                pred.tier, pred.real_winner, pred.league, f"{pred.ic:.2f}"
            ])
            return True
        except Exception as err:
            _log_debug(f"log_prediction[attempt={attempt + 1}]", err)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # backoff: 1s, 2s, 4s...
    return False


def get_pending_matches():
    sheet = get_sheet()
    if not sheet:
        return []
    try:
        all_rows = sheet.get_all_values()
        pending = []
        for i, row in enumerate(all_rows):
            if i == 0 or len(row) < 4:
                continue
            winner = row[9] if len(row) > 9 else ""
            if winner.strip() == "":
                pending.append({"match_id": row[0], "p1": row[2], "p2": row[3]})
        return pending
    except Exception as err:
        _log_debug("get_pending_matches", err)
        return []


def save_predictions_callback():
    partidos = st.session_state.get('last_procesados', [])
    logged = 0
    for p in partidos:
        try:
            pred = Prediction(
                match_id=p['match_id'], p1=p['p1'], p2=p['p2'],
                p_mod=p['pmod'], p_casa=p['nv1'], odd=p['odd1'],
                ev_val=p['ev1'], tier=p['tier'], league=p['league'],
                ic=p['ic'], real_winner=""
            )
            if log_prediction(pred):
                logged += 1
        except Exception as err:
            _log_debug("save_predictions_callback", err)
    st.session_state.save_msg = f"¡{logged} partidos guardados en el Oráculo!"


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title=f"Quantum Tennis {APP_VERSION}", page_icon="🎾", layout="wide")
    st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    p, span, div, h1, h2, h3, h4, h5, h6, caption, li, td, th { color: #FFFFFF !important; }
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        background-color: #262730; color: white !important; border: 1px solid #4CAF50;
    }
    .stButton>button { background-color: #1E1E1E; color: #4CAF50 !important;
        border: 1px solid #4CAF50; font-weight: bold; }
    .stButton>button:hover { background-color: #4CAF50; color: white !important; }
    </style>
    """, unsafe_allow_html=True)

    st.title(f"🎾 Quantum Tennis Engine {APP_VERSION} — Quant Risk Edition")
    st.caption(
        "Log5 · Markov MC 10k · Oráculo 40k+ (2022-2024) · "
        "Kill-Switches EV · Shadow Bot · Kelly 1/4 · Gemini 2.5 Pro"
    )

    with st.expander("👁️ VER EL GRAN ORÁCULO DE GOOGLE SHEETS", expanded=False):
        sheet_key = _sheets_key()
        if sheet_key:
            components.iframe(
                f"https://docs.google.com/spreadsheets/d/{sheet_key}/edit?usp=sharing",
                height=500, scrolling=True
            )
        else:
            st.info("Configura SHEETS_KEY en st.secrets o .env para ver el Oráculo.")

    if st.session_state.get("save_msg"):
        st.success(st.session_state.save_msg)
        st.balloons()
        st.session_state.save_msg = ""

    if not gemini_available():
        st.error(
            "⚠️ Falta GOOGLE_API_KEY. Configúralo en `.env` (local) o en `st.secrets` "
            "(Streamlit Cloud). Genera tu key en https://aistudio.google.com/app/apikey"
        )
        st.stop()

    if "txt" not in st.session_state:
        st.session_state.txt = ""

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        best_of = st.radio("Formato:", [3, 5], horizontal=True,
                           format_func=lambda x: f"Mejor de {x}")
    with c2:
        db_path = st.text_input("BD", value=DB_PATH, label_visibility="collapsed")
    with c3:
        if st.button("🗑️ Limpiar", use_container_width=True):
            st.session_state.txt = ""
            st.session_state.analisis_ejecucion = None
            st.rerun()

    txt = st.text_area(
        "📋 Pega los partidos (formato Caliente o 'Sinner -120 vs Alcaraz +100'):",
        key="txt", height=160,
        placeholder="Alcaraz C\nSinner J\n-140\n+110",
    )

    btn_analizar = st.button("🚀 Analizar", type="primary", use_container_width=True)

    # ── AGENTE LIQUIDADOR ────────────────────────────────────────────────────
    st.divider()
    with st.container():
        st.subheader("🎯 AGENTE LIQUIDADOR — 100% Automático")
        st.write("Un click. Gemini buscará en Flashscore y subirá al Oráculo los ganadores.")

        if st.button("🤖 Iniciar Liquidación Automática en Internet"):
            with st.spinner("Conectando al Oráculo, buscando y subiendo resultados..."):
                pendientes = get_pending_matches()
                if pendientes:
                    sugerencias = batch_guess_winners_gemini(pendientes)
                    reporte_acciones = []
                    for p in pendientes:
                        win = sugerencias.get(p['match_id'], "Ninguno")
                        real_winner = "Ninguno"
                        if win != "Ninguno":
                            if p['p1'].lower() in win.lower() or win.lower() in p['p1'].lower():
                                real_winner = p['p1']
                            elif p['p2'].lower() in win.lower() or win.lower() in p['p2'].lower():
                                real_winner = p['p2']

                        if real_winner != "Ninguno":
                            pred = Prediction(
                                match_id=p['match_id'], p1=p['p1'], p2=p['p2'],
                                p_mod=0, p_casa=0, odd=None, ev_val=None,
                                tier="", league="", ic=0, real_winner=real_winner
                            )
                            ok = log_prediction(pred)
                            if ok:
                                reporte_acciones.append(f"✅ **{p['p1']} vs {p['p2']}** ➔ Subido: **{real_winner}**")
                            else:
                                reporte_acciones.append(f"❌ **{p['p1']} vs {p['p2']}** ➔ Fallo de guardado.")
                        else:
                            reporte_acciones.append(f"⏳ **{p['p1']} vs {p['p2']}** ➔ Pendiente.")
                    st.session_state.pending_error = ""
                    st.session_state.reporte_liq = reporte_acciones
                else:
                    st.session_state.pending_error = (
                        "⚠️ No se encontró la llave de Google Sheets o no hay partidos pendientes."
                    )
                    st.session_state.reporte_liq = []
            st.rerun()

        if st.session_state.get("pending_error"):
            st.warning(st.session_state.pending_error)
            st.session_state.pending_error = ""
        if st.session_state.get("reporte_liq"):
            st.success("Operación Automática Finalizada. Resultados:")
            for linea in st.session_state.reporte_liq:
                st.markdown(linea)
        else:
            st.write("Haz click en Iniciar Liquidación para conectar con el Oráculo.")

    st.divider()

    # ── ANÁLISIS ─────────────────────────────────────────────────────────────
    if btn_analizar:
        if not txt.strip():
            st.warning("Pega al menos un partido.")
        else:
            st.session_state.analisis_ejecucion = txt

    if not st.session_state.get("analisis_ejecucion"):
        # Panel debug si hay errores acumulados
        if st.session_state.get("debug_errors"):
            with st.expander("🐞 Debug — errores acumulados"):
                for e in st.session_state["debug_errors"][-20:]:
                    st.code(e)
        return

    partidos = parse_matches(st.session_state.analisis_ejecucion)
    if not partidos:
        st.error("No se encontraron pares.")
        return

    # B11: oráculo + índice
    oracle, index, n_skipped = load_oracle(db_path)
    if not oracle:
        st.error(f"Oráculo vacío o no encontrado: {db_path}")
        return
    st.caption(
        f"📂 Oráculo cargado: {len(oracle):,} registros en RAM · "
        f"índice: {len(index):,} apellidos"
        + (f" · ⚠️ {n_skipped} registros corruptos descartados" if n_skipped else "")
    )

    partidos_procesados = []

    for p1_raw, odd1, p2_raw, odd2, league in partidos:
        p1, p2 = p1_raw or "Jugador 1", p2_raw or "Jugador 2"

        # B15: UTR con word boundary
        if UTR_RE.search(p1) or UTR_RE.search(p2):
            st.error(f"🚫 EXCLUIDO | TORNEO UTR: {p1} vs {p2}")
            continue

        dec1_check = american_to_decimal(odd1) if odd1 else 2.0
        if dec1_check < 1.18:
            st.warning(f"🚫 EXCLUIDO | SUPER-FAVORITO (cuota {dec1_check:.2f}): {p1}")
            continue

        st.divider()
        st.subheader(f"⚡ {p1}  vs  {p2}")

        with st.spinner("🌍 Detectando torneo y físico…"):
            ctx = extract_match_context(p1, p2)

        surface = ctx.get("surface", "Hard")
        tourney = ctx.get("tourney", "Unknown")
        altitude = ctx.get("altitude", 0)
        level = ctx.get("level", league)
        p1_hnd = ctx.get("p1_hand", "R")
        p1_hgt = ctx.get("p1_height", 185)
        p1_loc = ctx.get("p1_local", False)
        p2_hnd = ctx.get("p2_hand", "R")
        p2_hgt = ctx.get("p2_height", 185)
        p2_loc = ctx.get("p2_local", False)
        # B2: removidas variables muertas p1_top100 / p2_top100

        st.caption(
            f"📍 {tourney} | {surface} | {level} | Alt: {altitude}m | "
            f"P1: {p1_hgt}cm ({p1_hnd}) vs P2: {p2_hgt}cm ({p2_hnd})"
        )

        if check_tier_drop(p1, level, oracle, index):
            st.error(f"🚫 TIER DROP: {p1} (Top 100 en {level})")
            continue

        c_s1, c_s2 = st.columns(2)
        with c_s1:
            with st.spinner(f"Stats {p1}…"):
                s1 = get_stats(p1, surface, level, oracle, index)
                if not s1:
                    st.info(f"🌐 {p1} no en oráculo → web…")
                    s1 = gemini_stats(p1)
        with c_s2:
            with st.spinner(f"Stats {p2}…"):
                s2 = get_stats(p2, surface, level, oracle, index)
                if not s2:
                    st.info(f"🌐 {p2} no en oráculo → web…")
                    s2 = gemini_stats(p2)

        if not s1 or not s2:
            st.error("Stats insuficientes.")
            continue

        h2h = get_h2h(p1, p2, oracle, index)
        tank_p1 = check_tanking(p1, oracle, index)
        tank_p2 = check_tanking(p2, oracle, index)

        sum_adj, ic, adj_notes = compute_adjustments(s1, s2, h2h, ctx)

        adj_spw1, adj_rpw1, env1 = apply_environment(
            s1["spw"], s1["rpw"], s1["n"], altitude, s1["fatigue_mins"], surface, tourney
        )
        adj_spw2, adj_rpw2, env2 = apply_environment(
            s2["spw"], s2["rpw"], s2["n"], altitude, s2["fatigue_mins"], surface, tourney
        )

        pA = log5_serve(adj_spw1, adj_rpw2)
        pB = log5_serve(adj_spw2, adj_rpw1)
        seed = make_seed(p1, p2)

        with st.spinner(f"Corriendo {MC_ITERATIONS:,} simulaciones…"):
            mc_A = sim_match(pA, pB, best_of, seed=seed)
            mc_B = 1 - mc_A

        pmod_final = max(0.01, min(0.99, mc_A * (1 + sum_adj)))

        st.markdown("#### 🧮 Motor Matemático")
        m1, m2 = st.columns(2)
        ev_lines = []
        nv1_val = ev1_val = tier1_val = None

        for col, name, p_srv, mc_w, odd, stats, adj_spw, adj_rpw in [
            (m1, p1, pA, pmod_final, odd1, s1, adj_spw1, adj_rpw1),
            (m2, p2, pB, 1 - pmod_final, odd2, s2, adj_spw2, adj_rpw2),
        ]:
            col.markdown(f"**{name}**")
            tag_str = " · ".join(stats.get("tags", []))
            col.caption(
                f"Fuente: {stats['source']} · "
                f"n={stats['n']} (sup) / {stats.get('n_total', stats['n'])} (total) · "
                f"SPW {stats['spw']}% → {adj_spw:.1f}% · "
                f"RPW {stats['rpw']}% → {adj_rpw:.1f}% · "
                f"Hold {stats['hold']}% · "
                f"Clutch {stats.get('clutch', 0):.0f} · "
                f"DR5 {stats.get('dr_last5', 0):.3f}"
                + (f" · ⚠️ {tag_str}" if tag_str else "")
            )
            col.metric("P(srv/punto) Log5", f"{p_srv * 100:.1f}%")
            col.metric("P(match) Modelo", f"{mc_w * 100:.1f}%")

            if odd:
                mc_w_cal = calibrate_probability(mc_w, odd)
                _nv1, _nv2, _f1, _f2 = no_vig(odd1 or 100, odd2 or 100)
                nv_this = _nv1 if col is m1 else _nv2
                fair = _f1 if col is m1 else _f2
                dec_odd = american_to_decimal(odd)
                ev_val = calc_ev(mc_w_cal, dec_odd)
                edge = (mc_w_cal - nv_this) * 100
                # Q3: pasar n_total para Shadow Bot sparse-data filter
                tier, emoji, stake_pct = classify(
                    mc_w_cal, ev_val * 100, dec_odd, odd, level,
                    n_total=stats.get("n_total", 999),
                )

                if mc_w_cal != mc_w:
                    col.metric("P(match) Calibrada", f"{mc_w_cal * 100:.1f}%",
                               f"Real Log5: {mc_w * 100:.1f}%", delta_color="off")

                col.metric("P(match) Casa No-Vig", f"{nv_this * 100:.1f}%",
                           f"Fair odd {fair}", delta_color="off")
                col.metric("💰 Expected Value", f"{ev_val * 100:+.1f}%",
                           delta_color="normal" if ev_val > 0 else "inverse")
                col.metric("Edge vs Casa", f"{edge:+.1f}%")

                if stake_pct > 0:
                    col.success(f"**Stake Sugerido: {stake_pct:.2f}% de Bankroll**")
                else:
                    col.error("**NO APOSTAR**")

                if col is m1:
                    nv1_val = _nv1
                    ev1_val = ev_val
                    tier1_val = tier
                    pmod_final = mc_w_cal
                    ev_lines.append(
                        f"{name}: EV={ev_val * 100:+.2f}% | Edge={edge:+.1f}% | "
                        f"{tier} {emoji} | Stake {stake_pct:.1f}%"
                    )
            else:
                col.info("Sin cuota → EV no calculable")

        st.markdown("---")

        if odd1 and odd2 and ev1_val is not None:
            dec1 = american_to_decimal(odd1)
            emoji1 = classify(
                pmod_final, ev1_val * 100, dec1, odd1, level,
                n_total=s1.get("n_total", 999),
            )[1]
            st.markdown(f"""
#### 📊 Salida Quantum Engine

| Campo | Valor |
|---|---|
| **Partido** | {p1} vs {p2} |
| **Torneo** | {tourney} ({level}) |
| **Superficie** | {surface} |
| **Pmod (Real Prob)** | {pmod_final * 100:.1f}% |
| **Implied Obj (Cuota)** | {nv1_val * 100:.1f}% |
| **EV Edge** | {ev1_val * 100:+.1f}% |
| **IC Confianza** | {ic:.2f} {' '.join('[' + t + ']' for t in s1.get('tags', []))} |
| **STATUS FINAL** | {tier1_val} {emoji1} |
| **TANKING CHECK** | {'⚠️ SÍ' if tank_p1 else 'NO'} ({p1}) \\| {'⚠️ SÍ' if tank_p2 else 'NO'} ({p2}) |
| **H2H reciente** | {h2h['p1_wins']}-{h2h['p2_wins']} ({h2h['total']} partidos ≤36m) |
""")

            with st.expander("🔧 Ajustes Dinámicos (Físicos V7 + Matemática)"):
                for note in adj_notes + env1 + env2:
                    st.write(f"• {note}")
                st.write(f"**sum_adj total: {sum_adj:+.3f}**")

            partidos_procesados.append({
                "match_id": f"{p1}_{p2}_{datetime.now().strftime('%Y%m%d')}",
                "p1": p1, "p2": p2, "pmod": pmod_final,
                "nv1": nv1_val, "odd1": odd1, "ev1": ev1_val,
                "tier": tier1_val, "league": level, "ic": ic,
            })

        st.markdown("#### 🤖 Gemini — Análisis H48")
        report = (
            f"{p1}: SPW={s1['spw']}%→{adj_spw1:.1f}% RPW={s1['rpw']}%→{adj_rpw1:.1f}% "
            f"Hold={s1['hold']}% (n={s1['n']}, {s1['source']})\n"
            f"{p2}: SPW={s2['spw']}%→{adj_spw2:.1f}% RPW={s2['rpw']}%→{adj_rpw2:.1f}% "
            f"Hold={s2['hold']}% (n={s2['n']}, {s2['source']})\n"
            f"Log5: {p1}={pA:.3f} | {p2}={pB:.3f}\n"
            f"MC ({MC_ITERATIONS} iter, Bo{best_of}): "
            f"{p1}={pmod_final * 100:.1f}% | {p2}={(1 - pmod_final) * 100:.1f}%\n"
            f"H2H: {h2h['p1_wins']}-{h2h['p2_wins']} (≤36m)\n"
            + ("\n".join(ev_lines) if ev_lines else "Sin cuotas") + "\n"
            f"IC: {ic:.2f} | sum_adj: {sum_adj:+.3f}"
        )
        with st.spinner("Gemini analizando…"):
            st.markdown(gemini_full_analysis(p1, p2, report))

    if partidos_procesados:
        st.session_state.last_procesados = partidos_procesados
        st.button(
            "💾 Guardar en Oráculo",
            on_click=save_predictions_callback,
            type="primary", use_container_width=True
        )

    # Panel debug siempre al final
    if st.session_state.get("debug_errors"):
        with st.expander("🐞 Debug — errores de esta sesión"):
            for e in st.session_state["debug_errors"][-20:]:
                st.code(e)


if __name__ == "__main__":
    main()
