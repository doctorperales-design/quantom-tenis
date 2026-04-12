"""
Quantum Tennis Engine v9.0 — Hardened Audit Edition
Markov Chains · Shrinkage Bayesiano · Oráculo Enriquecido 2022-2025

CHANGELOG v9.0 vs v8.1:
──────────────────────────────────────────────────────────────────────
BUG  #1  classify() tenía huecos lógicos: underdog con EV +8% salía BASURA,
         favorito con pmod=0.55 y EV +7% salía BASURA. Reescrito completo.
BUG  #2  Doble penalización de fatiga: apply_environment() reduce SPW/RPW
         Y compute_adjustments() reduce sum_adj. Fatiga ahora SOLO en environment.
BUG  #3  MC_ITERATIONS = 50000 hacía la app inutilizable en móvil (~15s por partido).
         Reducido a 10000 (error estándar ~0.5%, suficiente para EV de apuestas).
BUG  #4  _name_match() demasiado permisivo: "Ru" matcheaba "Rune H", "Ruud C", "Rublev A".
         Ahora exige match de apellido completo o nombre completo.
BUG  #5  Sin seed en Monte Carlo: resultados diferentes en cada run. Agregado seed
         determinístico basado en nombres de jugadores para reproducibilidad.
BUG  #6  Shrinkage proporcional (ratio SPW/RPW fijo) impedía que un RPW anormalmente
         bajo se corrigiera hacia la media. Cambiado a shrinkage independiente por métrica.
BUG  #7  BD leída N veces por partido (stats×2 + h2h + tanking×2 + tier_drop×2 = 8 lecturas
         de 40k líneas). Ahora se carga UNA vez en memoria con @st.cache_data.
BUG  #8  get_gemini_stats_fallback() definido pero nunca llamado (dead code). Eliminado.
BUG  #9  Court Pace multiplicativo (v8.0: *=1.03) distorsionaba más en valores altos.
         Cambiado a aditivo ±1.5 pp (correcto en v8.1, preservado).
BUG  #10 sim_set: condiciones gA==7 / gB==7 son inalcanzables (el tiebreak ya retorna).
         Eliminado dead code.

MEJORA #1  Oráculo cargado en RAM una sola vez → 8x más rápido por partido.
MEJORA #2  classify() con lógica EV-first + Pmod-guard completa sin huecos.
MEJORA #3  Seed determinístico para MC: mismo partido = mismo resultado siempre.
MEJORA #4  Alerta de varianza en cuotas cercanas (tiebreak-heavy → solo PARLAY).
MEJORA #5  compute_adjustments() limpia: fatiga solo en environment, sin doble conteo.
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
from datetime import datetime, timedelta
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
GEMINI_MODEL  = "gemini-2.5-pro"
DB_PATH       = "matches_comprimidos.csv"
MC_ITERATIONS = 10000  # BUG #3 FIX: 50k→10k. Error estándar ~0.5% (suficiente para EV)

# ─────────────────────────────────────────────────────────────────────────────
# FALLBACKS EMPÍRICOS — 40,225 registros (2022-2024)
# ─────────────────────────────────────────────────────────────────────────────
SURFACE_FALLBACKS = {
    "Hard":   {"hold": 74.0, "spw": 68.9, "rpw": 42.8},
    "Clay":   {"hold": 70.5, "spw": 66.1, "rpw": 45.9},
    "Grass":  {"hold": 77.6, "spw": 70.9, "rpw": 39.5},
    "Carpet": {"hold": 75.8, "spw": 69.5, "rpw": 41.1},
}

FAST_COURTS = {
    "dubai", "cincinnati", "brisbane", "stuttgart", "s hertogenbosch",
    "queens", "halle", "wimbledon", "us open", "tokyo", "vienna",
    "basel", "paris masters", "atp finals",
}
SLOW_COURTS = {
    "miami", "indian wells", "rome", "madrid", "roland garros",
    "monte carlo", "barcelona", "canadian open", "montreal",
    "buenos aires", "rio", "houston", "umag",
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
# ORÁCULO EN RAM — BUG #7 FIX / MEJORA #1
# Carga el archivo completo UNA sola vez y lo mantiene en caché.
# Todas las funciones (get_stats, get_h2h, check_tanking, check_tier_drop)
# operan sobre esta lista en memoria en lugar de releer el disco.
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_oracle(path: str) -> list[list]:
    """Carga el oráculo completo en RAM. Se ejecuta UNA vez."""
    records = []
    try:
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('['):
                    try:
                        rec = json.loads(line)
                        if len(rec) >= 20:
                            records.append(rec)
                    except Exception:
                        pass
                else:
                    # Fallback CSV
                    try:
                        parts = next(csv.reader([line]))
                        parsed = []
                        for p in parts:
                            p = p.strip()
                            if p.lstrip('-').isdigit():
                                parsed.append(int(p))
                            else:
                                try:
                                    parsed.append(float(p))
                                except ValueError:
                                    parsed.append(p)
                        if len(parsed) >= 20:
                            records.append(parsed)
                    except Exception:
                        pass
    except FileNotFoundError:
        st.warning(f"⚠️ Archivo no encontrado: {path}")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACCIÓN AUTÓNOMA (GEMINI)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def extract_match_context(p1: str, p2: str) -> dict:
    client = get_gemini_client()
    default = {
        "surface": "Hard", "tourney": "Unknown", "altitude": 0, "level": "ATP",
        "p1_hand": "R", "p1_height": 185, "p1_local": False,
        "p2_hand": "R", "p2_height": 185, "p2_local": False
    }
    if not client:
        return default
    try:
        prompt = (
            f'Investiga el partido de tenis entre "{p1}" (P1) y "{p2}" (P2).\n'
            f'Detecta 4 cosas crudas:\n'
            f'1. Torneo, superficie (Hard, Clay, Grass, Carpet), nivel (ATP/Masters/GrandSlam/Challenger/ITF), altitud en metros.\n'
            f'2. Mano Dominante de ambos (L para Zurdo, R para Diestro).\n'
            f'3. Estatura (altura) de ambos en cm (entero).\n'
            f'4. Localía: ¿Algún jugador es de la misma nacionalidad/país sede del torneo? (True/False).\n'
            f'Responde SOLO con JSON crudo:\n'
            f'{{"surface": "Hard", "tourney": "Monte Carlo", "altitude": 35, "level": "Masters", "p1_hand": "R", "p1_height": 185, "p1_local": false, "p2_hand": "L", "p2_height": 198, "p2_local": true}}'
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
    except Exception:
        pass
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
        
        prompt = "Eres un analista de resultados de tenis. Ve y busca en internet (usa Flashscore, ATP Tour, WTA Tour u otros sitios de tenis en vivo) el resultado de los siguientes partidos recientes:\n"
        for m in chunk:
            prompt += f"- ID '{m['match_id']}': {m['p1']} vs {m['p2']}\n"
        prompt += """
Devuelve ÚNICAMENTE un JSON crudo con las respuestas. 
Usa la estructura: {"match_id": "Nombre Exacto del Ganador"}.
Usa estrictamente los nombres de los jugadores tal y como te los envié.
Si el partido no se ha jugado, se canceló, o no encuentras resultado oficial, pon "Ninguno".
Ejemplo: {"id_1": "Carlos Alcaraz", "id_2": "Ninguno"}
"""
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
        except Exception:
            pass
            
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
    BUG #4 FIX: Match estricto por apellido.
    'Sinner'   → matchea 'Sinner J' (apellido completo)
    'Sinner J' → matchea 'Sinner J' (exacto)
    'Sin'      → NO matchea 'Sinner J' (parcial de apellido)
    """
    q = query.lower().strip()
    r = record_name.lower().strip()
    if q == r:
        return True
    # Query es apellido solo → el record debe empezar con ese apellido + espacio
    if ' ' not in q:
        return r.startswith(q + ' ') or r == q
    # Query tiene apellido + inicial → match exacto o record contiene query
    q_parts = q.split()
    r_parts = r.split()
    if len(q_parts) >= 2 and len(r_parts) >= 2:
        return q_parts[0] == r_parts[0] and q_parts[1][0] == r_parts[1][0]
    return q == r


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
# ORÁCULO — STATS CON FILTROS (opera sobre RAM)
# ─────────────────────────────────────────────────────────────────────────────
def get_stats(name: str, surface: str, level: str, oracle: list[list]) -> dict | None:
    allowed_levels = LEVEL_GROUPS.get(level, {1, 2, 3, 4})
    srf_code = {"Hard": 1, "Clay": 2, "Grass": 3, "Carpet": 4}.get(surface, 1)

    sv_pts = sv_won = sv_gms = sv_held = 0
    rt_pts = rt_won = rt_gms = rt_brk = 0
    n_total = n_surface = 0
    bp_saved_total = bp_faced_total = bp_conv_won = bp_conv_total = 0
    recent_minutes = []
    recent_results = []

    name_lower = name.lower()

    for rec in oracle:
        wn = str(rec[2])
        ln = str(rec[4])
        if name_lower not in wn.lower() and name_lower not in ln.lower():
            continue

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

    # MUESTRA MIXTA: ampliar si superficie < 10 pero total >= 10
    if n_surface < 10 and n_total >= 10:
        return _get_stats_all_surfaces(name, level, oracle, mixed_sample=True)

    if n_surface == 0:
        if level != "ITF":
            return get_stats(name, surface, "ITF", oracle)
        return None

    spw = sv_won / sv_pts * 100 if sv_pts else 0
    rpw = rt_won / rt_pts * 100 if rt_pts else 0
    hold = sv_held / sv_gms * 100 if sv_gms else 0
    brk = rt_brk / rt_gms * 100 if rt_gms else 0

    fb = get_fallback(surface)
    if spw <= 25.0:
        spw = fb["spw"]
    if rpw <= 25.0:
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
                            mixed_sample: bool = False) -> dict | None:
    allowed_levels = LEVEL_GROUPS.get(level, {1, 2, 3, 4})
    sv_pts = sv_won = sv_gms = sv_held = 0
    rt_pts = rt_won = rt_gms = rt_brk = 0
    n = 0

    for rec in oracle:
        wn, ln = str(rec[2]), str(rec[4])
        is_w = _name_match(name, wn)
        is_l = _name_match(name, ln)
        if not is_w and not is_l:
            continue
        if _safe_int(rec[1]) not in allowed_levels:
            continue

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
        n += 1

    if n == 0:
        if level != "ITF":
            return _get_stats_all_surfaces(name, "ITF", oracle, mixed_sample)
        return None

    spw = max(25.1, sv_won / sv_pts * 100 if sv_pts else 55.0)
    rpw = max(25.1, rt_won / rt_pts * 100 if rt_pts else 40.0)

    tags = ["MUESTRA MIXTA"]
    if n < 10:
        tags.append("FALLBACK")

    return {
        "n": n, "n_total": n,
        "hold": round(sv_held / sv_gms * 100, 1) if sv_gms else 0,
        "brk": round(rt_brk / rt_gms * 100, 1) if rt_gms else 0,
        "spw": round(spw, 1), "rpw": round(rpw, 1),
        "clutch": 100.0, "dr_last5": 1.0, "last5_wins": 0,
        "fatigue_mins": 0, "tags": tags, "source": "Oráculo Local (mixta)",
    }


# ─────────────────────────────────────────────────────────────────────────────
# H2H (≤36 MESES) — opera sobre RAM
# ─────────────────────────────────────────────────────────────────────────────
def get_h2h(p1: str, p2: str, oracle: list[list], months_limit: int = 36) -> dict:
    wins_p1 = wins_p2 = 0
    matches = []
    cutoff = datetime.now() - timedelta(days=months_limit * 30)

    for rec in oracle:
        wn, ln = str(rec[2]), str(rec[4])
        p1_is_w = _name_match(p1, wn) and _name_match(p2, ln)
        p1_is_l = _name_match(p1, ln) and _name_match(p2, wn)
        if not p1_is_w and not p1_is_l:
            continue

        rec_date = _safe_int(rec[20]) if len(rec) > 20 else 0
        if rec_date > 0:
            match_dt = _date_int_to_dt(rec_date)
            if match_dt < cutoff:
                continue

        surface = SRF_MAP.get(_safe_int(rec[0]), "?")
        tourney = str(rec[24]) if len(rec) > 24 else "?"

        if p1_is_w:
            wins_p1 += 1
            matches.append({"date": rec_date, "winner": p1,
                            "surface": surface, "tourney": tourney})
        else:
            wins_p2 += 1
            matches.append({"date": rec_date, "winner": p2,
                            "surface": surface, "tourney": tourney})

    return {
        "p1_wins": wins_p1, "p2_wins": wins_p2,
        "total": wins_p1 + wins_p2,
        "matches": sorted(matches, key=lambda x: x["date"], reverse=True),
    }


# ─────────────────────────────────────────────────────────────────────────────
# TANKING CHECK & TIER DROP — opera sobre RAM
# ─────────────────────────────────────────────────────────────────────────────
def check_tanking(name: str, oracle: list[list], last_n: int = 5) -> bool:
    losses = []
    for rec in oracle:
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


def check_tier_drop(name: str, level: str, oracle: list[list]) -> bool:
    if level not in ("Challenger", "ITF"):
        return False

    ranks = []
    for rec in oracle:
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
# FÍSICA Y ENTORNO — BUG #6 FIX (shrinkage independiente)
# ─────────────────────────────────────────────────────────────────────────────
def apply_environment(spw: float, rpw: float, n: int,
                      altitude_m: int, fatigue_mins: int,
                      surface: str, tourney: str = "") -> tuple[float, float, list]:
    """
    BUG #6 FIX: Shrinkage independiente por métrica.
    SPW y RPW se acercan a SUS respectivas medias, no de forma proporcional.
    """
    fb = get_fallback(surface)
    AVG_SPW, AVG_RPW = fb["spw"], fb["rpw"]
    adjustments = []

    # Shrinkage Bayesiano independiente (threshold 20 partidos)
    confidence = min(n / 20.0, 1.0) if n > 0 else 0.0
    shrink = 0.35 * (1.0 - confidence)

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

    # Court Pace (aditivo ±1.5 pp)
    tourney_low = tourney.lower() if tourney else ""
    if any(t in tourney_low for t in FAST_COURTS):
        adj_spw += 1.5
        adj_rpw -= 1.5
        adjustments.append(f"Court Pace RÁPIDA ({tourney})")
    elif any(t in tourney_low for t in SLOW_COURTS):
        adj_spw -= 1.5
        adj_rpw += 1.5
        adjustments.append(f"Court Pace LENTA ({tourney})")

    # Fatiga (BUG #2 FIX: ÚNICA ubicación de penalización de fatiga)
    if fatigue_mins > 150:
        penalty = min((fatigue_mins - 150) / 100.0 * 0.5, 2.0)
        adj_spw -= penalty
        adj_rpw -= penalty * 0.3
        adjustments.append(f"Fatiga -{penalty:.2f} SPW ({fatigue_mins} min recientes)")

    return adj_spw, adj_rpw, adjustments


# ─────────────────────────────────────────────────────────────────────────────
# FASE CERO: AJUSTES DINÁMICOS — BUG #2 FIX (sin fatiga aquí)
# ─────────────────────────────────────────────────────────────────────────────
def compute_adjustments(s1: dict, s2: dict, h2h: dict, ctx: dict = None) -> tuple[float, float, list]:
    """
    BUG #2 FIX: Fatiga ELIMINADA de aquí (ya está en apply_environment).
    Se restauran Ajustes V7 (Zurdo, Altura, Clay Trap) 100% automáticos vía ctx.
    """
    sum_adj = 0.0
    ic = 1.00
    notes = []
    
    # ── Ajustes Físicos (Restauración V7) ──
    if ctx:
        p1_h = ctx.get("p1_hand", "R")
        p2_h = ctx.get("p2_hand", "R")
        p1_ht = ctx.get("p1_height", 0)
        p2_ht = ctx.get("p2_height", 0)
        p1_l = ctx.get("p1_local", False)
        p2_l = ctx.get("p2_local", False)
        surf = ctx.get("surface", "Hard")
        lvl = ctx.get("level", "ATP")

        # Factor Zurdo
        if p2_h == "L" and p1_h != "L":
            sum_adj -= 0.05
            notes.append("Factor V3: Oponente (P2) es Zurdo → -0.05")
        if p1_h == "L" and p2_h != "L":
            sum_adj += 0.05
            notes.append("Factor V3: P1 es Zurdo → +0.05")

        # Altura en pistas rápidas (Hard/Grass/Carpet)
        if surf in ["Hard", "Grass", "Carpet"]:
            if isinstance(p1_ht, (int, float)) and p1_ht > 193:
                sum_adj += 0.03
                notes.append(f"Ajuste Físico: P1 alto ({p1_ht}cm) en {surf} → +0.03")
            if isinstance(p2_ht, (int, float)) and p2_ht > 193:
                sum_adj -= 0.03
                notes.append(f"Ajuste Físico: P2 alto ({p2_ht}cm) en {surf} → -0.03")

        # Clay Trap (Local en Arcilla de torneos menores)
        if surf == "Clay" and lvl in ["Challenger", "ITF"]:
            if p2_l:
                sum_adj -= 0.06
                notes.append("Clay Trap: P2 es Local en Challenger arcilla → -0.06")
            if p1_l:
                sum_adj += 0.06
                notes.append("Clay Trap: P1 es Local en Challenger arcilla → +0.06")

    # ── Tags (MUESTRA MIXTA, FALLBACK, FUENTE WEB) ──
    for tag in s1.get("tags", []) + s2.get("tags", []):
        if "MIXTA" in tag and ic > 0.85:
            ic -= 0.05
            notes.append(f"[MUESTRA MIXTA] IC -0.05")
        if "FALLBACK" in tag:
            ic -= 0.15
            notes.append(f"[FALLBACK] IC -0.15")
        if "WEB" in tag:
            sum_adj -= 0.03
            notes.append(f"[FUENTE WEB] sum_adj -0.03")

    # Muestra pequeña
    if s1.get("n", 0) < 10 or s2.get("n", 0) < 10:
        ic = min(ic, 0.85)
        notes.append(f"Muestra <10 en superficie → IC ≤ 0.85")

    # H2H
    if h2h["total"] == 0:
        ic = min(ic, 0.90)
        notes.append(f"H2H = 0 partidos → IC ≤ 0.90")
    elif h2h["total"] >= 3:
        ratio = h2h["p1_wins"] / h2h["total"]
        if ratio >= 0.7:
            sum_adj += 0.03
            notes.append(f"H2H dominante ({h2h['p1_wins']}-{h2h['p2_wins']}) → +0.03")
        elif ratio <= 0.3:
            sum_adj -= 0.03
            notes.append(f"H2H desfavorable ({h2h['p1_wins']}-{h2h['p2_wins']}) → -0.03")

    # Dominance Ratio
    dr1 = s1.get("dr_last5", 1.0)
    wins1 = s1.get("last5_wins", 0)
    if dr1 > 1.25:
        sum_adj += 0.04
        notes.append(f"DR últimos 5 = {dr1:.3f} → +0.04")
    if wins1 <= 1:
        sum_adj -= 0.04
        notes.append(f"Solo {wins1}/5 victorias recientes → -0.04")

    # Clutch Differential
    c1 = s1.get("clutch", 100.0)
    c2 = s2.get("clutch", 100.0)
    if c1 > 110 and c2 < 90:
        sum_adj += 0.05
        notes.append(f"Clutch ventaja ({c1:.0f} vs {c2:.0f}) → +0.05")
    elif c1 < 90 and c2 > 110:
        sum_adj -= 0.05
        notes.append(f"Clutch desventaja ({c1:.0f} vs {c2:.0f}) → -0.05")

    return sum_adj, max(ic, 0.50), notes


# ─────────────────────────────────────────────────────────────────────────────
# CLASIFICACIÓN (SEMÁFORO) — BUG #1 FIX COMPLETO
# Lógica: EV-first con guardia de probabilidad mínima.
# Sin huecos: todo rango de pmod × EV tiene una categoría asignada.
# ─────────────────────────────────────────────────────────────────────────────
def classify(pmod: float, ev_pct: float, odd_dec: float) -> tuple[str, str]:
    """
    BUG #1 FIX: Reescrito sin huecos lógicos.
    Antes: underdog pmod=0.50 EV=+8% → BASURA (incorrecto)
    Ahora: underdog pmod=0.50 EV=+8% → DERECHA ✅
    """
    # Guard absoluto: modelo dice que pierde más de lo que gana
    if pmod < 0.45:
        return "BASURA", "🗑️"

    # ── Underdog pagado (cuota >= 2.0) ───────────────────────────────────────
    if odd_dec >= 2.0:
        if pmod >= 0.65:
            return "BOMBA NUCLEAR", "💣"
        if ev_pct >= 15.0:
            return "FRANCOTIRADOR", "🎯"
        if ev_pct >= 6.5:
            return "DERECHA", "✅"
        if ev_pct >= 4.0:
            return "PARLAY", "🟡"
        return "BASURA", "🗑️"

    # ── Favorito dominante (modelo dice ≥70%) ────────────────────────────────
    if pmod >= 0.70:
        if ev_pct >= 3.0:
            return "SUPER DERECHA", "🟢"
        if ev_pct >= 0.0:
            return "DERECHA", "✅"
        return "FAVORITO SOBREVENDIDO", "🔵"

    # ── Favorito moderado (45-70%, cuota < 2.0) ─────────────────────────────
    if ev_pct >= 6.5:
        return "DERECHA", "✅"
    if ev_pct >= 4.0:
        return "PARLAY", "🟡"
    if ev_pct >= 0.0 and pmod >= 0.55:
        return "VALOR MARGINAL", "⚠️"

    return "BASURA", "🗑️"


# ─────────────────────────────────────────────────────────────────────────────
# PARSER DE SPORTSBOOK
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

    return results


# ─────────────────────────────────────────────────────────────────────────────
# MATEMÁTICA LOCAL — BUG #5 FIX (seed determinístico)
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


def log5_serve(spw: float, rpw: float) -> float:
    A, B = spw / 100, rpw / 100
    d = A * (1 - B) + (1 - A) * B
    return (A * (1 - B)) / d if d else 0.5


def game_prob(p: float) -> float:
    """Probabilidad analítica (Markov) de ganar un game al saque con prob p de ganar punto."""
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    q = 1 - p
    p_deuce = (p ** 2) / (p ** 2 + q ** 2)
    return (p ** 4) * (1 + 4 * q + 10 * q ** 2) + 20 * (p ** 3) * (q ** 3) * p_deuce


def sim_tiebreak(p_first: float, p_second: float) -> int:
    a = b = total = 0
    while True:
        if total == 0:
            win = random.random() < p_first
        else:
            block = (total - 1) // 2
            win = random.random() < (p_second if block % 2 == 0 else p_first)
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
            a_first: bool = True) -> tuple[int, bool]:
    """
    BUG #10 FIX: eliminadas condiciones inalcanzables gA==7 / gB==7.
    pg_A/pg_B = probabilidad analítica de ganar game (Markov).
    pA/pB = probabilidad de ganar punto (para tiebreak).
    """
    gA = gB = 0
    a_serves = a_first
    while True:
        if gA == 6 and gB == 6:
            tf = pA if a_serves else pB
            ts = pB if a_serves else pA
            w = sim_tiebreak(tf, ts)
            if w:
                gA += 1
            else:
                gB += 1
            return (1 if gA > gB else 0), not a_serves

        if a_serves:
            if random.random() < pg_A:
                gA += 1
            else:
                gB += 1
        else:
            if random.random() < pg_B:
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
    """
    BUG #5 FIX: seed determinístico. Mismo partido = mismo resultado.
    MEJORA #3: reproducibilidad sin sacrificar calidad estadística.
    """
    random.seed(seed)
    needed = 2 if best_of == 3 else 3
    wins = 0
    pg_A = game_prob(pA)
    pg_B = game_prob(pB)
    for _ in range(n):
        sA = sB = 0
        a_srv = True
        while sA < needed and sB < needed:
            w, a_srv = sim_set(pg_A, pg_B, pA, pB, a_srv)
            if w:
                sA += 1
            else:
                sB += 1
        if sA == needed:
            wins += 1
    return wins / n


def make_seed(p1: str, p2: str) -> int:
    """MEJORA #3: seed basado en nombres para reproducibilidad."""
    key = f"{p1.lower().strip()}|{p2.lower().strip()}"
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16)


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
            "spw": round(max(25.1, float(data.get("spw_pct", 62.0))), 1),
            "rpw": round(max(25.1, float(data.get("rpw_pct", 38.0))), 1),
            "clutch": 100.0, "dr_last5": 1.0, "last5_wins": 0,
            "fatigue_mins": 0, "tags": ["FUENTE WEB"], "source": "Gemini Search",
        }
    except Exception:
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
    except Exception as e:
        return f"Error Gemini: {e}"


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


def log_prediction(match_id, p1, p2, p_mod, p_casa, odd, ev_val,
                   tier, league, ic, real_winner=""):
    sheet = get_sheet()
    if not sheet:
        return False
    try:
        col_ids = sheet.col_values(1)
        if match_id in col_ids:
            if real_winner:
                idx = col_ids.index(match_id) + 1
                sheet.update_cell(idx, 10, real_winner)
            return True
        sheet.append_row([
            match_id, datetime.now().strftime("%Y-%m-%d %H:%M"),
            p1, p2, f"{p_mod * 100:.1f}%",
            f"{p_casa * 100:.1f}%" if p_casa else "S/C",
            str(odd) if odd else "S/C",
            f"{ev_val * 100:+.1f}%" if ev_val is not None else "S/C",
            tier, real_winner, league, f"{ic:.2f}"
        ])
        return True
    except Exception:
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
    except Exception:
        return []


def save_predictions_callback():
    partidos = st.session_state.get('last_procesados', [])
    logged = 0
    for p in partidos:
        try:
            if log_prediction(
                p['match_id'], p['p1'], p['p2'], p['pmod'], p['nv1'],
                p['odd1'], p['ev1'], p['tier'], p['league'], p['ic'], ""
            ):
                logged += 1
        except Exception:
            pass
    st.session_state.save_msg = f"¡{logged} partidos guardados en el Oráculo!"


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Quantum Tennis v9.0", page_icon="🎾", layout="wide")
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

    st.title("🎾 Quantum Tennis Engine v9.0 — Hardened Audit")
    st.caption(
        "Log5 · Markov MC 10k · Oráculo 40k+ (2022-2024) · "
        "Filtros Nivel/Superficie/Fecha · H2H · Clutch · DR · Fatiga · Court Pace · "
        "Gemini 2.5 Pro"
    )

    with st.expander("👁️ VER EL GRAN ORÁCULO DE GOOGLE SHEETS", expanded=False):
        components.iframe(
            "https://docs.google.com/spreadsheets/d/1kciFhxjiVOeScsu_7e6UZvJ36ungHyeQxjIWMBu5CYs"
            "/edit?usp=sharing",
            height=500, scrolling=True
        )

    if st.session_state.get("save_msg"):
        st.success(st.session_state.save_msg)
        st.balloons()
        st.session_state.save_msg = ""

    if not gemini_available():
        st.error("⚠️ Falta GOOGLE_API_KEY en tu archivo .env")
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
        st.write("Un click. Gemini buscará en Flashscore y subirá al Oráculo los ganadores automáticamente.")
        
        if st.button("🤖 Iniciar Liquidación Automática en Internet"):
            with st.spinner("Conectando al Oráculo, buscando en Flashscore y subiendo resultados..."):
                pendientes = get_pending_matches()
                
                if pendientes:
                    sugerencias = batch_guess_winners_gemini(pendientes)
                    reporte_acciones = []
                    
                    for p in pendientes:
                        win = sugerencias.get(p['match_id'], "Ninguno")
                        real_winner = "Ninguno"
                        if win != "Ninguno":
                            if p['p1'].lower() in win.lower() or win.lower() in p['p1'].lower(): real_winner = p['p1']
                            elif p['p2'].lower() in win.lower() or win.lower() in p['p2'].lower(): real_winner = p['p2']
                        
                        if real_winner != "Ninguno":
                            ok = log_prediction(p['match_id'], p['p1'], p['p2'], 0, 0, None, None, "", "", 0, real_winner)
                            if ok:
                                reporte_acciones.append(f"✅ **{p['p1']} vs {p['p2']}** ➔ Subido a Sheets: **{real_winner}**")
                            else:
                                reporte_acciones.append(f"❌ **{p['p1']} vs {p['p2']}** ➔ Encontró a {real_winner} pero falló el guardado.")
                        else:
                            reporte_acciones.append(f"⏳ **{p['p1']} vs {p['p2']}** ➔ Aún sin resultado oficial / Pendiente.")
                    
                    st.session_state.pending_error = ""
                    st.session_state.reporte_liq = reporte_acciones
                else:
                    st.session_state.pending_error = "⚠️ No se encontró la llave de Google Sheets (st.secrets) en este entorno, o no hay NINGÚN partido pendiente."
                    st.session_state.reporte_liq = []
            st.rerun()

        if st.session_state.get("pending_error"):
            st.warning(st.session_state.pending_error)
            st.session_state.pending_error = ""
            
        if st.session_state.get("reporte_liq"):
            st.success("Operación Automática Finalizada. Aquí tienes los resultados:")
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
        return

    partidos = parse_matches(st.session_state.analisis_ejecucion)
    if not partidos:
        st.error("No se encontraron pares.")
        return

    # MEJORA #1: cargar oráculo UNA vez
    oracle = load_oracle(db_path)
    if not oracle:
        st.error(f"Oráculo vacío o no encontrado: {db_path}")
        return
    st.caption(f"📂 Oráculo cargado: {len(oracle):,} registros en RAM")

    partidos_procesados = []

    for p1_raw, odd1, p2_raw, odd2, league in partidos:
        p1, p2 = p1_raw or "Jugador 1", p2_raw or "Jugador 2"

        # ── LÍNEAS ROJAS ─────────────────────────────────────────────────────
        if "utr" in p1.lower() or "utr" in p2.lower():
            st.error(f"🚫 EXCLUIDO | TORNEO UTR: {p1} vs {p2}")
            continue

        dec1_check = american_to_decimal(odd1) if odd1 else 2.0
        if dec1_check < 1.18:
            st.warning(f"🚫 EXCLUIDO | SUPER-FAVORITO (cuota {dec1_check:.2f}): {p1}")
            continue

        st.divider()
        st.subheader(f"⚡ {p1}  vs  {p2}")

        # ── Contexto ─────────────────────────────────────────────────────────
        with st.spinner("🌍 Detectando torneo y físico…"):
            ctx = extract_match_context(p1, p2)

        surface = ctx.get("surface", "Hard")
        tourney = ctx.get("tourney", "Unknown")
        altitude = ctx.get("altitude", 0)
        level = ctx.get("level", league)
        p1_hnd, p1_hgt, p1_t100, p1_loc = ctx["p1_hand"], ctx["p1_height"], ctx["p1_top100"], ctx["p1_local"]
        p2_hnd, p2_hgt, p2_t100, p2_loc = ctx["p2_hand"], ctx["p2_height"], ctx["p2_top100"], ctx["p2_local"]

        st.caption(f"📍 {tourney} | {surface} | {level} | Alt: {altitude}m | P1: {p1_hgt}cm ({p1_hnd}) vs P2: {p2_hgt}cm ({p2_hnd})")

        # Tier Drop
        if check_tier_drop(p1, level, oracle):
            st.error(f"🚫 TIER DROP: {p1} (Top 100 en {level})")
            continue

        # ── Stats ────────────────────────────────────────────────────────────
        c_s1, c_s2 = st.columns(2)
        with c_s1:
            with st.spinner(f"Stats {p1}…"):
                s1 = get_stats(p1, surface, level, oracle)
                if not s1:
                    st.info(f"🌐 {p1} no en oráculo → web…")
                    s1 = gemini_stats(p1)
        with c_s2:
            with st.spinner(f"Stats {p2}…"):
                s2 = get_stats(p2, surface, level, oracle)
                if not s2:
                    st.info(f"🌐 {p2} no en oráculo → web…")
                    s2 = gemini_stats(p2)

        if not s1 or not s2:
            st.error("Stats insuficientes.")
            continue

        # ── H2H + Tanking ────────────────────────────────────────────────────
        h2h = get_h2h(p1, p2, oracle)
        tank_p1 = check_tanking(p1, oracle)
        tank_p2 = check_tanking(p2, oracle)

        # ── Ajustes ──────────────────────────────────────────────────────────
        sum_adj, ic, adj_notes = compute_adjustments(s1, s2, h2h, ctx)

        adj_spw1, adj_rpw1, env1 = apply_environment(
            s1["spw"], s1["rpw"], s1["n"], altitude, s1["fatigue_mins"], surface, tourney
        )
        adj_spw2, adj_rpw2, env2 = apply_environment(
            s2["spw"], s2["rpw"], s2["n"], altitude, s2["fatigue_mins"], surface, tourney
        )

        # ── Log5 + Monte Carlo ───────────────────────────────────────────────
        pA = log5_serve(adj_spw1, adj_rpw2)
        pB = log5_serve(adj_spw2, adj_rpw1)
        seed = make_seed(p1, p2)

        with st.spinner(f"Corriendo {MC_ITERATIONS:,} simulaciones…"):
            mc_A = sim_match(pA, pB, best_of, seed=seed)
            mc_B = 1 - mc_A

        pmod_final = max(0.01, min(0.99, mc_A * (1 + sum_adj)))

        # ── UI Resultados ────────────────────────────────────────────────────
        st.markdown("#### 🧮 Motor Matemático")
        m1, m2 = st.columns(2)
        ev_lines = []

        # Variables para guardar después
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
                _nv1, _nv2, _f1, _f2 = no_vig(odd1 or 100, odd2 or 100)
                nv_this = _nv1 if col is m1 else _nv2
                fair = _f1 if col is m1 else _f2
                dec_odd = american_to_decimal(odd)
                ev_val = calc_ev(mc_w, dec_odd)
                edge = (mc_w - nv_this) * 100
                tier, emoji = classify(mc_w, ev_val * 100, dec_odd)

                col.metric("P(match) Casa No-Vig", f"{nv_this * 100:.1f}%",
                           f"Fair odd {fair}", delta_color="off")
                col.metric("💰 Expected Value", f"{ev_val * 100:+.1f}%",
                           delta_color="normal" if ev_val > 0 else "inverse")
                col.metric("Edge vs Casa", f"{edge:+.1f}%")

                if col is m1:
                    nv1_val = _nv1
                    ev1_val = ev_val
                    tier1_val = tier
                    ev_lines.append(
                        f"{name}: EV={ev_val * 100:+.2f}% | Edge={edge:+.1f}% | {tier} {emoji}"
                    )
            else:
                col.info("Sin cuota → EV no calculable")

        # ── Panel de auditoría ────────────────────────────────────────────────
        st.markdown("---")

        if odd1 and odd2 and ev1_val is not None:
            dec1 = american_to_decimal(odd1)
            emoji1 = classify(pmod_final, ev1_val * 100, dec1)[1]

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

        # ── Gemini H48 ───────────────────────────────────────────────────────
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

    # ── Guardar ──────────────────────────────────────────────────────────────
    if partidos_procesados:
        st.session_state.last_procesados = partidos_procesados
        st.button(
            "💾 Guardar en Oráculo",
            on_click=save_predictions_callback,
            type="primary", use_container_width=True
        )


if __name__ == "__main__":
    main()
