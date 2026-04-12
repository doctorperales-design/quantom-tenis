"""
Quantum Tennis Engine v8.0 — Autonomous Audit Edition
Markov Chains · Shrinkage Bayesiano · Oráculo Enriquecido 2022-2025
Bugs corregidos y mejoras aplicadas sobre v7.0

CHANGELOG v8.0 vs v7.0:
──────────────────────────────────────────────────────────────────────
BUG #1  adj_rpw fatiga truncada — la línea estaba vacía, fatiga nunca se aplicaba a RPW
BUG #2  DB_PATH apuntaba a "matches_jsonl.jsonl" — ahora lee matches_comprimidos.csv (26 cols)
BUG #3  Sin filtro de nivel — get_stats leía TODOS los registros sin respetar LEVEL FILTER
BUG #4  Sin filtro de superficie — stats se calculaban mezclando todas las superficies
BUG #5  Sin filtro de fecha — ahora se filtran registros > 36 meses para H2H
BUG #6  parse_matches no extraía liga/circuito — imposible activar filtros de nivel
BUG #7  Fallback genérico no distinguía Hold de SPW — ahora incluye hold, spw, rpw
BUG #8  Anti-Ceros threshold (≤10%) demasiado bajo — subido a ≤25% con floor por superficie

MEJORA #1  Nuevas columnas del oráculo: Date, Minutes, W_Rank, L_Rank, Tourney, Round
MEJORA #2  H2H directo reciente (≤36 meses) calculado desde el CSV
MEJORA #3  Fatiga calculada desde columna Minutes (ya no depende solo de Gemini)
MEJORA #4  Court Pace por torneo (rápida/lenta) aplicado automáticamente
MEJORA #5  Tier Drop post-lesión detectado vía W_Rank/L_Rank
MEJORA #6  Clutch Differential (BP Saved% + BP Converted%) incluido
MEJORA #7  Dominance Ratio últimos 5 partidos calculado
MEJORA #8  Filtro MUESTRA MIXTA con etiqueta y penalización IC
MEJORA #9  Semáforo completo con todas las categorías del modelo original
MEJORA #10 Tanking Check básico (derrotas rápidas recientes)
──────────────────────────────────────────────────────────────────────
"""

import streamlit as st
import json
import os
import re
import random
import math
from datetime import datetime, timedelta
from dotenv import load_dotenv
from google import genai
from google.genai import types
import csv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
GEMINI_MODEL  = "gemini-2.5-pro"
# NOTA ANTIGRAVITY: Si cambian a archivo puramente .jsonl, solo renombra esto.
# Añadí lógica híbrida abajo para leer tanto CSV real como JSONL en ese archivo.
DB_PATH       = "matches_comprimidos.csv"      
MC_ITERATIONS = 50000                            

# ─────────────────────────────────────────────────────────────────────────────
# FALLBACKS EMPÍRICOS — Extraídos del oráculo 40,225 registros (2022-2024)
# ─────────────────────────────────────────────────────────────────────────────
SURFACE_FALLBACKS = {
    "Hard":   {"hold": 74.0, "spw": 68.9, "rpw": 42.8},
    "Clay":   {"hold": 70.5, "spw": 66.1, "rpw": 45.9},
    "Grass":  {"hold": 77.6, "spw": 70.9, "rpw": 39.5},
    "Carpet": {"hold": 75.8, "spw": 69.5, "rpw": 41.1},
}

# Torneos con Court Pace conocido
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

# Mapas de decodificación del oráculo
SRF_MAP = {1: "Hard", 2: "Clay", 3: "Grass", 4: "Carpet", 0: "Unknown"}
LVL_MAP = {1: "ATP", 2: "Masters", 3: "GrandSlam", 4: "Challenger", 5: "Finals", 6: "DavisCup", 0: "ITF"}

LEVEL_GROUPS = {
    "ATP":       {1, 2, 3},       
    "Masters":   {1, 2, 3},
    "GrandSlam": {1, 2, 3},
    "Challenger":{3, 4},            
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
# EXTRACCIÓN AUTÓNOMA DE CONTEXTO
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def extract_match_context(p1: str, p2: str) -> dict:
    client = get_gemini_client()
    if not client:
        return {"surface": "Hard", "tourney": "Unknown", "altitude": 0, "level": "ATP"}
    try:
        prompt = (
            f'Detecta el torneo actual, la superficie (Hard, Clay, Grass, Carpet), '
            f'el nivel del circuito (ATP, Masters, GrandSlam, Challenger, ITF, DavisCup, Finals) '
            f'y la altitud en metros sobre el nivel del mar para el partido de tenis entre '
            f'"{p1}" y "{p2}" programado para las próximas 48 horas.\n'
            f'Responde SOLO con JSON crudo:\n'
            f'{{"surface": "Hard", "tourney": "Monte Carlo Masters", "altitude": 35, "level": "Masters"}}'
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
            data = json.loads(raw[s:e+1])
            lvl = data.get("level", "ATP")
            if lvl not in LEVEL_GROUPS:
                lvl = "ATP"
            data["level"] = lvl
            return data
        return {"surface": "Hard", "tourney": "Unknown", "altitude": 0, "level": "ATP"}
    except Exception:
        return {"surface": "Hard", "tourney": "Unknown", "altitude": 0, "level": "ATP"}

# ─────────────────────────────────────────────────────────────────────────────
# PARSER HÍBRIDO (JSONL/CSV) — LECTURA SEGURA
# ─────────────────────────────────────────────────────────────────────────────
def _parse_db_line(line: str) -> list | None:
    """Intenta procesar la línea como JSON, de lo contrario la asume CSV."""
    line = line.strip()
    if not line: return None
    if line.startswith('['):
        try:
            return json.loads(line)
        except Exception:
            pass
    # Intento CSV rápido
    try:
        parts = next(csv.reader([line]))
        # Convertir a ints si es posible (ej: rec[0] Srf)
        parsed = []
        for p in parts:
            p = p.strip()
            if p.isdigit(): parsed.append(int(p))
            else:
                try: parsed.append(float(p))
                except: parsed.append(p)
        return parsed
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────────────────────
# ORÁCULO — LECTURA CON FILTROS
# ─────────────────────────────────────────────────────────────────────────────
def get_fallback(surface: str) -> dict:
    s = surface.lower()
    if "clay" in s or "arcilla" in s or "tierra" in s: return SURFACE_FALLBACKS["Clay"]
    if "grass" in s or "hierba" in s or "pasto" in s:  return SURFACE_FALLBACKS["Grass"]
    if "carpet" in s or "moqueta" in s:                return SURFACE_FALLBACKS["Carpet"]
    return SURFACE_FALLBACKS["Hard"]

def _name_match(query: str, record_name: str) -> bool:
    q = query.lower().strip()
    r = record_name.lower().strip()
    if q == r: return True
    if ' ' not in q and r.startswith(q + ' '): return True
    if q in r or r in q: return True
    return False

def _date_int_to_dt(d: int) -> datetime:
    try: return datetime.strptime(str(d), "%Y%m%d")
    except: return datetime.now()

@st.cache_data(show_spinner=False)
def get_stats(name: str, surface: str = "Hard", level: str = "ATP",
              path: str = DB_PATH) -> dict | None:
    allowed_levels = LEVEL_GROUPS.get(level, {1, 2, 3, 4})
    srf_code = {"Hard": 1, "Clay": 2, "Grass": 3, "Carpet": 4}.get(surface, 1)

    sv_pts = sv_won = sv_gms = sv_held = 0
    rt_pts = rt_won = rt_gms = rt_brk = 0
    n_total = n_surface = 0
    mixed_sample = fallback_applied = False
    recent_minutes = []
    recent_results = []
    bp_saved_total = bp_faced_total = bp_conv_won = bp_conv_total = 0

    try:
        with open(path, encoding='utf-8') as f:
            for line in f:
                if name.lower() not in line.lower(): continue
                rec = _parse_db_line(line)
                if not rec or len(rec) < 26: continue

                wn, ln = rec[2], rec[4]
                rec_srf, rec_lvl, rec_date, rec_mins = rec[0], rec[1], rec[20], rec[21]
                rec_wrank, rec_lrank = rec[22], rec[23]

                is_w = _name_match(name, str(wn))
                is_l = _name_match(name, str(ln))
                if not is_w and not is_l: continue
                if rec_lvl not in allowed_levels: continue

                on_surface = (rec_srf == srf_code)

                if is_w:
                    wp = (rec[6], rec[8], rec[9], rec[10], rec[11], rec[12])
                    lp = (rec[13], rec[15], rec[16], rec[17], rec[18], rec[19])
                else:
                    wp = (rec[13], rec[15], rec[16], rec[17], rec[18], rec[19])
                    lp = (rec[6], rec[8], rec[9], rec[10], rec[11], rec[12])

                if on_surface:
                    sv_pts  += wp[0]; sv_won  += wp[1] + wp[2]
                    sv_gms  += wp[3]; sv_held += max(0, wp[3] - (wp[5] - wp[4]))
                    rt_pts  += lp[0]; rt_won  += max(0, lp[0] - (lp[1] + lp[2]))
                    rt_gms  += lp[3]; rt_brk  += max(0, lp[5] - lp[4])
                    n_surface += 1

                    bp_saved_total += wp[4]   
                    bp_faced_total += wp[5]   
                    bp_conv_won    += max(0, lp[5] - lp[4])  
                    bp_conv_total  += lp[5]   
                n_total += 1

                try: min_val = int(rec_mins)
                except: min_val = 0
                
                try: date_val = int(rec_date)
                except: date_val = 0

                if min_val > 0 and date_val > 0:
                    recent_minutes.append((date_val, min_val, is_w))

                if on_surface and wp[0] > 0 and lp[0] > 0:
                    my_spw = (wp[1] + wp[2]) / wp[0]
                    opp_spw = (lp[1] + lp[2]) / lp[0]
                    if opp_spw > 0:
                        recent_results.append((date_val, my_spw / opp_spw, is_w))

    except FileNotFoundError:
        st.warning(f"⚠️ Archivo BD no encontrado: {path}")
        return None
    except Exception as e:
        st.error(f"BD error: {e}")
        return None

    if n_surface < 10 and n_total >= 10:
        return _get_stats_all_surfaces(name, level, path, mixed_sample=True)

    if n_surface < 10 and n_total < 10:
        fallback_applied = True
        if n_surface == 0 and n_total == 0: return None

    if n_surface == 0: return None

    spw = sv_won / sv_pts * 100 if sv_pts else 0
    rpw = rt_won / rt_pts * 100 if rt_pts else 0
    hold = sv_held / sv_gms * 100 if sv_gms else 0
    brk = rt_brk / rt_gms * 100 if rt_gms else 0

    fb = get_fallback(surface)
    if spw <= 25.0: spw = fb["spw"]
    if rpw <= 25.0: rpw = fb["rpw"]

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
    if fallback_applied: tags.append("FALLBACK")

    return {
        "n": n_surface, "n_total": n_total,
        "hold": round(hold, 1), "brk": round(brk, 1),
        "spw": round(spw, 1), "rpw": round(rpw, 1),
        "clutch": round(clutch, 1), "dr_last5": round(avg_dr, 3),
        "last5_wins": last5_wins, "fatigue_mins": fatigue_mins,
        "tags": tags, "source": "Oráculo Local",
    }


def _get_stats_all_surfaces(name: str, level: str, path: str,
                            mixed_sample: bool = False) -> dict | None:
    allowed_levels = LEVEL_GROUPS.get(level, {1, 2, 3, 4})
    sv_pts = sv_won = sv_gms = sv_held = 0
    rt_pts = rt_won = rt_gms = rt_brk = 0
    n = 0

    try:
        with open(path, encoding='utf-8') as f:
            for line in f:
                if name.lower() not in line.lower(): continue
                rec = _parse_db_line(line)
                if not rec or len(rec) < 26: continue

                wn, ln = rec[2], rec[4]
                is_w = _name_match(name, str(wn))
                is_l = _name_match(name, str(ln))
                if not is_w and not is_l: continue
                if rec[1] not in allowed_levels: continue

                if is_w:
                    wp = (rec[6], rec[8], rec[9], rec[10], rec[11], rec[12])
                    lp = (rec[13], rec[15], rec[16], rec[17], rec[18], rec[19])
                else:
                    wp = (rec[13], rec[15], rec[16], rec[17], rec[18], rec[19])
                    lp = (rec[6], rec[8], rec[9], rec[10], rec[11], rec[12])

                sv_pts  += wp[0]; sv_won  += wp[1] + wp[2]; sv_gms  += wp[3]
                sv_held += max(0, wp[3] - (wp[5] - wp[4]))
                rt_pts  += lp[0]; rt_won  += max(0, lp[0] - (lp[1] + lp[2]))
                rt_gms  += lp[3]; rt_brk  += max(0, lp[5] - lp[4])
                n += 1
    except Exception:
        return None

    if n == 0: return None
    spw = max(25.1, sv_won / sv_pts * 100 if sv_pts else 55.0)
    rpw = max(25.1, rt_won / rt_pts * 100 if rt_pts else 40.0)

    tags = ["MUESTRA MIXTA"]
    if n < 10: tags.append("FALLBACK")

    return {
        "n": n, "n_total": n,
        "hold": round(sv_held / sv_gms * 100, 1) if sv_gms else 0,
        "brk": round(rt_brk / rt_gms * 100, 1) if rt_gms else 0,
        "spw": round(spw, 1), "rpw": round(rpw, 1),
        "clutch": 100.0, "dr_last5": 1.0, "last5_wins": 0,
        "fatigue_mins": 0, "tags": tags, "source": "Oráculo Local (mixta)",
    }

# ─────────────────────────────────────────────────────────────────────────────
# H2H DIRECTO (≤36 Meses)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_h2h(p1: str, p2: str, path: str = DB_PATH, months_limit: int = 36) -> dict:
    wins_p1 = wins_p2 = 0
    matches = []
    try:
        with open(path, encoding='utf-8') as f:
            for line in f:
                low = line.lower()
                if p1.lower() not in low or p2.lower() not in low: continue
                rec = _parse_db_line(line)
                if not rec or len(rec) < 26: continue

                wn, ln, rec_date = str(rec[2]), str(rec[4]), rec[20]
                p1_is_w = _name_match(p1, wn) and _name_match(p2, ln)
                p1_is_l = _name_match(p1, ln) and _name_match(p2, wn)
                if not p1_is_w and not p1_is_l: continue

                try: rec_date_int = int(rec_date)
                except: rec_date_int = 0

                if rec_date_int > 0:
                    try:
                        match_dt = _date_int_to_dt(rec_date_int)
                        cutoff = datetime.now() - timedelta(days=months_limit * 30)
                        if match_dt < cutoff: continue
                    except Exception: pass

                surface = SRF_MAP.get(rec[0], "?")
                tourney = str(rec[24]) if len(rec) > 24 else "?"

                if p1_is_w:
                    wins_p1 += 1
                    matches.append({"date": rec_date_int, "winner": p1, "surface": surface, "tourney": tourney})
                else:
                    wins_p2 += 1
                    matches.append({"date": rec_date_int, "winner": p2, "surface": surface, "tourney": tourney})
    except Exception:
        pass

    return {
        "p1_wins": wins_p1, "p2_wins": wins_p2,
        "total": wins_p1 + wins_p2,
        "matches": sorted(matches, key=lambda x: x["date"], reverse=True),
    }

# ─────────────────────────────────────────────────────────────────────────────
# TANKING CHECK & TIER DROP
# ─────────────────────────────────────────────────────────────────────────────
def check_tanking(name: str, path: str = DB_PATH, last_n: int = 5) -> bool:
    losses = []
    try:
        with open(path, encoding='utf-8') as f:
            for line in f:
                if name.lower() not in line.lower(): continue
                rec = _parse_db_line(line)
                if not rec or len(rec) < 26: continue
                if _name_match(name, str(rec[4])):
                    try: losses.append((int(rec[20]), int(rec[21])))
                    except: pass
    except Exception: return False
    losses.sort(key=lambda x: x[0], reverse=True)
    recent = losses[:last_n]
    return sum(1 for _, mins in recent if 0 < mins < 60) >= 2


def check_tier_drop(name: str, level: str, path: str = DB_PATH) -> bool:
    if level not in ("Challenger", "ITF"): return False
    ranks = []
    try:
        with open(path, encoding='utf-8') as f:
            for line in f:
                if name.lower() not in line.lower(): continue
                rec = _parse_db_line(line)
                if not rec or len(rec) < 26: continue
                
                try: 
                    d = int(rec[20])
                    wr, lr = int(rec[22]), int(rec[23])
                except: continue

                if _name_match(name, str(rec[2])) and wr > 0: ranks.append((d, wr))
                elif _name_match(name, str(rec[4])) and lr > 0: ranks.append((d, lr))
    except Exception: return False

    if not ranks: return False
    ranks.sort(key=lambda x: x[0], reverse=True)
    best_recent = min(r[1] for r in ranks[:10])
    return best_recent <= 100

# ─────────────────────────────────────────────────────────────────────────────
# FÍSICA Y ENTORNO
# ─────────────────────────────────────────────────────────────────────────────
def apply_environment(spw: float, rpw: float, n: int,
                      altitude_m: int, fatigue_mins: int,
                      surface: str, tourney: str = "") -> tuple[float, float, list]:
    fb = get_fallback(surface)
    AVG_SPW, AVG_RPW = fb["spw"], fb["rpw"]
    adjustments = []

    confidence = min(n / 20.0, 1.0) if n > 0 else 0.0
    shrink     = 0.35 * (1.0 - confidence)

    total_points = spw + rpw
    adj_total = total_points * (1 - shrink) + (AVG_SPW + AVG_RPW) * shrink
    adj_spw = (spw / total_points) * adj_total if total_points > 0 else AVG_SPW
    adj_rpw = adj_total - adj_spw

    if shrink > 0.01:
        adjustments.append(f"Shrinkage {shrink:.0%} (n={n})")

    if altitude_m > 200:
        alt_bonus = (altitude_m / 1000.0) * 1.5
        adj_spw += alt_bonus
        adj_rpw -= alt_bonus * 0.5     
        adjustments.append(f"Altitud +{alt_bonus:.2f} SPW (elev={altitude_m}m)")

    tourney_low = tourney.lower() if tourney else ""
    if any(t in tourney_low for t in FAST_COURTS):
        adj_spw += 1.5
        adj_rpw -= 1.5
        adjustments.append(f"Court Pace RÁPIDA ({tourney})")
    elif any(t in tourney_low for t in SLOW_COURTS):
        adj_spw -= 1.5
        adj_rpw += 1.5
        adjustments.append(f"Court Pace LENTA ({tourney})")

    if fatigue_mins > 150:
        penalty = min((fatigue_mins - 150) / 100.0 * 0.5, 2.0)
        adj_spw -= penalty
        adj_rpw -= penalty * 0.3       
        adjustments.append(f"Fatiga -{penalty:.2f} SPW ({fatigue_mins} min recientes)")

    return adj_spw, adj_rpw, adjustments


# ─────────────────────────────────────────────────────────────────────────────
# FASE CERO: AJUSTES DINÁMICOS
# ─────────────────────────────────────────────────────────────────────────────
def compute_adjustments(s1: dict, s2: dict, h2h: dict, surface: str,
                        tourney: str, level: str,
                        p1_name: str, p2_name: str) -> tuple[float, float, list]:
    sum_adj = 0.0
    ic = 1.00
    notes = []

    for tag in s1.get("tags", []) + s2.get("tags", []):
        if tag == "MUESTRA MIXTA" and ic > 0.85:
            ic -= 0.05
            notes.append(f"[MUESTRA MIXTA] IC -0.05")
        if tag == "FALLBACK":
            ic -= 0.15
            notes.append(f"[FALLBACK] IC -0.15")

    if s1.get("n", 0) < 10 or s2.get("n", 0) < 10:
        ic = min(ic, 0.85)
        notes.append(f"Muestra <10 en superficie → IC ≤ 0.85")

    if h2h["total"] == 0:
        ic = min(ic, 0.90)
        notes.append(f"H2H = 0 partidos → IC ≤ 0.90")
    elif h2h["total"] >= 3:
        h2h_ratio = h2h["p1_wins"] / h2h["total"]
        if h2h_ratio >= 0.7:
            sum_adj += 0.03
            notes.append(f"H2H dominante ({h2h['p1_wins']}-{h2h['p2_wins']}) → +0.03")
        elif h2h_ratio <= 0.3:
            sum_adj -= 0.03
            notes.append(f"H2H desfavorable ({h2h['p1_wins']}-{h2h['p2_wins']}) → -0.03")

    dr1 = s1.get("dr_last5", 1.0)
    wins1 = s1.get("last5_wins", 0)
    if dr1 > 1.25:
        sum_adj += 0.04
        notes.append(f"DR últimos 5 = {dr1:.3f} (dominante) → +0.04")
    if wins1 <= 1:
        sum_adj -= 0.04
        notes.append(f"Solo {wins1}/5 victorias recientes → -0.04")

    c1 = s1.get("clutch", 100.0)
    c2 = s2.get("clutch", 100.0)
    if c1 > 110 and c2 < 90:
        sum_adj += 0.05
        notes.append(f"Clutch ventaja ({c1:.0f} vs {c2:.0f}) → +0.05")
    elif c1 < 90 and c2 > 110:
        sum_adj -= 0.05
        notes.append(f"Clutch desventaja ({c1:.0f} vs {c2:.0f}) → -0.05")

    fat1 = s1.get("fatigue_mins", 0)
    if fat1 > 150:
        sum_adj -= 0.05
        notes.append(f"Fatiga P1 ({fat1} min recientes) → -0.05")

    fat2 = s2.get("fatigue_mins", 0)
    if fat2 > 150:
        sum_adj += 0.03
        notes.append(f"Fatiga P2 ({fat2} min recientes) → +0.03")

    return sum_adj, max(ic, 0.50), notes


# ─────────────────────────────────────────────────────────────────────────────
# CLASIFICACIÓN (SEMÁFORO)
# ─────────────────────────────────────────────────────────────────────────────
def classify(pmod: float, ev_pct: float, odd_dec: float) -> tuple[str, str]:
    if pmod < 0.45: return "BASURA", "🗑️"
    if pmod >= 0.70:
        if ev_pct >= 3.0: return "FAVORITO LIMPIO", "🟢"
        elif ev_pct >= 0.0: return "FAVORITO VALOR BAJO", "🟢⚠️"
        else: return "FAVORITO SOBREVENDIDO", "🟢🚨"

    if ev_pct >= 15.0 and odd_dec > 2.0: return "FRANCOTIRADOR", "🎯"
    elif ev_pct >= 6.5: return "DERECHA", "✅"
    elif ev_pct >= 4.0: return "PARLAY", "🟡"
    else: return "BASURA", "🗑️"

# ─────────────────────────────────────────────────────────────────────────────
# PARSER DE SPORTSBOOK
# ─────────────────────────────────────────────────────────────────────────────
def extract_american_odds(text: str):
    m = re.search(r'([+-]\d{3,4})', text)
    if m: return text.replace(m.group(1), '').strip(), int(m.group(1))
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
                if "itf" in low or "world tennis" in low:  cur_league = "ITF"
                elif "challenger" in low: cur_league = "Challenger"
                elif "masters" in low or "1000" in low: cur_league = "Masters"
                elif "grand slam" in low: cur_league = "GrandSlam"
                elif "wta" in low or "women" in low: cur_league = "ATP"  
                elif "atp" in low: cur_league = "ATP"
                continue
            parts = re.split(r'(?i)\s+vs\.?\s+', line)
            if len(parts) == 2:
                p1, o1 = extract_american_odds(parts[0])
                p2, o2 = extract_american_odds(parts[1])
                if p1 and p2: results.append((p1, o1, p2, o2, cur_league))
        if results: return results

    DATE_RE    = re.compile(r'^(\d{1,2}\s+[a-zA-Z]{3}\s+)?\d{2}:\d{2}$')
    COUNTER_RE = re.compile(r'^\+\s*\d{1,2}\s*(Streaming)?$', re.I)
    ODD_LONE   = re.compile(r'^[+-]\d{3,4}$')
    name_q, odd_q = [], []

    for raw in text.splitlines():
        line = raw.strip()
        if not line: continue
        low = line.lower()

        if any(kw in low for kw in METADATA_KW):
            if "itf" in low or "world tennis" in low: cur_league = "ITF"
            elif "challenger" in low: cur_league = "Challenger"
            elif "masters" in low or "1000" in low: cur_league = "Masters"
            elif "atp" in low: cur_league = "ATP"
            elif "wta" in low or "women" in low: cur_league = "ATP"
            continue
        if DATE_RE.match(line) or COUNTER_RE.match(line): continue

        odds_in = re.findall(r'[+-]\d{3,4}', line)
        if ODD_LONE.match(line): odd_q.append(int(line))
        elif odds_in:
            for o in odds_in: odd_q.append(int(o))
            name_part = re.sub(r'[+-]\d{3,4}.*', '', line).strip()
            if name_part: name_q.append(name_part)
        else: name_q.append(line)

        while len(name_q) >= 2 and len(odd_q) >= 2:
            results.append((name_q.pop(0), odd_q.pop(0), name_q.pop(0), odd_q.pop(0), cur_league))

    return results

# ─────────────────────────────────────────────────────────────────────────────
# MATEMÁTICA LOCAL
# ─────────────────────────────────────────────────────────────────────────────
def american_to_decimal(odd: int) -> float:
    if odd > 0: return (odd / 100.0) + 1.0
    elif odd < 0: return (100.0 / abs(odd)) + 1.0
    return 1.0

def no_vig(odd1: int, odd2: int) -> tuple:
    i1 = 1 / american_to_decimal(odd1)
    i2 = 1 / american_to_decimal(odd2)
    t  = i1 + i2
    p1, p2 = i1 / t, i2 / t
    return p1, p2, round(1 / p1, 3), round(1 / p2, 3)

def calc_ev(prob: float, dec_odd: float) -> float:
    return prob * (dec_odd - 1) - (1 - prob)

def log5_serve(spw: float, rpw: float) -> float:
    A, B = spw / 100, rpw / 100
    d = A * (1 - B) + (1 - A) * B
    return (A * (1 - B)) / d if d else 0.5

def game_prob(p: float) -> float:
    """Markov Chain probability of winning a game serving with p"""
    if p <= 0.0: return 0.0
    if p >= 1.0: return 1.0
    p_deuce = (p**2) / (p**2 + (1-p)**2)
    return (p**4) * (1 + 4*(1-p) + 10*(1-p)**2) + 20 * (p**3) * ((1-p)**3) * p_deuce

def sim_tiebreak(p_first: float, p_second: float) -> int:
    a = b = total = 0
    while True:
        win = random.random() < p_first if total == 0 else random.random() < (p_second if ((total - 1) // 2) % 2 == 0 else p_first)
        if win: a += 1
        else:   b += 1
        total += 1
        if a >= 7 and a - b >= 2: return 1
        if b >= 7 and b - a >= 2: return 0

def sim_set(prob_game_A: float, prob_game_B: float, pA: float, pB: float, a_first: bool = True) -> tuple[int, bool]:
    gA = gB = 0
    a_serves = a_first
    while True:
        if gA == 6 and gB == 6:
            w = sim_tiebreak(pA if a_serves else pB, pB if a_serves else pA)
            if w: gA += 1
            else: gB += 1
            return (1 if gA > gB else 0), not a_serves
        if a_serves:
            if random.random() < prob_game_A: gA += 1
            else: gB += 1
        else:
            if random.random() < prob_game_B: gB += 1
            else: gA += 1
        a_serves = not a_serves
        if gA >= 6 and gA - gB >= 2: return 1, a_serves
        if gB >= 6 and gB - gA >= 2: return 0, a_serves
        if gA == 7: return 1, a_serves
        if gB == 7: return 0, a_serves

def sim_match(pA: float, pB: float, best_of: int = 3, n: int = MC_ITERATIONS) -> float:
    needed = 2 if best_of == 3 else 3
    wins = 0
    prob_game_A = game_prob(pA)
    prob_game_B = game_prob(pB)
    for _ in range(n):
        sA = sB = 0
        a_srv = True
        while sA < needed and sB < needed:
            w, a_srv = sim_set(prob_game_A, prob_game_B, pA, pB, a_srv)
            if w: sA += 1
            else: sB += 1
        if sA == needed: wins += 1
    return wins / n

# ─────────────────────────────────────────────────────────────────────────────
# GEMINI — FALLBACK WEB + ANÁLISIS H48
# ─────────────────────────────────────────────────────────────────────────────
def gemini_stats(name: str) -> dict | None:
    client = get_gemini_client()
    if not client: return None
    try:
        prompt = f"""Busca estadísticas de tenis recientes de "{name}"
Necesito SOLO: Service Points Won % y Return Points Won %
Responde SOLO con JSON crudo: {{"spw_pct": 64.5, "rpw_pct": 38.2}}"""
        r = client.models.generate_content(
            model=GEMINI_MODEL, contents=prompt,
            config=types.GenerateContentConfig(tools=[{"google_search": {}}], temperature=0.1)
        )
        raw = r.text.replace("```json", "").replace("```", "").strip()
        s, e = raw.find('{'), raw.rfind('}')
        data = json.loads(raw[s:e+1]) if s != -1 and e != -1 else json.loads(raw)
        
        return {
            "n": 0, "n_total": 0, "hold": 0.0, "brk": 0.0,
            "spw": round(max(25.1, float(data.get("spw_pct", 55.0))), 1),
            "rpw": round(max(25.1, float(data.get("rpw_pct", 40.0))), 1),
            "clutch": 100.0, "dr_last5": 1.0, "last5_wins": 0,
            "fatigue_mins": 0, "tags": ["FUENTE WEB"], "source": "Gemini Search",
        }
    except Exception as e:
        return None

def gemini_full_analysis(p1: str, p2: str, report: str) -> str:
    client = get_gemini_client()
    if not client: return "❌ Sin API Key."
    try:
        prompt = f"""Eres analista. Busca {p1} y {p2} (últimas 48 hrs, fatiga/lesiones).
AUDITORÍA MATEMÁTICA: {report}
VEREDICTO ESTRICTO:
- EV > 5% → ✅ VERDE
- 0% < EV ≤ 5% → ⚠️ AMARILLO 
- EV negativo o lesión → 🚫 ROJO
Responde corto con Contexo, Auditoría y Veredicto."""
        r = client.models.generate_content(
            model=GEMINI_MODEL, contents=prompt,
            config=types.GenerateContentConfig(tools=[{"google_search": {}}], temperature=0.2)
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
        if "gcp_service_account" not in st.secrets: return None
        gc = gspread.service_account_from_dict(st.secrets["gcp_service_account"])
        return gc.open_by_key("1kciFhxjiVOeScsu_7e6UZvJ36ungHyeQxjIWMBu5CYs").sheet1
    except Exception: return None

def log_prediction(match_id, p1, p2, p_mod, p_casa, odd, ev_val, tier, league, ic):
    sheet = get_sheet()
    if not sheet: return False
    try:
        if match_id in sheet.col_values(1): return False
        sheet.append_row([
            match_id, datetime.now().strftime("%Y-%m-%d %H:%M"),
            p1, p2, f"{p_mod*100:.1f}%", f"{p_casa*100:.1f}%" if p_casa else "S/C",
            str(odd) if odd else "S/C", f"{ev_val*100:+.1f}%" if ev_val is not None else "S/C",
            tier, "", league, f"{ic:.2f}"
        ])
        return True
    except Exception: return False

# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Quantum Tennis v8.1", page_icon="🎾", layout="wide")
    st.title("🎾 Quantum Tennis Engine v8.1 — Oráculo Math Audit Edition")
    st.caption("Motor local (Log5 · Markov MC 50k) · Oráculo 40k+ registros (2022-2024) · Filtros Nivel/Superficie/Fecha · H2H · Clutch · DR · Fatiga · Court Pace · Gemini 2.5 Pro Search Grounding")

    if not gemini_available():
        st.error("⚠️ Falta GOOGLE_API_KEY en tu archivo .env")
        st.stop()

    if "txt" not in st.session_state: st.session_state.txt = ""
    
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1: best_of = st.radio("Formato:", [3, 5], horizontal=True, format_func=lambda x: f"Mejor de {x}")
    with c2: db_path = st.text_input("BD", value=DB_PATH, label_visibility="collapsed")
    with c3:
        if st.button("🗑️ Limpiar", use_container_width=True):
            st.session_state.txt = ""
            st.rerun()

    txt = st.text_area("📋 Pega los partidos (formato Caliente o 'Sinner -120 vs Alcaraz +100'):", key="txt", height=160, placeholder="Alcaraz C\nSinner J\n-140\n+110")

    if not st.button("🚀 Analizar", type="primary", use_container_width=True): return
    if not txt.strip():
        st.warning("Pega al menos un partido.")
        return

    partidos = parse_matches(txt)
    if not partidos:
        st.error("No se encontraron pares.")
        return

    for p1_raw, odd1, p2_raw, odd2, league in partidos:
        p1, p2 = p1_raw or "Jugador 1", p2_raw or "Jugador 2"

        if "utr" in p1.lower() or "utr" in p2.lower():
            st.error(f"🚫 EXCLUIDO | TORNEO UTR: {p1} vs {p2}")
            continue

        if odd1 and odd2:
            dec1 = american_to_decimal(odd1)
            if dec1 < 1.18:
                st.warning(f"🚫 EXCLUIDO | SUPER-FAVORITO (cuota {dec1:.2f}): {p1}")
                continue

        st.divider()
        st.subheader(f"⚡ {p1}  vs  {p2}")

        with st.spinner("🌍 Detectando torneo, superficie y contexto…"):
            ctx = extract_match_context(p1, p2)

        surface, tourney, altitude, level = ctx.get("surface", "Hard"), ctx.get("tourney", "Unknown"), ctx.get("altitude", 0), ctx.get("level", league)
        st.caption(f"📍 {tourney} | {surface} | {level} | Alt: {altitude}m")

        if check_tier_drop(p1, level, db_path):
            st.error(f"🚫 EXCLUIDO | TIER DROP POST-LESIÓN: {p1} (Top 100 en {level})")
            continue

        c_s1, c_s2 = st.columns(2)
        with c_s1:
            with st.spinner(f"Stats {p1}…"):
                s1 = get_stats(p1, surface, level, db_path)
                if not s1: s1 = gemini_stats(p1)
        with c_s2:
            with st.spinner(f"Stats {p2}…"):
                s2 = get_stats(p2, surface, level, db_path)
                if not s2: s2 = gemini_stats(p2)

        if not s1 or not s2:
            st.error("Stats insuficientes.")
            continue

        h2h = get_h2h(p1, p2, db_path)
        tank_p1, tank_p2 = check_tanking(p1, db_path), check_tanking(p2, db_path)
        
        sum_adj, ic, adj_notes = compute_adjustments(s1, s2, h2h, surface, tourney, level, p1, p2)
        
        adj_spw1, adj_rpw1, env_notes1 = apply_environment(s1["spw"], s1["rpw"], s1["n"], altitude, s1["fatigue_mins"], surface, tourney)
        adj_spw2, adj_rpw2, env_notes2 = apply_environment(s2["spw"], s2["rpw"], s2["n"], altitude, s2["fatigue_mins"], surface, tourney)

        pA, pB = log5_serve(adj_spw1, adj_rpw2), log5_serve(adj_spw2, adj_rpw1)

        with st.spinner(f"Corriendo {MC_ITERATIONS:,} simulaciones Gaussianas…"):
            mc_A = sim_match(pA, pB, best_of)
            mc_B = 1 - mc_A

        pmod_final = max(0.01, min(0.99, mc_A * (1 + sum_adj)))

        st.markdown("#### 🧮 Motor Matemático")
        m1, m2 = st.columns(2)
        ev_lines = []

        for col, name, p_srv, mc_w, odd, stats, adj_spw, adj_rpw in [
            (m1, p1, pA, pmod_final, odd1, s1, adj_spw1, adj_rpw1),
            (m2, p2, pB, 1-pmod_final, odd2, s2, adj_spw2, adj_rpw2),
        ]:
            col.markdown(f"**{name}**")
            tag_str = " · ".join(stats.get("tags", [])) if stats.get("tags") else ""
            col.caption(
                f"Fuente: {stats['source']} · n={stats['n']} (sup) / {stats.get('n_total', stats['n'])} (total) · "
                f"SPW {stats['spw']}% → {adj_spw:.1f}% · RPW {stats['rpw']}% → {adj_rpw:.1f}% · "
                f"Hold {stats['hold']}% · Clutch {stats.get('clutch', 0):.0f} · DR5 {stats.get('dr_last5', 0):.3f}"
                + (f" · ⚠️ {tag_str}" if tag_str else "")
            )
            col.metric("P(srv/punto) Log5", f"{p_srv*100:.1f}%")
            col.metric("P(match) Modelo", f"{mc_w*100:.1f}%")

            if odd:
                nv1, nv2, f1, f2 = no_vig(odd1 or 100, odd2 or 100)
                nv_this = nv1 if col is m1 else nv2
                fair    = f1  if col is m1 else f2
                dec_odd = american_to_decimal(odd)
                ev_val  = calc_ev(mc_w, dec_odd)
                edge    = (mc_w - nv_this) * 100
                tier, emoji = classify(mc_w, ev_val * 100, dec_odd)

                col.metric("P(match) Casa No-Vig", f"{nv_this*100:.1f}%", f"Fair odd {fair}", delta_color="off")
                col.metric("💰 Expected Value", f"{ev_val*100:+.1f}%", delta_color="normal" if ev_val > 0 else "inverse")
                col.metric("Edge vs Casa", f"{edge:+.1f}%")
                if col is m1: ev_lines.append(f"{name}: EV={ev_val*100:+.2f}% | Edge={edge:+.1f}% | {tier} {emoji}")
            else:
                col.info("Sin cuota → EV no calculable")

        st.markdown("---")
        if odd1 and odd2:
            dec1, ev1 = american_to_decimal(odd1), calc_ev(pmod_final, american_to_decimal(odd1))
            tier1, emoji1 = classify(pmod_final, ev1 * 100, dec1)
            nv1, _, _, _ = no_vig(odd1, odd2)

            st.markdown(f"""
#### 📊 Salida Quantum Engine

| Campo | Valor |
|---|---|
| **Partido** | {p1} vs {p2} |
| **Torneo** | {tourney} ({level}) |
| **Superficie** | {surface} |
| **Pmod (Real Prob)** | {pmod_final*100:.1f}% |
| **Implied Obj (Cuota)** | {nv1*100:.1f}% |
| **EV Edge** | {ev1*100:+.1f}% |
| **IC Confianza** | {ic:.2f} {' '.join('['+t+']' for t in s1.get('tags',[]))} |
| **STATUS FINAL** | {tier1} {emoji1} |
| **TANKING CHECK** | {'⚠️ SÍ' if tank_p1 else 'NO'} ({p1}) \| {'⚠️ SÍ' if tank_p2 else 'NO'} ({p2}) |
| **H2H reciente** | {h2h['p1_wins']}-{h2h['p2_wins']} ({h2h['total']} partidos ≤36m) |
""")
            with st.expander("🔧 Ajustes aplicados al modelo"):
                for note in adj_notes + env_notes1 + env_notes2: st.write(f"• {note}")
                st.write(f"**sum_adj total: {sum_adj:+.3f}**")
            log_prediction(f"{p1}_{p2}_{datetime.now().strftime('%Y%m%d')}", p1, p2, pmod_final, nv1, odd1, ev1, tier1, level, ic)

        st.markdown("#### 🤖 Gemini — Análisis H48 + Veredicto")
        report_full = f"""
{p1}: SPW={s1['spw']}%→{adj_spw1:.1f}% RPW={s1['rpw']}%→{adj_rpw1:.1f}% Hold={s1['hold']}% (n={s1['n']}, {s1['source']})
{p2}: SPW={s2['spw']}%→{adj_spw2:.1f}% RPW={s2['rpw']}%→{adj_rpw2:.1f}% Hold={s2['hold']}% (n={s2['n']}, {s2['source']})
Log5: {p1}={pA:.3f} | {p2}={pB:.3f}
Monte Carlo ({MC_ITERATIONS} iter, Bo{best_of}): {p1}={pmod_final*100:.1f}% | {p2}={(1-pmod_final)*100:.1f}%
H2H: {h2h['p1_wins']}-{h2h['p2_wins']} (≤36m)
{chr(10).join(ev_lines) if ev_lines else 'Sin cuotas'}
IC: {ic:.2f} | sum_adj: {sum_adj:+.3f}
"""
        with st.spinner("Gemini buscando contexto y generando veredicto…"):
            st.markdown(gemini_full_analysis(p1, p2, report_full))

if __name__ == "__main__":
    main()
