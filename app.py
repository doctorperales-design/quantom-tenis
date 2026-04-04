import streamlit as st
import json
import os
import re
import random
import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types
import anthropic
from openai import OpenAI

load_dotenv()

# --- MATH ENGINE (PYTHON LOCAL) ---

def extract_american_odds(text):
    match = re.search(r'([+-]\d{3,4})', text)
    if match:
        odd = int(match.group(1))
        name = text.replace(match.group(1), '').strip()
        return name, odd
    return text.strip(), None

def parse_caliente_blocks(text):
    """
    Parsea bloques crudos copiados desde apps (ej. Caliente)
    Busca patrones de líneas apiladas incluso si están interrumpidas por metadata.
    """
    matches = []
    
    # 1. Chequeo de formato simple con 'vs' (Fallback)
    if ' vs ' in text.lower() and ('+' in text or '-' in text):
        for line in text.split('\n'):
            line = line.strip()
            if not line or ' vs ' not in line.lower(): continue
            parts = re.split(r'(?i) vs ', line)
            p1, o1 = extract_american_odds(parts[0])
            p2, o2 = extract_american_odds(parts[1])
            matches.append((p1, o1, p2, o2))
        return matches

    # 2. Heurística Extrema (Máquina de estados para tokens saltados)
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    
    names_buffer = []
    odds_buffer = []

    for line in lines:
        odds_in_line = re.findall(r'[-+]\d{3,4}', line)
        
        # Filtrar basuras de las casas de apuestas (metadatos)
        is_metadata = any(x in line.lower() for x in ['local', 'visita', 'empate', 'sencillos', 'dobles', 'vivo', 'apuestas', 'streaming', 'women', 'men', 'tour ', 'challenger', 'atp', 'wta', 'itf'])
        is_time = bool(re.search(r'\d{2}:\d{2}', line))
        is_counter = line.startswith('+') and len(line) <= 5 and len(odds_in_line) == 0
        
        if odds_in_line and not is_counter:
            for o in odds_in_line:
                if len(o) >= 3: # Validación extra para no agarrar +4 por error
                    odds_buffer.append(int(o))
            
            # Si completamos un par de cuotas y teníamos al menos 2 nombres rezagados
            if len(odds_buffer) >= 2 and len(names_buffer) >= 2:
                # Los nombres siempre son los últimos 2 registrados antes de las cuotas
                p1 = names_buffer[-2]
                p2 = names_buffer[-1]
                matches.append((p1, odds_buffer[0], p2, odds_buffer[1]))
                # Reset
                names_buffer = []
                odds_buffer = []
        else:
            if not is_metadata and not is_time and not is_counter:
                # Es muy probable que sea el nombre de un jugador
                names_buffer.append(line)

    return matches

def american_to_decimal(odd):
    if odd > 0: return (odd / 100.0) + 1.0
    elif odd < 0: return (100.0 / abs(odd)) + 1.0
    return 1.0

def no_vig_probs(odd1, odd2):
    dec1 = american_to_decimal(odd1)
    dec2 = american_to_decimal(odd2)
    imp1 = 1.0 / dec1
    imp2 = 1.0 / dec2
    total = imp1 + imp2
    true_prob1 = imp1 / total
    true_prob2 = imp2 / total
    return true_prob1, true_prob2, 1.0/true_prob1, 1.0/true_prob2

def calculate_ev(prob_win, decimal_odd):
    profit_if_win = decimal_odd - 1.0
    prob_loss = 1.0 - prob_win
    return (prob_win * profit_if_win) - (prob_loss * 1.0)

def log5_point_prob(spw_server, rpw_returner):
    A = spw_server / 100.0
    B = rpw_returner / 100.0
    # Prob Server wins point vs Returner
    return (A * (1 - B)) / (A * (1 - B) + (1 - A) * B)

def simulate_game(p_serve):
    pts_srv, pts_ret = 0, 0
    while True:
        if random.random() < p_serve: pts_srv += 1
        else: pts_ret += 1
        if pts_srv >= 4 and pts_srv - pts_ret >= 2: return 1
        if pts_ret >= 4 and pts_ret - pts_srv >= 2: return 0

def simulate_set(p_serve_A, p_serve_B):
    games_A, games_B = 0, 0
    serve_turn = 0 
    while True:
        if games_A == 6 and games_B == 6:
            pts_A, pts_B, tb_serve_turn = 0, 0, 0
            while True:
                if tb_serve_turn % 4 in [0, 3]:
                    if random.random() < p_serve_A: pts_A += 1
                    else: pts_B += 1
                else:
                    if random.random() < p_serve_B: pts_B += 1
                    else: pts_A += 1
                if pts_A >= 7 and pts_A - pts_B >= 2: return 1
                if pts_B >= 7 and pts_B - pts_A >= 2: return 0
                tb_serve_turn += 1
        
        if serve_turn == 0:
            games_A += simulate_game(p_serve_A)
            serve_turn = 1
        else:
            games_B += simulate_game(p_serve_B)
            serve_turn = 0
            
        if games_A >= 6 and games_A - games_B >= 2: return 1
        if games_B >= 6 and games_B - games_A >= 2: return 0
        if games_A == 7: return 1
        if games_B == 7: return 0

def simulate_match(p_serve_A, p_serve_B, best_of=3, iterations=2000):
    sets_to_win = 2 if best_of == 3 else 3
    A_wins = 0
    for _ in range(iterations):
        sets_A, sets_B = 0, 0
        while sets_A < sets_to_win and sets_B < sets_to_win:
            if simulate_set(p_serve_A, p_serve_B): sets_A += 1
            else: sets_B += 1
        if sets_A == sets_to_win: A_wins += 1
    return A_wins / iterations

def calculate_stats(player_name, file_path="matches_jsonl.jsonl"):
    svc_games_total, svc_games_won = 0, 0
    svc_pts_total, svc_pts_won = 0, 0
    ret_games_total, ret_games_won = 0, 0
    ret_pts_total, ret_pts_won = 0, 0
    matches_found = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if player_name.lower() not in line.lower(): continue
                match = json.loads(line)
                w_name, l_name = match[2], match[4]
                
                if player_name.lower() in w_name.lower():
                    matches_found += 1
                    w_svpt, w_1st_won, w_2nd_won, w_svgms, w_bpsaved, w_bpfaced = match[6], match[8], match[9], match[10], match[11], match[12]
                    l_svpt, l_1st_won, l_2nd_won, l_svgms, l_bpsaved, l_bpfaced = match[13], match[15], match[16], match[17], match[18], match[19]
                    
                    svc_pts_total += w_svpt
                    svc_pts_won += w_1st_won + w_2nd_won
                    svc_games_total += w_svgms
                    svc_games_won += (w_svgms - (w_bpfaced - w_bpsaved))
                    
                    ret_pts_total += l_svpt
                    ret_pts_won += (l_svpt - (l_1st_won + l_2nd_won))
                    ret_games_total += l_svgms
                    ret_games_won += (l_bpfaced - l_bpsaved)
                    
                elif player_name.lower() in l_name.lower():
                    matches_found += 1
                    w_svpt, w_1st_won, w_2nd_won, w_svgms, w_bpsaved, w_bpfaced = match[6], match[8], match[9], match[10], match[11], match[12]
                    l_svpt, l_1st_won, l_2nd_won, l_svgms, l_bpsaved, l_bpfaced = match[13], match[15], match[16], match[17], match[18], match[19]
                    
                    svc_pts_total += l_svpt
                    svc_pts_won += l_1st_won + l_2nd_won
                    svc_games_total += l_svgms
                    svc_games_won += (l_svgms - (l_bpfaced - l_bpsaved))
                    
                    ret_pts_total += w_svpt
                    ret_pts_won += (w_svpt - (w_1st_won + w_2nd_won))
                    ret_games_total += w_svgms
                    ret_games_won += (w_bpfaced - w_bpsaved)
    except Exception as e:
        st.error(f"Error procesando base de datos: {e}")
        return None
        
    if matches_found == 0: return None
        
    hold_pct = (svc_games_won / svc_games_total * 100) if svc_games_total else 0
    break_pct = (ret_games_won / ret_games_total * 100) if ret_games_total else 0
    spw_pct = (svc_pts_won / svc_pts_total * 100) if svc_pts_total else 0
    rpw_pct = (ret_pts_won / ret_pts_total * 100) if ret_pts_total else 0
    
    # Filtro Anti-Ceros (Si Sackmann tiene un Walkover o partido sin Stats registradas)
    if spw_pct <= 10.0: spw_pct = 55.0
    if rpw_pct <= 10.0: rpw_pct = 40.0
    
    return {
        "matches": matches_found, "hold_pct": round(hold_pct, 1),
        "break_pct": round(break_pct, 1), "spw_pct": round(spw_pct, 1),
        "rpw_pct": round(rpw_pct, 1)
    }

def fetch_stats_gemini_fallback(player_name):
    """
    Fallback Inteligente: Si el jugador no existe localmente, Gemini usa Search Grounding
    para buscar sus estadísticas ITF/Challenger y escupir un JSON estricto.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key or api_key == "tu_llave_aqui":
        st.warning(f"⚠️ {player_name} no está en BD local. Requiere 'GOOGLE_API_KEY' para rastreo web.")
        return None
        
    try:
        st.info(f"🌐 Rastreando stats de {player_name} en la Web Profunda con Gemini...")
        client = genai.Client(api_key=api_key)
        
        prompt = f"""
Busca en bases de datos públicas (TennisAbstract, CoreTennis, FlashScore, Sofascore, ATP/ITF) las estadísticas generales recientes de este tenista ITF/Challenger:
"{player_name}"

Necesito EXCLUSIVAMENTE sus dos porcentajes de rendimiento subyacente:
1. Puntos Ganados con su Saque (Service Points Won %)
2. Puntos Ganados al Resto (Return Points Won %)

Si no hay un número exacto en la web, infiere promedios realistas basados en su Win Rate global o biografía (ej. ¿es gran sacador o de arcilla?) para nunca dejarlo en cero.
Responde ÚNICA Y ESTRICTAMENTE con un solo objeto JSON crudo en este formato (nada de texto, explicaciones, ni backticks):
{{
  "spw_pct": 65.5,
  "rpw_pct": 39.2
}}
"""
        response = client.models.generate_content(
            model='gemini-2.5-pro',
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[{"google_search": {}}],
                temperature=0.0
            ) # temperature 0 y search grounding
        )
        
        raw = response.text.replace('```json', '').replace('```', '').strip()
        data = json.loads(raw)
        
        spw = float(data.get("spw_pct", 50.0))
        rpw = float(data.get("rpw_pct", 50.0))
        
        # Filtro Anti-Cortocircuito (Si Gemini escupe Zeros, lo forzamos a métricas estándar de ITF)
        if spw <= 10.0: spw = 55.0 
        if rpw <= 10.0: rpw = 40.0
        
        return {
            "matches": "Web Scraping",
            "hold_pct": 0.0,
            "break_pct": 0.0,
            "spw_pct": spw,
            "rpw_pct": rpw
        }
    except Exception as e:
        st.error(f"Falla en el Rescate de Gemini para {player_name}: {e}")
        return None

# --- LLM AGENTS ---

def call_gemini_search(p1, p2):
    try:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key or api_key == "tu_llave_aqui": return "❌ No API KEY"
        client = genai.Client(api_key=api_key)
        
        prompt = f"""Busca información de las últimas 48h sobre {p1} y {p2}.
Reporta estrictamente hechos: 
1. Partidos muy largos.
2. Lesiones médicas recientes.
No des tu opinión predictiva, solo contexto estructurado en viñetas."""
        
        response = client.models.generate_content(
            model='gemini-2.5-pro', contents=prompt,
            config=types.GenerateContentConfig(tools=[{"google_search": {}}], temperature=0.0)
        )
        return response.text
    except Exception as e: return f"Error Gemini: {e}"

def call_claude_auditor(local_math_report):
    try:
        client = anthropic.Anthropic()
        prompt = f"""Eres un Auditor Matemático de apuestas. Analiza el siguiente reporte físico-matemático generado localmente en Python (que incluye Log5 Point Probs, Monte Carlo Win% y Expected Value local):
        
{local_math_report}

Tu trabajo es auditar la coherencia. Responde brevemente si las matemáticas tienen sentido según las varianzas históricas y si el Expected Value calculado suena creíble de acuerdo a la discrepancia entre Mercado y Monte Carlo.
Da tu respuesta en no más de 4 renglones en formato de tabla de auditoría."""
        response = client.messages.create(
            model="claude-4-6-sonnet-latest", max_tokens=250, temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception as e: return f"Error Claude: {e}"

def call_openai_consensus(local_math_report, gemini_context):
    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key or api_key == "tu_llave_aqui": return "❌ No API KEY"
        client = OpenAI(api_key=api_key)
        prompt = f"""Eres el 'Juez de Consenso Reglado'.

Matemáticas Locales Python (Expected Value):
{local_math_report}

Contexto Médico (Gemini):
{gemini_context}

Tus Reglas:
1. SI (EV Python de un jugador es > 5%) Y (Gemini reporta cero lesiones/fatiga severa para ese jugador): EMITIR DICTAMEN VERDE (APOSTAR).
2. SI (EV > 0 y <= 5%) O (buen EV pero hay noticias de pequeña fatiga): EMITIR AMARILLO.
3. SI (EV negativo) O (lesión confirmada grave): EMITIR ROJO (EVITAR).

No expliques narrativa. Imprime un veredicto frío, binario y exacto obedeciendo tus reglas."""
        response = client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": prompt}], temperature=0, max_tokens=150
        )
        return response.choices[0].message.content
    except Exception as e: return f"Error OpenAI: {e}"

def main():
    st.set_page_config(page_title="Quantum Tennis Engine", page_icon="🎾", layout="wide")
    st.title("🎾 Quantum Tennis Engine v4.0 (Math-Safe)")
    st.markdown("Ahora Python procesa *toda* la matemática de Log5, No-Vig y simulaciones Monte Carlo puras localmente.")
    
    best_of = st.radio("Formato del Torneo:", [3, 5], index=0, format_func=lambda x: f"Mejor de {x} Sets", horizontal=True)
    matches_text = st.text_area("📋 Pega tus partidos (con cuotas opcionales Ejemplo: Sinner -120 vs Alcaraz +100):", height=150)
    
    if st.button("🚀 Analizar Encuentros", use_container_width=True):
        if not matches_text.strip():
            st.warning("Ingresa un partido válido.")
            return

        # Usamos el nuevo Mega-Parser (que ahora soporta guiones especiales de iPhone/Mac)
        
        # Pre-limpiamos guiones raros que mete Apple por diseño tipográfico a guiones estándar
        limpio = matches_text.replace('–', '-').replace('—', '-').replace('−', '-')
        parsed_matches = parse_caliente_blocks(limpio)
        
        if not parsed_matches:
            st.error("🤖 El Parser no encontró pares de Jugadores y Cuotas. Asegúrate de haber copiado desde el Nombre del Jugador y no solo las cuotas.")
            with st.expander("🔎 Ver exactamente qué texto leyó el sistema (Debug)"):
                st.code(matches_text)
            return

        for p1_name, odd1, p2_name, odd2 in parsed_matches:
            # Fallbacks obligatorios
            if not p1_name: p1_name = "Desconocido 1"
            if not p2_name: p2_name = "Desconocido 2"
            
            # --- FILTROS DE SEGURIDAD (Excluir Ruido) ---
            if "utr" in p1_name.lower() or "utr" in p2_name.lower():
                st.warning(f"🚫 Saltando partido UTR Pro ({p1_name} vs {p2_name}) por falta de rigor ATP/WTA.")
                continue
                
            if odd1 and odd2:
                if (odd1 <= -500) or (odd2 <= -500):
                    st.warning(f"🚫 Saltando {p1_name} vs {p2_name} por estar fuertemente decidido (Cuota de {min(odd1, odd2)} <= -500).")
                    continue

            with st.expander(f"Análisis: {p1_name} vs {p2_name}", expanded=True):
                st.markdown("### 🧮 Motor Matemático (Python Local)")
                st.caption(f"Cuotas Detectadas: {p1_name} ({odd1}) vs {p2_name} ({odd2})")
                
                stats_p1 = calculate_stats(p1_name)
                if not stats_p1:
                    stats_p1 = fetch_stats_gemini_fallback(p1_name)
                    
                stats_p2 = calculate_stats(p2_name)
                if not stats_p2:
                    stats_p2 = fetch_stats_gemini_fallback(p2_name)
                
                if stats_p1 and stats_p2:
                    # Log 5 Probs serve
                    pA_serve = log5_point_prob(stats_p1['spw_pct'], stats_p2['rpw_pct'])
                    pB_serve = log5_point_prob(stats_p2['spw_pct'], stats_p1['rpw_pct'])
                    
                    # Monte Carlo
                    st.text("Corriendo 2,000 simulaciones Monte Carlo internas...")
                    mc_win_p1 = simulate_match(pA_serve, pB_serve, best_of=best_of)
                    mc_win_p2 = 1.0 - mc_win_p1
                    
                    c1, c2 = st.columns(2)
                    
                    if odd1 and odd2:
                        nv_prob1, nv_prob2, f_odd1, f_odd2 = no_vig_probs(odd1, odd2)
                        ev1 = calculate_ev(mc_win_p1, american_to_decimal(odd1))
                        ev2 = calculate_ev(mc_win_p2, american_to_decimal(odd2))
                        
                        # ----- Columna 1: Jugador 1 -----
                        c1.markdown(f"#### 🎾 {p1_name}")
                        c1.metric("1. Prob. de ganar su Saque", f"{pA_serve * 100:.1f}%")
                        
                        p1_modelo = mc_win_p1 * 100
                        p1_casa = nv_prob1 * 100
                        edge1 = p1_modelo - p1_casa
                        c1.metric("2. Prob. Modelo (Monte Carlo)", f"{p1_modelo:.1f}%", f"{edge1:+.1f}% Edge vs Casa🏆")
                        c1.metric("3. Prob. Casa (Bookie No-Vig)", f"{p1_casa:.1f}%", f"Fair Odds: {f_odd1:.2f}", delta_color="off")
                        c1.metric("💰 Expected Value (ROI)", f"{(ev1 * 100):+.1f}%")

                        # ----- Columna 2: Jugador 2 -----
                        c2.markdown(f"#### 🎾 {p2_name}")
                        c2.metric("1. Prob. de ganar su Saque", f"{pB_serve * 100:.1f}%")
                        
                        p2_modelo = mc_win_p2 * 100
                        p2_casa = nv_prob2 * 100
                        edge2 = p2_modelo - p2_casa
                        c2.metric("2. Prob. Modelo (Monte Carlo)", f"{p2_modelo:.1f}%", f"{edge2:+.1f}% Edge vs Casa🏆")
                        c2.metric("3. Prob. Casa (Bookie No-Vig)", f"{p2_casa:.1f}%", f"Fair Odds: {f_odd2:.2f}", delta_color="off")
                        c2.metric("💰 Expected Value (ROI)", f"{(ev2 * 100):+.1f}%")
                        
                        ev_info = f"EV P1: {(ev1 * 100):+.2f}%, EV P2: {(ev2 * 100):+.2f}% (Bookie Odds: {odd1} / {odd2})"
                    else:
                        c1.markdown(f"#### 🎾 {p1_name}")
                        c1.metric("Prob. Saque", f"{pA_serve * 100:.1f}%")
                        c1.metric("Prob. Modelo", f"{mc_win_p1 * 100:.1f}%")
                        
                        c2.markdown(f"#### 🎾 {p2_name}")
                        c2.metric("Prob. Saque", f"{pB_serve * 100:.1f}%")
                        c2.metric("Prob. Modelo", f"{mc_win_p2 * 100:.1f}%")
                        
                        st.warning("No se detectaron cuotas válidas (-120, +250) para cruzar contra la Casa.")
                        ev_info = "Sin cuotas ingresadas"

                    local_report = f"""
Stats P1 ({p1_name}): SPW {stats_p1['spw_pct']}%, RPW {stats_p1['rpw_pct']}%
Stats P2 ({p2_name}): SPW {stats_p2['spw_pct']}%, RPW {stats_p2['rpw_pct']}%
Log5 Serve Point Probs: P1={pA_serve:.3f}, P2={pB_serve:.3f}
Monte Carlo Win%: P1={mc_win_p1*100:.1f}%, P2={mc_win_p2*100:.1f}%
Expected Value (si aplica): {ev_info}
"""
                else:
                    st.error("Stats insuficientes para simulaciones.")
                    local_report = "Error. No stats."

                st.divider()
                st.markdown("### 🤖 Auditoría & Consenso")
                if "Error" not in local_report:
                    gemini_result = call_gemini_search(p1_name, p2_name)
                    claude_result = call_claude_auditor(local_report)
                    openai_result = call_openai_consensus(local_report, gemini_result)
                    
                    rc1, rc2, rc3 = st.columns(3)
                    with rc1:
                        st.info(f"**Gemini (Contexto H48):**\n\n{gemini_result}")
                    with rc2:
                        st.success(f"**Claude (Auditor):**\n\n{claude_result}")
                    with rc3:
                        st.warning(f"**GPT-4o (Tribunal):**\n\n{openai_result}")

if __name__ == "__main__":
    main()
