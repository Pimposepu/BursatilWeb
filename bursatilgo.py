"""
BursatilGO — App Local Simplificada v3
=======================================
UN SOLO ARCHIVO. Sin módulos externos propios.
Corré con:
  python -m streamlit run bursatilgo.py --server.address=0.0.0.0
Desde el celu: abrí http://IP-DE-TU-PC:8501
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="BursatilGO", page_icon="📊", layout="wide")

st.markdown("""
<style>
body, .main { background-color: #0e1117; color: white; }
.stButton>button { background-color: #00BCD4; color: black; font-weight: bold; border-radius: 8px; }
.stTextInput>div>input { background-color: #1e1e2e; color: white; border: 1px solid #333; }
h1 { color: #00BCD4 !important; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════
#  DESCARGA — compatible con yfinance v0.2+
# ════════════════════════════════════════════

def descargar_datos(simbolo, periodo="2y"):
    try:
        df = yf.download(
            simbolo, period=periodo, interval="1d",
            progress=False, auto_adjust=True, actions=False
        )
        if df is None or df.empty:
            return None

        # yfinance nuevo devuelve MultiIndex (Price, Ticker)
        # Lo aplanamos tomando siempre el primer nivel
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [str(col[0]).strip() for col in df.columns]

        # Normalizar a Capitalize
        df.columns = [str(c).strip().capitalize() for c in df.columns]

        # Verificar columnas mínimas
        for col in ['Open', 'High', 'Low', 'Close']:
            if col not in df.columns:
                return None

        df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
        df.index = pd.to_datetime(df.index)
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        return df

    except Exception as e:
        st.error(f"Error al descargar: {e}")
        return None


# ════════════════════════════════════════════
#  INDICADORES
# ════════════════════════════════════════════

def calcular_indicadores(df):
    df = df.copy()
    close = df['Close']

    df['RSI']       = ta.momentum.RSIIndicator(close, window=14).rsi()

    macd            = ta.trend.MACD(close)
    df['MACD']      = macd.macd()
    df['MACD_Sig']  = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()

    bb              = ta.volatility.BollingerBands(close, window=20)
    df['BB_Upper']  = bb.bollinger_hband()
    df['BB_Lower']  = bb.bollinger_lband()
    df['BB_Mid']    = bb.bollinger_mavg()

    df['MM8']       = close.rolling(8).mean()
    df['MM21']      = close.rolling(21).mean()
    df['MM200']     = close.rolling(200).mean()

    if 'Volume' in df.columns:
        df['Vol_Avg'] = df['Volume'].rolling(20).mean()

    df['Soporte']     = df['Low'].rolling(20,  min_periods=1).min()
    df['Resistencia'] = df['High'].rolling(20, min_periods=1).max()

    return df


# ════════════════════════════════════════════
#  PATRONES DE VELAS
# ════════════════════════════════════════════

def detectar_patron(df):
    if len(df) < 3:
        return "Sin datos", 0
    try:
        i  = len(df) - 1
        o,  c,  h,  l  = float(df['Open'].iloc[i]),   float(df['Close'].iloc[i]),  float(df['High'].iloc[i]),  float(df['Low'].iloc[i])
        o1, c1, h1, l1 = float(df['Open'].iloc[i-1]), float(df['Close'].iloc[i-1]), float(df['High'].iloc[i-1]), float(df['Low'].iloc[i-1])
        o2, c2         = float(df['Open'].iloc[i-2]), float(df['Close'].iloc[i-2])

        rango    = h - l if h != l else 1e-9
        cuerpo   = abs(c - o)
        mech_inf = min(o, c) - l
        mech_sup = h - max(o, c)

        if cuerpo / rango < 0.10:
            if mech_inf > rango * 0.6 and mech_sup < rango * 0.10:
                return "Dragonfly Doji", 1
            elif mech_sup > rango * 0.6 and mech_inf < rango * 0.10:
                return "Lapida Doji", -1
            return "Doji", 0

        if mech_inf >= cuerpo * 2 and mech_sup < cuerpo:
            return "Martillo", 1

        if mech_sup >= cuerpo * 2 and mech_inf < cuerpo:
            return "Estrella Fugaz", -1

        if c1 < o1 and c > o and c > o1 and o < c1:
            return "Envolvente Alcista", 1

        if c1 > o1 and c < o and c < o1 and o > c1:
            return "Envolvente Bajista", -1

        if h < h1 and l > l1 and cuerpo < abs(c1 - o1):
            return ("Harami Alcista", 1) if c1 < o1 else ("Harami Bajista", -1)

        if abs(l - l1) / rango < 0.05 and c1 < o1 and c > o:
            return "Pinzas Alcistas", 1
        if abs(h - h1) / rango < 0.05 and c1 > o1 and c < o:
            return "Pinzas Bajistas", -1

        if c2 < o2 and abs(c1-o1) < abs(c2-o2)*0.5 and c > o and c > (o2+c2)/2:
            return "Estrella Manana", 2

        if c2 > o2 and abs(c1-o1) < abs(c2-o2)*0.5 and c < o and c < (o2+c2)/2:
            return "Estrella Tarde", -2

        return "Ninguno", 0
    except Exception:
        return "Error", 0


# ════════════════════════════════════════════
#  FIBONACCI
# ════════════════════════════════════════════

def calcular_fibonacci(df, ventana=60):
    sub = df['Close'].tail(ventana)
    mn, mx = float(sub.min()), float(sub.max())
    r = mx - mn
    return {
        'fib_0':   mn,
        'fib_236': mn + 0.236 * r,
        'fib_382': mn + 0.382 * r,
        'fib_500': mn + 0.500 * r,
        'fib_618': mn + 0.618 * r,
        'fib_100': mx,
    }


# ════════════════════════════════════════════
#  SIMILITUDES HISTORICAS
# ════════════════════════════════════════════

def similitud_historica(serie, ventana=10, top=5):
    if len(serie) < ventana * 2:
        return []

    def norm(s):
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn + 1e-9)

    actual = norm(serie[-ventana:])
    sims = []
    for i in range(len(serie) - ventana * 2):
        dist = float(np.linalg.norm(actual - norm(serie[i:i+ventana])))
        sims.append((dist, i))
    sims.sort(key=lambda x: x[0])

    result = []
    for _, idx in sims[:top]:
        idx_fin = idx + ventana
        if idx_fin < len(serie) - 1:
            var = (serie[idx_fin+1] - serie[idx_fin]) / serie[idx_fin] * 100
            result.append({'idx': idx_fin, 'var_pct': round(float(var), 2)})
    return result


# ════════════════════════════════════════════
#  VOTOS
# ════════════════════════════════════════════

def safe_float(val, default=0.0):
    try:
        v = float(val)
        return default if (v != v) else v  # NaN check
    except Exception:
        return default


def calcular_votos(df, fib):
    u      = df.iloc[-1]
    u_prev = df.iloc[-2] if len(df) > 1 else u
    precio = safe_float(u['Close'])

    mm200       = safe_float(u.get('MM200'),     precio)
    mm21        = safe_float(u.get('MM21'),      precio)
    soporte     = safe_float(u.get('Soporte'),   precio * 0.95)
    resistencia = safe_float(u.get('Resistencia'), precio * 1.05)
    rsi         = safe_float(u.get('RSI'),       50)
    bb_lower    = safe_float(u.get('BB_Lower'),  precio * 0.95)
    bb_upper    = safe_float(u.get('BB_Upper'),  precio * 1.05)
    macd_hist   = safe_float(u.get('MACD_Hist'), 0)
    macd_hist_p = safe_float(u_prev.get('MACD_Hist'), 0)

    # Estructura
    precio_ini = float(df['Close'].iloc[-30]) if len(df) >= 30 else float(df['Close'].iloc[0])
    cambio_pct = (precio - precio_ini) / precio_ini * 100
    try:
        adx_val = float(ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14).adx().iloc[-1])
        if adx_val != adx_val: adx_val = 20
    except Exception:
        adx_val = 20

    if adx_val > 25:
        if cambio_pct > 2:   estructura = "TENDENCIA ALCISTA"
        elif cambio_pct < -2: estructura = "TENDENCIA BAJISTA"
        else:                 estructura = "RANGO"
    else:
        estructura = "RANGO / LATERAL"

    votos = {}

    # Pilar 1 — Tendencia
    votos['Tendencia'] = 1 if 'ALCISTA' in estructura else (-1 if 'BAJISTA' in estructura else 0)
    votos['MM200']     = 1 if precio > mm200 else -1
    votos['MM21']      = 1 if precio > mm21  else -1

    # Pilar 2 — Nivel
    cerca_sop = abs(precio - soporte) / precio < 0.025
    cerca_res = abs(precio - resistencia) / precio < 0.025
    votos['Nivel S/R']  = 1 if cerca_sop else (-1 if cerca_res else 0)
    cerca_fib = (abs(precio - fib['fib_500']) / precio < 0.025 or
                 abs(precio - fib['fib_618']) / precio < 0.025)
    votos['Fibonacci']  = 1 if (cerca_fib and votos['Tendencia'] == 1) else \
                         -1 if (cerca_fib and votos['Tendencia'] == -1) else 0
    votos['Bollinger']  = 1 if precio <= bb_lower else (-1 if precio >= bb_upper else 0)

    # Pilar 3 — Senal
    _, voto_velas = detectar_patron(df)
    votos['Velas'] = voto_velas
    votos['RSI']   = 1 if rsi < 40 else (-1 if rsi > 70 else 0)

    if   macd_hist > 0 and macd_hist_p <= 0: votos['MACD'] = 1
    elif macd_hist < 0 and macd_hist_p >= 0: votos['MACD'] = -1
    else: votos['MACD'] = 1 if macd_hist > 0 else -1 if macd_hist < 0 else 0

    # Externos (con fix MultiIndex)
    try:
        spy = yf.download("SPY", period="5d", interval="1d", progress=False, auto_adjust=True)
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = [str(c[0]).strip().capitalize() for c in spy.columns]
        votos['SPY'] = 1 if float(spy['Close'].iloc[-1]) > float(spy['Close'].iloc[-2]) else -1
    except Exception:
        votos['SPY'] = 0

    try:
        vix = yf.download("^VIX", period="2d", interval="1d", progress=False, auto_adjust=True)
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = [str(c[0]).strip().capitalize() for c in vix.columns]
        vix_val = float(vix['Close'].iloc[-1])
        votos['VIX'] = 1 if vix_val < 20 else (-1 if vix_val > 30 else 0)
    except Exception:
        votos['VIX'] = 0

    pesos = {'Tendencia':2,'MM200':1,'MM21':1,'Nivel S/R':2,'Fibonacci':1,
             'Bollinger':1,'Velas':2,'RSI':1,'MACD':1,'SPY':1,'VIX':1}

    score     = sum(votos[k] * pesos.get(k, 1) for k in votos)
    score_max = sum(pesos.values())

    if score >= 5:     decision, color = "COMPRAR",           "#26a69a"
    elif score <= -5:  decision, color = "VENDER / EVITAR",   "#ef5350"
    else:              decision, color = "ESPERAR / NEUTRO",  "#FFA726"

    return {
        'votos': votos, 'score': score, 'score_max': score_max,
        'decision': decision, 'color': color,
        'estructura': estructura, 'rsi': rsi,
        'precio': precio, 'soporte': soporte, 'resistencia': resistencia,
    }


# ════════════════════════════════════════════
#  GRAFICOS
# ════════════════════════════════════════════

def grafico_precio(df, soporte, resistencia, fib618):
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#1e1e2e')
    tail = df.tail(60)
    ax.plot(tail.index, tail['Close'],   color='white',   linewidth=1.8, label='Precio', zorder=3)
    ax.plot(tail.index, tail['MM8'],     color='#FFD700', linewidth=1.0, linestyle='--', label='MM8',  alpha=0.8)
    ax.plot(tail.index, tail['MM21'],    color='#00BCD4', linewidth=1.0, linestyle='-.', label='MM21', alpha=0.8)
    ax.fill_between(tail.index, tail['BB_Upper'], tail['BB_Lower'], alpha=0.08, color='#9C27B0')
    ax.axhline(soporte,     color='#2196F3', linestyle=':', linewidth=1.2, label='Soporte')
    ax.axhline(resistencia, color='#FF9800', linestyle=':', linewidth=1.2, label='Resistencia')
    ax.axhline(fib618,      color='#9C27B0', linestyle='--', linewidth=0.8, label='Fib 61.8%')
    ax.tick_params(colors='#888888')
    for sp in ax.spines.values(): sp.set_edgecolor('#333')
    ax.legend(fontsize=7, facecolor='#1e1e2e', labelcolor='white', framealpha=0.7, loc='upper left')
    ax.set_title('Precio - MM8 - MM21 - Bollinger', color='#888888', fontsize=10)
    plt.tight_layout()
    return fig


def grafico_rsi(df):
    fig, ax = plt.subplots(figsize=(10, 2))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#1e1e2e')
    tail_idx = df.tail(60).index
    tail_rsi = df['RSI'].tail(60)
    ax.plot(tail_idx, tail_rsi, color='#FF9800', linewidth=1.4)
    ax.axhline(70, color='#ef5350', linestyle='--', linewidth=0.8)
    ax.axhline(40, color='#26a69a', linestyle='--', linewidth=0.8)
    ax.fill_between(tail_idx, tail_rsi, 40, where=(tail_rsi < 40), color='#26a69a', alpha=0.25)
    ax.fill_between(tail_idx, tail_rsi, 70, where=(tail_rsi > 70), color='#ef5350', alpha=0.25)
    ax.set_ylim(0, 100)
    ax.set_title('RSI (14)', color='#888888', fontsize=10)
    ax.tick_params(colors='#888888')
    for sp in ax.spines.values(): sp.set_edgecolor('#333')
    plt.tight_layout()
    return fig


def grafico_macd(df):
    fig, ax = plt.subplots(figsize=(10, 2))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#1e1e2e')
    tail = df.tail(60)
    hist = tail['MACD_Hist']
    colors_bars = ['#26a69a' if v >= 0 else '#ef5350' for v in hist]
    ax.bar(tail.index, hist, color=colors_bars, alpha=0.8, width=0.8)
    ax.plot(tail.index, tail['MACD'],    color='#2196F3', linewidth=1.0, label='MACD')
    ax.plot(tail.index, tail['MACD_Sig'], color='#FF9800', linewidth=1.0, label='Senal')
    ax.axhline(0, color='white', linewidth=0.5)
    ax.set_title('MACD', color='#888888', fontsize=10)
    ax.tick_params(colors='#888888')
    ax.legend(fontsize=7, facecolor='#1e1e2e', labelcolor='white', framealpha=0.7)
    for sp in ax.spines.values(): sp.set_edgecolor('#333')
    plt.tight_layout()
    return fig


# ════════════════════════════════════════════
#  INTERFAZ
# ════════════════════════════════════════════

st.title("BursatilGO - IA Maestra")
st.caption("Sistema de 3 Pilares: Tendencia - Nivel - Senal | Basado en La Biblia de las Velas Japonesas")

c1, c2, c3 = st.columns([3, 1, 1])
with c1:
    simbolo = st.text_input("Ticker", value="AAPL", placeholder="Ej: AAPL, BTC-USD, ALUA.BA", label_visibility="collapsed").upper().strip()
with c2:
    periodo = st.selectbox("Periodo", ["1y", "2y", "5y"], index=1, label_visibility="collapsed")
with c3:
    analizar = st.button("ANALIZAR", use_container_width=True)

st.divider()

if analizar:
    if not simbolo:
        st.error("Ingresa un ticker.")
        st.stop()

    with st.spinner("Descargando datos de " + simbolo + "..."):
        df = descargar_datos(simbolo, periodo)

    if df is None or df.empty:
        st.error("No se encontraron datos para " + simbolo + ". Probá: AAPL, BTC-USD, MSFT, ALUA.BA")
        st.stop()

    with st.spinner("Calculando indicadores..."):
        df = calcular_indicadores(df)
        fib = calcular_fibonacci(df)
        patron, _ = detectar_patron(df)
        res = calcular_votos(df, fib)

    # Decision
    dec   = res['decision']
    col   = res['color']
    sc    = res['score']
    scmax = res['score_max']
    pct   = round(sc / scmax * 100)
    emoji = "✅" if "COMPRAR" in dec else "❌" if "VENDER" in dec else "⚪"

    st.markdown(
        "<div style='background:" + col + "22; border-left:4px solid " + col + "; padding:16px; border-radius:8px; margin-bottom:16px'>"
        "<span style='font-size:24px; font-weight:bold; color:" + col + "'>" + emoji + " " + dec + "</span>"
        "<span style='color:#aaa; font-size:14px; margin-left:16px'>Score: " + str(sc) + " / " + str(scmax) + "  (" + str(pct) + "%)</span>"
        "</div>",
        unsafe_allow_html=True
    )

    # Metricas
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Precio",      "$" + str(round(res['precio'], 2)))
    m2.metric("Soporte",     "$" + str(round(res['soporte'], 2)))
    m3.metric("Resistencia", "$" + str(round(res['resistencia'], 2)))
    m4.metric("RSI(14)",     str(round(res['rsi'], 1)))
    m5.metric("Fib 61.8%",  "$" + str(round(fib['fib_618'], 2)))
    m6.metric("Estructura",  res['estructura'])

    # Patron
    st.info("Patron detectado: " + patron)

    # Graficos
    st.subheader("Graficos")
    st.pyplot(grafico_precio(df, res['soporte'], res['resistencia'], fib['fib_618']))
    col_rsi, col_macd = st.columns(2)
    with col_rsi:
        st.pyplot(grafico_rsi(df))
    with col_macd:
        st.pyplot(grafico_macd(df))

    # Votos
    st.subheader("Votos Detallados — 3 Pilares")
    pilares = {
        "Pilar 1 - TENDENCIA": ['Tendencia', 'MM200', 'MM21'],
        "Pilar 2 - NIVEL":     ['Nivel S/R', 'Fibonacci', 'Bollinger'],
        "Pilar 3 - SENAL":     ['Velas', 'RSI', 'MACD', 'SPY', 'VIX'],
    }
    p1, p2, p3 = st.columns(3)
    for columna, (titulo, keys) in zip([p1, p2, p3], pilares.items()):
        with columna:
            st.markdown("**" + titulo + "**")
            for k in keys:
                v = res['votos'].get(k, 0)
                icon = "✅✅" if v >= 2 else "✅" if v == 1 else "❌❌" if v <= -2 else "❌" if v == -1 else "⚪"
                st.write(icon + "  " + k)

    # Similitudes
    st.subheader("Similitudes Historicas")
    similares = similitud_historica(df['Close'].values, ventana=10)
    if similares:
        rows = []
        for s in similares:
            idx = s['idx']
            if idx < len(df):
                fecha = str(df.index[idx])[:10]
                valor = round(float(df['Close'].iloc[idx]), 2)
                var   = s['var_pct']
                rows.append({
                    "Fecha": fecha,
                    "Precio": "$" + str(valor),
                    "Var. siguiente dia": ("+" if var > 0 else "") + str(var) + "%",
                    "Senal": "Subio" if var > 0 else "Bajo"
                })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.write("No hay suficiente historial.")

    st.caption("pimpoSoluciones.")

with st.expander("Como abrirla desde el celular"):
    st.markdown("""
    1. Conecta el celu a la misma WiFi que tu PC
    2. En PowerShell corre: `python -m streamlit run bursatilgo.py --server.address=0.0.0.0`
    3. En Windows: abri cmd y escribi `ipconfig`, buscas "Direccion IPv4" (algo como 192.168.1.15)
    4. En el celu abri el navegador y escribi: `http://192.168.1.15:8501`
    """)