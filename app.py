import os
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from definiciones import DARK, LIGHT

# ─── Configuración de página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Simulador de Correlación",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Tema: inicializar antes de cualquier widget ──────────────────────────────
if "tema_claro" not in st.session_state:
    st.session_state.tema_claro = False

C = LIGHT if st.session_state.tema_claro else DARK

PLOTLY_LAYOUT = dict(
    paper_bgcolor=C["paper"],
    plot_bgcolor=C["bg"],
    font=dict(color=C["font"], family="Courier New, monospace"),
    margin=dict(l=50, r=20, t=40, b=40),
    hovermode="x unified",
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=C["legend_border"]),
)
AXIS_STYLE = dict(gridcolor=C["grid"], zerolinecolor=C["grid"])

# ─── Estilo (badges) ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    .badge-ok {
        background: #10b98118;
        border: 1px solid #10b98144;
        border-radius: 8px;
        padding: 10px 14px;
        color: #10b981;
        font-family: 'Courier New', monospace;
    }
    .badge-warn {
        background: #f59e0b18;
        border: 1px solid #f59e0b44;
        border-radius: 8px;
        padding: 10px 14px;
        color: #f59e0b;
        font-family: 'Courier New', monospace;
    }
    .badge-danger {
        background: #ef444418;
        border: 1px solid #ef444444;
        border-radius: 8px;
        padding: 10px 14px;
        color: #ef4444;
        font-family: 'Courier New', monospace;
    }
</style>
""", unsafe_allow_html=True)

# ─── Helpers ───────────────────────────────────────────────────────────────────
def add_white_noise_to(signal, snr_db, seed):
    """Suma ruido blanco gaussiano a la señal con SNR especificado en dB.

    SNR_dB = 10·log10(P_señal / P_ruido)
    """
    rng = np.random.default_rng(seed)
    p_signal = float(np.mean(signal ** 2))
    if p_signal < 1e-12:
        p_signal = 1.0
    p_noise = p_signal / (10 ** (snr_db / 10))
    return signal + rng.normal(0.0, np.sqrt(p_noise), size=signal.shape)

def _overlap_count(N, M):
    """Número de muestras solapadas en cada lag de np.correlate(a, b, 'full')
    con len(a)=N, len(b)=M. En los extremos del eje de lag el solapamiento
    decae linealmente — esa es la fuente del envoltorio triangular en los
    bordes del resultado.
    """
    L = N + M - 1
    k = np.arange(L)
    return np.minimum.reduce([
        (k + 1).astype(float),
        np.full(L, M, dtype=float),
        np.full(L, N, dtype=float),
        (L - k).astype(float),
    ])

def cross_corr_overlap_mean(rx, ref):
    """Media de rx·ref en cada lag: np.correlate(full) / n_solape. Sin normalizar por RMS."""
    full = np.correlate(rx, ref, mode="full").astype(float)
    n_ov = np.maximum(_overlap_count(len(rx), len(ref)), 1.0)
    return full / n_ov


def cross_corr_normalized(rx, ref, kind="stationary"):
    """Correlación cruzada con compensación del recorte de integración.

    El integral teórico se evalúa sobre dominio infinito; en una observación
    finita el solape decae hacia los bordes y aparece la típica caída
    triangular. Se compensa eligiendo la normalización adecuada al tipo de
    señal:

      kind="stationary" → estimador insesgado (divide por el solape de cada
        lag). Para señales periódicas / estacionarias: senoidales, ruido,
        sumas, secuencias pseudoaleatorias.
      kind="finite" → estimador sesgado (divide por la energía de la
        referencia). Para señales de energía finita y soporte localizado
        (pulsos, ecos), donde el triángulo del solape es la respuesta
        correcta.
    """
    full = np.correlate(rx, ref, mode="full").astype(float)
    if kind == "stationary":
        n_ov = np.maximum(_overlap_count(len(rx), len(ref)), 1.0)
        unbiased = full / n_ov
        norm = np.sqrt(np.mean(rx ** 2) * np.mean(ref ** 2)) + 1e-12
        return unbiased / norm
    norm = float(np.sum(ref ** 2)) + 1e-12
    return full / norm

def autocorr_normalized(x, kind="stationary"):
    """Autocorrelación con compensación de bordes (ver cross_corr_normalized).
    Pico en lag 0 = 1 en ambas variantes.
    """
    full = np.correlate(x, x, mode="full").astype(float)
    if kind == "stationary":
        n_ov = np.maximum(_overlap_count(len(x), len(x)), 1.0)
        unbiased = full / n_ov
        return unbiased / (np.mean(x ** 2) + 1e-12)
    peak = float(np.max(np.abs(full))) + 1e-12
    return full / peak

def info_box(html, font_color=None):
    color = font_color or C["zone_label"]
    st.markdown(
        f"""
        <div style="font-family:'Courier New',monospace; font-size:12px; color:{color};
                    line-height:1.7; padding:12px; background:{C['info_bg']};
                    border:1px solid {C['info_border']}; border-radius:8px;">
        {html}
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    if os.path.exists("FaIn UNCo Logo 2.png"):
        st.image("FaIn UNCo Logo 2.png", width="stretch")
    st.markdown("## ⚙️ Parámetros")

    add_noise = st.toggle("🔊 Sumar ruido blanco a la señal recibida", value=True, key="ruido_on")
    snr_db = st.slider("SNR (dB)", -10, 30, 6, 1, disabled=not add_noise, key="snr_db")
    st.caption("_El ruido se aplica a las señales recibidas. El patrón de sincronismo siempre permanece limpio._")
    st.divider()

    st.toggle("☀️ Tema claro", key="tema_claro")
    st.divider()

    st.markdown(
        f"""
        <div style="font-family:'Courier New',monospace; font-size:12px; color:{C['zone_label']}; line-height:1.8">
        <b style="color:{C['font']}">📐 Notación</b><br>
        Correlación cruzada:<br>
        &nbsp;&nbsp;<span style="color:{C['filtered']}">R<sub>xy</sub>(τ) = ∫ x(t) y(t+τ) dt</span><br>
        Autocorrelación:<br>
        &nbsp;&nbsp;<span style="color:{C['filtered']}">R<sub>xx</sub>(τ) = ∫ x(t) x(t+τ) dt</span><br>
        SNR = P<sub>señal</sub> / P<sub>ruido</sub>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─── Cuerpo principal ─────────────────────────────────────────────────────────
st.markdown(
    f"<h1 style='text-align:center; font-family:sans-serif; "
    f"color:{C['signal']}; margin-bottom:2px'>"
    f"Simulador de Correlación</h1>"
    f"<p style='text-align:center; color:{C['zone_label']}; font-family:monospace; font-size:13px; margin-top:0'>"
    f"Correlación de señales · Sincronismo de trama</p>",
    unsafe_allow_html=True,
)

# ─── Habilitación de tabs ─────────────────────────────────────────────────────
# Por defecto solo se muestran "Correlación simple" y "Sincronismo de trama".
# Para volver a habilitar los tabs "Correlación cruzada" y "Autocorrelación",
# cambiar la siguiente línea a True:
MOSTRAR_TABS_AVANZADOS = False

if MOSTRAR_TABS_AVANZADOS:
    tab1, tab2, tab_simple, tab3 = st.tabs([
        "📈 Correlación cruzada",
        "🔄 Autocorrelación",
        "🧪 Correlación simple",
        "🎯 Sincronismo de trama",
    ])
else:
    tab_simple, tab3 = st.tabs([
        "🧪 Correlación",
        "🎯 Sincronismo de trama",
    ])

# ─── Tab 1: Correlación cruzada ───────────────────────────────────────────────
if MOSTRAR_TABS_AVANZADOS:
    with tab1:
        st.markdown("#### Correlación cruzada de dos señales")
        st.caption(
            "Elegí un caso didáctico. La _señal A_ es la referencia y la _señal B_ es la "
            "recibida (sobre la que se aplica el ruido si está activado)."
        )

        casos_xc = [
            "Pulso retardado (radar / ranging)",
            "Senoidales · misma frecuencia (con desfase)",
            "Senoidal vs cuadrada (misma frecuencia)",
            "Frecuencias distintas (caso ortogonal)",
            "Señal vs ruido (caso negativo)",
        ]
        caso = st.selectbox("Caso", casos_xc, key="caso_xcorr")

        # Eje de tiempo: se calcula sobre una ventana extendida (T_calc) y se
        # muestra solo la ventana visible (T_disp). El "ciclo extra de integración"
        # evita que la correlación caiga artificialmente en los bordes del eje
        # τ por falta de muestras solapadas.
        fs_int = 2000                         # Hz internos para la simulación
        T_disp = 1.0                          # ventana visible (s)
        T_calc = 3.0                          # ventana de cómputo (s) — 3× la visible
        t = np.arange(0, T_calc, 1.0 / fs_int)
        N = len(t)
        N_disp = int(T_disp * fs_int)         # muestras visibles
        t_disp = t[:N_disp]                   # eje visible para señales

        retardo_real_ms = None                # τ verdadero, si aplica
        es_caso_ruido = (caso == "Señal vs ruido (caso negativo)")
        # Tipo de señal para elegir el estimador (insesgado para estacionarias,
        # sesgado para señales de energía finita y soporte localizado).
        kind = "finite" if caso == "Pulso retardado (radar / ranging)" else "stationary"

        if caso == "Pulso retardado (radar / ranging)":
            c1, c2 = st.columns(2)
            delay_ms = c1.slider("Retardo del eco τ (ms)", 0, 800, 250, 10, key="delay_radar")
            ancho_ms = c2.slider("Ancho del pulso (ms)", 5, 200, 40, 5, key="ancho_radar")
            ancho_n = max(1, int(ancho_ms / 1000 * fs_int))
            delay_n = int(delay_ms / 1000 * fs_int)

            a = np.zeros(N)
            a[0:ancho_n] = 1.0
            b = np.zeros(N)
            end_b = min(N, delay_n + ancho_n)
            if delay_n < N:
                b[delay_n:end_b] = 1.0

            nombre_a, nombre_b = "Pulso emitido", "Eco recibido"
            retardo_real_ms = float(delay_ms)
            explicacion = (
                "La <b>correlación cruzada</b> entre el pulso emitido y el eco recibido "
                "presenta un pico en τ igual al retardo del eco. "
                "Es la base del radar y del LIDAR: medir el tiempo del pico permite "
                "calcular la distancia al blanco."
            )

        elif caso == "Senoidales · misma frecuencia (con desfase)":
            c1, c2 = st.columns(2)
            f0 = c1.slider("Frecuencia f₀ (Hz)", 1, 30, 5, 1, key="f_seno_xc")
            fase = c2.slider("Desfase φ (°)", -180, 180, 60, 5, key="fase_seno_xc")
            a = np.sin(2 * np.pi * f0 * t)
            b = np.sin(2 * np.pi * f0 * t + np.deg2rad(fase))
            nombre_a = "sin(2π·f₀·t)"
            nombre_b = f"sin(2π·f₀·t + {fase}°)"
            # τ que maximiza la correlación: B(t) = A(t + τ_real) ⇒ τ_real = -φ/(2π·f₀)
            retardo_real_ms = -np.deg2rad(fase) / (2 * np.pi * f0) * 1000.0
            explicacion = (
                "La correlación de dos senoidales de igual frecuencia es <b>cosenoidal</b> en τ "
                "y alcanza el máximo en el desfase temporal entre ambas. "
                "Permite estimar el desfase de un canal coherente."
            )

        elif caso == "Senoidal vs cuadrada (misma frecuencia)":
            f0 = st.slider("Frecuencia f₀ (Hz)", 1, 30, 5, 1, key="f_sq_xc")
            a = np.sin(2 * np.pi * f0 * t)
            b = np.sign(np.sin(2 * np.pi * f0 * t))
            nombre_a = "sin(2π·f₀·t)"
            nombre_b = "sgn(sin(2π·f₀·t))"
            retardo_real_ms = 0.0
            explicacion = (
                "Aunque las formas son distintas, ambas comparten la <b>fundamental</b>. "
                "La correlación es alta y máxima en τ=0 — la cuadrada se descompone en "
                "armónicos impares y solo la fundamental aporta al producto interno con la senoidal."
            )

        elif caso == "Frecuencias distintas (caso ortogonal)":
            c1, c2 = st.columns(2)
            f1 = c1.slider("Frecuencia A (Hz)", 1, 30, 5, 1, key="f1_orto_xc")
            f2 = c2.slider("Frecuencia B (Hz)", 1, 30, 8, 1, key="f2_orto_xc")
            a = np.sin(2 * np.pi * f1 * t)
            b = np.sin(2 * np.pi * f2 * t)
            nombre_a = f"sin(2π·{f1}·t)"
            nombre_b = f"sin(2π·{f2}·t)"
            explicacion = (
                "Sobre un intervalo entero de períodos comunes, las senoidales de "
                "<b>frecuencias distintas</b> son ortogonales: la correlación oscila alrededor de "
                "cero y no presenta un pico claro. Es el principio de la modulación FDM/OFDM."
            )

        else:  # Señal vs ruido
            f0 = st.slider("Frecuencia f₀ (Hz)", 1, 30, 5, 1, key="f_ruido_xc")
            a = np.sin(2 * np.pi * f0 * t)
            rng = np.random.default_rng(7)
            b = rng.normal(0, 1, size=N)
            nombre_a = "sin(2π·f₀·t)"
            nombre_b = "ruido blanco"
            explicacion = (
                "El ruido blanco es <b>estadísticamente independiente</b> de cualquier señal "
                "determinista. La correlación cruzada tiende a cero (con fluctuaciones por la "
                "longitud finita de la observación). Por eso la correlación se usa como filtro "
                "adaptado contra el ruido."
            )

        # Ruido en la señal recibida (B). En "señal vs ruido" no se vuelve a sumar.
        b_clean = b.copy()
        if add_noise and not es_caso_ruido:
            b = add_white_noise_to(b, snr_db, seed=13)

        # Correlación cruzada con compensación del recorte de integración.
        # Se computa sobre la ventana extendida (3× la visible) y se muestra
        # solo el rango |τ| ≤ T_disp, donde el solape sigue siendo amplio.
        r_full = cross_corr_normalized(b, a, kind=kind)
        lags_full = np.arange(-(N - 1), N)
        tau_full_ms = lags_full * 1000.0 / fs_int
        mask = np.abs(lags_full) <= N_disp
        tau_ms = tau_full_ms[mask]
        r_xc = r_full[mask]

        # Argmax "inteligente" para señales periódicas: la correlación insesgada
        # equilibra la altura de los picos secundarios, por lo que un argmax global
        # puede caer en cualquiera de ellos. Si conocemos τ esperado, buscamos el
        # pico principal en una ventana acotada alrededor de ese valor.
        if kind == "stationary" and retardo_real_ms is not None:
            i_target = int(np.argmin(np.abs(tau_ms - retardo_real_ms)))
            half_w = max(20, len(tau_ms) // 30)
            lo, hi = max(0, i_target - half_w), min(len(tau_ms), i_target + half_w + 1)
            i_max = lo + int(np.argmax(r_xc[lo:hi]))
        else:
            i_max = int(np.argmax(r_xc))
        tau_pico_ms = float(tau_ms[i_max])
        nivel_pico = float(r_xc[i_max])

        # ─── Gráfico ───
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                f"Señal A (referencia): {nombre_a}",
                f"Señal B (recibida): {nombre_b}" + ("  ·  con ruido" if (add_noise and not es_caso_ruido) else "  ·  sin ruido"),
                "Correlación cruzada R<sub>BA</sub>(τ)",
            ),
            vertical_spacing=0.11,
        )
        fig.add_trace(
            go.Scatter(x=t_disp * 1000, y=a[:N_disp],
                       line=dict(color=C["signal"], width=2), showlegend=False),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=t_disp * 1000, y=b[:N_disp],
                       line=dict(color=C["received"], width=1.5), showlegend=False),
            row=2, col=1,
        )
        fig.add_trace(
            go.Scatter(x=tau_ms, y=r_xc, line=dict(color=C["filtered"], width=2), showlegend=False),
            row=3, col=1,
        )
        fig.add_vline(
            x=tau_pico_ms, line_dash="dash", line_color=C["peak"], line_width=1.5,
            annotation_text=f"τ* = {tau_pico_ms:.1f} ms<br>R = {nivel_pico:.2f}",
            annotation_font_color=C["peak"], annotation_position="top right",
            row=3, col=1,
        )
        if retardo_real_ms is not None:
            fig.add_vline(
                x=retardo_real_ms, line_dash="dot", line_color=C["real"], line_width=1.2,
                annotation_text=f"τ real = {retardo_real_ms:.1f} ms",
                annotation_font_color=C["real"], annotation_position="bottom right",
                row=3, col=1,
            )

        fig.update_layout(**PLOTLY_LAYOUT, height=620)
        fig.update_xaxes(title_text="Tiempo (ms)", row=1, col=1, **AXIS_STYLE)
        fig.update_yaxes(title_text="Amplitud", row=1, col=1, **AXIS_STYLE)
        fig.update_xaxes(title_text="Tiempo (ms)", row=2, col=1, **AXIS_STYLE)
        fig.update_yaxes(title_text="Amplitud", row=2, col=1, **AXIS_STYLE)
        fig.update_xaxes(title_text="Lag τ (ms)", row=3, col=1, **AXIS_STYLE)
        fig.update_yaxes(title_text="R(τ)", row=3, col=1, **AXIS_STYLE)
        st.plotly_chart(fig, width="stretch", key="chart_xcorr")

        c1, c2, c3 = st.columns(3)
        c1.metric("Pico de correlación", f"{nivel_pico:.3f}")
        c2.metric("τ del pico", f"{tau_pico_ms:.2f} ms")
        if retardo_real_ms is not None:
            c3.metric("τ real", f"{retardo_real_ms:.2f} ms",
                      delta=f"{tau_pico_ms - retardo_real_ms:+.2f} ms")
        else:
            c3.metric("τ real", "—")

        info_box(explicacion)

# ─── Tab 2: Autocorrelación ───────────────────────────────────────────────────
if MOSTRAR_TABS_AVANZADOS:
    with tab2:
        st.markdown("#### Autocorrelación de una señal")
        st.caption(
            "La autocorrelación R<sub>xx</sub>(τ) mide la similitud de una señal "
            "consigo misma desplazada en τ. Revela periodicidades y propiedades "
            "estadísticas — y es la base del diseño de buenos códigos de sincronismo.",
            unsafe_allow_html=True,
        )

        casos_ac = [
            "Senoidal pura",
            "Ruido blanco",
            "Pulso rectangular",
            "Senoidal sumergida en ruido",
            "Secuencia pseudoaleatoria (PRBS)",
        ]
        caso2 = st.selectbox("Tipo de señal", casos_ac, key="caso_acorr")

        # Eje extendido para compensar el recorte de integración (ver Tab 1).
        fs_int2 = 2000
        T_disp2 = 1.0                          # ventana visible (s)
        T_calc2 = 3.0                          # ventana de cómputo (s) — 3× la visible
        t2 = np.arange(0, T_calc2, 1.0 / fs_int2)
        N2 = len(t2)
        N_disp2 = int(T_disp2 * fs_int2)
        t2_disp = t2[:N_disp2]

        es_caso_ruido2 = (caso2 == "Ruido blanco")
        kind2 = "finite" if caso2 == "Pulso rectangular" else "stationary"

        if caso2 == "Senoidal pura":
            f0 = st.slider("Frecuencia (Hz)", 1, 30, 5, 1, key="f_seno_ac")
            x = np.sin(2 * np.pi * f0 * t2)
            explicacion2 = (
                "La autocorrelación de una senoidal de frecuencia f₀ es "
                "<b>otra cosenoidal</b> de la misma frecuencia: "
                "R<sub>xx</sub>(τ) = ½·cos(2π·f₀·τ). "
                "La señal mantiene memoria infinita de sí misma."
            )

        elif caso2 == "Ruido blanco":
            rng = np.random.default_rng(7)
            x = rng.normal(0, 1, size=N2)
            explicacion2 = (
                "El ruido blanco ideal tiene autocorrelación tipo <b>delta de Dirac</b> "
                "en τ=0: muestras en instantes distintos son estadísticamente independientes. "
                "Para una observación finita aparecen pequeñas fluctuaciones laterales."
            )

        elif caso2 == "Pulso rectangular":
            ancho_ms = st.slider("Ancho del pulso (ms)", 10, 500, 100, 10, key="pulso_ac")
            ancho_n = max(1, int(ancho_ms / 1000 * fs_int2))
            x = np.zeros(N2)
            # Centrar el pulso en la ventana visible (no en la extendida).
            start = max(0, (N_disp2 - ancho_n) // 2)
            x[start:start + ancho_n] = 1.0
            explicacion2 = (
                "La autocorrelación de un pulso rectangular de ancho T es un "
                "<b>triángulo</b> de base 2T y vértice en τ=0. "
                "El ancho de la autocorrelación es proporcional a la duración del pulso."
            )

        elif caso2 == "Senoidal sumergida en ruido":
            f0 = st.slider("Frecuencia oculta f₀ (Hz)", 1, 30, 5, 1, key="f_ocul_ac")
            amp = st.slider("Amplitud de la senoidal (rel. al ruido)", 0.05, 1.0, 0.3, 0.05, key="amp_ocul_ac")
            rng = np.random.default_rng(11)
            x = amp * np.sin(2 * np.pi * f0 * t2) + rng.normal(0, 1, size=N2)
            explicacion2 = (
                "Aunque en el dominio temporal la senoidal está oculta por el ruido, "
                "la <b>autocorrelación promedia</b> el ruido (que tiende a cero fuera de τ=0) "
                "y deja visible la componente periódica. "
                "Es uno de los métodos más potentes para detectar señales en ruido."
            )

        else:  # PRBS
            bits_per_sym = 32
            nbits = N2 // bits_per_sym
            rng = np.random.default_rng(3)
            bits = rng.integers(0, 2, size=nbits)
            x = np.repeat(np.where(bits == 1, 1.0, -1.0), bits_per_sym)
            if len(x) < N2:
                x = np.concatenate([x, np.zeros(N2 - len(x))])
            else:
                x = x[:N2]
            explicacion2 = (
                "Una secuencia pseudoaleatoria (PRBS / m-sequence) tiene autocorrelación con "
                "un <b>pico estrecho en τ=0</b> y lóbulos laterales muy bajos. "
                "Es la propiedad clave para diseñar códigos de sincronismo, CDMA y radar de "
                "compresión de pulso."
            )

        # Aplicar ruido (excepto cuando la señal ya es ruido)
        x_proc = x.copy()
        if add_noise and not es_caso_ruido2:
            x_proc = add_white_noise_to(x, snr_db, seed=99)

        # Autocorrelación con compensación de bordes y recorte al rango visible.
        rxx_full = autocorr_normalized(x_proc, kind=kind2)
        lags2_full = np.arange(-(N2 - 1), N2)
        tau_ms2_full = lags2_full * 1000.0 / fs_int2
        mask2 = np.abs(lags2_full) <= N_disp2
        tau_ms2 = tau_ms2_full[mask2]
        rxx = rxx_full[mask2]

        fig2 = make_subplots(
            rows=2, cols=1,
            subplot_titles=(
                f"Señal x(t): {caso2}" + ("  ·  con ruido sumado" if (add_noise and not es_caso_ruido2) else ""),
                "Autocorrelación normalizada R<sub>xx</sub>(τ)",
            ),
            vertical_spacing=0.18,
            row_heights=[0.4, 0.6],
        )
        fig2.add_trace(
            go.Scatter(x=t2_disp * 1000, y=x_proc[:N_disp2],
                       line=dict(color=C["signal"], width=1.2), showlegend=False),
            row=1, col=1,
        )
        fig2.add_trace(
            go.Scatter(x=tau_ms2, y=rxx, line=dict(color=C["filtered"], width=2), showlegend=False),
            row=2, col=1,
        )
        fig2.add_vline(x=0, line_dash="dot", line_color=C["peak"], line_width=1.0, row=2, col=1)

        fig2.update_layout(**PLOTLY_LAYOUT, height=520)
        fig2.update_xaxes(title_text="Tiempo (ms)", row=1, col=1, **AXIS_STYLE)
        fig2.update_yaxes(title_text="Amplitud", row=1, col=1, **AXIS_STYLE)
        fig2.update_xaxes(title_text="Lag τ (ms)", row=2, col=1, **AXIS_STYLE)
        fig2.update_yaxes(title_text="R(τ) / R(0)", row=2, col=1, **AXIS_STYLE)
        st.plotly_chart(fig2, width="stretch", key="chart_acorr")

        info_box(explicacion2)

# ─── Tab nuevo: Correlación simple ────────────────────────────────────────────
with tab_simple:
    st.markdown("#### Correlación de dos señales (versión didáctica)")
    st.caption(
        "Elegí dos señales con los menús desplegables y observá su correlación cruzada. "
        "Activando _autocorrelación_ se inhabilita la Señal B y se correlaciona la "
        "Señal A consigo misma. La correlación se calcula **sin normalizar** (solo media "
        "en el solape) sobre un tramo de tiempo **más largo** que el gráfico, para reducir "
        "el sesgo de bordes en los retardos mostrados."
    )

    TIPOS = [
        "Senoidal",
        "Tren de pulsos cuadrados",
        "Tren de pulsos triangulares",
        "Diente de sierra (medio ciclo)",
    ]

    auto_on = st.toggle(
        "🔁 Autocorrelación (correlaciona la Señal A consigo misma; deshabilita Señal B)",
        value=False, key="auto_simple",
    )

    cA, cB = st.columns(2)
    with cA:
        st.markdown("**Señal A**")
        tipo_a = st.selectbox("Tipo", TIPOS, index=0, key="tipo_a_simple")
        f_a = st.slider("Frecuencia (Hz)", 1, 30, 5, 1, key="f_a_simple")
        amp_a = st.slider("Amplitud", 0.1, 2.0, 1.0, 0.1, key="amp_a_simple")
    with cB:
        st.markdown("**Señal B**" + (" — _deshabilitada_" if auto_on else ""))
        tipo_b = st.selectbox("Tipo", TIPOS, index=1, key="tipo_b_simple", disabled=auto_on)
        f_b = st.slider("Frecuencia (Hz)", 1, 30, 5, 1, key="f_b_simple", disabled=auto_on)
        amp_b = st.slider("Amplitud", 0.1, 2.0, 1.0, 0.1, key="amp_b_simple", disabled=auto_on)

    # ─── Generación de las señales ─────────────────────────────────────────────
    # Solo se grafica T_disp en los paneles de señales; la correlación usa t en [0, T_calc)
    # con T_calc >> T_disp para integrar muchos períodos de la frecuencia más baja activa
    # y que, en el rango de τ mostrado, el solape sea amplio (menos error de extremos).
    fs = 2000                                  # frecuencia de muestreo (Hz)
    T_disp = 1.0                               # ventana visible (s)
    f_min_hz = max(min(float(f_a), float(f_b) if not auto_on else float(f_a)), 0.5)
    # Al menos ~48 períodos de la fundamental más lenta; nunca menos que 3× lo visible;
    # tope 12 s para no crecer demasiado en RAM/tiempo.
    T_calc = min(12.0, max(3.0 * T_disp, 48.0 / f_min_hz))
    t = np.arange(0, T_calc, 1.0 / fs)
    N = len(t)
    N_disp = int(T_disp * fs)
    t_vis = t[:N_disp]                         # eje visible

    def generar_senal(tipo, freq, amp, t):
        """Genera una de las señales pedidas con frecuencia y amplitud dadas."""
        if tipo == "Senoidal":
            return amp * np.sin(2 * np.pi * freq * t)
        if tipo == "Tren de pulsos cuadrados":
            return amp * np.sign(np.sin(2 * np.pi * freq * t))
        if tipo == "Tren de pulsos triangulares":
            # Onda triangular simétrica entre -amp y +amp. fase ∈ [0,1) recorre
            # un período; el valor sube de -1 a 1 y vuelve a -1. Se desplaza un
            # cuarto de período para que cruce por cero en t=0 (igual que sin).
            fase = (freq * t + 0.25) % 1.0
            return amp * (1.0 - 4.0 * np.abs(fase - 0.5))
        # Diente de sierra de medio ciclo: rampa lineal de 0 a amp durante la
        # primera mitad del período y cero durante la segunda mitad.
        fase = (freq * t) % 1.0
        return amp * np.where(fase < 0.5, 2.0 * fase, 0.0)

    señal_a = generar_senal(tipo_a, f_a, amp_a, t)
    if auto_on:
        # Autocorrelación: la "segunda señal" es la misma A.
        señal_b_limpia = señal_a.copy()
        nombre_b = f"= Señal A ({tipo_a})"
    else:
        señal_b_limpia = generar_senal(tipo_b, f_b, amp_b, t)
        nombre_b = f"{tipo_b}  ·  f={f_b} Hz  ·  A={amp_b}"

    # Sumar ruido a la señal recibida (B). La señal A queda siempre limpia.
    if add_noise:
        señal_b = add_white_noise_to(señal_b_limpia, snr_db, seed=21)
    else:
        señal_b = señal_b_limpia.copy()

    # ─── Cálculo de la correlación (solo esta pestaña: dimensional por solape) ──
    # np.correlate(full) y en cada τ se divide solo por el número de muestras
    # solapadas (media de b·a). No se divide por RMS → el eje y es acorde a las amplitudes.
    correl = cross_corr_overlap_mean(señal_b, señal_a)

    # Eje de lags en ms; recorte al semieje positivo 0 ≤ τ ≤ T_disp (amplio solape).
    lags = np.arange(-(N - 1), N)
    tau_ms = lags * 1000.0 / fs
    mascara = (lags >= 0) & (lags <= N_disp)
    tau_vis_ms = tau_ms[mascara]
    correl_vis = correl[mascara]

    # Pico de la correlación (en el rango visible).
    i_pico = int(np.argmax(correl_vis))
    tau_pico = float(tau_vis_ms[i_pico])
    nivel_pico = float(correl_vis[i_pico])

    # ─── Gráfico ───────────────────────────────────────────────────────────────
    titulo_a = f"Señal A · {tipo_a}  ·  f={f_a} Hz  ·  A={amp_a}"
    titulo_b = f"Señal B · {nombre_b}" + (
        f"  ·  con ruido (SNR={snr_db} dB)" if add_noise else "  ·  sin ruido"
    )
    titulo_corr = (
        "Autocorrelación R<sub>AA</sub>(τ)" if auto_on else "Correlación cruzada R<sub>BA</sub>(τ)"
    )

    fig_s = make_subplots(
        rows=3, cols=1,
        subplot_titles=(titulo_a, titulo_b, titulo_corr),
        vertical_spacing=0.11,
    )
    fig_s.add_trace(
        go.Scatter(x=t_vis * 1000, y=señal_a[:N_disp],
                   line=dict(color=C["signal"], width=2), showlegend=False),
        row=1, col=1,
    )
    fig_s.add_trace(
        go.Scatter(x=t_vis * 1000, y=señal_b[:N_disp],
                   line=dict(color=C["received"], width=1.5), showlegend=False),
        row=2, col=1,
    )
    fig_s.add_trace(
        go.Scatter(x=tau_vis_ms, y=correl_vis,
                   line=dict(color=C["filtered"], width=2), showlegend=False),
        row=3, col=1,
    )
    fig_s.add_vline(
        x=tau_pico, line_dash="dash", line_color=C["peak"], line_width=1.5,
        annotation_text=f"τ* = {tau_pico:.1f} ms<br>R = {nivel_pico:.2f}",
        annotation_font_color=C["peak"], annotation_position="top right",
        row=3, col=1,
    )

    fig_s.update_layout(**PLOTLY_LAYOUT, height=620)
    fig_s.update_xaxes(title_text="Tiempo (ms)", row=1, col=1, **AXIS_STYLE)
    fig_s.update_yaxes(title_text="Amplitud", row=1, col=1, **AXIS_STYLE)
    fig_s.update_xaxes(title_text="Tiempo (ms)", row=2, col=1, **AXIS_STYLE)
    fig_s.update_yaxes(title_text="Amplitud", row=2, col=1, **AXIS_STYLE)
    fig_s.update_xaxes(title_text="Lag τ (ms)", row=3, col=1, **AXIS_STYLE)
    fig_s.update_yaxes(title_text="Media b·a", row=3, col=1, **AXIS_STYLE)
    st.plotly_chart(fig_s, width="stretch", key="chart_simple")

    c1, c2 = st.columns(2)
    c1.metric("Pico de correlación", f"{nivel_pico:.3f}")
    c2.metric("τ del pico", f"{tau_pico:.2f} ms")

#    info_box(
#        "Correlación: <code>np.correlate</code> y división por muestras solapadas (media de b·a), "
#        "sin RMS. La ventana temporal de cómputo es más larga que la mostrada (muchos períodos "
#        "de la frecuencia más baja) para que en el τ visible el solape sea amplio."
#    )

# ─── Tab 3: Sincronismo de trama ──────────────────────────────────────────────
with tab3:
    st.markdown("#### Detección de sincronismo por correlación cruzada")
    st.caption(
        "Se busca un patrón conocido de 8 bits dentro de una trama recibida. "
        "El patrón se mantiene siempre limpio (referencia ideal) y la correlación "
        "presenta un pico en el instante en que el patrón comienza dentro de la trama — "
        "ese tiempo permite sincronizar el receptor."
    )

    # ─── Parámetros del patrón y la trama ──────────────────────────────────────
    c1, c2, c3 = st.columns([1.2, 1, 1])
    sync_str = c1.text_input(
        "🔑 Patrón de sincronismo (8 bits, 0 y 1)",
        value="11001011",
        max_chars=12,
        key="sync_str",
        help="Códigos sugeridos: 11001011, 01011001 (Barker-7 + 1 bit), 11110000, 10110100",
    )
    n_bits = c2.slider("Longitud de la trama (bits)", 16, 96, 48, 4, key="n_bits_trama")
    bitrate_mbps = c3.slider("Tasa de bits (Mbps)", 1, 50, 10, 1, key="bitrate_mbps")

    # Validación del patrón
    sync_str_clean = "".join(c for c in sync_str if c in "01")
    if len(sync_str_clean) != 8:
        st.markdown(
            '<div class="badge-warn"><strong>⚠ Patrón inválido</strong><br>'
            '<span style="font-size:11px">Debe tener exactamente 8 bits binarios. '
            'Se usa "11001011" por defecto.</span></div>',
            unsafe_allow_html=True,
        )
        sync_str_clean = "11001011"
    sync_bits = [int(b) for b in sync_str_clean]

    # Posición de inserción
    max_pos = max(0, n_bits - 8)
    pos_default = min(20, max_pos)
    c1, c2 = st.columns([1, 1])
    pos_inj = c1.slider(
        "📍 Posición donde se embebe el patrón en la trama (bit)",
        0, max_pos, pos_default, 1, key="pos_inj",
    )
    seed_trama = c2.number_input(
        "Semilla del relleno aleatorio", 0, 9999, 7, 1, key="seed_trama",
        help="Cambiar la semilla genera otros bits aleatorios alrededor del patrón.",
    )

    # ─── Construcción de las señales ───────────────────────────────────────────
    # Trama: bits aleatorios con el patrón embebido en pos_inj
    rng_t = np.random.default_rng(int(seed_trama))
    bits_trama = rng_t.integers(0, 2, size=n_bits).tolist()
    bits_trama[pos_inj:pos_inj + 8] = sync_bits

    # Conversión a NRZ ±1 con sobremuestreo por bit
    samples_per_bit = 32
    Tb = 1.0 / (bitrate_mbps * 1e6)             # duración de un bit (s)
    fs_sim = bitrate_mbps * 1e6 * samples_per_bit

    def bits_to_nrz(bits, sb=samples_per_bit):
        return np.repeat(np.where(np.asarray(bits) == 1, 1.0, -1.0), sb)

    trama = bits_to_nrz(bits_trama)             # señal enviada (limpia)
    sync_sig = bits_to_nrz(sync_bits)           # patrón de referencia (siempre limpio)
    t_trama = np.arange(len(trama)) / fs_sim
    t_sync = np.arange(len(sync_sig)) / fs_sim

    # Trama recibida: igual a la enviada + ruido (si está activado)
    if add_noise:
        trama_rx = add_white_noise_to(trama, snr_db, seed=42)
    else:
        trama_rx = trama.copy()

    # ─── Correlación cruzada ───────────────────────────────────────────────────
    # El patrón se mantiene limpio (referencia) y normalizamos por su energía,
    # de modo que un match perfecto dé R = 1. Se recorta el eje de lag al
    # rango válido: aquellas posiciones donde el patrón cabe ENTERO dentro de
    # la trama (lag ∈ [0, N_trama − N_patrón]). Más allá de ese rango el
    # solape es parcial y produciría la típica caída triangular de borde —
    # un artefacto, no información útil para el sincronismo.
    full = np.correlate(trama_rx, sync_sig, mode="full")
    norm = float(np.sum(sync_sig ** 2))         # = len(sync_sig) para NRZ ±1
    corr_full = full / (norm + 1e-12)
    lag0 = len(sync_sig) - 1
    n_valid = len(trama_rx) - len(sync_sig) + 1
    corr = corr_full[lag0:lag0 + n_valid]       # rango válido (sin tail tapered)
    t_corr = np.arange(len(corr)) / fs_sim

    i_max = int(np.argmax(corr))
    nivel_max = float(corr[i_max])
    tiempo_max = i_max / fs_sim                 # τ medido (s)
    bit_detectado = i_max // samples_per_bit
    tiempo_real = pos_inj * Tb                  # τ esperado (s)
    error_bits = int(round(bit_detectado - pos_inj))

    # Unidades de tiempo legibles según el bitrate
    if Tb >= 1e-3:
        t_unit, t_scale, t_fmt = "ms", 1e3, "{:.3f}"
    elif Tb >= 1e-6:
        t_unit, t_scale, t_fmt = "µs", 1e6, "{:.3f}"
    else:
        t_unit, t_scale, t_fmt = "ns", 1e9, "{:.0f}"

    # ─── Gráfico de 4 paneles ──────────────────────────────────────────────────
    fig3 = make_subplots(
        rows=4, cols=1,
        subplot_titles=(
            f"Trama enviada · {n_bits} bits @ {bitrate_mbps} Mbps  (patrón en bit {pos_inj}–{pos_inj + 7})",
            "Trama recibida" + (f"  ·  SNR = {snr_db} dB" if add_noise else "  ·  sin ruido"),
            f"Patrón de sincronismo (limpio): {sync_str_clean}",
            "Correlación cruzada — pico = retardo del sincronismo",
        ),
        vertical_spacing=0.09,
        row_heights=[0.22, 0.27, 0.18, 0.33],
    )

    # 1) Trama enviada (limpia) + zona resaltada del patrón
    fig3.add_vrect(
        x0=pos_inj * Tb * t_scale,
        x1=(pos_inj + 8) * Tb * t_scale,
        fillcolor=C["code"], opacity=0.18, line_width=0,
        annotation_text="patrón embebido",
        annotation_font_color=C["code"], annotation_font_size=10,
        annotation_position="top left",
        row=1, col=1,
    )
    fig3.add_trace(
        go.Scatter(x=t_trama * t_scale, y=trama, line=dict(color=C["signal"], width=1.5),
                   name="enviada", showlegend=False),
        row=1, col=1,
    )

    # 2) Trama recibida (con ruido)
    fig3.add_trace(
        go.Scatter(x=t_trama * t_scale, y=trama_rx, line=dict(color=C["received"], width=1.0),
                   name="recibida", showlegend=False),
        row=2, col=1,
    )

    # 3) Patrón de sincronismo limpio
    fig3.add_trace(
        go.Scatter(x=t_sync * t_scale, y=sync_sig, line=dict(color=C["code"], width=2.2),
                   name="patrón", showlegend=False),
        row=3, col=1,
    )

    # 4) Correlación cruzada
    fig3.add_trace(
        go.Scatter(x=t_corr * t_scale, y=corr, line=dict(color=C["filtered"], width=2),
                   name="R(τ)", showlegend=False),
        row=4, col=1,
    )
    fig3.add_vline(
        x=tiempo_max * t_scale, line_dash="dash", line_color=C["peak"], line_width=1.6,
        annotation_text=f"τ* = {t_fmt.format(tiempo_max * t_scale)} {t_unit}<br>R = {nivel_max:.2f}",
        annotation_font_color=C["peak"], annotation_position="top right",
        row=4, col=1,
    )
    fig3.add_vline(
        x=tiempo_real * t_scale, line_dash="dot", line_color=C["real"], line_width=1.2,
        annotation_text=f"τ real = {t_fmt.format(tiempo_real * t_scale)} {t_unit}",
        annotation_font_color=C["real"], annotation_position="bottom right",
        row=4, col=1,
    )
    fig3.add_hline(y=0, line_color=C["grid"], line_width=0.8, row=4, col=1)

    fig3.update_layout(**PLOTLY_LAYOUT, height=820)
    for r in (1, 2, 3, 4):
        fig3.update_xaxes(row=r, col=1, **AXIS_STYLE)
        fig3.update_yaxes(row=r, col=1, **AXIS_STYLE)
    fig3.update_xaxes(title_text=f"Tiempo ({t_unit})", row=2, col=1, **AXIS_STYLE)
    fig3.update_xaxes(title_text=f"Tiempo ({t_unit})", row=3, col=1, **AXIS_STYLE)
    fig3.update_xaxes(title_text=f"Lag τ ({t_unit})", row=4, col=1, **AXIS_STYLE)
    fig3.update_yaxes(range=[-1.6, 1.6], row=1, col=1, **AXIS_STYLE)
    fig3.update_yaxes(range=[-1.6, 1.6], row=3, col=1, **AXIS_STYLE)

    st.plotly_chart(fig3, width="stretch", key="chart_sincronismo")

    # ─── Resultado de la detección ─────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if error_bits == 0:
            st.markdown(
                f'<div class="badge-ok"><strong>✅ Sincronismo correcto</strong><br>'
                f'<span style="font-size:11px">Detectado en bit {bit_detectado}</span></div>',
                unsafe_allow_html=True,
            )
        elif abs(error_bits) == 1:
            st.markdown(
                f'<div class="badge-warn"><strong>⚠ Error de {error_bits:+d} bit</strong><br>'
                f'<span style="font-size:11px">Re-sincronización marginal</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="badge-danger"><strong>❌ Falla de sincronismo</strong><br>'
                f'<span style="font-size:11px">Error de {error_bits:+d} bits</span></div>',
                unsafe_allow_html=True,
            )
    c2.metric("τ medido (pico)", f"{t_fmt.format(tiempo_max * t_scale)} {t_unit}")
    c3.metric("τ real (esperado)", f"{t_fmt.format(tiempo_real * t_scale)} {t_unit}",
              delta=f"{(tiempo_max - tiempo_real) * t_scale:+.3f} {t_unit}")
    c4.metric("Pico de correlación", f"{nivel_max:.3f}", help="Valor 1.00 = coincidencia perfecta")

    # ─── Autocorrelación del propio patrón (calidad del código) ────────────────
    st.markdown("##### Autocorrelación del patrón elegido (calidad del código)")
    rcc = autocorr_normalized(sync_sig, kind="finite")
    lags_cc = np.arange(-(len(sync_sig) - 1), len(sync_sig))
    tau_cc = lags_cc / fs_sim * t_scale

    # Estimación de la relación pico / lóbulos secundarios
    half = len(sync_sig) - 1
    abs_rcc = np.abs(rcc)
    # excluir el pico central (3 muestras a cada lado para evitar la cima)
    mask = np.ones_like(abs_rcc, dtype=bool)
    mask[max(0, half - 2):half + 3] = False
    sidelobe = float(np.max(abs_rcc[mask])) if np.any(mask) else 0.0
    psr_db = 20 * np.log10(1.0 / max(sidelobe, 1e-6))

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=tau_cc, y=rcc, line=dict(color=C["code"], width=2),
                              name="R(τ) del patrón", showlegend=False))
    fig4.add_vline(x=0, line_dash="dot", line_color=C["peak"], line_width=1.0)
    fig4.update_layout(
        **PLOTLY_LAYOUT, height=260,
        title=dict(
            text=f"Pico/lóbulo lateral = {psr_db:.1f} dB  ·  un valor alto indica un buen código de sincronismo",
            font=dict(size=12),
        ),
    )
    fig4.update_xaxes(title_text=f"Lag τ ({t_unit})", **AXIS_STYLE)
    fig4.update_yaxes(title_text="R(τ)/R(0)", **AXIS_STYLE)
    st.plotly_chart(fig4, width="stretch", key="chart_code_acorr")

    info_box(
        "<b>¿Por qué funciona el sincronismo por correlación?</b><br>"
        "El patrón limpio actúa como <b>filtro adaptado</b> de la trama recibida. "
        "Cuando el patrón se alinea exactamente con su copia dentro de la trama, "
        "los productos punto a punto suman coherentemente y aparece un <b>pico</b> en R(τ). "
        "El ruido, en cambio, se suma incoherentemente y queda muy por debajo del pico — "
        "por eso la correlación tolera SNR bajos. "
        "El instante del pico indica el <b>retardo</b> con que el patrón llegó al receptor, "
        "información necesaria para alinear el reloj de bits y demodular correctamente."
    )

# ─── Pie ──────────────────────────────────────────────────────────────────────
st.markdown(
    f"<p style='text-align:center; color:{C['zone_label']}; font-size:11px; "
    f"font-family:monospace; margin-top:8px'>"
    f"Ingeniería Electrónica · Sistemas de Comunicaciones · Medios de Enlace</p>",
    unsafe_allow_html=True,
)
