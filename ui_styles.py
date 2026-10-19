"""Estilos y componentes visuales reutilizables de la plataforma.

Solo se usan selectores estables de Streamlit (data-testid); nunca clases
`st-emotion-cache-*`, que cambian en cada versión.
"""
import html

import streamlit as st

ACCENT = "#D32F2F"

_CSS = """
<style>
/* ---------- Estructura general ---------- */
[data-testid="stMainBlockContainer"] {
    max-width: 1240px;
    padding: 4rem 2.5rem 4rem 2.5rem;
}
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
    padding-top: 1.5rem;
}
h1, h2, h3 { letter-spacing: -0.01em; }

/* ---------- Encabezado ---------- */
.upd-hero {
    display: flex; align-items: center; gap: 1.1rem;
    padding: 1.1rem 1.4rem; margin-bottom: 1.4rem;
    border: 1px solid rgba(128,128,128,.25); border-left: 5px solid #D32F2F;
    border-radius: 14px;
    background: linear-gradient(90deg, rgba(211,47,47,.10), rgba(211,47,47,0));
}
.upd-hero img { width: 64px; height: 64px; flex: none; }
.upd-hero h1 { font-size: 1.7rem; line-height: 1.2; margin: 0; padding: 0; }
.upd-hero p { margin: .25rem 0 0 0; opacity: .75; font-size: .95rem; }

/* ---------- Encabezado de sección (idéntico en todas las tarjetas) ---------- */
.upd-section {
    display: flex; align-items: center; gap: .8rem; margin: 0 0 1rem 0;
}
.upd-section .upd-chip {
    width: 42px; height: 42px; flex: none; border-radius: 12px;
    display: flex; align-items: center; justify-content: center; font-size: 1.35rem;
    background: rgba(211,47,47,.12); border: 1px solid rgba(211,47,47,.25);
}
.upd-section .upd-t { font-weight: 700; font-size: 1.15rem; line-height: 1.2; }
.upd-section .upd-s { opacity: .7; font-size: .85rem; margin-top: .1rem; }

/* ---------- Tarjetas (st.container(border=True)) ---------- */
[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 14px;
}

/* ---------- Imágenes ---------- */
[data-testid="stImage"] { display: flex; justify-content: center; }
[data-testid="stImage"] img {
    max-height: 460px; width: auto; max-width: 100%;
    object-fit: contain; border-radius: 10px;
}

/* ---------- Etiquetas de resultado ---------- */
.upd-badge {
    display: inline-block; padding: .3rem .8rem; border-radius: 999px;
    font-weight: 600; font-size: .85rem; color: #fff; background: #D32F2F;
}
.upd-badge.ok { background: #2E7D32; }
.upd-caption { text-align: center; margin: .2rem 0 .6rem 0; }

/* ---------- Estados (sin cajas / modelo apagado) ---------- */
.upd-status {
    text-align: center; padding: 2rem 1rem;
}
.upd-status .upd-dot {
    width: 72px; height: 72px; border-radius: 50%; margin: 0 auto 1rem auto;
    display: flex; align-items: center; justify-content: center;
    font-size: 2.2rem; font-weight: 700; color: #fff;
}
.upd-status.ok .upd-dot { background: #2E7D32; box-shadow: 0 4px 12px rgba(46,125,50,.35); }
.upd-status.info .upd-dot { background: #1976D2; box-shadow: 0 4px 12px rgba(25,118,210,.35); }
.upd-status .upd-title { font-size: 1.1rem; font-weight: 700; }
.upd-status .upd-text { opacity: .75; margin-top: .3rem; font-size: .9rem; }

/* ---------- Pantalla de bienvenida ---------- */
.upd-steps { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: .5rem; }
.upd-step {
    border: 1px solid rgba(128,128,128,.25); border-radius: 14px; padding: 1.2rem;
}
.upd-step .upd-num {
    width: 34px; height: 34px; border-radius: 50%; background: #D32F2F; color: #fff;
    display: flex; align-items: center; justify-content: center; font-weight: 700; margin-bottom: .7rem;
}
.upd-step b { display: block; margin-bottom: .25rem; }
.upd-step span { opacity: .75; font-size: .9rem; }
@media (max-width: 900px) { .upd-steps { grid-template-columns: 1fr; } }

/* ---------- Cargador de archivos (Streamlit >= 1.4x) ----------
   Los textos nativos en inglés se ocultan con font-size:0 y se reemplazan
   mediante ::after; no depende de la jerarquía exacta de elementos. */
[data-testid="stFileUploaderDropzone"] {
    display: flex; flex-direction: column; align-items: stretch; gap: .7rem;
    padding: 1rem; border: 2px dashed rgba(211,47,47,.6); border-radius: 14px;
    background: rgba(211,47,47,.05); text-align: center;
}
[data-testid="stFileUploaderDropzone"]:hover { background: rgba(211,47,47,.10); }
[data-testid="stFileUploaderDropzone"] button { width: 100%; justify-content: center; }
[data-testid="stFileUploaderDropzone"] button p { font-size: 0; }
[data-testid="stFileUploaderDropzone"] button p::after {
    content: "Cargar imágenes"; font-size: 1rem;
}
[data-testid="stFileUploaderDropzoneInstructions"] { justify-content: center; }
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small { font-size: 0; }
[data-testid="stFileUploaderDropzoneInstructions"] span::after,
[data-testid="stFileUploaderDropzoneInstructions"] small::after {
    content: "Arrastra y suelta aquí · JPG, PNG · máx. 5 MB";
    font-size: .8rem; opacity: .75; display: block; white-space: normal; text-align: center;
}
[data-testid="stFileUploaderDropzoneInstructions"] > div { width: 100%; }
/* Si hay muchos archivos, la lista se desplaza en lugar de alargar la barra lateral */
[data-testid="stFileUploader"] > div:last-child { max-height: 240px; overflow-y: auto; }

/* Separación compacta entre bloques de la barra lateral */
[data-testid="stSidebar"] hr { margin: .2rem 0 !important; }
[data-testid="stSidebar"] [data-testid="stHeading"] h3 { padding-top: .25rem; }

/* ---------- Botones ---------- */
[data-testid="stSidebar"] .stButton button,
[data-testid="stSidebar"] .stDownloadButton button { font-weight: 600; }
[data-testid="stSidebar"] .stButton button[kind="primary"] {
    background-color: #D32F2F !important; border: 1px solid #D32F2F !important;
    box-shadow: 0 1px 3px rgba(211,47,47,.35);
}
[data-testid="stSidebar"] .stButton button[kind="primary"]:hover { background-color: #B71C1C !important; border-color: #B71C1C !important; }
[data-testid="stApp"] .st-key-delete_button button { border: 1.5px solid #D32F2F !important; }
[data-testid="stApp"] .st-key-delete_button button p { color: #D32F2F !important; }
[data-testid="stApp"] .st-key-delete_button button:hover { background-color: rgba(211,47,47,.12) !important; }
/* Cabecera de la barra lateral compacta: acerca el título a la flecha de contraer */
[data-testid="stSidebarHeader"] {
    position: absolute; top: 0; right: 0; width: auto; height: auto; min-height: 0;
    padding: .4rem .5rem; margin: 0; z-index: 2;
}
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding-top: 1.6rem !important; }
/* Botón de tema en la cabecera */
.st-key-theme_toggle button {
    width: 2.8rem; height: 2.8rem; min-height: 0; padding: 0; border-radius: 50%;
    font-size: 1.3rem; margin-left: auto; display: flex;
}
.st-key-theme_toggle { display: flex; justify-content: flex-end; }
/* Colores que nunca deben cambiar con el tema */
.upd-badge, .upd-dot, .upd-num { color: #fff !important; }
button[kind="primary"] p { color: #fff !important; }
</style>
"""

# Paletas para forzar claro u oscuro con independencia del tema del sistema/navegador.
_PALETTES = {
    "dark": dict(bg="#0E1117", side="#171A23", text="#FAFAFA", field="#1E222D",
                 border="rgba(250,250,250,.22)", scheme="dark"),
    "light": dict(bg="#FFFFFF", side="#F0F2F6", text="#31333F", field="#FFFFFF",
                  border="rgba(49,51,63,.35)", scheme="light"),
}

_THEME_TEMPLATE = """
<style>
html { color-scheme: %(scheme)s; }
[data-testid="stApp"], [data-testid="stHeader"] { background-color: %(bg)s !important; }
[data-testid="stSidebar"], [data-testid="stSidebar"] > div { background-color: %(side)s !important; }
[data-testid="stApp"] { color: %(text)s; }
[data-testid="stApp"] :is(p, span, label, li, small, h1, h2, h3, h4, h5, h6, legend, summary, .upd-t, .upd-s, .upd-title, .upd-text, .upd-step b, .upd-step span, .upd-hero h1, .upd-hero p) { color: %(text)s; }
[data-testid="stApp"] :is(input, textarea) { color: %(text)s !important; background-color: %(field)s !important; }
[data-testid="stApp"] :is([data-baseweb="select"] > div, [data-baseweb="input"], [data-baseweb="base-input"]) {
    background-color: %(field)s !important; border-color: %(border)s !important;
}
[data-testid="stApp"] [data-baseweb="select"] :is(div, span, input) { color: %(text)s !important; }
[data-testid="stApp"] [data-baseweb="select"] svg { fill: %(text)s !important; }
[data-testid="stApp"] :is([data-testid="stSelectbox"], [data-testid="stMultiSelect"]) div:has(> input) {
    background-color: %(field)s !important; border-color: %(border)s !important;
}
[data-testid="stApp"] :is([data-testid="stSelectbox"], [data-testid="stMultiSelect"]) :is(div, span, input) { color: %(text)s !important; }
[data-testid="stApp"] :is([data-testid="stSelectbox"], [data-testid="stMultiSelect"]) svg { fill: %(text)s !important; color: %(text)s !important; }
[data-testid="stSelectboxVirtualDropdown"], [data-testid="stSelectboxVirtualDropdown"] * { background-color: %(field)s !important; color: %(text)s !important; }
[data-testid="stSelectboxVirtualDropdown"] li:hover, [data-testid="stSelectboxVirtualDropdown"] [aria-selected="true"] { background-color: rgba(211,47,47,.25) !important; }
[data-baseweb="popover"], [data-baseweb="popover"] > div, [data-baseweb="menu"] { background-color: %(field)s !important; }
[data-baseweb="popover"] :is(ul, li, div[role="listbox"], [role="option"]) { background-color: %(field)s !important; color: %(text)s !important; }
[data-baseweb="popover"] li:hover, [data-baseweb="popover"] [aria-selected="true"] { background-color: rgba(211,47,47,.25) !important; }
/* Barra de herramientas de imágenes (botón de pantalla completa) */
/* Iconos de ayuda (?) y barra de herramientas de imágenes */
[data-testid="stApp"] [data-testid="stTooltipIcon"] :is(span, button) { background: transparent !important; box-shadow: none !important; }
[data-testid="stApp"] [data-testid="stTooltipIcon"] svg { stroke: %(text)s !important; fill: none !important; opacity: .75; }
[data-testid="stApp"] [data-testid="stTooltipIcon"] svg :is(circle, path, line) { fill: none !important; stroke: %(text)s !important; }
[data-testid="stApp"] [data-testid="stTooltipIcon"] button:hover svg { opacity: 1; }
[data-testid="stElementToolbar"], [data-testid="stElementToolbar"] > div,
[data-testid="stElementToolbarButtonContainer"] {
    background: transparent !important; border: none !important; box-shadow: none !important;
}
[data-testid="stElementToolbar"] :is(button, svg) { color: %(text)s !important; }
[data-testid="stElementToolbar"] svg path:not([fill="none"]) { fill: %(text)s !important; stroke: none !important; }
[data-testid="stElementToolbar"] svg path[fill="none"] { fill: none !important; }
[data-testid="stElementToolbar"] button {
    background-color: %(field)s !important; border: 1px solid %(border)s !important;
    border-radius: 8px; box-shadow: none !important;
}
[data-testid="stElementToolbar"] button:hover { background-color: rgba(211,47,47,.25) !important; }
/* Indicadores «Running…» / spinners (p. ej. load_models) */
[data-testid="stApp"] code { background-color: %(field)s !important; color: %(text)s !important; border: 1px solid %(border)s; }
[data-testid="stSpinner"], [data-testid="stSpinner"] > div {
    background: %(field)s !important; background-image: none !important; border-radius: 10px;
}
[data-testid="stSpinner"] * { color: %(text)s !important; }
[data-testid="stApp"] [data-testid="stExpander"] :is(details, summary) { background-color: transparent !important; }
[data-testid="stApp"] [data-testid="stBaseButton-secondary"],
[data-testid="stApp"] [data-testid="stBaseButton-secondaryFormSubmit"],
[data-testid="stApp"] [data-testid="stDownloadButton"] button {
    background-color: %(field)s !important; border-color: %(border)s !important;
}
[data-testid="stApp"] [data-testid="stExpander"] details,
[data-testid="stApp"] [data-testid="stVerticalBlockBorderWrapper"] { border-color: %(border)s !important; }
[data-testid="stApp"] [data-testid="stFileChip"] { background-color: %(field)s !important; border: 1px solid %(border)s; }
[data-testid="stApp"] [data-testid="stFileChip"] * { color: %(text)s !important; }
/* Miniatura: caja suave con el icono en rojo (antes quedaba oscuro sobre oscuro en modo claro) */
[data-testid="stApp"] [data-testid="stFileChip"] > div:first-child { background-color: rgba(211,47,47,.14) !important; }
[data-testid="stApp"] [data-testid="stFileChip"] > div:first-child :is(svg, path) { color: #D32F2F !important; }
[data-testid="stApp"] [data-testid="stFileChip"] > div:first-child path:not([fill="none"]) { fill: #D32F2F !important; }
[data-testid="stApp"] [data-testid="stFileChipDeleteBtn"] button { background-color: transparent !important; border-radius: 50%%; }
[data-testid="stApp"] [data-testid="stFileChipDeleteBtn"] button:hover { background-color: rgba(211,47,47,.18) !important; }
[data-testid="stApp"] [data-testid="stFileChipDeleteBtn"] button:hover :is(svg, path) { color: #D32F2F !important; fill: #D32F2F !important; }
</style>
"""


def theme_css(mode: str) -> str:
    palette = _PALETTES.get(mode)
    return _THEME_TEMPLATE % palette if palette else ""


def inject_styles(theme_mode: str = "auto") -> None:
    """Inyecta el CSS global y, si se fuerza, el tema claro u oscuro."""
    st.markdown(_CSS + theme_css(theme_mode), unsafe_allow_html=True)


def current_theme() -> str:
    """Tema activo: el elegido con el botón o, al inicio, el del sistema/navegador."""
    if "theme_mode" not in st.session_state:
        try:
            detected = st.context.theme.type
        except Exception:
            detected = None
        st.session_state["theme_mode"] = "dark" if detected == "dark" else "light"
    return st.session_state["theme_mode"]


def _toggle_theme() -> None:
    st.session_state["theme_mode"] = "light" if st.session_state["theme_mode"] == "dark" else "dark"


def theme_toggle() -> None:
    """Botón único luna/sol para la cabecera."""
    dark = current_theme() == "dark"
    st.button(
        "☀️" if dark else "🌙", key="theme_toggle", on_click=_toggle_theme,
        help="Cambiar a modo claro" if dark else "Cambiar a modo oscuro",
    )


def render_header(logo_b64: str, title: str, subtitle: str) -> None:
    logo = f'<img src="data:image/png;base64,{logo_b64}" alt="">' if logo_b64 else ""
    col_header, col_theme = st.columns([14, 1], vertical_alignment="center")
    with col_header:
        st.markdown(
            f'<div class="upd-hero">{logo}<div><h1>{html.escape(title)}</h1>'
            f"<p>{html.escape(subtitle)}</p></div></div>",
            unsafe_allow_html=True,
        )
    with col_theme:
        theme_toggle()


def section_title(title: str, icon: str = "", subtitle: str = "") -> None:
    chip = f'<div class="upd-chip">{html.escape(icon)}</div>' if icon else ""
    sub = f'<div class="upd-s">{html.escape(subtitle)}</div>' if subtitle else ""
    st.markdown(
        f'<div class="upd-section">{chip}<div><div class="upd-t">{html.escape(title)}</div>{sub}</div></div>',
        unsafe_allow_html=True,
    )


def badge(text: str, ok: bool = False) -> str:
    cls = "upd-badge ok" if ok else "upd-badge"
    return f'<div class="upd-caption"><span class="{cls}">{html.escape(text)}</span></div>'


def status_card(kind: str, title: str, text: str) -> None:
    """kind: 'ok' (verde, ✓) o 'info' (azul, i)."""
    symbol = "✓" if kind == "ok" else "i"
    st.markdown(
        f'<div class="upd-status {kind}"><div class="upd-dot">{symbol}</div>'
        f'<div class="upd-title">{html.escape(title)}</div>'
        f'<div class="upd-text">{html.escape(text)}</div></div>',
        unsafe_allow_html=True,
    )


def render_welcome() -> None:
    st.markdown(
        """
        <div class="upd-steps">
            <div class="upd-step"><div class="upd-num">1</div><b>Carga las imágenes</b>
                <span>Usa el cargador de la barra lateral (hasta 20 imágenes JPG o PNG).</span></div>
            <div class="upd-step"><div class="upd-num">2</div><b>Analiza</b>
                <span>Pulsa «Analizar» para detectar, clasificar y segmentar la lesión.</span></div>
            <div class="upd-step"><div class="upd-num">3</div><b>Revisa y exporta</b>
                <span>Consulta los resultados, el área de la lesión y descarga el informe.</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
