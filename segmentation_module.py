"""
Módulo de segmentación de la lesión para "Cuidado Inteligente del Pie Diabético v3.0".

Segmenta la IMAGEN COMPLETA con SegFormer-B2 (primera etapa de la Estrategia C), superpone la
máscara sobre la imagen y calcula el área de la lesión. No modifica el flujo de clasificación
de la versión 2.0 (detección con YOLOv8 + ViT).

Uso desde app.py (ver las instrucciones de integración): 
    from segmentation_module import render_segmentation_panel
    render_segmentation_panel(source_img)      # source_img: archivo subido por st.file_uploader

Requisitos: conexión a Hugging Face la primera vez; los pesos (best_segformer-b2.pth del repo
daoliver/segformer-b2) se descargan y quedan en la caché local de huggingface_hub.
"""
import io
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

import settings
from ui_styles import section_title

IMAGE_SIZE = 256                                   # entrada del segmentador (igual que en el entrenamiento)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
RATIO_MIN_REGION = 0.05                            # se ignoran regiones < 5 % de la región mayor
MAX_LADO_VISUALIZACION = 1600                      # px; solo afecta a lo que se muestra, no al cálculo


# --------------------------------------------------------------------------
# Modelo
# --------------------------------------------------------------------------
def _weights_cached() -> bool:
    from huggingface_hub import try_to_load_from_cache
    path = try_to_load_from_cache(settings.SEGMENTATION_REPO, settings.SEGMENTATION_FILE)
    return isinstance(path, str)


@st.cache_resource(show_spinner=False)
def prefetch_segmenter_weights():
    """Inicia en segundo plano la descarga de los pesos (solo la primera vez) para no bloquear la interfaz."""
    import threading
    from huggingface_hub import hf_hub_download

    def _download():
        try:
            hf_hub_download(repo_id=settings.SEGMENTATION_REPO, filename=settings.SEGMENTATION_FILE)
        except Exception:  # noqa: BLE001 - el error se muestra al segmentar
            pass

    if not _weights_cached():
        threading.Thread(target=_download, daemon=True).start()
    return True


@st.cache_resource(show_spinner="Cargando el segmentador SegFormer-B2...")
def load_segmenter():
    """Construye SegFormer-B2 (1 clase) y carga los pesos entrenados desde Hugging Face."""
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import SegformerConfig, SegformerForSemanticSegmentation

    try:
        weights_path = hf_hub_download(repo_id=settings.SEGMENTATION_REPO, filename=settings.SEGMENTATION_FILE)
    except Exception as ex:  # noqa: BLE001
        raise RuntimeError(
            f"No se pudo descargar «{settings.SEGMENTATION_FILE}» de Hugging Face ({settings.SEGMENTATION_REPO}). "
            "Compruebe que el repositorio es público y el nombre es correcto; si es privado, defina la variable "
            f"de entorno HF_TOKEN con un token de lectura. Detalle: {type(ex).__name__}"
        ) from ex

    config = SegformerConfig(
        num_labels=1, num_channels=3, num_encoder_blocks=4,
        depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], hidden_sizes=[64, 128, 320, 512],
        patch_sizes=[7, 3, 3, 3], strides=[4, 2, 2, 2], num_attention_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4], decoder_hidden_size=768,
    )
    model = SegformerForSemanticSegmentation(config)

    try:
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    except TypeError:  # versiones antiguas de torch
        checkpoint = torch.load(weights_path, map_location="cpu")
    state = checkpoint
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                state = checkpoint[key]
                break
    state = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in state.items()}

    result = model.load_state_dict(state, strict=False)
    faltan = [k for k in result.missing_keys if not k.endswith("num_batches_tracked")]
    if faltan or result.unexpected_keys:
        raise RuntimeError(
            "Los pesos de segmentación no coinciden con SegFormer-B2 (1 clase). "
            f"Faltan {len(faltan)} claves y sobran {len(result.unexpected_keys)}."
        )
    return model.eval()


def _load_rgb(image_bytes: bytes) -> Image.Image:
    """Abre la imagen respetando la orientación EXIF, para segmentar lo mismo que ve el usuario."""
    return ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")


def segment_image(model, pil_rgb: Image.Image, threshold: float = 0.5) -> np.ndarray:
    """Segmenta la imagen completa. Devuelve una máscara binaria (0/1) del tamaño de la imagen original."""
    import torch
    import torch.nn.functional as F

    arr = np.array(pil_rgb)
    alto, ancho = arr.shape[:2]
    redim = cv2.resize(arr, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    x = (redim.astype(np.float32) / 255.0 - MEAN) / STD
    x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0)
    with torch.no_grad():
        logits = model(pixel_values=x).logits                      # (1, 1, H/4, W/4)
        logits = F.interpolate(logits, size=(alto, ancho), mode="bilinear", align_corners=False)
        prob = torch.sigmoid(logits)[0, 0].numpy()
    return (prob > threshold).astype(np.uint8)


@st.cache_data(show_spinner=False)
def segment_cached(image_bytes: bytes, threshold: float) -> np.ndarray:
    """Segmentación con caché: no se repite al interactuar con la interfaz."""
    model = load_segmenter()
    return segment_image(model, _load_rgb(image_bytes), threshold)


# --------------------------------------------------------------------------
# Postprocesamiento, visualización y área
# --------------------------------------------------------------------------
def clean_mask(mask: np.ndarray, ratio_min: float = RATIO_MIN_REGION):
    """Conserva las regiones con área >= ratio_min * área de la región mayor (descarta falsos positivos
    diminutos). Devuelve la máscara limpia y la lista de áreas (px) de las regiones conservadas."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros_like(mask, dtype=np.uint8), []
    areas = stats[1:, cv2.CC_STAT_AREA]
    mayor = int(areas.max())
    keep = [i + 1 for i, a in enumerate(areas) if a >= ratio_min * mayor]
    limpia = np.isin(lab, keep).astype(np.uint8)
    return limpia, sorted((int(stats[i, cv2.CC_STAT_AREA]) for i in keep), reverse=True)


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color=(255, 0, 0), alpha: float = 0.35) -> np.ndarray:
    """Superpone la máscara sobre la imagen: relleno semitransparente y contorno."""
    out = rgb.copy()
    relleno = np.zeros_like(rgb)
    relleno[mask > 0] = color
    sel = mask > 0
    out[sel] = (rgb[sel].astype(np.float32) * (1 - alpha) + relleno[sel].astype(np.float32) * alpha).astype(np.uint8)
    contornos, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    grosor = max(2, int(0.004 * max(rgb.shape[:2])))
    cv2.drawContours(out, contornos, -1, color, grosor)
    return out


def lesion_area(mask: np.ndarray, region_areas, px_per_cm: float = 0.0) -> dict:
    """Área de la lesión: en píxeles, como porcentaje de la imagen y, si se conoce la escala, en cm²."""
    alto, ancho = mask.shape[:2]
    pixeles = int(mask.sum())
    res = {
        "pixeles": pixeles,
        "porcentaje": 100.0 * pixeles / (alto * ancho),
        "regiones": len(region_areas),
        "areas_regiones_px": list(region_areas),
        "cm2": None,
    }
    if px_per_cm and px_per_cm > 0:
        res["cm2"] = pixeles / (px_per_cm ** 2)
    return res


def _para_mostrar(arr: np.ndarray) -> np.ndarray:
    alto, ancho = arr.shape[:2]
    escala = MAX_LADO_VISUALIZACION / max(alto, ancho)
    if escala >= 1:
        return arr
    return cv2.resize(arr, (int(ancho * escala), int(alto * escala)), interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------
# Interfaz de Streamlit
# --------------------------------------------------------------------------
REFERENCIAS_CM = {
    "Otro objeto (indicar su longitud)": None,
    "Tarjeta bancaria o de identidad (largo 8,56 cm)": 8.56,
    "Regla o cinta (10 cm)": 10.0,
}

EXPLICACION_ESCALA = """
**¿Por qué hace falta una escala?** Una foto no sabe qué tamaño tiene lo que muestra: el mismo pie
ocupa más o menos píxeles según la cámara y la distancia. Por eso, sin más datos, el área solo puede
darse como **porcentaje de la imagen**. Si en la foto aparece un objeto de medida conocida, la app
puede convertir píxeles en centímetros.

**La escala es una equivalencia:** *cuántos píxeles ocupa 1 cm en esta foto* (px/cm).

**Cómo obtenerla, paso a paso**
1. **Al fotografiar**, coloque junto a la lesión, en el mismo plano y a la misma distancia, un objeto
   de longitud conocida: regla, cinta métrica o una tarjeta bancaria (8,56 cm de largo).
2. **Aquí abajo**, elija *Medir sobre la imagen* y coloque los puntos A y B en los dos extremos de
   ese objeto. La app calcula cuántos píxeles hay entre ellos.
3. **Indique su longitud real** en cm. La app obtiene los px/cm y el área en cm².

**Ejemplo:** una regla de 5 cm mide 250 px en la foto → escala = 250 / 5 = **50 px/cm**.
Una lesión de 12 500 px² tiene un área de 12 500 / 50² = **5 cm²**.

⚠️ La foto debe tomarse **de frente** (cámara perpendicular al pie). Si se toma en ángulo, o el objeto
no está al mismo nivel que la lesión, el valor en cm² es solo aproximado.
"""


def _dibujar_medicion(base: np.ndarray, a: tuple, b: tuple) -> np.ndarray:
    out = base.copy()
    grosor = max(3, int(0.004 * max(base.shape[:2])))
    radio = grosor * 3
    cv2.line(out, a, b, (255, 214, 0), grosor)
    for punto, texto in ((a, "A"), (b, "B")):
        cv2.circle(out, punto, radio, (0, 120, 255), -1)
        cv2.putText(out, texto, (punto[0] + radio, punto[1] - radio), cv2.FONT_HERSHEY_SIMPLEX,
                    grosor / 2.2, (255, 255, 255), max(2, grosor // 2), cv2.LINE_AA)
    return out


def _escala_por_medicion(superpuesta: np.ndarray, nombre: str) -> float:
    """Permite marcar dos puntos sobre la imagen y devuelve los px/cm (0 si aún no es válido)."""
    alto, ancho = superpuesta.shape[:2]
    st.caption(f"Imagen original: {ancho} × {alto} px. Mueva los puntos hasta los extremos del objeto de referencia.")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Punto A**")
        ax = st.slider("A · horizontal (px)", 0, ancho - 1, int(0.15 * ancho), key=f"ax_{nombre}")
        ay = st.slider("A · vertical (px)", 0, alto - 1, int(0.5 * alto), key=f"ay_{nombre}")
    with col_b:
        st.markdown("**Punto B**")
        bx = st.slider("B · horizontal (px)", 0, ancho - 1, int(0.45 * ancho), key=f"bx_{nombre}")
        by = st.slider("B · vertical (px)", 0, alto - 1, int(0.5 * alto), key=f"by_{nombre}")

    st.image(_para_mostrar(_dibujar_medicion(superpuesta, (ax, ay), (bx, by))),
             caption="Línea de medición (amarilla) entre A y B", use_container_width=True)
    distancia = float(np.hypot(bx - ax, by - ay))
    st.markdown(f"Distancia entre A y B: **{distancia:.0f} px**")

    ref = st.selectbox("Objeto de referencia", list(REFERENCIAS_CM), key=f"ref_{nombre}")
    longitud_cm = REFERENCIAS_CM[ref]
    if longitud_cm is None:
        longitud_cm = st.number_input("Longitud real entre A y B (cm)", min_value=0.0, value=0.0, step=0.1,
                                      key=f"cm_{nombre}")
    else:
        st.caption(f"Se usará {longitud_cm:.2f} cm como longitud real entre A y B.")

    if distancia > 0 and longitud_cm > 0:
        escala = distancia / longitud_cm
        st.success(f"Escala calculada: {escala:.1f} px/cm")
        return escala
    st.info("Falta la longitud real (cm) o los puntos A y B coinciden.")
    return 0.0


def render_segmentation_panel(source_file, threshold: float = 0.5) -> None:
    """Muestra la imagen con la máscara superpuesta y el área de la lesión."""
    with st.container(border=True):
        col_title, col_loading = st.columns([3, 2], vertical_alignment="center")
        with col_title:
            section_title("Segmentación de la lesión", "🩺", "Máscara superpuesta y área estimada")

        try:
            data = source_file.getvalue()
            rgb = np.array(_load_rgb(data))
            texto = ("Segmentando la imagen..." if _weights_cached() else
                     "Descargando el modelo de segmentación (~105 MB, solo la primera vez)...")
            with col_loading, st.spinner(texto, show_time=True):
                mask = segment_cached(data, threshold)
        except Exception as ex:  # noqa: BLE001
            st.error("No se pudo segmentar la imagen.")
            st.error(ex)
            return

        limpia, areas = clean_mask(mask)
        if not areas:
            st.info("El segmentador no identificó ninguna lesión en esta imagen.")
            return

        superpuesta = overlay_mask(rgb, limpia)
        col_img, col_info = st.columns([3, 2], gap="large")
        with col_img:
            st.image(_para_mostrar(superpuesta), caption="Máscara de segmentación superpuesta",
                     use_container_width=True)
        # El botón de descarga va en la barra lateral, justo debajo de «Exportar».
        buf = io.BytesIO()
        Image.fromarray(superpuesta).save(buf, format="PNG")
        st.sidebar.download_button("🩺 Descargar máscara", data=buf.getvalue(),
                                   file_name=f"{Path(source_file.name).stem}_mascara.png", mime="image/png",
                                   help="Descarga la imagen con la máscara de segmentación superpuesta",
                                   use_container_width=True, key="download_mask")
        # Se reserva el sitio de las métricas; se rellenan tras conocer la escala (más abajo).
        metrics_box = col_info.container()

        with st.expander("📏 Obtener el área en cm² (opcional) — ¿qué es la escala y cómo se calcula?"):
            st.markdown(EXPLICACION_ESCALA)
            st.divider()
            modo = st.radio("¿Cómo quiere indicar la escala?",
                            ["Medir sobre la imagen", "Ya conozco los píxeles por cm"],
                            horizontal=True, key=f"modo_escala_{source_file.name}")
            if modo == "Medir sobre la imagen":
                px_per_cm = _escala_por_medicion(superpuesta, source_file.name)
            else:
                px_per_cm = st.number_input("Píxeles por cm (px/cm)", min_value=0.0, value=0.0, step=1.0,
                                            key=f"escala_{source_file.name}")

        area = lesion_area(limpia, areas, px_per_cm)
        with metrics_box:
            st.metric("Área de la lesión (% de la imagen)", f"{area['porcentaje']:.2f} %")
            st.metric("Área en píxeles", f"{area['pixeles']:,}".replace(",", " "))
            if area["cm2"] is not None:
                st.metric("Área estimada", f"{area['cm2']:.2f} cm²")
                st.caption(f"Calculada con una escala de {px_per_cm:.1f} px/cm.")
            else:
                st.info("Para ver el área en cm², abra «Obtener el área en cm²» (abajo) e indique la escala.")
            if area["regiones"] > 1:
                st.caption(f"Regiones detectadas: {area['regiones']} "
                           f"(áreas en px: {', '.join(str(a) for a in area['areas_regiones_px'])})")