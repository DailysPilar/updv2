import streamlit as st
from helper import load_pt_model, get_image_download_buffer, draw_bounding_boxes, crop_images
from keras.api.models import load_model as load_h5_model
from pathlib import Path
import numpy as np
import PIL
import settings
import zipfile
import io
from torch.nn.functional import interpolate
import csv
from typing import List, Dict, Any
from PIL import Image
import base64
import time
from tqdm import tqdm 
from streamlit.components.v1 import html as html_component
import torch  # Agregar esta importación
from transformers import ViTForImageClassification, ViTImageProcessor
import matplotlib.pyplot as plt

# Constantes globales de la aplicación
IOU_THRES = 0.5  # Umbral de IoU para la supresión de no máximos
CLASSES_NAME = ['both', 'infection', 'ischaemia', 'none']  # Nombres de clases en inglés
CLASSES_NAME_ES = ['INFECCIÓN E ISQUEMIA', 'INFECCIÓN', 'ISQUEMIA', 'SANO']  # Nombres de clases en español
# Decorador para cachear el modelo de detección y evitar recargas innecesarias
@st.cache_resource
def load_models():
    """
    Carga todos los modelos necesarios y los cachea para evitar recargas innecesarias.
    """
    try:
        det_model = load_pt_model(Path(settings.DETECTION_MODEL))
        model_path = 'weights/vit3con32'
        processor = ViTImageProcessor.from_pretrained(model_path)
        clf_model = ViTForImageClassification.from_pretrained(
            model_path,
            num_labels=len(CLASSES_NAME),
            ignore_mismatched_sizes=True,
            output_attentions=True
        )
        return det_model, processor, clf_model
    except Exception as ex:
        st.error("No se pudo cargar los modelos. Verifique las rutas especificadas")
        st.error(ex)
        return None, None, None

def initialize_session() -> None:
    """
    Inicializa todas las variables necesarias en el estado de la sesión de Streamlit.
    Esta función se ejecuta al inicio de la aplicación y cada vez que se limpia la sesión.
    """
    # Lista para almacenar las imágenes subidas por el usuario
    if 'uploaded_images' not in st.session_state:
        st.session_state.uploaded_images = []
    
    # Lista para almacenar las imágenes procesadas por el modelo
    if 'processed_images' not in st.session_state:
        st.session_state.processed_images = []
    
    # Valor de confianza inicial para el modelo de detección
    if 'confidence' not in st.session_state:
        st.session_state.confidence = 30
    
    # Clave única para el cargador de archivos, se incrementa para forzar su reinicialización
    if 'uploader_key' not in st.session_state:
        st.session_state.uploader_key = 0

    # Modo de procesamiento (detección o clasificación directa)
    if 'use_detection' not in st.session_state:
        st.session_state.use_detection = True

    # Inicializar el estado de show_attention si no existe
    if 'show_attention' not in st.session_state:
        st.session_state.show_attention = True

    # Diccionario para almacenar los resultados de clasificación por imagen
    if 'classification_cache' not in st.session_state:
        st.session_state.classification_cache = {}

def clear_session() -> None:
    """
    Limpia el estado de la sesión, eliminando las imágenes cargadas y procesadas.
    """
    if 'uploaded_images' in st.session_state:
        del st.session_state.uploaded_images
    
    if 'processed_images' in st.session_state:
        del st.session_state.processed_images
    
    if 'classification_cache' in st.session_state:
        del st.session_state.classification_cache
    
    # Resetear el estado del toggle de atención cuando se limpia la sesión
    st.session_state.show_attention = True
    
    st.session_state.uploader_key += 1

def style_language_uploader():
    """
    Personaliza el estilo y texto del cargador de archivos en español.
    Modifica el CSS para cambiar los textos del botón y las instrucciones.
    """
    # Diccionario con los textos en español
    languages = {
        "es": {
            "button": "Cargar archivos",
            "instructions": "Arrastra y suelta archivos aquí",
            "limits": "Límite 5MB por archivo • JPG, JPEG, PNG",
        },
    }

    # CSS personalizado para el cargador de archivos
    hide_label = (
        """
        <style>
            /* Estilo para el botón principal */
            div[data-testid="stFileUploader"]>section[data-testid="stFileUploaderDropzone"]>button[data-testid="stBaseButton-secondary"] {
                width: 100%;
                visibility: hidden;
                position: relative;
            }
            
            /* Texto personalizado para el botón */
            div[data-testid="stFileUploader"]>section[data-testid="stFileUploaderDropzone"]>button[data-testid="stBaseButton-secondary"]::after {
                content: "BUTTON_TEXT";
                visibility: visible;
                display: block;
                position: absolute;
                width: 100%;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                font-family: inherit;
                font-size: inherit;
                color: inherit;
                background-color: inherit;
                border: inherit;
                border-radius: 8px;
                padding: 5px 10px;
            }
            
 /* Ajustar altura del contenedor del cargador */
            div[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"] {
                min-height: 100px !important;
                padding: 10px !important;
            }
            /* Estilo para las instrucciones */
            div[data-testid="stFileUploaderDropzoneInstructions"]>div>span {
                visibility: hidden;
            }
            div[data-testid="stFileUploaderDropzoneInstructions"]>div>span::after {
                content: "INSTRUCTIONS_TEXT";
                visibility: visible;
                display: block;
            }
            
            /* Estilo para los límites de archivo */
            div[data-testid="stFileUploaderDropzoneInstructions"]>div>small {
                visibility: hidden;
            }
            div[data-testid="stFileUploaderDropzoneInstructions"]>div>small::before {
                content: "FILE_LIMITS";
                visibility: visible;
                display: block;
            }
        </style>
        """
        .replace("BUTTON_TEXT", languages.get('es').get("button"))
        .replace("INSTRUCTIONS_TEXT", languages.get('es').get("instructions"))
        .replace("FILE_LIMITS", languages.get('es').get("limits"))
    )
    st.markdown(hide_label, unsafe_allow_html=True)

def write_csv(processed_images: List[Dict[str, Any]], classes_name: List[str]) -> str:
    """Genera un archivo CSV con las coordenadas de las cajas delimitadoras de las imágenes procesadas.

    Args:
        processed_images (List[Dict[str, Any]]): Lista de diccionarios donde cada diccionario contiene el
            nombre de archivo, las cajas delimitadoras y las clasificaciones de una imagen procesada.
        classes_name (List[str]): Nombre perteneciente a cada clase. 

    Returns:
        str: El contenido del archivo CSV como una cadena de texto, con el nombre del archivo, 
        las coordenadas de las cajas delimitadoras (xmin, ymin, xmax, ymax) y la clase para cada objeto detectado.
    """
    # Crear un archivo CSV en memoria para almacenar las coordenadas
    csv_buffer = io.StringIO()
    csv_writer = csv.writer(csv_buffer)
    
    # Escribir la cabecera del CSV según el modo
    if st.session_state.use_detection:
        csv_writer.writerow(['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'class'])
    else:
        csv_writer.writerow(['filename', 'class'])

    for img in processed_images:
        if st.session_state.use_detection:
            # Modo detección: escribir coordenadas y clase para cada detección
            for box, clf in zip(img['boxes'], img['classes']):
                xmin, ymin, xmax, ymax = [round(coord.item(), 2) for coord in box.xyxy[0]]
                csv_writer.writerow([img['filename'], xmin, ymin, xmax, ymax, classes_name[clf]])
        else:
            # Modo clasificación: escribir solo el nombre del archivo y la clase
            class_idx = img['classes'][0]  # Tomar la primera clase ya que solo hay una
            csv_writer.writerow([img['filename'], classes_name[class_idx]])

    # Devolver el contenido del CSV como una cadena de texto
    return csv_buffer.getvalue()

def check_duplicates(uploaded_files: list):
    """
    Verifica si hay archivos duplicados en la lista de archivos subidos.
    
    Args:
        uploaded_files: Lista de archivos subidos por el usuario
        
    Returns:
        Lista con los nombres de los archivos duplicados
    """
    seen_files = set()  # Conjunto para almacenar nombres de archivos vistos
    duplicate_files = []  # Lista para almacenar nombres de archivos duplicados

    for file in uploaded_files:
        if file.name in seen_files:
            duplicate_files.append(file.name)
        else:
            seen_files.add(file.name)

    return duplicate_files

def process_images(det_model, clf_model, processor, confidence: float, iou_thres: float, classes_name: List[str]) -> None:
    """
    Procesa las imágenes usando los modelos de detección y clasificación de manera más eficiente.
    """
    # Limpiar el estado antes de procesar nuevas imágenes
    st.session_state.processed_images = []
    st.session_state.show_attention = True
    
    target_size = (224, 224)
    processed_results = []  # Lista temporal para almacenar resultados
    
    for image in st.session_state.uploaded_images:
        image_name = image.name
        
        # Verificar si ya tenemos los resultados de clasificación en caché
        if image_name in st.session_state.classification_cache:
            cached_results = st.session_state.classification_cache[image_name]
            
            if st.session_state.use_detection:
                # Si estamos en modo detección, usar los resultados de detección
                detections = cached_results['detections']
                bboxes = cached_results['boxes']
                classes = cached_results['classes']
            else:
                # Si estamos en modo clasificación directa, usar la clasificación de la imagen completa
                detections = [cached_results['full_image_classification']]
                bboxes = []
                classes = [cached_results['full_image_class']]
        else:
            # Si no hay resultados en caché, procesar la imagen
            uploaded_image = PIL.Image.open(image)
            
            # Realizar la clasificación de la imagen completa primero
            with torch.no_grad():
                resized_image = uploaded_image.resize(target_size)
                inputs = processor(images=[resized_image], return_tensors="pt")
                outputs = clf_model(**inputs)
                full_image_class = outputs.logits.argmax(-1).item()
                
                # Procesar atenciones para la imagen completa
                attentions = outputs.attentions[-1][0]
                full_image_attention = process_attention_maps(attentions, uploaded_image)
                
                full_image_detection = {
                    'original_image': uploaded_image,
                    'attention_map': full_image_attention,
                    'class': CLASSES_NAME_ES[full_image_class]
                }
            
            if st.session_state.use_detection:
                # Modo con detección
                det_res = det_model.predict(uploaded_image, conf=confidence, iou=iou_thres)
                bboxes = det_res[0].boxes
                cropped_images = crop_images(uploaded_image, bboxes)
                
                # Clasificar cada detección
                detections = []
                classes = []
                
                with torch.no_grad():
                    for cropped_image in cropped_images:
                        resized_crop = cropped_image.resize(target_size)
                        inputs = processor(images=[resized_crop], return_tensors="pt")
                        outputs = clf_model(**inputs)
                        class_idx = outputs.logits.argmax(-1).item()
                        classes.append(class_idx)
                        
                        # Procesar atenciones para la detección
                        attentions = outputs.attentions[-1][0]
                        attention_map = process_attention_maps(attentions, cropped_image)
                        
                        detections.append({
                            'original_image': cropped_image,
                            'attention_map': attention_map,
                            'class': CLASSES_NAME_ES[class_idx]
                        })
            else:
                # Modo sin detección
                detections = [full_image_detection]
                bboxes = []
                classes = [full_image_class]
            
            # Guardar resultados en caché
            st.session_state.classification_cache[image_name] = {
                'full_image_classification': full_image_detection,
                'full_image_class': full_image_class,
                'detections': detections,
                'boxes': bboxes,
                'classes': classes
            }
        
        # Almacenar los resultados en la lista temporal
        processed_results.append({
            'image': PIL.Image.open(image),
            'filename': image_name,
            'boxes': bboxes,
            'classes': classes,
            'detections': detections
        })
    
    # Actualizar el estado de la sesión con todos los resultados procesados
    st.session_state.processed_images = processed_results

def process_attention_maps(attentions, image):
    """
    Procesa los mapas de atención de manera eficiente.
    """
    attention_maps_interpolated = []
    
    # Calcular importancia de cabezas y obtener las 3 principales
    cls_attention = attentions[:, 0, 1:]
    head_importance = cls_attention.mean(dim=1)
    top_heads = torch.argsort(head_importance, descending=True)[:3]
    
    # Normalizar los pesos de las cabezas
    head_weights = head_importance[top_heads]
    head_weights = head_weights / head_weights.sum()

    # Procesar cada una de las 3 cabezas principales
    for head_idx, weight in zip(top_heads, head_weights):
        attention = attentions[head_idx][0, 1:]
        grid_size = int(np.sqrt(attention.shape[0]))
        attention = attention.reshape(grid_size, grid_size)
        
        # Redimensionar el mapa de atención
        attention = interpolate(
            attention.unsqueeze(0).unsqueeze(0),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False
        ).squeeze()
        
        # Normalizar y aplicar umbral
        attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
        threshold = torch.quantile(attention, 0.7)
        attention_binary = (attention > threshold).float() * weight.item()
        
        attention_maps_interpolated.append(attention_binary.numpy())

    # Crear mapa combinado ponderado
    combined_attention = np.sum(attention_maps_interpolated, axis=0) if attention_maps_interpolated else np.zeros(image.size[::-1])
    return np.clip(combined_attention, 0, 1)

def compute_head_importance(images):
    num_heads = clf_model.config.num_attention_heads
    importance = {cls: np.zeros(num_heads) for cls in CLASSES_NAME}
    counts = {cls: 0 for cls in CLASSES_NAME}
    
    for image in images:
        # Obtener la predicción del modelo
        with torch.no_grad():
            inputs = processor(images=Image.fromarray(image), return_tensors="pt")
            outputs = clf_model(**inputs)
            predicted_class_idx = outputs.logits.argmax(-1).item()
            predicted_class = CLASSES_NAME[predicted_class_idx]
            
            counts[predicted_class] += 1
            
            # Atención de la última capa
            attentions = outputs.attentions[-1][0]  # [num_heads, seq_len, seq_len]
            cls_attention = attentions[:, 0, 1:]    # [num_heads, num_patches]
            
            importance[predicted_class] += cls_attention.mean(dim=1).numpy()
    
    # Normalizar
    for cls in CLASSES_NAME:
        if counts[cls] > 0:
            importance[cls] /= counts[cls]
    
    return importance

def get_attention_map(image, model, processor, head_idx=None, layer_idx=-1, aggregate_layers=False):
    pil_image = Image.fromarray(image)
    inputs = processor(images=pil_image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    if aggregate_layers:
        # Promedio ponderado de todas las capas
        attention_maps = []
        weights = torch.linspace(0.5, 1.0, len(outputs.attentions))
        for layer_weight, layer_attention in zip(weights, outputs.attentions):
            attention = layer_attention[0]
            if head_idx is not None:
                attention = attention[head_idx][0, 1:]
            else:
                attention = attention.mean(dim=0)[0, 1:]
            
            grid_size = int(np.sqrt(attention.shape[0]))
            attention = attention.reshape(grid_size, grid_size)
            attention_maps.append(attention * layer_weight)
        
        attention = torch.stack(attention_maps).mean(dim=0)
    else:
        # Comportamiento con una sola capa
        attentions = outputs.attentions[layer_idx][0]
        if head_idx is not None:
            attention = attentions[head_idx][0, 1:]
        else:
            attention = attentions.mean(dim=0)[0, 1:]
        
        grid_size = int(np.sqrt(attention.shape[0]))
        attention = attention.reshape(grid_size, grid_size)
    
    # Redimensionar y normalizar el mapa de atención
    attention = interpolate(
        attention.unsqueeze(0).unsqueeze(0),
        size=image.shape[:2],
        mode="bicubic",
        align_corners=False
    ).squeeze().numpy()
    
    return np.clip((attention - attention.min()) / (attention.max() - attention.min() + 1e-8), 0, 1)

def export_results(processed_images: List[Dict[str, Any]]) -> None:
    """
    Exporta las imágenes procesadas y sus anotaciones en un archivo ZIP.
    
    Args:
        processed_images: Lista de diccionarios con las imágenes procesadas y sus metadatos
    """
    # Crear un archivo ZIP en memoria
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        # Agregar cada imagen procesada al ZIP
        for processed in processed_images:
            # Crear figura con matplotlib para la visualización
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            
            # Mostrar la imagen original
            img_array = np.array(processed['image'])
            ax.imshow(img_array)
            
            # Superponer el mapa de atención
            if not st.session_state.use_detection or st.session_state.show_attention:
                for detection in processed['detections']:
                    mask = np.ma.masked_where(detection['attention_map'] == 0, detection['attention_map'])
                    ax.imshow(mask, cmap='Reds', alpha=0.6, vmin=0, vmax=1)
            
            ax.axis('off')
            plt.tight_layout(pad=0)
            
            # Convertir la figura a bytes
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
            plt.close(fig)
            buf.seek(0)
            
            # Guardar en el ZIP con un nombre descriptivo
            filename = processed['filename']
            base_name = Path(filename).stem
            extension = Path(filename).suffix
            output_name = f"{base_name}_processed{extension}"
            zip_file.writestr(output_name, buf.getvalue())

        # Agregar el archivo CSV con las anotaciones
        zip_file.writestr('anotaciones.csv', write_csv(processed_images, CLASSES_NAME))

    # Preparar el archivo ZIP para la descarga
    zip_buffer.seek(0)
    zip_data = zip_buffer.getvalue()

    # Mostrar el botón de descarga
    try:
        st.sidebar.download_button(
            use_container_width=True,
            help='Exportar imágenes procesadas y anotaciones',
            label="📥 Exportar",
            data=zip_data,
            file_name="upd.zip",
            mime="application/zip"
        )
    except Exception as ex:
        st.error("¡No se ha subido ninguna imagen aún!")
        st.error(ex)

def image_to_base64(image_path):
    """
    Convierte una imagen a formato base64 para mostrarla en la interfaz.
    
    Args:
        image_path: Ruta a la imagen
        
    Returns:
        String en formato base64 de la imagen
    """
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        st.error(f"No se encontró la imagen en: {image_path}")
        return ""

def create_visualization(detection, show_attention):
    """
    Crea la visualización de manera más eficiente usando matplotlib.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Mostrar la imagen original
    img_array = np.array(detection['original_image'])
    ax.imshow(img_array)
    
    # Superponer el mapa de atención si es necesario
    if show_attention:
        mask = np.ma.masked_where(detection['attention_map'] == 0, detection['attention_map'])
        ax.imshow(mask, cmap='Reds', alpha=0.6, vmin=0, vmax=1)
    
    ax.axis('off')
    plt.tight_layout(pad=0)
    
    # Convertir la figura a imagen PIL de manera más eficiente
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close(fig)
    buf.seek(0)
    
    return PIL.Image.open(buf)

def main():
    """
    Función principal que ejecuta la aplicación Streamlit.
    Configura la interfaz y maneja la lógica principal de la aplicación.
    """
    # Configuración de la página - DEBE ser el primer comando de Streamlit
    image_path = Path(__file__).parent / "huellas-humanas.png"
    st.set_page_config(
        page_title="UPD - Úlceras de Pie Diabético",
        page_icon=str(image_path),
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Inicializar el estado de la sesión y variables - MOVIDO AQUÍ
    initialize_session()
    source_imgs = []
    
    # Cargar los modelos
    try:
        det_model, processor, clf_model = load_models()
    except Exception as ex:
        st.error("No se pudo cargar los modelos. Verifique las rutas especificadas")
        st.error(ex)
        return
    
    # Mostrar logo y título
    try:
        base64_logo = image_to_base64(image_path)
        
        # Crear dos columnas para el logo y el toggle
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(
                f"""
            <style>
            .logo-title {{
                display: flex;
                align-items: center;
                gap: 15px;
            }}
            /* Eliminar padding del encabezado de la barra lateral */
            .st-emotion-cache-16idsys p {{
                padding-top: 0;
                padding-bottom: 0;
                margin: 0;
            }}
            /* Ajustar padding del contenedor principal */
            div[data-testid="stMainBlockContainer"] {{
                padding: 4rem 1rem 10rem 3rem 3rem;
            }}
            .st-emotion-cache-1jicfl2 {{
                padding-left: 3rem;
                padding-right: 3rem;
            }}
            .st-emotion-cache-kgpedg {{
                padding: 0 !important;
            }}
            </style>
            
            <div class="logo-title">
                <img src="data:image/png;base64,{base64_logo}" width="80">
                <h2>Cuidado inteligente del pie diabético</h2>
            </div>
            <br>
                """,unsafe_allow_html=True
            )
    except Exception as e:
        st.error(f"Error al cargar la imagen: {e}")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("<h2>Cuidado inteligente del pie diabético</h2>", unsafe_allow_html=True)
    
    # Mostrar toggle de atención solo si hay imágenes procesadas y el detector está activado
    with col2:
        # El toggle solo debe aparecer si el detector está activado Y hay imágenes procesadas
        if st.session_state.use_detection and len(st.session_state.processed_images)>=1:
            st.markdown("""
                <style>
                    .st-emotion-cache-1mo46gi {
                        display: flex !important;
                        justify-content: flex-end !important;
                        align-items: center !important;
                        width: 100% !important;
                        padding: 0 !important;
                        margin: 0 !important;
                    }
                    .st-emotion-cache-4mtp6l {
                        margin: 0 !important;
                        width: fit-content !important;
                        display: flex !important;
                        justify-content: flex-end !important;
                        align-items: center !important;
                    }
                    .edwcd611 {
                        margin: 0 !important;
                        width: fit-content !important;
                    }
                </style>
            """, unsafe_allow_html=True)
            
            show_attention = st.toggle(
                "Mostrar áreas de atención",
                value=st.session_state.show_attention,
                help="Activa/Desactiva la visualización de las áreas críticas",
                key="show_attention_toggle"
            )
            st.session_state.show_attention = show_attention
        else:
            # Deshabilitar el toggle de atención cuando no se cumplan las condiciones
            if "show_attention_toggle" in st.session_state:
                del st.session_state["show_attention_toggle"]
            st.session_state.show_attention = False
    
    # Configuración del sidebar
    with st.sidebar:
        # Título de la barra lateral
        st.header("⚙️ Configuración del modelo")

        # Toggle para elegir el modo de procesamiento
        use_detection = st.toggle(
            "Usar modelo de detección",
            value=st.session_state.use_detection,
            help="Activa/Desactiva el uso del modelo de detección.",
            key="detection_toggle",
            on_change=lambda: update_detection_mode()
        )
        
        # Actualizar el estado si cambió el toggle
        if use_detection != st.session_state.use_detection:
            st.session_state.use_detection = use_detection
            # Actualizar la visualización de las imágenes procesadas si existen
            if 'processed_images' in st.session_state and len(st.session_state.processed_images) > 0:
                for processed in st.session_state.processed_images:
                    image_name = processed['filename']
                    if image_name in st.session_state.classification_cache:
                        cached_results = st.session_state.classification_cache[image_name]
                        if st.session_state.use_detection:
                            # Si estamos en modo detección, usar los resultados de detección
                            processed['detections'] = cached_results['detections']
                            processed['boxes'] = cached_results['boxes']
                            processed['classes'] = cached_results['classes']
                        else:
                            # Si estamos en modo clasificación directa, usar la clasificación de la imagen completa
                            processed['detections'] = [cached_results['full_image_classification']]
                            processed['boxes'] = []
                            processed['classes'] = [cached_results['full_image_class']]

        # Control deslizante para ajustar la confianza del modelo
        confidence = st.slider( 
            label="📍 Seleccionar confianza de detección",
            min_value=0,
            max_value=100, 
            value=st.session_state.confidence,
            help='Probabilidad de certeza en la detección de la úlcera',
            disabled=not st.session_state.use_detection
        )

        # Actualizar la confianza solo si está en modo detección
        if confidence != st.session_state.confidence and st.session_state.use_detection:
            st.session_state.confidence = confidence
            clear_session()

        

        # Contenedor para el cargador de archivos
        uploader_container = st.container()
        with uploader_container:
            source_imgs = st.file_uploader(
                label="📍 Seleccionar imágenes",
                help='Imagen del pie que desea analizar',
                type=("jpg", "jpeg", "png"),
                accept_multiple_files=True,
                key=f"image_uploader_{st.session_state.uploader_key}"
            )

        # Personalizar el estilo del cargador
        style_language_uploader()

        # Verificar límite de imágenes
        if source_imgs and len(source_imgs) > 20:
            st.toast('⚠️ Ha superado el límite de imágenes')
            time.sleep(.5)        

        # Verificar duplicados
        if source_imgs:
            duplicate_files = check_duplicates(source_imgs)
            if duplicate_files:
                if len(duplicate_files) == 1:
                    st.error(f"El siguiente archivo está duplicado: {duplicate_files[0]}")
                else:
                    st.error(f"⚠️ Los siguientes archivos están duplicados: \n{', '.join(duplicate_files)}")
            
        # Botones de control
        if source_imgs and len(source_imgs) != 0:
            delete_button = st.button(
                "🗑️ Eliminar imágenes",
                use_container_width=True,
                help="Eliminar todas las imágenes cargadas",
                key="delete_button"
            )

            # Botón de análisis y procesamiento de imágenes
            if len(source_imgs) <= 20 and not check_duplicates(source_imgs):
                text_btn = '🔍 Analizar imágenes' if len(source_imgs) > 1 else '🔍 Analizar imagen'
                process_image_button = st.button(
                    label=text_btn,
                    help="Analizar las imágenes seleccionadas",
                    use_container_width=True,
                    key="process_button"
                )
            else:
                process_image_button = False
        else:
            delete_button = False
            process_image_button = False

    # Mostrar mensaje cuando no hay imágenes
    if not source_imgs or len(source_imgs) == 0:
        camera_svg = '''
            <svg xmlns="http://www.w3.org/2000/svg" fill="gray" viewBox="0 0 24 24" width="24" height="24">
                <circle cx="16" cy="8.011" r="2.5"/>
                <path d="M23,16a1,1,0,0,0-1,1v2a3,3,0,0,1-3,3H17a1,1,0,0,0,0,2h2a5.006,5.006,0,0,0,5-5V17A1,1,0,0,0,23,16Z"/>
                <path d="M1,8A1,1,0,0,0,2,7V5A3,3,0,0,1,5,2H7A1,1,0,0,0,7,0H5A5.006,5.006,0,0,0,0,5V7A1,1,0,0,0,1,8Z"/>
                <path d="M7,22H5a3,3,0,0,1-3-3V17a1,1,0,0,0-2,0v2a5.006,5.006,0,0,0,5,5H7a1,1,0,0,0,0-2Z"/>
                <path d="M19,0H17a1,1,0,0,0,0,2h2a3,3,0,0,1,3,3V7a1,1,0,0,0,2,0V5A5.006,5.006,0,0,0,19,0Z"/>
                <path d="M18.707,17.293,11.121,9.707a3,3,0,0,0-4.242,0L4.586,12A2,2,0,0,0,4,13.414V16a3,3,0,0,0,3,3H18a1,1,0,0,0,.707-1.707Z"/>
            </svg>'''

        with st.container(border=True):
            st.markdown(
                f"<div style=' font-size: 16px; display: flex; justify-content: center; align-items: center; padding: 0 0 10px 0; gap: 15px; border-radius: 8px;'>"
                f"{camera_svg}"
                "No ha seleccionado una imagen para su procesamiento"
                "</div>",
                unsafe_allow_html=True
            )

    # Procesar y mostrar las imágenes
    if source_imgs and len(source_imgs) != 0:
        st.session_state.uploaded_images = source_imgs

        # Selector de imagen para visualización
        if len(st.session_state.uploaded_images) > 1:
            image_filenames = [img.name for img in st.session_state.uploaded_images]
            selected_image = st.selectbox(
                "",
                image_filenames,
                help="Selecciona la imagen que desea visualizar"
            )
            original_image_index = image_filenames.index(selected_image)
            source_img = source_imgs[original_image_index]
        else:
            selected_image = source_imgs[0].name
            source_img = source_imgs[0]

        # Mostrar imágenes en columnas
        col1, col2 = st.columns([1,1], gap="medium")
        with col1:
            try:
                st.markdown("""
                    <div style='text-align: center; font-size: 1.5em;'>
                        📸 Imagen Original
                    </div>
                """, unsafe_allow_html=True)
                with st.container():
                    st.markdown("""
                        <style>
                            .stImage img {
                                max-height: 400px !important;
                                height: auto !important;
                                max-width: 400px !important;
                                object-fit: contain !important;
                                overflow: hidden;
                                margin: 0 auto !important;
                                display: block !important;
                            }
                            .st-emotion-cache-1kyxreq {
                                width: 100% !important;
                                margin: 0 !important;
                                padding: 0 !important;
                                display: flex !important;
                                justify-content: center !important;
                                align-items: center !important;
                            }
                            .st-emotion-cache-1v0mbdj {
                                width: 100% !important;
                                margin: 0 !important;
                                padding: 0 !important;
                                display: flex !important;
                                justify-content: center !important;
                                align-items: center !important;
                            }
                            div[data-testid="stImage"] {
                                display: flex !important;
                                justify-content: center !important;
                                align-items: center !important;
                                width: 100% !important;
                            }
                        </style>
                    """, unsafe_allow_html=True)
                    st.image(source_img, use_column_width=True)
            except Exception as ex:
                st.error("Ocurrió un error al abrir la imagen.")
                st.error(ex)

        # Procesar y mostrar resultados
        with col2:
            if delete_button:
                clear_session()
                st.toast('Todas las imágenes han sido eliminadas', icon='✅')
                time.sleep(0.6)
                st.rerun()

            if process_image_button:
                # Procesar las imágenes primero
                process_images(
                    det_model=det_model,
                    clf_model=clf_model,
                    processor=processor,
                    confidence=st.session_state.confidence/100,
                    iou_thres=IOU_THRES,
                    classes_name=CLASSES_NAME_ES
                )
                # Forzar una actualización de la interfaz
                #st.rerun()

            # Mostrar imágenes procesadas
            if st.session_state.processed_images:
                for processed in st.session_state.processed_images:
                    if processed['filename'] == selected_image:
                        if len(processed['detections']) > 0:
                            # Determinar el título y el estilo según el modo
                            if st.session_state.use_detection:
                                title = "🔍 Úlceras Detectadas"
                            else:
                                title = "🔍 Clasificación de la Imagen"
                            
                            st.markdown(f"""
                                <div style='text-align: center; font-size: 1.5em; margin-bottom: 0.5rem;'>
                                    {title}
                                </div>
                            """, unsafe_allow_html=True)
                            
                            with st.container():
                                # Mostrar cada detección verticalmente
                                for idx, detection in enumerate(processed['detections']):
                                    # Crear figura con matplotlib para la visualización
                                    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                                    
                                    # Mostrar la imagen original
                                    img_array = np.array(detection['original_image'])
                                    ax.imshow(img_array)
                                    
                                    # Superponer el mapa de atención
                                    # Si el modelo de detección está desactivado, siempre mostrar atención
                                    # Si está activado, respetar el toggle
                                    if not st.session_state.use_detection or st.session_state.show_attention:
                                        mask = np.ma.masked_where(detection['attention_map'] == 0, detection['attention_map'])
                                        ax.imshow(mask, cmap='Reds', alpha=0.6, vmin=0, vmax=1)
                                    
                                    ax.axis('off')
                                    plt.tight_layout(pad=0)  # Eliminar el padding
                                    
                                    # Convertir la figura a imagen PIL
                                    buf = io.BytesIO()
                                    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100)  # Eliminar el padding al guardar
                                    plt.close(fig)
                                    buf.seek(0)
                                    display_image = PIL.Image.open(buf)
                                    
                                    # Contenedor para la imagen
                                    st.image(
                                        display_image,
                                        use_column_width=True
                                    )
                                    
                                    # Badge de clasificación con estilo actualizado
                                    st.markdown(
                                        f"""
                                        <div style="
                                            background-color: #FF4B4B;
                                            color: white;
                                            padding: 8px 12px;
                                            border-radius: 6px;
                                            text-align: center;
                                            margin: 5px auto 15px auto;
                                            font-weight: 600;
                                            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                                            width: 80%;
                                            box-sizing: border-box;
                                        ">
                                            {f'Detección {idx + 1}: ' if st.session_state.use_detection else ''}{detection['class']}
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )
                        else:
                            st.markdown("""
                                <div style="
                                    background-color: transparent;
                                    padding: 2rem;
                                    text-align: center;
                                    margin: 1rem 0;
                                ">
                                    <div style="
                                        width: 100px;
                                        height: 100px;
                                        background-color: #00C853;
                                        border-radius: 50%;
                                        display: flex;
                                        align-items: center;
                                        justify-content: center;
                                        margin: 3rem auto 1rem auto;
                                        box-shadow: 0 4px 8px rgba(0,200,83,0.3);
                                    ">
                                        <div style="
                                            font-size: 4rem;
                                            color: white;
                                            transform: scale(1.2);
                                        ">
                                            ✓
                                        </div>
                                    </div>
                                    <div style="
                                        font-size: 1.2rem;
                                        color: var(--text-color);
                                        font-weight: 800;
                                    ">
                                        No se han detectado ulceraciones
                                    </div>
                                    <div style="
                                        font-size: 0.9rem;
                                        color: var(--text-color);
                                        margin-top: 0.5rem;
                                        opacity: 0.8;
                                    ">
                                        La imagen analizada no presenta signos de úlceras
                                    </div>
                                </div>
                                <style>
                                    :root {
                                        --text-color: var(--text-color);
                                    }
                                    @media (prefers-color-scheme: dark) {
                                        :root {
                                            --text-color: #FFFFFF;
                                        }
                                    }
                                    @media (prefers-color-scheme: light) {
                                        :root {
                                            --text-color: #000000;
                                        }
                                    }
                                </style>
                            """, unsafe_allow_html=True)

            # Mostrar botón de exportación si hay imágenes procesadas
            if st.session_state.processed_images and len(st.session_state.processed_images) == len(st.session_state.uploaded_images):
                # Mostrar el botón de exportación en ambos modos
                export_results(st.session_state.processed_images)

def update_detection_mode():
    """
    Actualiza el modo de detección y la visualización de las imágenes procesadas.
    """
    # Actualizar la visualización de las imágenes procesadas si existen
    if 'processed_images' in st.session_state and len(st.session_state.processed_images) > 0:
        for processed in st.session_state.processed_images:
            image_name = processed['filename']
            if image_name in st.session_state.classification_cache:
                cached_results = st.session_state.classification_cache[image_name]
                if st.session_state.use_detection:
                    # Si estamos en modo detección, usar los resultados de detección
                    processed['detections'] = cached_results['detections']
                    processed['boxes'] = cached_results['boxes']
                    processed['classes'] = cached_results['classes']
                else:
                    # Si estamos en modo clasificación directa, usar la clasificación de la imagen completa
                    processed['detections'] = [cached_results['full_image_classification']]
                    processed['boxes'] = []
                    processed['classes'] = [cached_results['full_image_class']]
    
    # Actualizar el estado del toggle de atención
    if not st.session_state.use_detection:
        if "show_attention_toggle" in st.session_state:
            del st.session_state["show_attention_toggle"]
        st.session_state.show_attention = False

if __name__ == '__main__':
    main()