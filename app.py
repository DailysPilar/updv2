import os
os.environ["HF_HUB_DISABLE_XET"] = "1"

import streamlit as st
from helper import load_pt_model, get_image_download_buffer, draw_bounding_boxes, crop_images
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
import matplotlib.pyplot as plt
from segmentation_module import render_segmentation_panel, prefetch_segmenter_weights
from ui_styles import (inject_styles, render_header, render_welcome, section_title, badge, status_card,
                       current_theme)

# Constantes globales de la aplicación
IMAGES_DIR = Path(__file__).resolve().parent / "dist" / "images"
IOU_THRES = 0.5  # Umbral de IoU para la supresión de no máximos
CLASSES_NAME = ['both', 'infection', 'ischaemia', 'none']  # Nombres de clases en inglés
CLASSES_NAME_ES = ['INFECCIÓN E ISQUEMIA', 'INFECCIÓN', 'ISQUEMIA', 'SANO']  # Nombres de clases en español
# Decorador para cachear el modelo de detección y evitar recargas innecesarias
@st.cache_resource
def load_models():
    try:
        from transformers import ViTForImageClassification, ViTImageProcessor

        det_model = load_pt_model(Path(settings.DETECTION_MODEL))
        model_path = 'daoliver/Vit_upd'
        processor = ViTImageProcessor.from_pretrained(model_path, local_files_only=False)
        clf_model = ViTForImageClassification.from_pretrained(
            model_path,
            num_labels=len(CLASSES_NAME),
            ignore_mismatched_sizes=True,
            output_attentions=True,
            attn_implementation="eager",
            local_files_only=False
        )
        return det_model, processor, clf_model
    except Exception as ex:
        import traceback
        st.error("No se pudo cargar los modelos.")
        st.code(traceback.format_exc())
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

    # Toggle de detección se inicializa automáticamente por el widget

    # Control de visualización de atención (inicializado en True)
    if 'show_attention' not in st.session_state:
        st.session_state.show_attention = True

    # Diccionario para almacenar los resultados de clasificación por imagen
    if 'classification_cache' not in st.session_state:
        st.session_state.classification_cache = {}

    # Variable para la imagen seleccionada (para múltiples imágenes)
    if 'selected_image_name' not in st.session_state:
        st.session_state.selected_image_name = None

def clear_session() -> None:
    """
    Limpia el estado de la sesión, eliminando las imágenes cargadas y procesadas.
    """
    # Primero, eliminar las variables que no dependen de otras
    if 'uploaded_images' in st.session_state:
        del st.session_state.uploaded_images
    
    if 'processed_images' in st.session_state:
        del st.session_state.processed_images
    
    if 'classification_cache' in st.session_state:
        del st.session_state.classification_cache
    
    # Incrementar la clave del cargador
    st.session_state.uploader_key += 1
    
    # Reiniciar el estado de atención y detección después de limpiar las otras variables
    if 'show_attention' in st.session_state:
        del st.session_state.show_attention
    # detection_toggle se maneja automáticamente por el widget
    if 'selected_image_name' in st.session_state:
        del st.session_state.selected_image_name
    
    st.session_state.show_attention = True
    st.session_state.selected_image_name = None

def write_csv(processed_images: List[Dict[str, Any]], classes_name: List[str]) -> str:
    """Genera un archivo CSV con las coordenadas de las cajas delimitadoras de las imágenes procesadas.

    Args:
        processed_images (List[Dict[str, Any]]): Lista de diccionarios donde cada diccionario contiene el
            nombre de archivo, las cajas delimitadoras y las clasificaciones de una imagen procesada.
        classes_name (List[str]): Nombre perteneciente a cada clase. 

    Returns:
        str: El contenido del archivo CSV como una cadena de texto.
    """
    # Crear un archivo CSV en memoria para almacenar las coordenadas
    csv_buffer = io.StringIO()
    csv_writer = csv.writer(csv_buffer)
    
    # Escribir la cabecera del CSV según el modo
    if st.session_state.detection_toggle:
        csv_writer.writerow(['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'class', 'confidence'])
    else:
        csv_writer.writerow(['filename', 'class'])

    for img in processed_images:
        if st.session_state.detection_toggle:
            # Modo detección: escribir coordenadas y clase para cada detección
            for idx, (box, clf) in enumerate(zip(img['boxes'], img['classes']), 1):
                xmin, ymin, xmax, ymax = [round(coord.item(), 2) for coord in box.xyxy[0]]
                conf = round(float(box.conf[0]) * 100, 2)  # Convertir a porcentaje
                # Crear nombre de archivo con índice de detección
                base_name = Path(img['filename']).stem
                extension = Path(img['filename']).suffix
                detection_filename = f"{base_name}_{idx}{extension}"
                csv_writer.writerow([detection_filename, xmin, ymin, xmax, ymax, classes_name[clf], conf])
        else:
            # Modo clasificación: escribir solo el nombre del archivo y la clase
            class_name = img['detections'][0]['class']
            base_name = Path(img['filename']).stem
            extension = Path(img['filename']).suffix
            processed_filename = f"{base_name}_processed{extension}"
            csv_writer.writerow([processed_filename, class_name])

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
    Siempre realiza tanto la detección como la clasificación, pero muestra los resultados según el modo seleccionado.
    """
    # Validaciones previas
    if not det_model or not clf_model or not processor:
        st.error("Los modelos no están disponibles. Por favor, recarga la página.")
        return
    
    if 'uploaded_images' not in st.session_state or not st.session_state.uploaded_images:
        st.warning("No hay imágenes cargadas para procesar.")
        return
    
    # Limpiar el estado antes de procesar nuevas imágenes
    st.session_state.processed_images = []
    
    # Asegurar que el estado de atención esté sincronizado con el modo de detección
    if not st.session_state.detection_toggle:
        st.session_state.show_attention = True
    
    target_size = (224, 224)
    processed_results = []  # Lista temporal para almacenar resultados
    
    for image in st.session_state.uploaded_images:
        try:
            image_name = image.name
            uploaded_image = PIL.Image.open(image)
            
            # Verificar que la imagen se abrió correctamente
            if uploaded_image is None:
                st.error(f"No se pudo abrir la imagen: {image_name}")
                continue
                
        except Exception as e:
            st.error(f"Error al abrir la imagen {image.name}: {str(e)}")
            continue
        
        # Verificar si ya tenemos los resultados de clasificación en caché
        if ('classification_cache' in st.session_state and 
            image_name in st.session_state.classification_cache):
            cached_results = st.session_state.classification_cache[image_name]
            
            if st.session_state.detection_toggle:
                # Si estamos en modo detección, usar los resultados de detección
                detections = cached_results.get('detections', [])
                bboxes = cached_results.get('boxes', [])
                classes = cached_results.get('classes', [])
                det_res = cached_results.get('det_res', None)
            else:
                # Si estamos en modo clasificación directa, usar la clasificación de la imagen completa
                detections = [cached_results.get('full_image_classification', {})]
                bboxes = []
                classes = [cached_results.get('full_image_class', 3)]
        else:
            # Si no hay resultados en caché, procesar la imagen
            # Realizar la clasificación de la imagen completa primero
            try:
                with torch.no_grad():
                    # Asegurar que la imagen esté en formato RGB
                    if uploaded_image.mode != 'RGB':
                        uploaded_image = uploaded_image.convert('RGB')
                    
                    resized_image = uploaded_image.resize(target_size)
                    inputs = processor(images=resized_image, return_tensors="pt")
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
            except Exception as e:
                st.error(f"Error al procesar la imagen {image_name}: {str(e)}")
                # Crear una detección por defecto en caso de error
                full_image_detection = {
                    'original_image': uploaded_image,
                    'attention_map': np.zeros(uploaded_image.size[::-1]),
                    'class': 'ERROR'
                }
                full_image_class = 3  # Clase 'none' como fallback
            
            # Siempre realizar la detección y clasificación de las detecciones
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
            
            # Guardar resultados en caché
            cached_results = {
                'full_image_classification': full_image_detection,
                'full_image_class': full_image_class,
                'detections': detections,
                'boxes': bboxes,
                'classes': classes,
                'det_res': det_res
            }
            st.session_state.classification_cache[image_name] = cached_results
        
        # Almacenar los resultados en la lista temporal según el modo actual
        processed_results.append({
            'image': uploaded_image,
            'filename': image_name,
            'det_res': cached_results.get('det_res', None), # Asegúrate de que det_res esté disponible
            'boxes': bboxes if st.session_state.detection_toggle else [],
            'classes': classes if st.session_state.detection_toggle else [cached_results['full_image_class']],
            'detections': detections if st.session_state.detection_toggle else [cached_results['full_image_classification']]
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
            if st.session_state.detection_toggle:
                # En modo detección, guardar cada detección individualmente
                for idx, detection in enumerate(processed['detections'], 1):
                    # Crear figura con matplotlib para la visualización
                    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                    
                    # Mostrar la imagen original
                    img_array = np.array(detection['original_image'])
                    ax.imshow(img_array)
                    
                    # Superponer el mapa de atención solo si está activado
                    if st.session_state.show_attention:
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
                    base_name = Path(processed['filename']).stem
                    extension = Path(processed['filename']).suffix
                    output_name = f"{base_name}_{idx}{extension}"
                    zip_file.writestr(output_name, buf.getvalue())
            else:
                # En modo clasificación, guardar la imagen completa
                fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                
                # Mostrar la imagen original
                img_array = np.array(processed['image'])
                ax.imshow(img_array)
                
                # Superponer el mapa de atención solo si está activado
                if st.session_state.show_attention:
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
        # Determinar el nombre del archivo ZIP según el modo
        zip_filename = "updcondetector.zip" if st.session_state.detection_toggle else "updsindetector.zip"
        
        st.sidebar.download_button(
            use_container_width=True,
            help='Exportar imágenes procesadas y anotaciones',
            label="📥 Exportar",
            data=zip_data,
            file_name=zip_filename,
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

def create_visualization(detection):
    """
    Crea la visualización de manera más eficiente usando matplotlib.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Mostrar la imagen original
    img_array = np.array(detection['original_image'])
    ax.imshow(img_array)
    
    # Superponer el mapa de atención solo si show_attention es True y estamos en modo detección
    if st.session_state.detection_toggle and st.session_state.show_attention:
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
    image_path = IMAGES_DIR / "huellas-humanas.png"
    st.set_page_config(
        page_title="UPD - Úlceras de Pie Diabético",
        page_icon=str(image_path),
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Inicializar el estado de la sesión y variables
    initialize_session()
    inject_styles(current_theme())
    prefetch_segmenter_weights()
    source_imgs = []
    
    # Cargar los modelos
    try:
        det_model, processor, clf_model = load_models()
    except Exception as ex:
        st.error("No se pudo cargar los modelos. Verifique la conexión a internet y las rutas especificadas")
        st.error(ex)
        return
    
    # Mostrar logo y título
    render_header(
        image_to_base64(image_path),
        "Cuidado inteligente del pie diabético",
        "Detección, clasificación y segmentación de úlceras de pie diabético",
    )
    
    # Configuración del sidebar
    with st.sidebar:
        # Título de la barra lateral
        st.subheader("⚙️ Configuración del modelo")

        # Agregar un callback para el toggle de detección
        def on_detection_toggle():
            """
            Callback que se ejecuta cuando cambia el estado del toggle de detección
            """
            # Validar que los modelos estén cargados
            if not det_model or not clf_model or not processor:
                st.warning("Los modelos aún se están cargando. Intenta de nuevo en unos segundos.")
                return
            
            # Si se desactiva la detección, forzar show_attention a True
            if not st.session_state.detection_toggle:
                st.session_state.show_attention = True
            else:
                # Si se reactiva la detección, también forzar show_attention a True
                st.session_state.show_attention = True
            
            # Reprocesar las imágenes si hay alguna cargada
            if ('uploaded_images' in st.session_state and 
                st.session_state.uploaded_images and 
                len(st.session_state.uploaded_images) > 0):
                process_images(det_model, clf_model, processor, st.session_state.confidence/100, IOU_THRES, CLASSES_NAME)

        # Toggle para elegir el modo de procesamiento
        st.toggle(
            "Usar modelo de detección",
            value=True,
            key="detection_toggle",
            on_change=on_detection_toggle,
            help="Activar/Desactivar modelo de detección de UPD"
        )

        # Mostrar el toggle de atención solo si la detección está activada
        if st.session_state.detection_toggle:
            def on_attention_toggle():
                """
                Callback que se ejecuta cuando cambia el estado del toggle de atención
                """
                # Validar que los modelos estén cargados
                if not det_model or not clf_model or not processor:
                    st.warning("Los modelos aún se están cargando. Intenta de nuevo en unos segundos.")
                    return
                
                # Reprocesar las imágenes para actualizar la visualización
                if ('uploaded_images' in st.session_state and 
                    st.session_state.uploaded_images and 
                    len(st.session_state.uploaded_images) > 0):
                    process_images(det_model, clf_model, processor, st.session_state.confidence/100, IOU_THRES, CLASSES_NAME)

            # Inicializar show_attention en el session state si no existe
            if 'show_attention' not in st.session_state:
                st.session_state.show_attention = True

            st.toggle(
                "Mostrar zonas de atención",
                key="show_attention",
                on_change=on_attention_toggle,
                help="Activar/Desactivar la visualización de las zonas de atención"
            )
        else:
            # Si la detección está desactivada, forzar show_attention a True
            st.session_state.show_attention = True

        st.toggle(
            "Mostrar segmentación de la lesión",
            value=True,
            key="show_segmentation",
            help="Segmenta la imagen completa y superpone la máscara, con el área de la lesión",
        )
        # Control deslizante para ajustar la confianza del modelo
        def on_confidence_change():
            # Validar que los modelos estén cargados
            if not det_model or not clf_model or not processor:
                st.warning("Los modelos aún se están cargando. Intenta de nuevo en unos segundos.")
                return
                
            if st.session_state.detection_toggle:
                st.session_state.confidence = st.session_state.confidence_slider
                # Reprocesar las imágenes si hay alguna cargada
                if ('uploaded_images' in st.session_state and 
                    st.session_state.uploaded_images and 
                    len(st.session_state.uploaded_images) > 0):
                    process_images(det_model, clf_model, processor, st.session_state.confidence/100, IOU_THRES, CLASSES_NAME)

        st.slider( 
            label="Confianza de detección (%)",
            min_value=0,
            max_value=100, 
            value=st.session_state.confidence,
            help='Probabilidad de certeza en la detección de la úlcera',
            disabled=not st.session_state.detection_toggle,
            key="confidence_slider",
            on_change=on_confidence_change
        )

        st.divider()
        st.subheader("📂 Seleccionar imágenes")

        # Contenedor para el cargador de archivos
        uploader_container = st.container()
        with uploader_container:
            source_imgs = st.file_uploader(
                label="Imágenes del pie a analizar",
                label_visibility="collapsed",
                help='Imagen del pie que desea analizar',
                type=("jpg", "jpeg", "png"),
                accept_multiple_files=True,
                key=f"image_uploader_{st.session_state.uploader_key}"
            )

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
            # Botón de análisis y procesamiento de imágenes
            if len(source_imgs) <= 20 and not check_duplicates(source_imgs):
                text_btn = '🔍 Analizar imágenes' if len(source_imgs) > 1 else '🔍 Analizar imagen'
                process_image_button = st.button(
                    label=text_btn,
                    help="Analizar las imágenes seleccionadas",
                    type="primary",
                    use_container_width=True,
                    key="process_button"
                )
            else:
                process_image_button = False

            delete_button = st.button(
                "🗑️ Eliminar imágenes",
                use_container_width=True,
                help="Eliminar todas las imágenes cargadas",
                key="delete_button"
            )
        else:
            delete_button = False
            process_image_button = False

    # Pantalla de bienvenida cuando no hay imágenes
    if not source_imgs or len(source_imgs) == 0:
        render_welcome()
    # Procesar y mostrar las imágenes
    if source_imgs and len(source_imgs) != 0:
        st.session_state.uploaded_images = source_imgs

        # Lista de nombres de imágenes procesadas
        processed_filenames = [img['filename'] for img in st.session_state.processed_images] if st.session_state.processed_images else []
        uploaded_filenames = [img.name for img in st.session_state.uploaded_images]

        # Verificar si hay nuevas imágenes que procesar
        new_images = [name for name in uploaded_filenames if name not in processed_filenames]

        # Selector de imagen para visualización
        if len(st.session_state.uploaded_images) > 1:
            image_filenames = [img.name for img in st.session_state.uploaded_images]
            
            # Inicializar o actualizar el estado de la imagen seleccionada
            if 'selected_image_name' not in st.session_state or st.session_state.selected_image_name not in image_filenames:
                # Si hay nuevas imágenes, seleccionar la primera de ellas
                if new_images:
                    st.session_state.selected_image_name = new_images[0]
                else:
                    st.session_state.selected_image_name = image_filenames[0]

            # Usar el estado de la sesión para mantener la selección
            selected_image = st.selectbox(
                "Imagen a visualizar",
                image_filenames,
                index=image_filenames.index(st.session_state.selected_image_name),
                help="Selecciona la imagen que desea visualizar",
                key="image_selector",
                on_change=lambda: setattr(st.session_state, 'selected_image_name', st.session_state.image_selector)
            )
            
            # Actualizar el estado con la imagen seleccionada
            st.session_state.selected_image_name = selected_image
            
            # Encontrar la imagen correspondiente
            for img in source_imgs:
                if img.name == selected_image:
                    source_img = img
                    break
            else:
                source_img = source_imgs[0]  # Fallback a la primera imagen si no se encuentra
        else:
            selected_image = source_imgs[0].name
            source_img = source_imgs[0]

        # Mostrar imágenes en columnas (cada una dentro de su tarjeta)
        col1, col2 = st.columns([1,1], gap="medium")
        with col1, st.container(border=True):
            try:
                section_title("Imagen original", "🖼️", "Foto cargada, sin modificaciones")
                st.image(source_img, use_container_width=True)
            except Exception as ex:
                st.error("Ocurrió un error al abrir la imagen.")
                st.error(ex)

        with col2, st.container(border=True):
            if delete_button:
                clear_session()
                st.toast('Todas las imágenes han sido eliminadas', icon='✅')
                time.sleep(0.6)
                st.rerun()

            processed_filenames = [img['filename'] for img in st.session_state.processed_images] if st.session_state.processed_images else []
            uploaded_filenames = [img.name for img in st.session_state.uploaded_images]
            new_images = [name for name in uploaded_filenames if name not in processed_filenames]
            section_title("Cajas delimitadoras", "🎯", "Zonas donde el modelo detectó úlceras")
            if new_images or process_image_button:
                with st.spinner("Detectando úlceras en la imagen...", show_time=True):
                    process_images(
                        det_model=det_model,
                        clf_model=clf_model,
                        processor=processor,
                        confidence=st.session_state.confidence / 100,
                        iou_thres=IOU_THRES,
                        classes_name=CLASSES_NAME_ES
                    )

            # Mostrar imagen original o cartel según detecciones
            mostrar_cartel = False
            if st.session_state.detection_toggle and st.session_state.processed_images:
                for processed in st.session_state.processed_images:
                    if processed['filename'] == selected_image:
                        if len(processed['detections']) > 0:
                            # Mostrar la imagen con los cuadros delimitadores
                            try:
                                original_image = PIL.Image.open(source_img)
                                image_with_boxes = draw_bounding_boxes(
                                    original_image,
                                    processed['det_res'],
                                    processed['classes'],
                                    CLASSES_NAME_ES
                                )
                                st.image(image_with_boxes, use_container_width=True)
                            except Exception as ex:
                                st.error("Ocurrió un error al dibujar las cajas delimitadoras.")
                                st.error(ex)
                            mostrar_cartel = False
                        else:
                            mostrar_cartel = True
                        break
                if mostrar_cartel:
                    status_card("ok", "No se han detectado ulceraciones",
                                "La imagen analizada no presenta signos de úlceras")
            elif not st.session_state.detection_toggle and st.session_state.processed_images:
                status_card("info", "Modelo de detección desactivado",
                            "Activa el modelo de detección para ver las cajas delimitadoras")

            # Mostrar botón de exportación si hay imágenes procesadas
            if st.session_state.processed_images and len(st.session_state.processed_images) == len(st.session_state.uploaded_images):
                export_results(st.session_state.processed_images)

        # Fila inferior: recortes de cada detección dentro de una tarjeta (máximo 3 por fila)
        if st.session_state.processed_images:
            for processed in st.session_state.processed_images:
                if processed['filename'] == selected_image and len(processed['detections']) > 0:
                    detections = processed['detections']
                    with st.container(border=True):
                        if st.session_state.detection_toggle:
                            section_title("Úlceras detectadas", "🔍",
                                          f"{len(detections)} zona(s) clasificada(s) con su mapa de atención")
                        else:
                            section_title("Clasificación de la imagen", "🔍", "Resultado sobre la imagen completa")
                        per_row = 3
                        for start in range(0, len(detections), per_row):
                            cols = st.columns(per_row)
                            for offset, detection in enumerate(detections[start:start + per_row]):
                                idx = start + offset
                                with cols[offset]:
                                    label = f"Detección {idx + 1}: " if st.session_state.detection_toggle else ""
                                    healthy = str(detection['class']).upper() in ("SANO", "NONE")
                                    st.markdown(badge(f"{label}{detection['class']}", ok=healthy), unsafe_allow_html=True)
                                    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
                                    img_array = np.array(detection['original_image'])
                                    ax.imshow(img_array)
                                    if st.session_state.show_attention:
                                        mask = np.ma.masked_where(detection['attention_map'] == 0, detection['attention_map'])
                                        ax.imshow(mask, cmap='Reds', alpha=0.6, vmin=0, vmax=1)
                                    ax.axis('off')
                                    plt.tight_layout(pad=0)
                                    buf = io.BytesIO()
                                    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
                                    plt.close(fig)
                                    buf.seek(0)
                                    st.image(PIL.Image.open(buf), use_container_width=True)
                    break

        # v3.0: segmentación de la imagen completa con máscara superpuesta y área
        if st.session_state.get("show_segmentation", True):
            render_segmentation_panel(source_img)

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
                if st.session_state.detection_toggle:
                    # Si estamos en modo detección, usar los resultados de detección
                    processed['detections'] = cached_results['detections']
                    processed['boxes'] = cached_results['boxes']
                    processed['classes'] = cached_results['classes']
                else:
                    # Si estamos en modo clasificación directa, usar la clasificación de la imagen completa
                    processed['detections'] = [cached_results['full_image_classification']]
                    processed['boxes'] = []
                    processed['classes'] = [cached_results['full_image_class']]

if __name__ == '__main__':
    main()
