import streamlit as st
from helper import load_pt_model, get_image_download_buffer, draw_bounding_boxes, crop_images
from keras.api.models import load_model as load_h5_model
from pathlib import Path
import numpy as np
import PIL
import settings
import zipfile
import io
import csv
from typing import List, Dict, Any
from PIL import Image
import base64
import time
from tqdm import tqdm 
from streamlit.components.v1 import html as html_component
import torch  # Agregar esta importación
from transformers import ViTForImageClassification, ViTImageProcessor

# Constantes globales de la aplicación
IOU_THRES = 0.5  # Umbral de IoU para la supresión de no máximos
CLASSES_NAME = ['both', 'infection', 'ischaemia', 'none']  # Nombres de clases en inglés
CLASSES_NAME_ES = ['INFECCIÓN E ISQUEMIA', 'INFECCIÓN', 'ISQUEMIA', 'SANO']  # Nombres de clases en español
# Decorador para cachear el modelo de detección y evitar recargas innecesarias
@st.cache_resource
def load_det_model(model_path):
    """
    Carga el modelo de detección YOLO desde la ruta especificada.
    
    Args:
        model_path: Ruta al archivo del modelo de detección
        
    Returns:
        Modelo de detección YOLO cargado
    """
    return load_pt_model(model_path)

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

def clear_session() -> None:
    """
    Limpia el estado de la sesión, eliminando las imágenes cargadas y procesadas.
    Mantiene el estado del cargador de archivos y la confianza del modelo.
    """
    # Eliminar las imágenes subidas
    if 'uploaded_images' in st.session_state:
        del st.session_state.uploaded_images
    
    # Eliminar las imágenes procesadas
    if 'processed_images' in st.session_state:
        del st.session_state.processed_images
    
    # Incrementar la clave del cargador para forzar su reinicialización
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
    # Escribir la cabecera del CSV
    csv_writer.writerow(['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'class'])

    for img in processed_images:
        # Procesar cada caja delimitadora y escribir las coordenadas redondeadas
        for box, clf in zip(img['boxes'], img['classes']):
            xmin, ymin, xmax, ymax = [round(coord.item(), 2) for coord in box.xyxy[0]]
            csv_writer.writerow([img['filename'], xmin, ymin, xmax, ymax, classes_name[clf]])

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

# Importar las librerías necesarias para el modelo ViT
from transformers import ViTForImageClassification, ViTImageProcessor

# Cargar el modelo y el procesador
model_path = 'weights/vit3con32'
processor = ViTImageProcessor.from_pretrained(model_path)
clf_model = ViTForImageClassification.from_pretrained(
    model_path,
    num_labels=len(CLASSES_NAME),
    ignore_mismatched_sizes=True
)

def process_images(det_model, clf_model, confidence: float, iou_thres: float, classes_name: List[str]) -> None:
    """
    Procesa las imágenes usando los modelos de detección y clasificación.
    """
    for image in st.session_state.uploaded_images:
        # Abrir la imagen
        uploaded_image = PIL.Image.open(image)
        
        # Detectar úlceras en la imagen
        det_res = det_model.predict(uploaded_image, conf=confidence, iou=iou_thres)
        bboxes = det_res[0].boxes

        # Clasificar cada detección
        classes = []
        cropped_images = crop_images(uploaded_image, bboxes)
        clf_model.eval()
        
        detections = []
        
        with torch.no_grad():
            for cropped_image in cropped_images:
                # Redimensionar la imagen
                resized_image = cropped_image.resize((224, 224))
                
                # Preprocesar la imagen
                inputs = processor(images=resized_image, return_tensors="pt")
                
                # Realizar la predicción
                outputs = clf_model(**inputs)
                logits = outputs.logits
                predicted_class_idx = logits.argmax(-1).item()
                classes.append(predicted_class_idx)
                
                detections.append({
                    'image': cropped_image,
                    'class': CLASSES_NAME_ES[predicted_class_idx]
                })

        # Almacenar los resultados
        st.session_state.processed_images.append({
            'image': uploaded_image,
            'filename': image.name,
            'boxes': bboxes,
            'classes': classes,
            'detections': detections
        })

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
            # Convertir la imagen PIL a bytes
            img_buffer = io.BytesIO()
            processed['image'].save(img_buffer, format='JPEG')
            img_buffer.seek(0)
            
            # Guardar en el ZIP
            zip_file.writestr(processed['filename'], img_buffer.getvalue())

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

def main():
    """
    Función principal que ejecuta la aplicación Streamlit.
    Configura la interfaz y maneja la lógica principal de la aplicación.
    """
    # Configuración de la página
    image_path = Path(__file__).parent / "huellas-humanas.png"
    st.set_page_config(
        page_title="UPD - Úlceras de Pie Diabético",
        page_icon=str(image_path),
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Mostrar logo y título
    try:
        base64_logo = image_to_base64(image_path)
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
        .block-container..st-emotion-cache-1jicfl2 {{
            padding: 2rem 1rem 10rem 5rem 5rem;
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
    
    # Inicializar el estado de la sesión y variables
    initialize_session()
    source_imgs = []  # Lista para almacenar las imágenes subidas

    # Configuración del sidebar
    with st.sidebar:
        # Título de la barra lateral
        st.header("⚙️ Configuración del modelo")

        # Control deslizante para ajustar la confianza del modelo
        confidence = st.slider( 
            label="📍 Seleccionar confianza de detección",
            min_value=0,
            max_value=100, 
            value=st.session_state.confidence,
            help='Probabilidad de certeza en la detección de la úlcera'
        )

        # Actualizar la confianza si ha cambiado
        if confidence != st.session_state.confidence:
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

    # Cargar los modelos globalmente
    try:
        det_model = load_det_model(Path(settings.DETECTION_MODEL))
        model_path = 'weights/vit3con32'
        processor = ViTImageProcessor.from_pretrained(model_path)
        clf_model = ViTForImageClassification.from_pretrained(
            model_path,
            num_labels=len(CLASSES_NAME),
            ignore_mismatched_sizes=True
        )
    except Exception as ex:
        st.error("No se pudo cargar el modelo. Verifique la ruta especificada")
        st.error(ex)

    # Procesar y mostrar las imágenes
    if source_imgs and len(source_imgs) != 0:
        st.session_state.uploaded_images = source_imgs

        # Selector de imagen para visualización
        if len(st.session_state.uploaded_images) > 1:
            image_filenames = [img.name for img in st.session_state.uploaded_images]
            selected_image = st.selectbox("Selecciona la imagen que desea visualizar:", image_filenames)
            original_image_index = image_filenames.index(selected_image)
            source_img = source_imgs[original_image_index]
        else:
            selected_image = source_imgs[0].name
            source_img = source_imgs[0]

        # Mostrar imágenes en columnas
        col1, col2 = st.columns([1,1], gap="medium")
        with col1:
            try:
                st.markdown("### Imagen Original")
                with st.container(border=True):
                    st.markdown("""
                        <style>
                            .stImage img {
                                max-height: 400px !important;
                                height: auto !important;
                                width: 100% !important;
                                object-fit: contain !important;
                            }
                            .st-emotion-cache-1kyxreq {
                                width: 100% !important;
                                margin: 0 !important;
                                padding: 0 !important;
                            }
                            .st-emotion-cache-1v0mbdj {
                                width: 100% !important;
                                margin: 0 !important;
                                padding: 0 !important;
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
                st.session_state.processed_images = []
                process_images(
                    det_model=det_model,
                    clf_model=clf_model,
                    confidence=st.session_state.confidence/100,
                    iou_thres=IOU_THRES,
                    classes_name=CLASSES_NAME_ES
                )

            # Mostrar imágenes procesadas
            if st.session_state.processed_images:
                for processed in st.session_state.processed_images:
                    if processed['filename'] == selected_image:
                        if len(processed['detections']) > 0:
                            st.markdown("### Úlceras Detectadas")
                            
                            # Contenedor con borde para las detecciones
                            with st.container(border=True):
                                # Mostrar cada detección verticalmente
                                for idx, detection in enumerate(processed['detections']):
                                    # Contenedor para la imagen
                                    st.image(
                                        detection['image'],
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
                                            width: 100%;
                                            box-sizing: border-box;
                                        ">
                                            Detección #{idx + 1}: {detection['class']}
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )
                        else:
                            st.info('No se han detectado ulceraciones', icon="ℹ️")

            # Mostrar botón de exportación si hay detecciones
            if st.session_state.processed_images and len(st.session_state.processed_images) == len(st.session_state.uploaded_images):
                if any(len(p['boxes']) > 0 for p in st.session_state.processed_images):
                    export_results(st.session_state.processed_images)
                else:
                    st.info('No se han detectado ulceraciones', icon="ℹ️")

if __name__ == '__main__':
    main()