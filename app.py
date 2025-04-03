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

@st.cache_resource
def load_det_model(model_path):
    """Carga el modelo de detección desde la ruta especificada."""
    return load_pt_model(model_path)

@st.cache_resource
def load_clf_model(model_path):
    """Carga el modelo de clasificación desde la ruta especificada."""
    return load_h5_model(model_path)

def initialize_session() -> None:
    """Inicializa el estado de la sesión de Streamlit."""
    if 'uploaded_images' not in st.session_state:
        st.session_state.uploaded_images = []
    if 'processed_images' not in st.session_state:
        st.session_state.processed_images = []
    if 'confidence' not in st.session_state:
        st.session_state.confidence = 30  # Valor inicial de confianza

def clear_session() -> None:
    """Limpia las imágenes cargadas y procesadas del estado de sesión de Streamlit.

    Esta función restablece las claves 'uploaded_images' y 'processed_images' en 
    el estado de sesión, eliminando cualquier dato de imagen que haya sido subido o procesado.
    """
    if 'uploaded_images' in st.session_state:
        st.session_state.uploaded_images = []   # Limpiar imágenes cargadas
    if 'processed_images' in st.session_state:
        st.session_state.processed_images = []  # Limpiar imágenes procesadas

def style_language_uploader():
    languages = {
        "es": {
            "button": "Cargar archivos",
            "instructions": "Arrastra y suelta archivos aquí",
            "limits": "Límite 5MB por archivo • JPG, JPEG, PNG",
        },
    }

    hide_label = (
        """
        <style>
            div[data-testid="stFileUploader"]>section[data-testid="stFileUploaderDropzone"]>button[data-testid="stBaseButton-secondary"] {
                width: 100%;
                visibility: hidden; /* Oculta el texto original */
                position: relative;
            }
            div[data-testid="stFileUploader"]>section[data-testid="stFileUploaderDropzone"]>button[data-testid="stBaseButton-secondary"]::after {
                content: "BUTTON_TEXT";
                visibility: visible;
                display: block;
                position: absolute;
                width: 100%;
                top: 50%; /* Centra verticalmente */
                left: 50%; /* Centra horizontalmente */
                transform: translate(-50%, -50%); /* Ajusta la posición para centrar */
                font-family: inherit;
                font-size: inherit;
                color: inherit;
                background-color: inherit;
                border: inherit;
                border-radius: 8px;
                padding: 5px 10px;
            }
            div[data-testid="stFileUploaderDropzoneInstructions"]>div>span {
                visibility: hidden;
            }
            div[data-testid="stFileUploaderDropzoneInstructions"]>div>span::after {
                content: "INSTRUCTIONS_TEXT";
                visibility: visible;
                display: block;
            }
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
    Verifica si hay archivos duplicados basados en sus nombres.

    Args:
        uploaded_files (list): Lista de archivos subidos.

    Returns:
        List[str]: Una lista de archivos duplicados.
    """
    seen_files = set()
    duplicate_files = []

    for file in uploaded_files:
        if file.name in seen_files:
            duplicate_files.append(file.name)
        else:
            seen_files.add(file.name)

    return duplicate_files

def process_images(det_model, clf_model, confidence: float, iou_thres: float, classes_name: List[str]) -> None:
    """Realiza la detección y clasificación de úlceras en las imágenes cargadas, almacenando los resultados procesados.

    Args:
        det_model: Modelo de detección YOLO utilizado para detectar úlceras en las imágenes.
        clf_model: Modelo de clasificación utilizado para clasificar las áreas delimitadas por las cajas.
        confidence (float): Nivel de confianza mínimo para que el modelo considere una detección como válida.
        iou_thres (float): Umbral de IoU para aplicar la supresión de no máximos y eliminar detecciones duplicadas.
        classes_name (List[str]): Nombre perteneciente a cada clase.
    """
    for image in st.session_state.uploaded_images:
        # Proceso de detección
        uploaded_image = PIL.Image.open(image)
        det_res = det_model.predict(uploaded_image, conf=confidence, iou=iou_thres)  # Realiza la detección utilizando el modelo
        bboxes = det_res[0].boxes

        # Proceso de clasificación
        classes = []
        cropped_images = crop_images(uploaded_image, bboxes)  # Recorta las regiones de interés
        for cropped_image in cropped_images:
            # Redimensiona y normaliza el recorte
            resized_image = cropped_image.resize((224, 224))
            norm_image_np = np.array(resized_image) / 255.0
            norm_image_np = np.expand_dims(norm_image_np, axis=0)

            # Realiza la predicción y almacena el resultado
            pre = clf_model.predict(norm_image_np)
            pred = np.argmax(pre, axis=1)[0]
            classes.append(pred)

        # Dibujar imagen
        processed_image = draw_bounding_boxes(uploaded_image, det_res, classes, classes_name)

        # Almacena la imagen procesada y las cajas en el estado de la sesión
        st.session_state.processed_images.append({
            'image': processed_image,
            'filename': image.name,
            'boxes': bboxes,
            'classes': classes
        })

def export_results(processed_images: List[Dict[str, Any]]) -> None:
    """Permite descargar las imágenes procesadas y las anotaciones en formato CSV en un archivo ZIP.

    Args:
        processed_images (List[Dict[str, Any]]): Lista de diccionarios donde cada diccionario
            contiene el nombre de archivo y las cajas delimitadoras de una imagen procesada.
    """
    # Crear un archivo ZIP en memoria
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        for processed in processed_images:
            # Guardar cada imagen procesada en el ZIP
            img_buffer = get_image_download_buffer(processed['image']).getvalue()
            zip_file.writestr(processed['filename'], img_buffer)

        # Agregar el archivo CSV al ZIP
        zip_file.writestr('anotaciones.csv', write_csv(processed_images, classes_name))

    # Preparar el archivo ZIP para la descarga
    zip_buffer.seek(0)  # Volver al inicio del buffer
    zip_data = zip_buffer.getvalue()  # Convertir a bytes

    # Agrega un botón para descarga la imagen
    try:
        st.sidebar.download_button(
            use_container_width=True,
            help='Exportar imágenes procesadas y anotaciones',
            label="Exportar",
            data=zip_data,
            file_name="upd.zip",
            mime="application/zip"
        )
    except Exception as ex:
        st.error("¡No se ha subido ninguna imagen aún!")
        st.error(ex)

if __name__ == '__main__':
    # Constantes
    iou_thres = 0.5  # NMS
    classes_name = ['both', 'infection', 'ischaemia', 'none']  # Clases de las úlceras
    classes_name_es = ['INFECCIÓN E ISQUEMIA', 'INFECCIÓN', 'ISQUEMIA', 'SANO']  # Clases de las úlceras en español

    # Configuración del diseño de la página
    st.set_page_config(
        page_title="UPD",
        page_icon="🦶",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Título de la página principal
    st.title("Cuidado inteligente del pie diabético")

    #Inicializar estado de la sesión
    initialize_session()

    # Título de la barra lateral
    st.sidebar.header("Configuración del modelo")

    # Control deslizante para la confianza del modelo
    confidence = st.sidebar.slider( 
        label="Seleccionar confianza de detección",
        min_value=0,
        max_value=100, 
        value=st.session_state.confidence,
        help='Probabilidad de certeza en la detección de la úlcera'
    )

    # Revisa si ha cambiado el valor y ejecuta clear_session
    if confidence != st.session_state.confidence:
        st.session_state.confidence = confidence
        clear_session()

    # Cargador de archivos para seleccionar imágenes
    source_imgs = st.sidebar.file_uploader(
        label="Seleccionar imágenes",
        help='Imagen del pie que desea analizar',
        type=("jpg", "jpeg", "png"),
        accept_multiple_files=True)

    # Cambiar lenguaje de la app
    style_language_uploader()

    # Verificar que no se haya superado el límite de imágenes cargadas
    if len(source_imgs) > 20:
        st.sidebar.error('Ha superado el límite de imágenes')

    # Verificar que no existan imágenes duplicadas
    duplicate_files = check_duplicates(source_imgs)
    if duplicate_files:
        if len(duplicate_files) == 1:
            st.sidebar.error(f"El siguiente archivo está duplicado: {duplicate_files[0]}")
        else:
            st.sidebar.error(f"Los siguientes archivos están duplicados: {', '.join(duplicate_files)}")

    # Botón para analizar las imágenes, mostrar solo cuando se carguen las imágenes
    if len(source_imgs) != 0 and len(source_imgs) <= 20 and not duplicate_files:
        text_btn = 'Analizar imágenes' if len(source_imgs) > 1 else 'Analizar imagen'
        process_image_button = st.sidebar.button(  # Botón para iniciar la detección
            label=text_btn, 
            use_container_width=True,
            help='Iniciar procesamiento de las imágenes cargadas')

    # Cargar los modelos
    try:
        det_model = load_det_model(Path(settings.DETECTION_MODEL))
        clf_model = load_clf_model(Path(settings.CLASS_MODEL))
    except Exception as ex:
        st.error("No se pudo cargar el modelo. Verifique la ruta especificada")
        st.error(ex)

    # Verificar si la imagen original ha cambiado
    if 'uploaded_images' in st.session_state:
        if source_imgs is not None and st.session_state.uploaded_images != source_imgs:
            clear_session()  # Limpia el estado de la sesión

    if len(source_imgs) != 0:
        st.session_state.uploaded_images = source_imgs

        # Usar un selector para elegir la imagen a mostrar
        if len(st.session_state.uploaded_images) > 1:
            image_filenames = [img.name for img in st.session_state.uploaded_images]
            selected_image = st.selectbox("Selecciona la imagen que desea visualizar:", image_filenames)

            # Mostrar la imagen original correspondiente
            original_image_index = image_filenames.index(selected_image)
            source_img = source_imgs[original_image_index]
        else:
            selected_image = source_imgs[0].name
            source_img = source_imgs[0]

        col1, col2 = st.columns(2)   # Crear dos columnas

        # Crear columnas para mostrar las imágenes
        with col1:
            try:
                # Abrir y mostrar la imagen subida por el usuario
                st.image(source_img, caption="Imagen original", use_column_width='auto')
            except Exception as ex:
                st.error("Ocurrió un error al abrir la imagen.")
                st.error(ex)

        if len(source_imgs) <= 20 and not duplicate_files:
            with col2:
                # Procesar imágenes al presionar el botón
                if process_image_button:
                    st.session_state.processed_images = []  # Limpiar imágenes procesadas
                    process_images(det_model=det_model,
                                clf_model=clf_model,
                                confidence=st.session_state.confidence/100,
                                iou_thres=iou_thres,
                                classes_name=classes_name_es)

                # Mostrar imágenes procesadas
                for processed in st.session_state.processed_images:
                    if processed['filename'] == selected_image:
                        st.image(processed['image'], caption='Ulceraciones detectadas', use_column_width='auto')

                # Si las imágenes procesadas presentan úlceras mostrar botón para exportar
                if len(st.session_state.processed_images) == len(st.session_state.uploaded_images):  # Verificar que se procesen todas
                    # Verifica si alguna imagen procesada tiene cajas
                    if any(len(p['boxes']) > 0 for p in st.session_state.processed_images):
                        export_results(st.session_state.processed_images)
                    else:
                        st.info('No se han detectado ulceraciones', icon="ℹ️")
    else:
        camera_svg = '''
            <svg xmlns="http://www.w3.org/2000/svg" fill="gray" viewBox="0 0 24 24" width="24" height="24">
                <circle cx="16" cy="8.011" r="2.5"/><path d="M23,16a1,1,0,0,0-1,1v2a3,3,0,0,1-3,3H17a1,1,0,0,0,0,2h2a5.006,5.006,0,0,0,5-5V17A1,1,0,0,0,23,16Z"/><path d="M1,8A1,1,0,0,0,2,7V5A3,3,0,0,1,5,2H7A1,1,0,0,0,7,0H5A5.006,5.006,0,0,0,0,5V7A1,1,0,0,0,1,8Z"/><path d="M7,22H5a3,3,0,0,1-3-3V17a1,1,0,0,0-2,0v2a5.006,5.006,0,0,0,5,5H7a1,1,0,0,0,0-2Z"/><path d="M19,0H17a1,1,0,0,0,0,2h2a3,3,0,0,1,3,3V7a1,1,0,0,0,2,0V5A5.006,5.006,0,0,0,19,0Z"/><path d="M18.707,17.293,11.121,9.707a3,3,0,0,0-4.242,0L4.586,12A2,2,0,0,0,4,13.414V16a3,3,0,0,0,3,3H18a1,1,0,0,0,.707-1.707Z"/>
            </svg>'''

        # Mostrar marco temporal hasta que no se seleccionen las imágenes
        with st.container(border=True):
            st.markdown(
                f"<div style='font-size: 16px; display: flex; justify-content: center; align-items: center; padding: 0 0 10px 0; gap: 15px; border-radius: 8px;'>"
                    f"{camera_svg}"
                    "No ha seleccionado una imagen para su procesamiento"
                "</div>",
                unsafe_allow_html=True
            )