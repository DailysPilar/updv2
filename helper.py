import os
from ultralytics import YOLO
from io import BytesIO
import cv2
import PIL
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List

def load_pt_model(model_path: str) -> YOLO:
    """
    Carga un modelo de detección de objetos YOLO desde la ruta especificada.

    Args:
        model_path (str): La ruta al archivo del modelo YOLO.

    Returns:
        YOLO: El modelo de detección de objetos YOLO cargado.
    """
    return YOLO(model_path)

def crop_images(image: Image, bboxes: List) -> List:
    """Recorta las regiones de la imagen según las cajas delimitadoras.

    Args:
        image (Image): Imagen original en formato PIL.
        bboxes (List): Lista de cajas delimitadoras, donde cada caja es una tupla (xmin, ymin, xmax, ymax).

    Returns:
        List: Lista de imágenes recortadas correspondientes a cada caja delimitadora.
    """
    cropped_images = []
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = map(int, bbox.xyxy[0])  # Convertir coordenadas a enteros
        cropped_image = image.crop((xmin, ymin, xmax, ymax))  # Recorta la imagen según la caja delimitadora
        cropped_images.append(cropped_image)  # Agrega el recorte a la lista
    return cropped_images

def get_image_download_buffer(img_array: np.ndarray) -> BytesIO:
    """
    Convierte un array de imagen en un buffer de bytes descargable en formato JPEG.

    Args:
        img_array (numpy.ndarray): El array de la imagen a convertir.

    Returns:
        BytesIO: Un buffer de bytes de la imagen en formato JPEG, listo para descargar.
    """
    # Convierte el array de la imagen en una imagen PIL
    img = PIL.Image.fromarray(img_array)

    # Crea un objeto BytesIO que actuará como un buffer en memoria para la imagen
    buffered = BytesIO()

    # Guarda la imagen en el buffer en formato JPEG
    img.save(buffered, format="JPEG")

    # Retorna el puntero del buffer al inicio para asegurar que se pueda leer desde el principio
    buffered.seek(0)
    return buffered

def draw_bounding_boxes(image, det_results, classes, classes_name):
    """
    Dibuja los cuadros delimitadores en la imagen según los resultados de la predicción.

    Args:
        image: La imagen original en formato RGB en la que se dibujarán los cuadros.
        det_results (List): Resultados de la predicción del modelo YOLO, incluyendo las cajas, clases y confianza.
        classes (List[int]): Resultados de la clasificación de cada caja delimitadora.
        classes_name (List[str]): Nombre perteneciente a cada clase.

    Returns:
        numpy.ndarray: La imagen con los cuadros delimitadores dibujados, en formato RGB.
    """
    color = (124, 80, 0)  # Color de las cajas en formato RGB
    text_color = (255, 255, 255)  # Color del texto (blanco)

    # Convertir la imagen de OpenCV (NumPy) a PIL para poder dibujar con Pillow
    pil_image = Image.fromarray(cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    # Obtener el tamaño de la imagen en PIL (width, height)
    width, height = pil_image.size

    # Ajustar tamaño de fuente y grosor de línea en función de la resolución de la imagen
    base_font_size = 20
    base_line_width = 4
    base_baseline = 3
    scale_factor = max(width, height) / 1000  # Ajustar este valor según el tamaño de imagen
    font_size = int(base_font_size * scale_factor)
    line_width = int(base_line_width * scale_factor)
    baseline = int(base_baseline * scale_factor)

    # Cargar la fuente, ajustando el tamaño
    try:
        # Ruta absoluta al archivo de fuente
        base_dir = os.path.dirname(os.path.abspath(__file__))
        font_path = os.path.join(base_dir, "static/roboto.ttf")
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()

    # Iterar sobre cada resultado de la predicción
    for det_result, clf in zip(det_results, classes):
        boxes = det_result.boxes  # Cuadros delimitadores
        for box in boxes:
            xmin, ymin, xmax, ymax = box.xyxy[0]
            conf = box.conf[0]

            # Definir el texto con la etiqueta y la confianza
            label_clf = f'{classes_name[clf]}'

            # Medir el tamaño del texto para calcular el fondo
            text_bbox = draw.textbbox((xmin, ymin), label_clf, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            # Posición del rectángulo de fondo
            text_position = (xmin + baseline, ymin - text_height - baseline)
            background_position = [
                text_position[0] - baseline, text_position[1],
                text_position[0] + text_width + baseline, text_position[1] + text_height + baseline
            ]

            # Dibujar el cuadro delimitador del objeto
            draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=line_width)

            # Dibujar el rectángulo de fondo para el texto
            draw.rectangle(background_position, fill=color)

            # Dibujar el texto (etiqueta + confianza) en varias posiciones para simular negrita
            offsets = [(0, 0), (1, 0), (0, 1), (1, 1)]
            for offset in offsets:
                draw.text((text_position[0] + offset[0], text_position[1] + offset[1]), label_clf, font=font, fill=text_color)

    # Convertir la imagen de vuelta a formato OpenCV
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)