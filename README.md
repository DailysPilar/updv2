# Detección y clasificación de úlceras de pie diabético (UPD)

Este proyecto es una aplicación interactiva desarrollada en **Streamlit** que combina un modelo de **detección de objetos (YOLOv8)** con un modelo **Vision Transformer (ViT)** de clasificación, para identificar, localizar y clasificar úlceras de pie diabético (UPD) en imágenes, además de ofrecer explicabilidad visual sobre en qué zonas de la imagen se fijó el modelo para tomar su decisión.

> En la rama `detection` se realiza solamente la detección de las úlceras, y en la rama `master` se realiza la detección y luego la clasificación de cada una.

## Tabla de Contenidos

- [Detección y clasificación de úlceras de pie diabético (UPD)](#detección-y-clasificación-de-úlceras-de-pie-diabético-upd)
  - [Tabla de Contenidos](#tabla-de-contenidos)
  - [Descripción](#descripción)
  - [Cómo funciona el pipeline](#cómo-funciona-el-pipeline)
  - [Características principales](#características-principales)
  - [Estructura del proyecto](#estructura-del-proyecto)
  - [Requisitos previos](#requisitos-previos)
  - [Instalación](#instalación)
  - [Uso](#uso)
  - [Modos de operación](#modos-de-operación)
  - [Exportación de resultados](#exportación-de-resultados)
  - [Notas y solución de problemas](#notas-y-solución-de-problemas)
  - [Contribución](#contribución)
  - [Licencia](#licencia)

## Descripción

Este sistema automatiza el análisis de imágenes de pies con posibles úlceras diabéticas mediante inteligencia artificial, ayudando a profesionales de la salud a agilizar el diagnóstico. El flujo combina:

- **Preprocesamiento** de la imagen cargada por el usuario.
- **Detección** de regiones de interés con un modelo **YOLOv8** (`ultralytics`).
- **Clasificación** de cada región (o de la imagen completa) con un modelo **Vision Transformer (ViT)** alojado en Hugging Face (`daoliver/Vit_upd`).
- **Explicabilidad**: se extraen y combinan los mapas de atención de las cabezas más relevantes del ViT para resaltar visualmente en qué zonas se fijó el modelo al clasificar.

## Cómo funciona el pipeline

1. El usuario carga una o varias imágenes desde la barra lateral.
2. Cada imagen se clasifica primero de forma completa con el ViT (para tener siempre un resultado de respaldo).
3. Si el modo de detección está activo, YOLOv8 localiza las posibles úlceras en la imagen y recorta cada región detectada.
4. Cada recorte se pasa nuevamente por el ViT para clasificarlo individualmente en una de 4 clases: `infección e isquemia`, `infección`, `isquemia` o `sano`.
5. Para cada clasificación se calculan los mapas de atención de las 3 cabezas más relevantes del transformer, se normalizan, se umbralizan y se combinan en un mapa de calor.
6. Los resultados (imagen con cajas delimitadoras, recortes clasificados y mapas de atención superpuestos) se muestran en la interfaz y pueden exportarse.

Los resultados de cada imagen se cachean en la sesión (`classification_cache`) para evitar reprocesar innecesariamente al cambiar de modo o al ajustar la visualización de atención.

## Características principales

- **Doble modo de análisis**: detección + clasificación por regiones, o clasificación directa de la imagen completa (toggle en la barra lateral).
- **Carga múltiple de imágenes** (hasta 20 por sesión), con detección automática de archivos duplicados.
- **Control de confianza** del modelo de detección mediante un slider (0–100%).
- **Visualización de zonas de atención** activable/desactivable, superpuesta como mapa de calor sobre la imagen.
- **Selector de imagen** cuando se cargan varias, para revisar los resultados de cada una individualmente.
- **Interfaz totalmente en español**, incluyendo textos personalizados del cargador de archivos.
- **Exportación de resultados** en un archivo ZIP con las imágenes procesadas (con o sin mapa de atención) y un CSV con las coordenadas de las cajas delimitadoras, la clase y la confianza de cada detección.
- **Manejo de errores** robusto al cargar imágenes, procesar el modelo o generar visualizaciones.

## Estructura del proyecto

```
├── app.py                  # Lógica principal de la interfaz Streamlit
├── helper.py                # Funciones auxiliares: carga del modelo YOLO, recorte de imágenes, dibujo de cajas delimitadoras
├── settings.py               # Rutas de configuración (modelo de detección, etc.)
├── weights/                  # Carpeta con el modelo de detección (det_model.pt) 
├── static/                   # Recursos estáticos (p. ej. fuente roboto.ttf para las cajas delimitadoras)
├── requirements.txt           # Dependencias de Python
├── packages.txt               # Dependencias del sistema (libgl1, necesaria para OpenCV)
├── .streamlit/config.toml      # Configuración de Streamlit (límite de subida, etc.)
└── .devcontainer/devcontainer.json  # Configuración para entorno de desarrollo en contenedor / Codespaces
```

## Requisitos previos

- Python 3.11 (recomendado, ver `.devcontainer/devcontainer.json`).
- El archivo del modelo de detección `det_model.pt` colocado en la carpeta `weights/`.
- Conexión a internet en el primer uso, ya que el modelo de clasificación ViT se descarga automáticamente desde Hugging Face (`daoliver/Vit_upd`).
- La librería del sistema `libgl1` (ver `packages.txt`), requerida por OpenCV en entornos Linux/contenedores.

## Instalación

1. Clona el repositorio:
```bash
git clone https://github.com/DailysPilar/updv2.git
```
2. Navega al directorio del proyecto:
```bash
cd dfu-app
```
3. Instala las dependencias de Python:
```bash
pip install -r requirements.txt
```
4. Coloca el modelo de detección `det_model.pt` dentro de la carpeta `weights/` (créala si no existe).

> **Nota:** en Windows, si al instalar ves un error de red como `IncompleteRead` con algún paquete, intenta:
> ```bash
> pip install --default-timeout=100 --retries 10 --no-cache-dir -r requirements.txt
> ```

## Uso

Ejecuta la aplicación con:
```bash
streamlit run app.py
```

En la interfaz:

1. Sube una o varias imágenes de un pie con posibles úlceras (JPG, JPEG o PNG, máximo 5MB por archivo, hasta 20 imágenes por sesión).
2. Ajusta, si lo deseas, el modo de análisis y el nivel de confianza de detección desde la barra lateral.
3. Presiona **"🔍 Analizar imagen(es)"** para procesar.
4. Revisa los resultados: imagen original, cajas delimitadoras (si aplica), clasificación de cada úlcera detectada y mapas de atención.
5. Exporta los resultados con el botón **"📥 Exportar"**.

## Modos de operación

- **Con detección (por defecto):** YOLOv8 localiza las úlceras y cada una se clasifica por separado. Se muestran las cajas delimitadoras sobre la imagen y un cartel indicando si no se detectaron úlceras.
- **Sin detección:** se omite la localización y se clasifica la imagen completa directamente con el ViT. En este modo el mapa de atención siempre se muestra.

El slider de confianza solo está activo en modo detección, y cualquier cambio en el modo, la confianza o la visualización de atención reprocesa automáticamente las imágenes ya cargadas usando los resultados cacheados cuando es posible.

## Exportación de resultados

Al presionar el botón de exportación se genera un archivo ZIP que contiene:

- Cada imagen procesada (con o sin el mapa de atención superpuesto, según la opción activada), nombrada según el modo:
  - En modo detección: `nombre_1.ext`, `nombre_2.ext`, ... (una por cada úlcera detectada).
  - En modo clasificación directa: `nombre_processed.ext`.
- Un archivo `anotaciones.csv` con:
  - En modo detección: `filename, xmin, ymin, xmax, ymax, class, confidence`.
  - En modo clasificación directa: `filename, class`.

El nombre del ZIP cambia según el modo activo: `updcondetector.zip` o `updsindetector.zip`.

## Notas y solución de problemas

- El modelo de clasificación se descarga la primera vez desde Hugging Face (`daoliver/Vit_upd`); si falla la carga, verifica tu conexión a internet.
- En entornos Linux o contenedores, asegúrate de instalar la dependencia del sistema `libgl1` (incluida en `packages.txt`), necesaria para que OpenCV funcione correctamente.
- El límite de subida de archivos está configurado en 5MB por archivo (`.streamlit/config.toml`), y la interfaz limita a un máximo de 20 imágenes por sesión.

## Contribución

Si deseas contribuir al proyecto, sigue los siguientes pasos:

1. Haz un fork del repositorio.
2. Crea una rama para tu funcionalidad (`git checkout -b nueva-funcionalidad`).
3. Realiza los cambios y haz commit (`git commit -m 'Añadir nueva funcionalidad'`).
4. Sube tu rama (`git push origin nueva-funcionalidad`).
5. Abre un Pull Request.

## Licencia

Este proyecto está licenciado bajo la Licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más detalles.
