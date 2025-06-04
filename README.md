# Detección y clasificación de úlceras de pie diabético

Este proyecto utiliza modelos de inteligencia artificial para detectar y clasificar úlceras del pie diabético (UPD) a través de una interfaz interactiva desarrollada en **Streamlit**. Los usuarios pueden cargar imágenes y recibir predicciones visuales de las úlceras como la localización y clasificación de las mismas.

> En la rama `detection` se realiza solamente la detección de las úlceras, y en la rama `master` se realiza la detección y luego la clasificación de cada una.

## Tabla de Contenidos

- [Detección y clasificación de úlceras de pie diabético](#detección-y-clasificación-de-úlceras-de-pie-diabético)
  - [Tabla de Contenidos](#tabla-de-contenidos)
  - [Descripción](#descripción)
  - [Instalación](#instalación)
  - [Uso](#uso)
  - [Contribución](#contribución)
  - [Licencia](#licencia)

## Descripción
Este proyecto es un sistema de detección automática de úlceras del pie diabético (UPD) utilizando modelos de inteligencia artificial basados en _YOLOv8_ y _Vision_Transformer_ViT_. El sistema permite identificar y etiquetar úlceras en imágenes, ayudando a los profesionales de la salud a diagnosticar de manera más rápida y precisa.

## Instalación
1. Clona el repositorio:
```bash
git clone https://github.com/DailysPilar/updv2.git
```
2. Navega al directorio del proyecto:
```bash
cd dfu-app
```
3. Instala las dependencias:
```bash
pip install -r requirements.txt
```

## Uso
Para ejecutar la aplicación de Streamlit y cargar imágenes para la detección y clasificación de úlceras, sigue estos pasos:

Ejecuta la aplicación:
```bash
streamlit run app.py
```
En la interfaz, sube una imagen de un pie diabético con posibles úlceras. El modelo analizará la imagen y mostrará:

- Las cajas delimitadoras alrededor de las úlceras detectadas.
- Sobre las cajas se muestra la clasificación correspondiente de cada úlceras.

## Contribución
Si deseas contribuir al proyecto, sigue los siguientes pasos:

1. Haz un fork del repositorio.
2. Crea una rama para tu funcionalidad (``git checkout -b nueva-funcionalidad``).
3. Realiza los cambios y haz commit (``git commit -m 'Añadir nueva funcionalidad'``).
4. Sube tu rama (``git push origin nueva-funcionalidad``).
5. Abre un Pull Request.

## Licencia
Este proyecto está licenciado bajo la Licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más detalles.

