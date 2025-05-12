from pathlib import Path
import sys

# Obtiene la ruta absoluta del archivo actual
FILE = Path(__file__).resolve()
# Obtiene el directorio padre del archivo actual
ROOT = FILE.parent
# Agrega la ruta raíz a la lista sys.path si no está ya presente
if ROOT not in sys.path:
    sys.path.append(str(ROOT))
# Obtiene la ruta relativa del directorio raíz con respecto al directorio de trabajo actual
ROOT = ROOT.relative_to(Path.cwd())

# Directorio donde se almacenan los modelos
MODEL_DIR = ROOT / 'weights'
DETECTION_MODEL = MODEL_DIR / 'det_model.pt'
CLASS_MODEL = MODEL_DIR / 'vit3con32'
