import os
import numpy as np
import cv2
import torch
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from transformers import ViTForImageClassification, ViTImageProcessor
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torch.nn.functional import interpolate
from collections import defaultdict

# 1. Configuración inicial
nombre_clases = ['both', 'infection', 'ischaemia', 'none']
model_path = '../Entrenamiento/Transformer/modeloViTbase3con32/vit3con32'
# 2. Cargar modelo y procesador
processor = ViTImageProcessor.from_pretrained(model_path)
model = ViTForImageClassification.from_pretrained(
    model_path,
    num_labels=len(nombre_clases),
    ignore_mismatched_sizes=True,
    output_attentions=True
)

# 3. Función para cargar imágenes
def load_images_with_labels(dataset_path, class_order):
    images = []
    true_labels = []
    image_paths = []
    
    for label, class_name in enumerate(class_order):
        class_dir = os.path.join(dataset_path, class_name)
        for img_file in os.listdir(class_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_dir, img_file)
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (224, 224))
                images.append(image)
                true_labels.append(label)
                image_paths.append(img_path)
    return images, true_labels, image_paths

# 4. Función para calcular importancia de cabezas
def compute_head_importance(images, labels, model, processor):
    num_heads = model.config.num_attention_heads
    importance = {cls: np.zeros(num_heads) for cls in nombre_clases}
    counts = {cls: 0 for cls in nombre_clases}
    
    for image, label in zip(images, labels):
        class_name = nombre_clases[label]
        counts[class_name] += 1
        
        with torch.no_grad():
            inputs = processor(images=Image.fromarray(image), return_tensors="pt")
            outputs = model(**inputs)
        
        # Atención de la última capa
        attentions = outputs.attentions[-1][0]  # [num_heads, seq_len, seq_len]
        cls_attention = attentions[:, 0, 1:]    # [num_heads, num_patches]
        
        importance[class_name] += cls_attention.mean(dim=1).numpy()
    
    # Normalizar
    for cls in nombre_clases:
        if counts[cls] > 0:
            importance[cls] /= counts[cls]
    
    return importance

# 5. Función para obtener mapas de atención
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

# 6. Visualización mejorada
def visualize_attention(image, true_label, predicted_label, model, processor, optimal_heads, save_path=None):
    attention_maps = []
    
    fig = plt.figure(figsize=(10, 5))
    gs = plt.GridSpec(1, 2, figure=fig)
    
    # Imagen original
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image)
    ax1.set_title(f"Original\nClase: {nombre_clases[true_label]}")
    ax1.axis('off')
    
    # Generar mapas de atención y combinarlos
    for head in optimal_heads[:3]:
        attention = get_attention_map(image, model, processor, head_idx=head)
        threshold = np.percentile(attention, 70)
        attention_binary = np.where(attention > threshold, 1, 0)
        attention_maps.append(attention_binary)
    
    # Mapa combinado
    combined_attention = np.zeros_like(attention_maps[0])
    for attention_map in attention_maps:
        combined_attention = np.logical_or(combined_attention, attention_map).astype(float)
    
    # Mostrar mapa combinado
    ax_combined = fig.add_subplot(gs[0, 1])
    ax_combined.imshow(image)
    mask = np.ma.masked_where(combined_attention == 0, combined_attention)
    ax_combined.imshow(mask, cmap='RdBu_r', alpha=0.7)
    ax_combined.set_title(f"Mapa de Atención\nPredicción: {nombre_clases[predicted_label]}")
    ax_combined.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()

# 7. Cargar datos
test_dir = "../datasethospital/"
test_images, true_labels, image_paths = load_images_with_labels(test_dir, nombre_clases)

# 8. Calcular importancia de cabezas
print("Calculando importancia de cabezas de atención...")
head_importance = compute_head_importance(test_images, true_labels, model, processor)

# Determinar mejores cabezas por clase
optimal_heads = {}
for class_name in nombre_clases:
    top_heads = np.argsort(head_importance[class_name])[-3:][::-1]
    optimal_heads[class_name] = top_heads.tolist()


# 9. Procesar todas las imágenes
output_dir = "resultados_atencion3/"
os.makedirs(output_dir, exist_ok=True)

model.eval()
predicted_labels = []

print(f"\nProcesando {len(test_images)} imágenes...")
for idx, (image, img_path) in enumerate(zip(test_images, image_paths)):
     # Predicción
    with torch.no_grad():
        inputs = processor(images=Image.fromarray(image), return_tensors="pt")
        outputs = model(**inputs)
        predicted_class_idx = outputs.logits.argmax(-1).item()
        predicted_labels.append(predicted_class_idx)
    
    # Visualización
    true_class = nombre_clases[true_labels[idx]]
    save_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(img_path))[0]}_atencion.png")
    visualize_attention(
        image, 
        true_labels[idx], 
        predicted_class_idx, 
        model, 
        processor,
        optimal_heads[true_class],
        save_path
    )
    
    if (idx + 1) % 10 == 0:
        print(f"Procesadas {idx + 1}/{len(test_images)} imágenes")

# 10. Evaluación completa
print("\n--- Métricas de Evaluación ---")
print("Accuracy:", accuracy_score(true_labels, predicted_labels))
print("\nClassification Report:")
print(classification_report(true_labels, predicted_labels, target_names=nombre_clases))

# Matriz de confusión
cm = confusion_matrix(true_labels, predicted_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=nombre_clases, yticklabels=nombre_clases,
            linewidths=0.5, linecolor='gray')
plt.xlabel("Predicción", fontsize=12)
plt.ylabel("Real", fontsize=12)
plt.title("Matriz de Confusión Completa", fontsize=14)
plt.savefig(os.path.join(output_dir, "matriz_confusion.png"), bbox_inches='tight', dpi=150)
plt.show()

print(f"\nProceso completado. Resultados guardados en: {output_dir}")

