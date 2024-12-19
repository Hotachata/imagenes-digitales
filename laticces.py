import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar imagen en escala de grises
gray_img = cv2.imread('uvas.jpg', cv2.IMREAD_GRAYSCALE)

# Verifica si la imagen fue cargada correctamente
if gray_img is None:
    print("Error: No se pudo cargar la imagen. Verifica la ruta.")
else:
    # Mostrar la imagen original
    plt.imshow(gray_img, cmap='gray')
    plt.title("Imagen Original en Escala de Grises")
    plt.axis('off')
    plt.show()
    
    # Elemento estructurante
    kernel = np.ones((5, 5), np.uint8)
    # Dilatación
    dilated = cv2.dilate(gray_img, kernel, iterations=1)
    # Erosión
    eroded = cv2.erode(gray_img, kernel, iterations=1)
    # Mostrar resultados
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1), plt.imshow(dilated, cmap='gray')
    plt.title("Dilatación"), plt.axis('off')
    plt.subplot(1, 2, 2), plt.imshow(eroded, cmap='gray')
    plt.title("Erosión"), plt.axis('off')
    plt.show()

    # Apertura (erosión seguida de dilatación)
    opened = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, kernel)
    # Cierre (dilatación seguida de erosión)
    closed = cv2.morphologyEx(gray_img, cv2.MORPH_CLOSE, kernel)
    # Mostrar resultados
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1), plt.imshow(opened, cmap='gray')
    plt.title("Apertura - Suavizar Imagen"), plt.axis('off')
    plt.subplot(1, 2, 2), plt.imshow(closed, cmap='gray')
    plt.title("Cierre - Eliminar Agujeros"), plt.axis('off')
    plt.show()

    # Frontera en MMB
    # Frontera = Dilatación - Erosión
    border = cv2.subtract(dilated, eroded)
    # Mostrar frontera
    plt.imshow(border, cmap='gray')
    plt.title("Frontera en Escala de Grises")
    plt.axis('off')
    plt.show()
    # Gradiente morfológico en Laticces
    # Aplicar el gradiente morfológico simétrico
    gradiente_morfologico = cv2.morphologyEx(gray_img, cv2.MORPH_GRADIENT, kernel)
    # Mostrar el gradiente morfológico
    plt.imshow(gradiente_morfologico, cmap='gray')
    plt.title('Gradiente Morfológico')
    plt.show()
    # Aplicar el gradiente morfológico por dilatación, completar código...
    # Aplicar el gradiente morfológico por erosión, completar código...


