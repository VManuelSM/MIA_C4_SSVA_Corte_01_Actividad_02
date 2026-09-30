"""Métricas para comparar los filtros con números y no sólo a simple vista.

Un paso bajo siempre intercambia dos cosas: elimina ruido y elimina detalle.
Por eso se miden ambas por separado:

* **fidelidad** frente a una referencia limpia (MSE, PSNR, SSIM);
* **nitidez** de la imagen por sí misma (energía de gradiente, varianza del
  laplaciano, ancho de un borde).

Todas las métricas aceptan un `margen` para descartar el perímetro. En los
bordes de la imagen el vecindario se completa por reflexión y el resultado
depende de esa convención, no del filtro; con 2 píxeles basta para el núcleo
5×5.
"""

from __future__ import annotations

import cv2
import numpy as np


def interior(imagen: np.ndarray, margen: int) -> np.ndarray:
    """Recorta `margen` píxeles de cada lado."""
    if margen == 0:
        return imagen
    return imagen[margen:-margen, margen:-margen]


def mse(referencia: np.ndarray, prueba: np.ndarray, margen: int = 0) -> float:
    """Error cuadrático medio entre dos imágenes."""
    a = interior(referencia, margen).astype(np.float64)
    b = interior(prueba, margen).astype(np.float64)
    return float(np.mean((a - b) ** 2))


def psnr(referencia: np.ndarray, prueba: np.ndarray, margen: int = 0, pico: float = 255.0) -> float:
    """Relación señal a ruido de pico, en decibeles.

        PSNR = 10 · log₁₀(pico² / MSE)

    Es el MSE en escala logarítmica. Una mejora de 3 dB equivale a reducir el
    error cuadrático a la mitad; de 6 dB, a la cuarta parte.
    """
    error = mse(referencia, prueba, margen)
    return float("inf") if error == 0 else float(10.0 * np.log10(pico**2 / error))


def mapa_ssim(referencia: np.ndarray, prueba: np.ndarray) -> np.ndarray:
    """SSIM local de cada ventana (Wang, Bovik, Sheikh y Simoncelli, 2004).

    El MSE trata igual a un error que desplaza el brillo de toda una región
    y a uno que borra una textura; el ojo no. El SSIM compara, en ventanas
    locales, tres cosas por separado —luminancia media, contraste y
    correlación estructural—. Vale 1 donde las dos imágenes son idénticas.

    Se implementa con la configuración del artículo original: ventana
    gaussiana 11×11 con σ = 1.5, K₁ = 0.01, K₂ = 0.03 y rango dinámico 255.
    """
    a = referencia.astype(np.float64)
    b = prueba.astype(np.float64)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    def ventana(m: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(m, (11, 11), 1.5, borderType=cv2.BORDER_REFLECT_101)

    mu_a, mu_b = ventana(a), ventana(b)
    var_a = ventana(a * a) - mu_a**2
    var_b = ventana(b * b) - mu_b**2
    cov = ventana(a * b) - mu_a * mu_b
    return ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2))


def ssim(referencia: np.ndarray, prueba: np.ndarray, margen: int = 0) -> float:
    """SSIM global: el promedio del mapa local sobre la imagen.

    Cada ventana pesa lo mismo en el promedio, tenga o no estructura. En una
    imagen con grandes zonas lisas, esas zonas dominan el resultado.
    Las ventanas que tocan el perímetro se descartan, como en la
    implementación de referencia.
    """
    mapa = mapa_ssim(interior(referencia, margen), interior(prueba, margen))
    return float(interior(mapa, 5).mean())


def desviacion_local(imagen: np.ndarray) -> np.ndarray:
    """Desviación estándar de la ventana gaussiana 11×11 que usa el SSIM.

    Sirve para clasificar cada ventana como lisa o con estructura según el
    contenido de la imagen limpia.
    """
    f = imagen.astype(np.float64)

    def ventana(m: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(m, (11, 11), 1.5, borderType=cv2.BORDER_REFLECT_101)

    return np.sqrt(np.maximum(ventana(f * f) - ventana(f) ** 2, 0.0))


def energia_de_gradiente(imagen: np.ndarray, margen: int = 0) -> float:
    """Magnitud media del gradiente de Sobel: cuánto contraste hay en los bordes.

    Un paso bajo la reduce siempre, porque suaviza las transiciones; cuánto
    la reduce es una medida directa del detalle que se pierde.
    """
    f = imagen.astype(np.float64)
    gx = cv2.Sobel(f, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_64F, 0, 1, ksize=3)
    return float(interior(np.hypot(gx, gy), margen + 1).mean())


def varianza_del_laplaciano(imagen: np.ndarray, margen: int = 0) -> float:
    """Varianza de la respuesta al laplaciano, la medida clásica de enfoque.

    El laplaciano es un filtro paso alto de segundo orden: responde a la
    textura más fina y a los cambios bruscos. Su varianza cae rápidamente en
    cuanto la imagen se desenfoca, y lo hace con más sensibilidad que la
    energía de gradiente a las frecuencias más altas.
    """
    lap = cv2.Laplacian(imagen.astype(np.float64), cv2.CV_64F, ksize=1)
    return float(interior(lap, margen + 1).var())


def ancho_de_transicion(perfil: np.ndarray, bajo: float = 0.1, alto: float = 0.9) -> float:
    """Distancia, en píxeles, que tarda un perfil de borde en subir del 10 % al 90 %.

    Se normaliza el perfil a [0, 1] entre sus valores extremos y se localizan
    por interpolación lineal los cruces de ambos niveles. Un borde ideal
    tiene ancho de transición menor que un píxel; cada filtro lo ensancha.
    """
    p = perfil.astype(np.float64)
    p = (p - p.min()) / (p.max() - p.min())
    posiciones = np.arange(p.size, dtype=np.float64)
    # El perfil es monótono creciente a través del borde, así que `np.interp`
    # sobre (valor → posición) da directamente el punto de cruce.
    return float(np.interp(alto, p, posiciones) - np.interp(bajo, p, posiciones))
