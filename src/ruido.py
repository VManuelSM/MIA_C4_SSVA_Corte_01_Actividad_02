"""Modelos de ruido para evaluar el filtrado.

Todos los generadores reciben una semilla explícita. Sin ella, cada ejecución
del cuaderno produciría un ruido distinto y las cifras del reporte no podrían
reproducirse.
"""

from __future__ import annotations

import numpy as np


def campo_gaussiano(forma: tuple[int, ...], sigma: float, semilla: int = 0) -> np.ndarray:
    """Ruido blanco gaussiano de media cero, en punto flotante.

    Blanco significa independiente entre píxeles: su espectro es plano y
    contiene todas las frecuencias por igual, hasta Nyquist. Es el caso en que
    un paso bajo tiene algo que ganar, porque la mayor parte de la señal útil
    de una fotografía vive en frecuencias bajas y el ruido se reparte en todas.
    """
    return np.random.default_rng(semilla).normal(0.0, sigma, size=forma)


def gaussiano(imagen: np.ndarray, sigma: float, semilla: int = 0) -> np.ndarray:
    """Imagen con ruido gaussiano aditivo, redondeada y recortada a `uint8`.

    Es lo que entregaría una cámara: el ruido se suma a la señal y el
    resultado se cuantiza a ocho bits. El recorte en 0 y 255 hace que el
    modelo deje de ser exactamente aditivo en los extremos del rango.
    """
    ruidosa = imagen.astype(np.float64) + campo_gaussiano(imagen.shape, sigma, semilla)
    return np.clip(np.round(ruidosa), 0, 255).astype(np.uint8)


def sal_y_pimienta(imagen: np.ndarray, proporcion: float, semilla: int = 0) -> np.ndarray:
    """Sustituye una fracción de los píxeles por 0 o 255, en partes iguales.

    Modela fallas puntuales del sensor o de la transmisión. A diferencia del
    ruido gaussiano, no perturba un poco a todos los píxeles sino mucho a unos
    pocos, y un promedio no lo elimina: lo reparte entre los vecinos.
    """
    if not 0.0 <= proporcion <= 1.0:
        raise ValueError(f"proporcion debe estar en [0, 1]; se recibió {proporcion}")
    rng = np.random.default_rng(semilla)
    sorteo = rng.random(imagen.shape)
    resultado = imagen.copy()
    resultado[sorteo < proporcion / 2] = 0
    resultado[(sorteo >= proporcion / 2) & (sorteo < proporcion)] = 255
    return resultado
