"""Carga, preparación y escritura de imágenes.

Este módulo concentra todo el contacto con el sistema de archivos. Los filtros
(`convolucion.py`) trabajan exclusivamente sobre arreglos de NumPy y no saben
de rutas ni de disco: esa separación es la que permite probarlos sobre
imágenes sintéticas de pocos píxeles.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class ErrorDeImagen(Exception):
    """Falla al leer o escribir una imagen."""


def cargar(ruta: str | Path, color: bool = True) -> np.ndarray:
    """Lee una imagen del disco y verifica que realmente se haya cargado.

    `cv2.imread` no lanza excepción cuando la ruta no existe: devuelve `None`.
    Ese silencio es la causa más común de un `AttributeError` varias líneas
    después, lejos del origen real. Aquí se convierte en un error explícito.
    """
    ruta = Path(ruta)
    bandera = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE
    imagen = cv2.imread(str(ruta), bandera)
    if imagen is None:
        raise ErrorDeImagen(f"No se pudo leer la imagen: {ruta}")
    return imagen


def a_grises(imagen: np.ndarray) -> np.ndarray:
    """Convierte de BGR a escala de grises con la ponderación vista en clase.

        gris = 0.299 R + 0.587 G + 0.114 B

    Los coeficientes simulan la sensibilidad del ojo, que es mayor al verde.
    OpenCV guarda los canales en orden BGR, de ahí los índices invertidos.
    Se redondea en lugar de truncar para no sesgar la imagen medio nivel
    hacia el negro.
    """
    if imagen.ndim == 2:
        return imagen
    b, g, r = (imagen[:, :, c].astype(np.float64) for c in range(3))
    gris = 0.299 * r + 0.587 * g + 0.114 * b
    return np.clip(np.round(gris), 0, 255).astype(np.uint8)


def reducir(imagen: np.ndarray, lado_mayor: int) -> np.ndarray:
    """Escala la imagen para que su lado mayor mida `lado_mayor`, sin deformarla.

    Una fotografía de teléfono mide varios miles de píxeles por lado. A esa
    escala un núcleo de 3×3 abarca una fracción ínfima de cualquier detalle
    y su efecto es casi invisible; al reducir, cada píxel pasa a representar
    una porción mayor de la escena y el detalle fino llega hasta la frecuencia
    de Nyquist, que es donde los tres filtros se diferencian.

    El escalado usa `INTER_AREA`, que promedia los píxeles de origen: es en sí
    mismo un paso bajo y evita el aliasing que produciría tomar un píxel de
    cada tantos. Si la imagen ya es menor, se devuelve sin cambios.
    """
    alto, ancho = imagen.shape[:2]
    escala = lado_mayor / max(alto, ancho)
    if escala >= 1.0:
        return imagen
    nuevo = (round(ancho * escala), round(alto * escala))
    return cv2.resize(imagen, nuevo, interpolation=cv2.INTER_AREA)


def guardar(imagen: np.ndarray, ruta: str | Path) -> Path:
    """Escribe una imagen en disco, creando la carpeta destino si hace falta."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(ruta), imagen):
        raise ErrorDeImagen(f"No se pudo escribir la imagen: {ruta}")
    return ruta


def describir(imagen: np.ndarray, nombre: str = "imagen") -> str:
    """Resume forma, tipo y rango de una imagen en una línea legible."""
    return (
        f"{nombre}: forma={imagen.shape}, tipo={imagen.dtype}, "
        f"mín={imagen.min()}, máx={imagen.max()}, media={imagen.mean():.2f}"
    )
