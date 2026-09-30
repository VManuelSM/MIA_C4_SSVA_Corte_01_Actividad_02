"""Imágenes de prueba construidas por código, con contenido en frecuencia conocido.

Una fotografía mezcla todas las frecuencias a la vez y no permite atribuir un
efecto a una frecuencia concreta. Estas imágenes sí: en la placa zonal cada
posición corresponde a una frecuencia distinta, y en el escalón toda la
información está en un único borde. Son el equivalente, para un filtro, de
verificar una operación sobre un píxel antes de aplicarla a una imagen entera.
"""

from __future__ import annotations

import numpy as np

GRIS_MEDIO = 128


def placa_zonal(lado: int = 512, amplitud: float = 100.0, frecuencia_maxima: float = 0.5) -> np.ndarray:
    """Patrón de anillos concéntricos cuya frecuencia crece con la distancia al centro.

        I(r) = 128 + A · cos(π · f_máx · r² / R),    r ≤ R = lado / 2

    La frecuencia local, derivada de la fase entre 2π, es f(r) = f_máx · r / R:
    0 en el centro y f_máx en el borde del disco. Más importante, la frecuencia
    local en la posición (x, y) es un vector proporcional a (x, y), de modo que
    **la imagen es un mapa del plano de frecuencias**: filtrarla y mirar dónde
    desaparece el contraste equivale a dibujar la respuesta del filtro.

    Fuera del disco la imagen vale gris medio. Así ningún punto supera la
    frecuencia de Nyquist en ningún eje y el patrón de entrada está libre de
    aliasing; todo artefacto que aparezca después viene del filtro.

    La amplitud de 100 niveles deja la imagen en [28, 228], con margen para
    que el redondeo del filtrado nunca toque 0 ni 255.
    """
    radio = lado / 2.0
    y, x = np.mgrid[0:lado, 0:lado].astype(np.float64)
    x -= (lado - 1) / 2.0
    y -= (lado - 1) / 2.0
    r2 = x**2 + y**2
    imagen = GRIS_MEDIO + amplitud * np.cos(np.pi * frecuencia_maxima * r2 / radio)
    imagen[r2 > radio**2] = GRIS_MEDIO
    return np.clip(np.round(imagen), 0, 255).astype(np.uint8)


def frecuencia_local_de_placa(posicion_px: float, lado: int = 512, frecuencia_maxima: float = 0.5) -> float:
    """Frecuencia, en ciclos/píxel, del patrón de `placa_zonal` a esa distancia del centro."""
    return frecuencia_maxima * posicion_px / (lado / 2.0)


def escalon(lado: int = 64, oscuro: int = 60, claro: int = 190) -> np.ndarray:
    """Borde vertical ideal: la mitad izquierda oscura y la derecha clara.

    Es la entrada más exigente para un paso bajo, porque toda su energía de
    alta frecuencia está concentrada en una sola columna. El perfil de salida
    a través del borde es la integral del núcleo en esa dirección.
    """
    imagen = np.full((lado, lado), oscuro, dtype=np.uint8)
    imagen[:, lado // 2:] = claro
    return imagen


def franjas_de_nyquist(lado: int = 64, oscuro: int = 60, claro: int = 190, diagonal: bool = False) -> np.ndarray:
    """Patrón que alterna claro y oscuro en cada píxel: la frecuencia más alta representable.

    Con `diagonal=False` son franjas verticales (frecuencia 0.5 en x, 0 en y);
    con `diagonal=True` es un tablero de ajedrez (0.5 en ambos ejes).
    """
    y, x = np.mgrid[0:lado, 0:lado]
    fase = (x + y) % 2 if diagonal else x % 2
    return np.where(fase == 0, oscuro, claro).astype(np.uint8)
