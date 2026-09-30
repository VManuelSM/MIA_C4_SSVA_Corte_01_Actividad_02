"""Tres implementaciones de la misma operación de filtrado.

1. `filtrar_como_en_clase` generaliza el código compartido en clase a un
   núcleo de cualquier tamaño impar. Conserva sus decisiones: recorrido
   explícito píxel por píxel, aritmética entera, división al final con
   `round` de Python y perímetro sin evaluar (queda en 0).
2. `filtrar_vectorizado` hace exactamente las mismas cuentas pero desplaza la
   imagen completa en lugar de recorrerla, y sirve para comprobar que la
   versión de clase es correcta sin esperar a que termine muchas veces.
3. `filtrar` delega en `cv2.filter2D`, que es lo que se usaría en producción,
   y trata el borde reflejando la imagen en lugar de dejarlo en negro.

Una nota sobre nomenclatura. `cv2.filter2D` calcula una **correlación**, no
una convolución: no invierte el núcleo antes de recorrer la imagen. Los tres
núcleos de esta actividad son simétricos respecto al centro, así que
correlación y convolución dan el mismo resultado y el término se usa aquí
indistintamente.
"""

from __future__ import annotations

import cv2
import numpy as np

from nucleos import Nucleo


def filtrar_como_en_clase(imagen: np.ndarray, nucleo: Nucleo) -> np.ndarray:
    """Filtro por recorrido explícito, generalizando el código de clase a N×N.

    El código de clase nombra a mano los nueve vecinos de un núcleo 3×3
    (`arriba_izq`, `arriba`, …). Con un núcleo 5×5 serían veinticinco
    variables, así que aquí los vecinos se recorren con dos ciclos más. La
    aritmética es la misma: productos enteros, suma entera, una sola división
    al final.

    El perímetro no se evalúa, igual que en clase: con radio r, las primeras
    y últimas r filas y columnas no tienen vecindario completo y se quedan
    en 0. En el núcleo 5×5 ese marco negro mide dos píxeles.

    No se recorta el resultado a [0, 255] porque no hace falta: con pesos
    positivos que suman 1, la salida es un promedio ponderado de los vecinos
    y no puede salirse del intervalo que ellos ocupan. Es la diferencia de
    fondo con los núcleos de bordes vistos en clase, que tienen pesos
    negativos y sí necesitan `np.clip`.
    """
    alto, ancho = imagen.shape
    r = nucleo.radio
    k = nucleo.coeficientes
    salida = np.zeros((alto, ancho), dtype=np.uint8)

    for y in range(r, alto - r):
        for x in range(r, ancho - r):
            suma = 0
            for i in range(-r, r + 1):
                for j in range(-r, r + 1):
                    suma += int(imagen[y + i, x + j]) * int(k[i + r, j + r])
            # `round` de Python redondea los empates al par más cercano
            # (2.5 → 2, 3.5 → 4). Con divisor 16 los empates sí ocurren,
            # cuando la suma deja residuo 8.
            salida[y, x] = int(round(suma / nucleo.divisor))

    return salida


def filtrar_vectorizado(imagen: np.ndarray, nucleo: Nucleo) -> np.ndarray:
    """Mismas cuentas que `filtrar_como_en_clase`, sin recorrer píxel por píxel.

    En lugar de visitar cada píxel y sumar sus vecinos, se visita cada
    posición del núcleo y se suma la imagen completa desplazada a esa
    posición. El número de operaciones es el mismo; lo que cambia es que
    NumPy ejecuta cada suma de imagen completa en código compilado.

    `np.round` también redondea los empates al par, de modo que el resultado
    debe coincidir con la versión de clase bit a bit.
    """
    alto, ancho = imagen.shape
    r = nucleo.radio
    interior_alto, interior_ancho = alto - 2 * r, ancho - 2 * r
    fuente = imagen.astype(np.int64)

    acumulado = np.zeros((interior_alto, interior_ancho), dtype=np.int64)
    for i in range(nucleo.lado):
        for j in range(nucleo.lado):
            peso = int(nucleo.coeficientes[i, j])
            acumulado += peso * fuente[i:i + interior_alto, j:j + interior_ancho]

    salida = np.zeros((alto, ancho), dtype=np.uint8)
    salida[r:alto - r, r:ancho - r] = np.round(acumulado / nucleo.divisor).astype(np.uint8)
    return salida


def filtrar(imagen: np.ndarray, nucleo: Nucleo, borde: int = cv2.BORDER_REFLECT_101) -> np.ndarray:
    """Filtrado con OpenCV, conservando el tipo de la imagen de entrada.

    `BORDER_REFLECT_101` completa el vecindario que falta en el perímetro
    reflejando la imagen sin repetir el píxel del borde (…c b | a b c…). Así
    todos los píxeles reciben un valor filtrado y no aparece el marco negro
    de la versión de clase.
    """
    return cv2.filter2D(imagen, -1, nucleo.normalizado, borderType=borde)


def filtrar_real(campo: np.ndarray, nucleo: Nucleo, borde: int = cv2.BORDER_REFLECT_101) -> np.ndarray:
    """Filtrado en punto flotante, sin redondeo ni recorte a [0, 255].

    Se usa para analizar el filtro como sistema lineal: sobre un campo de
    ruido con media cero, por ejemplo, los valores negativos son legítimos y
    redondear a `uint8` los destruiría.
    """
    return cv2.filter2D(campo.astype(np.float64), cv2.CV_64F, nucleo.normalizado, borderType=borde)


def filtrar_separable(imagen: np.ndarray, columna: np.ndarray, fila: np.ndarray,
                      borde: int = cv2.BORDER_REFLECT_101) -> np.ndarray:
    """Filtrado en dos pasadas 1-D: primero por filas, después por columnas.

    Si el núcleo 2-D es el producto exterior `columna ⊗ fila`, filtrar con él
    equivale a filtrar con `fila` en horizontal y luego con `columna` en
    vertical. Un núcleo n×n pasa de n² a 2n multiplicaciones por píxel.
    """
    return cv2.sepFilter2D(imagen, -1, fila, columna, borderType=borde)
