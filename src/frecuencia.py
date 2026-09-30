"""Respuesta en frecuencia de un núcleo: qué deja pasar y qué atenúa.

«Paso bajo» es una afirmación sobre frecuencias, y la forma de verificarla es
calcular la respuesta del núcleo, su transformada de Fourier discreta:

    H(u, v) = Σₓ Σᵧ w(x, y) · exp(−i 2π (u x + v y))

donde (u, v) son frecuencias espaciales en ciclos por píxel. El intervalo útil
va de 0 (brillo uniforme) a 0.5 ciclos/píxel (la frecuencia de Nyquist: un
patrón que alterna claro y oscuro en cada píxel).

Los tres núcleos son simétricos respecto al centro, w(x, y) = w(−x, −y), y
para esos núcleos la parte imaginaria se cancela: H es **real**. Eso importa,
porque un valor real negativo no es una atenuación sino una **inversión de
contraste**: el patrón de esa frecuencia sale del filtro con claros y oscuros
intercambiados. La magnitud |H| esconde ese fenómeno; por eso aquí se
trabaja con H con signo.
"""

from __future__ import annotations

import numpy as np

from nucleos import Nucleo

NYQUIST = 0.5  # ciclos por píxel


def alcance(angulo_grados: float) -> float:
    """Frecuencia radial a la que una dirección sale del cuadrado de Nyquist.

    El plano útil es el cuadrado [−0.5, 0.5]²; un rayo con ángulo θ lo
    abandona en 0.5 / máx(|cos θ|, |sen θ|): 0.5 en los ejes, 0.707 en la
    diagonal.
    """
    theta = np.deg2rad(angulo_grados)
    return NYQUIST / max(abs(np.cos(theta)), abs(np.sin(theta)))


def respuesta_en_direccion(nucleo: Nucleo, frecuencias: np.ndarray, angulo_grados: float = 0.0) -> np.ndarray:
    """H evaluada a lo largo de una dirección del plano de frecuencias.

    Parameters
    ----------
    frecuencias : np.ndarray
        Frecuencia radial, en ciclos por píxel.
    angulo_grados : float
        0° es la dirección horizontal (patrones de franjas verticales); 45° la
        diagonal. En la diagonal la frecuencia radial llega hasta
        0.5·√2 ≈ 0.707 antes de alcanzar la esquina de Nyquist (0.5, 0.5).
    """
    theta = np.deg2rad(angulo_grados)
    w = nucleo.normalizado
    r = nucleo.radio
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    proyeccion = x * np.cos(theta) + y * np.sin(theta)
    # La parte imaginaria es cero por simetría; se suma sólo el coseno.
    fase = 2.0 * np.pi * np.multiply.outer(frecuencias, proyeccion)
    return (np.cos(fase) * w).sum(axis=(-2, -1))


def respuesta_2d(nucleo: Nucleo, n: int = 257) -> tuple[np.ndarray, np.ndarray]:
    """H sobre una rejilla n×n que cubre [−0.5, 0.5] ciclos/píxel en cada eje.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (frecuencias de un eje, matriz H con u en columnas y v en filas).
    """
    f = np.linspace(-NYQUIST, NYQUIST, n)
    u, v = np.meshgrid(f, f)
    w = nucleo.normalizado
    r = nucleo.radio
    H = np.zeros_like(u)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            H += w[dy + r, dx + r] * np.cos(2.0 * np.pi * (u * dx + v * dy))
    return f, H


def frecuencia_de_corte(nucleo: Nucleo, nivel: float = 1.0 / np.sqrt(2.0),
                        angulo_grados: float = 0.0) -> float:
    """Primera frecuencia en la que |H| cae por debajo de `nivel`.

    Con el nivel por omisión, 1/√2 ≈ 0.707, es la frecuencia de corte de
    −3 dB: la potencia de la señal que pasa se ha reducido a la mitad.
    """
    f = np.linspace(0.0, alcance(angulo_grados), 20001)
    h = respuesta_en_direccion(nucleo, f, angulo_grados)
    debajo = np.flatnonzero(h < nivel)
    return float(f[debajo[0]]) if debajo.size else float("nan")


def resumen(nucleo: Nucleo) -> dict[str, float]:
    """Indicadores de la respuesta en frecuencia que se reportan en tablas."""
    f = np.linspace(0.0, NYQUIST, 20001)
    horizontal = respuesta_en_direccion(nucleo, f, 0.0)
    ceros = np.flatnonzero(np.diff(np.sign(horizontal)) != 0)
    return {
        "corte_3db_horizontal": frecuencia_de_corte(nucleo, angulo_grados=0.0),
        "corte_3db_diagonal": frecuencia_de_corte(nucleo, angulo_grados=45.0),
        "h_nyquist_horizontal": float(horizontal[-1]),
        "h_nyquist_diagonal": float(respuesta_en_direccion(nucleo, np.array([alcance(45.0)]), 45.0)[0]),
        "h_minimo": float(horizontal.min()),
        "primer_cero": float(f[ceros[0] + 1]) if ceros.size else float("nan"),
    }


def respuesta_medida_en_placa(original: np.ndarray, filtrada: np.ndarray, frecuencia_maxima: float = 0.5,
                              fraccion_de_pico: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
    """Estima H(f) en la dirección horizontal a partir de una placa zonal filtrada.

    Sobre la fila central, la placa es una senoide cuya frecuencia crece
    linealmente con la distancia al centro. Donde la entrada está cerca de un
    máximo o un mínimo, el cociente

        (salida − 128) / (entrada − 128)

    es la ganancia del filtro a la frecuencia local, con su signo: negativo si
    el filtro invirtió el contraste. Se descartan los cruces por cero de la
    entrada, donde el cociente divide entre casi nada.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (frecuencias locales, ganancias medidas).
    """
    lado = original.shape[0]
    radio = lado / 2.0
    fila = lado // 2
    x = np.arange(lado // 2, lado)
    distancia = np.hypot(x - (lado - 1) / 2.0, fila - (lado - 1) / 2.0)
    entrada = original[fila, x].astype(np.float64) - 128.0
    salida = filtrada[fila, x].astype(np.float64) - 128.0
    amplitud = np.abs(entrada).max()
    util = (np.abs(entrada) >= fraccion_de_pico * amplitud) & (distancia < radio - 1)
    return frecuencia_maxima * distancia[util] / radio, salida[util] / entrada[util]
