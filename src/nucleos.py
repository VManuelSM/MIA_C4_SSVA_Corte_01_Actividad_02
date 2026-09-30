"""Los tres núcleos paso bajo de la actividad y sus propiedades medibles.

Cada núcleo se guarda como en la pizarra: una matriz de **enteros** y un
**divisor**. Separarlos no es cosmético. El código de clase suma productos
enteros y divide al final (`suma / 9`, `suma / 16`), y conservar esa forma
permite reproducir su aritmética exacta y, al mismo tiempo, obtener la versión
normalizada que necesitan OpenCV y el análisis en frecuencia.

Las propiedades que se calculan aquí son las que después explican el
comportamiento observado en las imágenes:

* la **dispersión espacial** σ, cuánto se extiende el promedio alrededor del
  píxel central;
* el **factor de ruido** √Σw², cuánto se reduce la desviación estándar de un
  ruido blanco al pasar por el filtro;
* la **separabilidad**, si el núcleo 2-D es el producto de dos núcleos 1-D.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Nucleo:
    """Un núcleo de convolución expresado como enteros sobre un divisor.

    Attributes
    ----------
    nombre : str
        Rótulo corto para tablas y figuras.
    coeficientes : np.ndarray
        Matriz cuadrada de enteros, de lado impar.
    divisor : int
        Constante de normalización. Debe coincidir con la suma de los
        coeficientes para que el filtro conserve el brillo medio.
    """

    nombre: str
    coeficientes: np.ndarray
    divisor: int

    def __post_init__(self) -> None:
        lado = self.coeficientes.shape[0]
        if self.coeficientes.shape != (lado, lado) or lado % 2 == 0:
            raise ValueError(f"{self.nombre}: el núcleo debe ser cuadrado y de lado impar")
        if int(self.coeficientes.sum()) != self.divisor:
            # Un divisor que no coincide con la suma cambia la ganancia en
            # continua: la imagen filtrada sale más clara o más oscura que la
            # original aunque el núcleo tenga la forma correcta.
            raise ValueError(
                f"{self.nombre}: la suma de coeficientes ({self.coeficientes.sum()}) "
                f"no coincide con el divisor ({self.divisor})"
            )

    @property
    def lado(self) -> int:
        return self.coeficientes.shape[0]

    @property
    def radio(self) -> int:
        """Píxeles de vecindario a cada lado del central: 1 en 3×3, 2 en 5×5."""
        return self.lado // 2

    @property
    def normalizado(self) -> np.ndarray:
        """Pesos reales que suman 1, en `float64`."""
        return self.coeficientes.astype(np.float64) / self.divisor


MEDIA_9 = Nucleo(
    nombre="Media 1/9",
    coeficientes=np.ones((3, 3), dtype=np.int64),
    divisor=9,
)

GAUSSIANO_16 = Nucleo(
    nombre="Gaussiano 1/16",
    coeficientes=np.array(
        [[1, 2, 1],
         [2, 4, 2],
         [1, 2, 1]], dtype=np.int64),
    divisor=16,
)

GAUSSIANO_273 = Nucleo(
    nombre="Gaussiano 1/273",
    coeficientes=np.array(
        [[1,  4,  7,  4, 1],
         [4, 16, 26, 16, 4],
         [7, 26, 41, 26, 7],
         [4, 16, 26, 16, 4],
         [1,  4,  7,  4, 1]], dtype=np.int64),
    divisor=273,
)

# Orden fijo en todo el trabajo: tablas, figuras y colores siguen esta
# secuencia para que cada filtro conserve su identidad visual.
NUCLEOS: tuple[Nucleo, ...] = (MEDIA_9, GAUSSIANO_16, GAUSSIANO_273)


def gaussiana_muestreada(lado: int, sigma: float) -> np.ndarray:
    """Gaussiana 2-D muestreada en una rejilla entera y normalizada a suma 1.

        G(x, y) = exp(-(x² + y²) / (2σ²)) / Σ exp(...)

    Sirve de referencia para medir cuánto se aparta de la gaussiana ideal un
    núcleo entero redondeado como el de 1/273.
    """
    radio = lado // 2
    y, x = np.mgrid[-radio:radio + 1, -radio:radio + 1]
    g = np.exp(-(x**2 + y**2) / (2.0 * sigma**2))
    return g / g.sum()


def dispersion(nucleo: Nucleo) -> float:
    """Desviación estándar espacial del núcleo en una dirección, en píxeles.

    Se trata el núcleo normalizado como una distribución de probabilidad sobre
    los desplazamientos (x, y) y se calcula la raíz de su segundo momento en x.
    Para una gaussiana continua coincide con su parámetro σ; para cualquier
    otro núcleo es el σ de la gaussiana que promedia con el mismo alcance.
    """
    w = nucleo.normalizado
    _, x = np.mgrid[-nucleo.radio:nucleo.radio + 1, -nucleo.radio:nucleo.radio + 1]
    return float(np.sqrt((w * x**2).sum()))


def factor_de_ruido(nucleo: Nucleo) -> float:
    """Cociente σ_salida / σ_entrada para un ruido blanco aditivo.

    Si el ruido es independiente entre píxeles con varianza σ², la salida del
    filtro es una suma ponderada de variables independientes y su varianza es
    σ² Σ w². La desviación estándar se multiplica entonces por √Σ w².

    Con pesos que suman 1 y un número fijo de posiciones, Σ w² es mínima
    cuando todos los pesos son iguales. Por eso el filtro de media es el
    núcleo 3×3 que más reduce el ruido, aunque no sea el que mejor se
    comporta en frecuencia.
    """
    return float(np.sqrt((nucleo.normalizado**2).sum()))


def valores_singulares(nucleo: Nucleo) -> np.ndarray:
    """Valores singulares de la matriz normalizada, de mayor a menor.

    Un núcleo es separable —igual al producto exterior de un vector columna
    por un vector fila— si y sólo si su matriz tiene rango 1, es decir, un
    único valor singular distinto de cero.
    """
    return np.linalg.svd(nucleo.normalizado, compute_uv=False)


def energia_de_rango_uno(nucleo: Nucleo) -> float:
    """Fracción de la energía del núcleo que captura su mejor aproximación separable."""
    s = valores_singulares(nucleo)
    return float(s[0] ** 2 / (s**2).sum())


def factores_separables(nucleo: Nucleo) -> tuple[np.ndarray, np.ndarray]:
    """Mejor par de vectores 1-D (columna, fila) cuyo producto aproxima el núcleo.

    Para un núcleo de rango 1 la aproximación es exacta; para uno que no lo es,
    es la aproximación de rango 1 de mínimo error cuadrático (teorema de
    Eckart–Young).
    """
    u, s, vt = np.linalg.svd(nucleo.normalizado)
    columna = u[:, 0] * np.sqrt(s[0])
    fila = vt[0, :] * np.sqrt(s[0])
    # El signo de los vectores singulares es arbitrario; se fija positivo.
    if columna.sum() < 0:
        columna, fila = -columna, -fila
    return columna, fila


def sigma_de_mejor_ajuste(nucleo: Nucleo, candidatos: np.ndarray | None = None) -> tuple[float, float]:
    """σ de la gaussiana muestreada más cercana al núcleo, y el error máximo.

    Returns
    -------
    tuple[float, float]
        (σ óptimo, máxima diferencia absoluta entre pesos normalizados).
    """
    if candidatos is None:
        candidatos = np.linspace(0.5, 2.0, 1501)
    errores = [
        np.abs(gaussiana_muestreada(nucleo.lado, s) - nucleo.normalizado).max()
        for s in candidatos
    ]
    i = int(np.argmin(errores))
    return float(candidatos[i]), float(errores[i])


def propiedades(nucleo: Nucleo) -> dict[str, float]:
    """Resumen numérico de un núcleo, en el orden en que se reporta."""
    return {
        "lado": nucleo.lado,
        "divisor": nucleo.divisor,
        "ganancia_dc": float(nucleo.normalizado.sum()),
        "peso_central": float(nucleo.normalizado[nucleo.radio, nucleo.radio]),
        "dispersion": dispersion(nucleo),
        "factor_de_ruido": factor_de_ruido(nucleo),
        "energia_rango_uno": energia_de_rango_uno(nucleo),
        "multiplicaciones_directas": nucleo.lado**2,
    }
