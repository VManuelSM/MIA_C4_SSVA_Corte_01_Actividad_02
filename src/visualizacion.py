"""Construcción de las figuras del reporte.

Cada filtro conserva el mismo color en todas las figuras —azul la media,
naranja el gaussiano 1/16, verde agua el gaussiano 1/273— y la referencia sin
filtrar va siempre en gris. La paleta se validó para daltonismo (las tres
parejas superan ΔE 9 en simulación de deuteranopía, protanopía y
tritanopía); como el verde agua tiene poco contraste sobre blanco, las curvas
llevan además rótulo directo y no dependen sólo del color.

Las figuras no llevan título general: el número y el título van en el pie de
figura del reporte, en formato APA, y repetirlos dentro de la imagen sólo
duplicaría el texto.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from nucleos import NUCLEOS, Nucleo

# --- Paleta -----------------------------------------------------------------

COLOR = {
    "Media 1/9": "#2a78d6",
    "Gaussiano 1/16": "#eb6834",
    "Gaussiano 1/273": "#1baf7a",
}
REFERENCIA = "#52514e"       # tinta secundaria: la imagen o señal sin filtrar
TINTA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
TINTA_TENUE = "#898781"
REJILLA = "#e1e0d9"
EJE = "#c3c2b7"

# Secuencial de un solo tono (azul claro → oscuro) para magnitudes.
SECUENCIAL = LinearSegmentedColormap.from_list(
    "azul", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"]
)
# Divergente rojo ← gris → azul: el gris neutro marca H = 0 y el rojo, la
# inversión de contraste.
DIVERGENTE = LinearSegmentedColormap.from_list(
    "rojo_azul", ["#a3302f", "#e34948", "#f0a9a8", "#f0efec", "#9ec5f4", "#3987e5", "#104281"]
)

PARAMETROS = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 9.5,
    "axes.titlesize": 10,
    "axes.labelsize": 9.5,
    "axes.edgecolor": EJE,
    "axes.labelcolor": TINTA_SECUNDARIA,
    "axes.titlecolor": TINTA,
    "axes.grid": False,
    "grid.color": REJILLA,
    "grid.linewidth": 0.6,
    "xtick.color": TINTA_TENUE,
    "ytick.color": TINTA_TENUE,
    "xtick.labelcolor": TINTA_SECUNDARIA,
    "ytick.labelcolor": TINTA_SECUNDARIA,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "lines.linewidth": 1.8,
    "figure.dpi": 110,
    "savefig.dpi": 170,
    "savefig.bbox": "tight",
}


def aplicar_estilo() -> None:
    """Fija el estilo común. Se invoca una vez al inicio del cuaderno."""
    plt.rcParams.update(PARAMETROS)


def _guardar(figura: plt.Figure, ruta: str | Path) -> Path:
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(ruta)
    plt.close(figura)
    return ruta


def _ejes_de_grafica(eje: plt.Axes) -> None:
    """Rejilla horizontal tenue, sin marco superior ni derecho."""
    eje.grid(True, axis="y")
    eje.set_axisbelow(True)
    for lado in ("top", "right"):
        eje.spines[lado].set_visible(False)


def _ejes_de_imagen(eje: plt.Axes) -> None:
    """Sin marcas, con un marco gris fino para separar la imagen del fondo."""
    eje.set_xticks([])
    eje.set_yticks([])
    for lado in eje.spines.values():
        lado.set_color("0.6")
        lado.set_linewidth(0.8)


# --- Imágenes -----------------------------------------------------------------


def mosaico(
    paneles: list[tuple[str, np.ndarray]],
    ruta: str | Path,
    columnas: int | None = None,
    ancho_panel: float = 3.2,
    cajas: list[tuple[int, int, int, int]] | None = None,
    rotulos_de_cajas: list[str] | None = None,
) -> Path:
    """Dispone imágenes en escala de grises en una rejilla y guarda la figura.

    El rango de despliegue se fija en [0, 255] para todas. Si se dejara que
    Matplotlib lo ajustara a cada imagen, una imagen suavizada —con menos
    contraste— se estiraría para ocupar todo el rango y la pérdida de
    contraste, que es justamente lo que se quiere mostrar, desaparecería.

    Parameters
    ----------
    cajas : list[tuple[int, int, int, int]], opcional
        Rectángulos (x, y, ancho, alto) que se dibujan sobre el primer panel
        para indicar de dónde salen las ampliaciones.
    rotulos_de_cajas : list[str], opcional
        Nombre de cada rectángulo, escrito sobre su esquina superior.
    """
    total = len(paneles)
    columnas = columnas or total
    filas = int(np.ceil(total / columnas))
    figura, ejes = plt.subplots(filas, columnas, figsize=(ancho_panel * columnas, ancho_panel * filas + 0.35))
    ejes = np.atleast_1d(ejes).ravel()

    for eje, (rotulo, imagen) in zip(ejes, paneles):
        eje.imshow(imagen, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        eje.set_title(rotulo)
        _ejes_de_imagen(eje)

    for k, (x, y, w, h) in enumerate(cajas or []):
        ejes[0].add_patch(Rectangle((x - 0.5, y - 0.5), w, h, fill=False, edgecolor="#eda100", linewidth=1.6))
        if rotulos_de_cajas:
            ejes[0].text(x, y - 6, rotulos_de_cajas[k], fontsize=9, color=TINTA, va="bottom",
                         bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "#eda100", "lw": 1.0})

    for eje in ejes[total:]:
        eje.axis("off")

    figura.tight_layout(h_pad=1.6 if filas > 1 else 1.08)
    return _guardar(figura, ruta)


def rejilla_de_ampliaciones(
    filas: list[tuple[str, list[np.ndarray]]],
    encabezados: list[str],
    ruta: str | Path,
    ancho_panel: float = 2.6,
    rango: tuple[float, float] = (0, 255),
    mapa: str = "gray",
) -> Path:
    """Tabla de imágenes: cada fila es una región o condición, cada columna un filtro.

    Es el formato que permite comparar los filtros entre sí sobre exactamente
    los mismos píxeles, que es lo que pide un análisis comparativo.
    """
    n_filas, n_columnas = len(filas), len(encabezados)
    figura, ejes = plt.subplots(
        n_filas, n_columnas, figsize=(ancho_panel * n_columnas + 0.5, ancho_panel * n_filas + 0.3), squeeze=False
    )
    for i, (rotulo_fila, imagenes) in enumerate(filas):
        for j, imagen in enumerate(imagenes):
            eje = ejes[i, j]
            eje.imshow(imagen, cmap=mapa, vmin=rango[0], vmax=rango[1], interpolation="nearest")
            _ejes_de_imagen(eje)
            if i == 0:
                eje.set_title(encabezados[j])
        ejes[i, 0].set_ylabel(rotulo_fila, color=TINTA, fontsize=9.5)
    figura.tight_layout(h_pad=1.2, w_pad=0.4)
    return _guardar(figura, ruta)


def rejilla_con_rotulos(
    celdas: list[list[tuple[str, np.ndarray]]],
    ruta: str | Path,
    ancho_panel: float = 2.6,
) -> Path:
    """Como `rejilla_de_ampliaciones`, pero cada imagen lleva su propio rótulo.

    Sirve cuando cada panel necesita reportar una cifra propia, por ejemplo el
    PSNR de esa imagen concreta.
    """
    n_filas, n_columnas = len(celdas), max(len(f) for f in celdas)
    figura, ejes = plt.subplots(
        n_filas, n_columnas, figsize=(ancho_panel * n_columnas, ancho_panel * n_filas + 0.45 * n_filas), squeeze=False
    )
    for i, fila in enumerate(celdas):
        for j, (rotulo, imagen) in enumerate(fila):
            eje = ejes[i, j]
            eje.imshow(imagen, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            eje.set_title(rotulo, fontsize=9)
            _ejes_de_imagen(eje)
        for j in range(len(fila), n_columnas):
            ejes[i, j].axis("off")
    figura.tight_layout(h_pad=1.0, w_pad=0.4)
    return _guardar(figura, ruta)


# --- Núcleos ------------------------------------------------------------------


def mapas_de_nucleos(nucleos: tuple[Nucleo, ...], ruta: str | Path) -> Path:
    """Los tres núcleos como mapas de calor, con sus coeficientes enteros escritos.

    Los tres se dibujan con la misma escala de color, la de los pesos ya
    normalizados. Así se compara directamente cuánto pesa el píxel central en
    cada uno: en la media pesa lo mismo que cualquier vecino; en los
    gaussianos, bastante más.
    """
    maximo = max(n.normalizado.max() for n in nucleos)
    anchos = [n.lado for n in nucleos]
    figura, ejes = plt.subplots(
        1, len(nucleos), figsize=(3.1 * len(nucleos) + 0.6, 3.2), gridspec_kw={"width_ratios": anchos}
    )
    for eje, n in zip(ejes, nucleos):
        imagen = eje.imshow(n.normalizado, cmap=SECUENCIAL, vmin=0, vmax=maximo)
        for (i, j), valor in np.ndenumerate(n.coeficientes):
            peso = n.normalizado[i, j]
            eje.text(j, i, f"{valor}", ha="center", va="center", fontsize=9 if n.lado > 3 else 11,
                     color="white" if peso > 0.55 * maximo else TINTA)
        eje.set_title(f"{n.nombre}\n(suma de coeficientes = {n.divisor})")
        eje.set_xticks(range(n.lado), [f"{d:+d}" if d else "0" for d in range(-n.radio, n.radio + 1)])
        eje.set_yticks(range(n.lado), [f"{d:+d}" if d else "0" for d in range(-n.radio, n.radio + 1)])
        eje.tick_params(length=0, labelsize=8)
        for lado in eje.spines.values():
            lado.set_visible(False)
    barra = figura.colorbar(imagen, ax=list(ejes), fraction=0.025, pad=0.02)
    barra.set_label("Peso normalizado")
    barra.outline.set_visible(False)
    return _guardar(figura, ruta)


# --- Frecuencia ---------------------------------------------------------------


def _con_menos(texto: str) -> str:
    """Sustituye el guion por el signo menos tipográfico."""
    return texto.replace("-", "−")


def _rotulo_final(eje: plt.Axes, x: float, y: float, texto: str, color: str, desplazamiento: float = 0.0) -> None:
    eje.annotate(texto, (x, y), xytext=(6, desplazamiento), textcoords="offset points",
                 va="center", ha="left", fontsize=8.5, color=TINTA_SECUNDARIA,
                 bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none"})
    eje.plot([x], [y], marker="o", markersize=4, color=color)


def cortes_de_respuesta(
    cortes: list[tuple[str, np.ndarray, dict[str, np.ndarray]]],
    ruta: str | Path,
    desplazamientos: dict[str, dict[str, float]] | None = None,
    mediciones: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] | None = None,
) -> Path:
    """H(f) a lo largo de una o más direcciones, un panel por dirección.

    Parameters
    ----------
    cortes : list
        Tuplas (título del panel, frecuencias, {nombre del filtro: H}).
    desplazamientos : dict, opcional
        Ajuste vertical, en puntos, del rótulo final de cada curva por panel,
        para que no se encimen cuando dos curvas terminan cerca.
    mediciones : dict, opcional
        Por panel y por filtro, pares (frecuencias, H medida) que se dibujan
        como círculos huecos encima de la curva teórica.
    """
    desplazamientos = desplazamientos or {}
    figura, ejes = plt.subplots(1, len(cortes), figsize=(5.0 * len(cortes) + 0.8, 3.6), sharey=True)
    ejes = np.atleast_1d(ejes)
    for eje, (titulo, f, curvas) in zip(ejes, cortes):
        _ejes_de_grafica(eje)
        eje.axhline(0.0, color=TINTA_TENUE, linewidth=0.9)
        eje.axhline(1.0 / np.sqrt(2.0), color=EJE, linewidth=0.9)
        eje.text(0.004, 1.0 / np.sqrt(2.0) + 0.03, "−3 dB", fontsize=8, color=TINTA_TENUE)
        for nombre, h in curvas.items():
            eje.plot(f, h, color=COLOR[nombre], label=nombre)
            _rotulo_final(eje, f[-1], h[-1], _con_menos(f"{h[-1]:+.3f}"), COLOR[nombre],
                          desplazamientos.get(titulo, {}).get(nombre, 0.0))
        for nombre, (fm, hm) in (mediciones or {}).get(titulo, {}).items():
            eje.plot(fm, hm, linestyle="none", marker="o", markersize=3.2, markerfacecolor="white",
                     markeredgecolor=COLOR[nombre], markeredgewidth=0.9, zorder=4)
        eje.set_xlim(0, f[-1] * 1.16)
        eje.set_ylim(-0.42, 1.05)
        eje.set_title(titulo)
        eje.set_xlabel("Frecuencia (ciclos por píxel)")
    ejes[0].set_ylabel("Respuesta H(f)")
    ejes[0].legend(loc="upper right")
    figura.tight_layout()
    return _guardar(figura, ruta)


def mapas_de_respuesta(respuestas: list[tuple[str, np.ndarray, np.ndarray]], ruta: str | Path) -> Path:
    """H(u, v) de cada núcleo sobre el plano de frecuencias, con escala divergente.

    Azul es paso, gris es bloqueo (H = 0) y rojo es inversión de contraste
    (H < 0). La curva negra marca el corte de −3 dB. Un filtro isótropo
    dibujaría círculos; cualquier otra forma delata una dirección
    privilegiada.
    """
    figura, ejes = plt.subplots(1, len(respuestas), figsize=(3.4 * len(respuestas) + 0.8, 3.5))
    for eje, (nombre, f, H) in zip(ejes, respuestas):
        imagen = eje.imshow(H, cmap=DIVERGENTE, vmin=-1, vmax=1, origin="lower",
                            extent=(f[0], f[-1], f[0], f[-1]))
        eje.contour(f, f, H, levels=[1.0 / np.sqrt(2.0)], colors=[TINTA], linewidths=1.0)
        # Sólo se traza el contorno de H = 0 cuando la inversión es apreciable.
        # El 1/273 tiene valores de −0.002 en las esquinas: dibujarlos con la
        # misma línea que los −0.333 de la media sugeriría un efecto que no
        # existe a escala de ocho bits.
        if H.min() < -0.01:
            eje.contour(f, f, H, levels=[0.0], colors=["#a3302f"], linewidths=0.9, linestyles="solid")
        eje.set_title(nombre)
        eje.set_xlabel("u (ciclos/píxel)")
        eje.set_xticks([-0.5, 0, 0.5])
        eje.set_yticks([-0.5, 0, 0.5])
    ejes[0].set_ylabel("v (ciclos/píxel)")
    barra = figura.colorbar(imagen, ax=list(ejes), fraction=0.02, pad=0.02)
    barra.set_label("H(u, v)")
    barra.outline.set_visible(False)
    return _guardar(figura, ruta)


# --- Perfiles y curvas --------------------------------------------------------


def perfiles(
    x: np.ndarray,
    referencia: np.ndarray,
    curvas: dict[str, np.ndarray],
    ruta: str | Path,
    etiqueta_referencia: str = "Sin filtrar",
    rotulos: dict[str, str] | None = None,
    xlabel: str = "Posición (píxeles)",
    ylabel: str = "Intensidad",
    titulo_de_leyenda: str | None = None,
) -> Path:
    """Perfiles de intensidad a lo largo de una línea de la imagen."""
    figura, eje = plt.subplots(figsize=(6.4, 3.4))
    _ejes_de_grafica(eje)
    eje.step(x, referencia, where="mid", color=REFERENCIA, linewidth=1.2, label=etiqueta_referencia)
    for nombre, perfil in curvas.items():
        eje.plot(x, perfil, color=COLOR[nombre], marker="o", markersize=4, label=nombre)
    if rotulos:
        eje.legend(handles=eje.get_lines(), labels=[etiqueta_referencia] + [f"{n}: {rotulos[n]}" for n in curvas],
                   title=titulo_de_leyenda, title_fontsize=8.5, alignment="left", loc="upper left")
    else:
        eje.legend(loc="upper left")
    eje.set_xlabel(xlabel)
    eje.set_ylabel(ylabel)
    figura.tight_layout()
    return _guardar(figura, ruta)


def curvas_contra_ruido(
    sigmas: np.ndarray,
    psnr: dict[str, np.ndarray],
    referencia: str,
    cruces: dict[str, float],
    ruta: str | Path,
    desplazamientos: dict[str, float] | None = None,
) -> Path:
    """PSNR contra nivel de ruido, en dos paneles con su propio eje.

    El panel izquierdo muestra el PSNR absoluto de cada filtro y de la imagen
    sin filtrar. El derecho resta a cada filtro el PSNR del filtro de
    `referencia`: ahí las diferencias de décimas de decibel, invisibles en la
    escala absoluta, se vuelven legibles, y cada cruce entre filtros aparece
    como un paso por cero.

    Parameters
    ----------
    psnr : dict[str, np.ndarray]
        Curvas por nombre; «Sin filtrar» se dibuja en gris y sólo en el panel
        izquierdo.
    cruces : dict[str, float]
        Para cada filtro distinto de la referencia, el σ en que su curva cruza
        la de la referencia. Se marca con un punto y su valor.
    """
    desplazamientos = desplazamientos or {}
    figura, (izq, der) = plt.subplots(1, 2, figsize=(11.2, 3.9), gridspec_kw={"width_ratios": [1.0, 1.0]})
    for eje in (izq, der):
        _ejes_de_grafica(eje)
        eje.set_xlabel("Desviación estándar del ruido σ (niveles de gris)")

    filtros = [k for k in psnr if k in COLOR]
    maximo = max(np.nanmax(psnr[k]) for k in filtros)
    minimo = min(np.nanmin(v) for v in psnr.values())
    for nombre, valores in psnr.items():
        color = COLOR.get(nombre, REFERENCIA)
        izq.plot(sigmas, valores, color=color, label=nombre, linewidth=1.8 if nombre in COLOR else 1.4)
        _rotulo_final(izq, sigmas[-1], valores[-1], nombre, color, desplazamientos.get(nombre, 0.0))
    izq.set_ylim(np.floor(minimo) - 1, np.ceil(maximo) + 3)
    izq.set_xlim(sigmas[0], sigmas[-1] * 1.42)
    izq.set_ylabel("PSNR (dB)")
    izq.set_title("PSNR absoluto")
    izq.legend(loc="upper right")

    der.plot(sigmas, np.zeros_like(sigmas), color=COLOR[referencia], label=referencia)
    _rotulo_final(der, sigmas[-1], 0.0, referencia, COLOR[referencia])
    for nombre in filtros:
        if nombre == referencia:
            continue
        diferencia = psnr[nombre] - psnr[referencia]
        der.plot(sigmas, diferencia, color=COLOR[nombre], label=nombre)
        _rotulo_final(der, sigmas[-1], diferencia[-1], nombre, COLOR[nombre])
        cruce = cruces.get(nombre)
        if cruce is not None and np.isfinite(cruce):
            der.plot([cruce], [0.0], marker="o", markersize=6, color=COLOR[nombre],
                     markeredgecolor="white", markeredgewidth=1.2, zorder=5)
            # El primer cruce se rotula arriba a la izquierda y el siguiente
            # abajo a la derecha, para que los dos textos no se encimen.
            arriba = cruce == min(c for c in cruces.values() if np.isfinite(c))
            der.annotate(f"σ = {cruce:.1f}", (cruce, 0.0), xytext=(-8, 9) if arriba else (8, -13),
                         textcoords="offset points", ha="right" if arriba else "left", fontsize=8.2,
                         color=TINTA_SECUNDARIA, bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none"})
    der.set_xlim(sigmas[0], sigmas[-1] * 1.42)
    der.set_ylabel(f"PSNR − PSNR del {referencia.lower()} (dB)")
    der.set_title(f"Diferencia respecto al {referencia.lower()}")
    figura.tight_layout(w_pad=2.5)
    return _guardar(figura, ruta)
