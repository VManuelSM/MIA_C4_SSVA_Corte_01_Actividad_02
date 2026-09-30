# MIA_C4_SSVA_Corte_01_Actividad_02

**Análisis comparativo de filtros paso bajo: media 1/9, gaussiano 1/16 y gaussiano 1/273**

| | |
|---|---|
| **Alumno** | Víctor Manuel Santos Martínez |
| **Matrícula** | 253220020 |
| **Materia** | Sistemas de Visión Artificial |
| **Profesor** | Dr. Ernesto García Amaro |
| **Actividad** | Actividad 2 — Filtro paso bajo, Corte 1 |
| **Universidad** | Universidad Politécnica Metropolitana de Hidalgo |
| **Fecha de entrega** | 30 de septiembre de 2026 |

---

## Descripción de la actividad

Realizar un análisis comparativo de los filtros paso bajo de media 1/9,
gaussiano 1/16 y gaussiano 1/273, a partir del código de convolución con núcleo
3×3 visto en clase.

Un filtro paso bajo reemplaza cada píxel por un promedio ponderado de su
vecindario. Los tres núcleos del curso cumplen esa definición y, sin embargo,
se comportan de forma distinta en cuanto a ruido, nitidez y respuesta en
frecuencia. El trabajo explica esas diferencias a partir de los coeficientes de
cada núcleo, las predice y las verifica con mediciones.

## Qué contiene el trabajo

La comparación se hace con cinco criterios. En cada uno, primero se deduce la
propiedad a partir del núcleo y después se comprueba sobre imágenes:

| Criterio | Evidencia |
|---|---|
| Estructura del núcleo | Peso central, dispersión σ, factor de ruido √Σw², separabilidad por descomposición en valores singulares |
| Implementación | Generalización del código de clase a N×N, verificación bit a bit contra una versión vectorizada y contra `cv2.filter2D`, tiempos |
| Respuesta en frecuencia | Transformada de cada núcleo, verificación experimental sobre una placa zonal y sobre patrones de Nyquist |
| Detalle | Fotografía propia, residuos, energía de gradiente, varianza del laplaciano, ancho de un borde |
| Ruido | Ruido gaussiano con predicción de σ de salida, barrido de PSNR con cruces predichos por una descomposición sesgo–varianza, ruido de sal y pimienta |

## Estructura del repositorio

```
.
├── filtros_paso_bajo.ipynb   Cuaderno principal, con salidas ejecutadas
├── src/
│   ├── nucleos.py            Los tres núcleos y sus propiedades medibles
│   ├── convolucion.py        Versión de clase, versión vectorizada y OpenCV
│   ├── frecuencia.py         Respuesta en frecuencia teórica y medida
│   ├── sinteticas.py         Placa zonal, escalón y patrones de Nyquist
│   ├── ruido.py              Ruido gaussiano y de sal y pimienta, con semilla
│   ├── metricas.py           MSE, PSNR, SSIM, nitidez y ancho de borde
│   ├── io_imagenes.py        Lectura, conversión a gris, escalado y escritura
│   └── visualizacion.py      Figuras del reporte
├── imagenes/                 Fotografía de entrada
├── salidas/                  Las 11 figuras generadas
├── requirements.txt
└── README.md
```

`nucleos.py` guarda cada núcleo como en la pizarra: una matriz de enteros y un
divisor. Esa forma permite reproducir la aritmética entera del código de clase
y, a la vez, obtener los pesos normalizados que necesitan OpenCV y el análisis
en frecuencia. Los filtros de `convolucion.py` reciben y devuelven arreglos de
NumPy y no conocen rutas, de modo que se pueden verificar sobre imágenes
sintéticas de pocos píxeles.

## Cómo reproducirlo

```bash
conda create -n sva-upmh python=3.12
conda activate sva-upmh
pip install -r requirements.txt

jupyter lab filtros_paso_bajo.ipynb
```

El cuaderno se ejecuta de principio a fin sin intervención y regenera las 11
figuras en `salidas/`. Todo el ruido se genera con semilla fija, de modo que
las cifras son reproducibles.

## Entorno de ejecución

Python 3.12.14 · OpenCV 5.0.0 · NumPy 2.5.3 · Matplotlib 3.11.2
