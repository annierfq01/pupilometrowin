# Pupilometer

**Pupilometer** es una aplicacion de escritorio para la deteccion y analisis de pupila e iris en imagenes y videos, desarrollada en Python con interfaz grafica Tkinter.

## Tabla de Contenidos

- [Descripcion](#descripcion)
- [Caracteristicas](#caracteristicas)
- [Instalacion](#instalacion)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Modulos Principales](#modulos-principales)
- [Uso](#uso)
- [Algoritmos de Deteccion](#algoritmos-de-deteccion)
- [Configuracion](#configuracion)
- [Desarrollo](#desarrollo)

## Descripcion

Pupilometer permite:

- **Deteccion automatica de pupila e iris** en imagenes y videos
- **Analisis de pupilometria** para estudios clinicos
- **Calibracion con regla** para medicion en milimetros
- **Soporte para IA (U-Net)** para deteccion mejorada
- **Grabacion de dataset** para entrenamiento de modelos
- **Exportacion de resultados** en CSV y TXT

## Caracteristicas

### Deteccion
- Multiples algoritmos de deteccion de pupila (Starburst, Swirski, Excuse, Canny, Threshold, DarkCircle, IA)
- Algoritmos de deteccion de iris (Gradient, Hough, Ellipse, Profile)
- Filtro de Kalman para suavizado de resultados en video
- Deteccion automatica de regla para calibracion

### Video
- Carga y procesamiento de videos (MP4, AVI, MOV, MKV, etc.)
- Seleccion de frames por fase (Basal, Iluminacion, Relajacion)
- Deteccion automatica de flash
- Recorte de region de interes
- Navegacion por frames

### Analisis
- Calculo de pupilometria con metricas clinicas
- Estimacion de frames indeterminados (ojos cerrados)
- Graficos de evolucion temporal
- Exportacion a CSV

### Interfaz
- Panel lateral con controles
- Canvas interactivo con edicion manual
- Ventanas de configuracion
- Soporte para atajos de teclado

## Instalacion

### Requisitos

- Python 3.8+
- OpenCV (`opencv-python`)
- NumPy
- Pillow
- Tkinter (incluido en Python para Windows/Mac)
- Matplotlib (para graficos)
- PyTorch (opcional, para IA)

### Instalacion rapida

```bash
pip install opencv-python numpy Pillow matplotlib
```

### Instalacion con soporte para IA

```bash
pip install torch torchvision
```

### Ejecucion

```bash
python -m pupilometer
```

o

```bash
python main.py
```

## Estructura del Proyecto

```
pupilometer/
├── __init__.py              # Inicializacion del paquete
├── __main__.py              # Punto de entrada
├── main.py                  # Aplicacion principal (Tkinter)
├── requirements.txt         # Dependencias
├── temp.txt                # Archivo temporal de referencia
├── prompt.txt              # Notas del desarrollador
│
├── core/                    # Modulos de deteccion (vision por computadora)
│   ├── __init__.py
│   ├── detection.py         # Deteccion basica
│   ├── advanced.py          # Detector avanzado
│   ├── darkcircle.py        # Detector dark circle
│   └── tracking.py          # Filtro de Kalman
│
├── analysis/                # Modulos de analisis clinico
│   ├── __init__.py
│   ├── pupillometry.py     # Calculos de pupilometria
│   ├── calibration.py       # Calibracion de regla
│   └── flash_detection.py   # Deteccion de flash en video
│
├── ai/                     # Modulos de inteligencia artificial
│   ├── __init__.py
│   ├── unet.py             # U-Net para segmentacion
│   └── dataset.py           # Grabador de dataset
│
├── ui/                     # Interfaz grafica (refactorizada)
│   ├── __init__.py
│   ├── main_window.py      # Ventana principal
│   ├── video_settings.py    # Ventana de ajustes de video
│   ├── analisis_video.py    # Ventana de analisis de video
│   └── dialogs.py           # Dialogos y ventanas
│
├── widgets/                # Widgets personalizados de Tkinter
│   ├── __init__.py
│   ├── range_slider.py      # Slider de rango dual
│   └── illumination_slider.py # Slider de iluminacion
│
├── services/               # Logica de negocio
│   ├── __init__.py
│   ├── settings_service.py # Gestion de configuraciones
│   ├── detection_service.py # Logica de deteccion
│   ├── video_service.py     # Procesamiento de video (legacy)
│   └── video_processor.py   # Procesamiento optimizado de video
│
└── utils/                  # Utilidades
    ├── __init__.py
    └── helpers.py           # Funciones auxiliares
```

## Modulos Principales

### `pupilometer.core`

Contiene los algoritmos de vision por computadora para la deteccion de pupila e iris.

**Clases principales:**
- `AdvancedEyeDetector`: Detector avanzado con multiples algoritmos
- `DetectionMode`: Enum con modos de deteccion
- `KalmanEye`: Filtro de Kalman para seguimiento

**Funciones:**
- `detect_all()`: Deteccion basica de pupila
- `draw_detections()`: Dibuja detecciones sobre imagen

### `pupilometer.analysis`

Contiene los modulos de analisis clinico.

**Funciones:**
- `compute()`: Calcula pupilometria
- `format_report()`: Genera informe de resultados
- `detect_flash_in_video()`: Detecta flash en video
- `calibrar_regla_rosa_fosforescente()`: Calibra con regla

### `pupilometer.ai`

Contiene los modulos de inteligencia artificial.

**Funciones:**
- `segment()`: Segmentacion con U-Net
- `train_model()`: Entrenamiento del modelo
- `extract_pupil_from_mask()`: Extrae pupila de mascara

### `pupilometer.ui`

Contiene la interfaz grafica modularizada.

**Modulos:**
- `main_window.py`: Ventana principal de la aplicacion
- `video_settings.py`: Ventana de configuracion de video
- `analisis_video.py`: Ventana de analisis de video
- `dialogs.py`: Dialogos reutilizables

### `pupilometer.services`

Contiene la logica de negocio separada de la interfaz.

**Modulos:**
- `settings_service.py`: Gestion de configuraciones persistentes
- `detection_service.py`: Logica de deteccion centralizada
- `video_service.py`: Procesamiento y carga de videos (legacy)
- `video_processor.py`: Procesamiento optimizado de video con streaming

**VideoProcessor**
Clase principal para procesamiento eficiente de video:
- `get_video_info()`: Obtiene metadatos sin cargar frames
- `calculate_phase_bounds()`: Calcula limites de fases
- `detect_flash_streaming()`: Detecta flash sin guardar frames
- `process()`: Procesa el video completo con streaming

### `pupilometer.widgets`

Contiene widgets personalizados para Tkinter.

**Clases:**
- `RangeSlider`: Slider de seleccion de rango dual
- `IlluminationSlider`: Slider para seleccion de iluminacion

## Uso

### Cargar una imagen

1. Archivo > Importar Imagen (o Ctrl+I)
2. Seleccionar la imagen
3. Presionar "Detectar" para analisis automatico

### Cargar un video

1. Archivo > Importar Video (o Ctrl+V)
2. Configurar las fases del video en la ventana de ajustes
3. Presionar "Aceptar"
4. Usar la barra de navegacion para explorar frames
5. Presionar "Detectar" para analisis

### Configurar deteccion

1. Ajustes > Analisis de Video
2. Configurar fps y duracion de cada fase
3. Configurar deteccion automatica de flash

### Calibrar con regla

1. Ajustes > Calibracion de Regla
2. Ingresar la distancia entre puntos de la regla
3. La regla rosa fosforescente se detectara automaticamente

### Usar IA (U-Net)

1. IA / Dataset > Usar IA en deteccion
2. Entrenar con dataset o cargar modelo pre-entrenado

## Algoritmos de Deteccion

### Algoritmo de Procesamiento de Video

El procesamiento de video en Pupilometer esta optimizado para manejar videos grandes sin consumir toda la memoria. Utiliza un enfoque de **streaming** que procesa los frames uno por uno sin necesidad de cargar todo el video en memoria.

#### Fases del Algoritmo

**1. Obtencion de Informacion del Video**
- Leer metadatos del video (FPS, duracion, resolucion)
- No se cargan frames en esta etapa
- Validar que el video se pueda abrir

**2. Calculo de Limites de Fases**
Se definen tres fases segun el protocolo de estimulacion:

```
Fase 1 (Basal):      [0, duracion_fase1]
Fase 2 (Iluminacion): [t1_end, t1_end + duracion_fase2]
Fase 3 (Relajacion):  [t2_end, t2_end + duracion_fase3]
```

Si la deteccion automatica de flash esta habilitada:
- Se detectan los tiempos de flash ON/OFF
- Los limites se ajustan automaticamente segun el flash detectado
- Fase 1 termina antes del flash
- Fase 2 abarca el periodo de flash
- Fase 3 comienza despues del flash

**3. Calculo de Frames Requeridos**
Para cada fase se calcula el numero de frames necesarios:

```
frames_fase = ceil(duracion_fase * min(fps_objetivo, fps_original))
```

**4. Procesamiento en Streaming**
- Se procesa el video frame por frame
- Para cada frame se determina a que fase pertenece segun su timestamp
- Se selecciona el frame si cumple con el intervalo de muestreo de su fase
- Los frames se almacenan en una lista solo si son necesarios
- Se usa el intervalo: `1.0 / min(fps_objetivo, fps_original)`

**5. Finalizacion**
- Se calculan los timestamps finales
- Se determina el FPS efectivo: `n_frames / duracion`
- Se asignan frames a sus fases para pupillometria

#### Compatibilidad con Pupillometria

Los timestamps se mantienen en segundos desde el inicio del video original. Esto asegura que:
- Los calculos de pupillometria usan tiempos reales
- Las posiciones relativas de los frames se mantienen
- El analisis de fases es preciso

#### Manejo de Casos Especiales

| Situacion | Comportamiento |
|-----------|----------------|
| Video mas corto que requerido | Escala las duraciones proporcionalmente |
| Video mas largo que requerido | Recorta al final de la fase 3 |
| FPS original menor al objetivo | Usa el FPS original (no se puede aumentar) |
| FPS original mayor al objetivo | Reduce segun el objetivo de cada fase |
| Flash no detectado | Usa los tiempos configurados por el usuario |
| Memoria insuficiente | El streaming evita este problema |

### Algoritmos de Pupila

| Algoritmo | Descripcion |
|-----------|-------------|
| `starburst` | Algoritmo starburst clasico |
| `swirski` | Algoritmo de Swirski |
| `excuse` | EXCUSE algorithm |
| `canny` | Deteccion con Canny |
| `threshold` | Deteccion por umbral |
| `darkcircle` | Detecta circulos oscuros |
| `ia` | U-Net (requiere entrenamiento) |

### Algoritmos de Iris

| Algoritmo | Descripcion |
|-----------|-------------|
| `gradient` | Deteccion por gradiente |
| `hough` | Transformada de Hough |
| `ellipse` | Ajuste de elipse |
| `profile` | Perfil de intensidad |

## Configuracion

### Parametros de Flash

Valores por defecto (definidos en `temp.txt`):

```python
CALIBRATION_FRAMES = 5      # Frames para calibracion
CHANGE_THRESHOLD = 5.0      # Umbral de cambio de brillo
SMOOTHING_WINDOW = 4        # Ventana de suavizado
HISTORY_MAX = 300           # Historial maximo
```

### Fases del Video

Configurables en la interfaz:

- **Fase 1 - Basal**: Pre-estimulo (default: 2 fps, 3 seg)
- **Fase 2 - Iluminacion**: Estimulo (default: 30 fps, 2 seg)
- **Fase 3 - Relajacion**: Post-estimulo (default: 10 fps, 7 seg)

### Protocolo de Estimulacion

Tiempos por defecto para el flash:

- Flash ON: 1.0 segundos
- Flash OFF: 1.2 segundos
- Duracion total: 6.0 segundos

## Desarrollo

### Agregar un nuevo algoritmo de deteccion

1. Implementar en `core/advanced.py`
2. Agregar al enum `DetectionMode`
3. Agregar opcion en la lista de algoritmos en `ui/main_window.py`

### Agregar una nueva ventana de dialogo

1. Crear en `ui/dialogs.py`
2. Importar y usar en `ui/main_window.py`

### Ejecutar tests

```bash
python -m pytest
```

### Construir dokumentacion

```bash
python -m pdoc --html pupilometer
```

## Licencia

Este proyecto es de uso interno/investigacion.

## Autores

Desarrollado para investigacion en pupilometria.
