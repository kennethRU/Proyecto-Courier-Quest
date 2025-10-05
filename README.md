Proyecto courier Quest
Este proyecto es un juego desarrollado en Python utilizando la librería Pygame. Incluye módulos para manejar la lógica del juego, inventario, clima, persistencia de datos y una interfaz de usuario.

Características principals

Sistema de inventario (inventory.py)
Módulo de clima dinámico (weather.py)
Gestión de datos persistentes (persistence.py)
Utilidades varias (utils.py, sorting.py)
Interfaz de usuario con Pygame (ui.py)
Lógica central del juego (game.py, main.py)
Integración con API externa (api.py)
Configuración centralizada (config.py)

Estructura del Proyecto 

│── api.py              # Manejo de API externas
│── config.py           # Configuración global del proyecto
│── game.py             # Mecánicas principales del juego
│── inventory.py        # Sistema de inventario
│── main.py             # Punto de entrada principal
│── models.py           # Clases y modelos de datos
│── persistence.py      # Guardado y carga de datos
│── sorting.py          # Algoritmos de ordenamiento
│── ui.py               # Interfaz de usuario con Pygame
│── utils.py            # Funciones auxiliares
│── weather.py          # Simulación del clima
│── requirements.txt    # Dependencias del proyecto
│── .venv/              # Entorno virtual de Python

Requisitos 

Python 3.10+
pygame 2.5.2
requests 2.32.3

Instalacion

clonar el repositorio con el commando: git clone
instalar dependencias con el comando: py -m pip install --user -r requirements.txt

Ejecución

Comando por consola: python main.py

Teclas de control del juego

↑ ↓ ← → : Mover al jugador por la ciudad.
A : Aceptar y recoger el pedido cercano
D : Entregar el pedido actualmente seleccionado en el inventario
c : Cancelar el pedido actualmente seleccionado
[ : Mover el cursor del inventario hacia atras
] : Mover el cursor del inventario hacia adelante
1 : Vista de inventario "natural"
2 : Vista de inventario por "prioridad"
3 : Vista de inventario por "deadline" (hora de entrega)
u : Deshacer acción (hasta 20 pasos)
s : Guardar la partida actual
L : Cargar la partida guardada
R : Reiniciar el juego (solo si el juego terminó)
H : Mostrar historial de partidas (solo si el juego terminó)
ESC : Si estás viendo el historial
     - cerrar historial. Si estás en el juego
     - salir del juego

Desarrollo 

Models → Define las estructuras de datos
Game / UI → Contienen la lógica principal y la interfaz
Persistence → Permite guardar y cargar partidas
Weather → Añade realismo dinámico con clima
Utils / Sorting → Encapsulan funciones auxiliares y algoritmos


Licencias/Uso

Proyecto de uso academico y libre de modificaciones

Autores/Creditos 

Kenneth Ramirez Ugalde
Fernan Mesen Barboza