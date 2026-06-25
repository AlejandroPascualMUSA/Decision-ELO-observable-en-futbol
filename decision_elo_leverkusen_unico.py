"""
Decision-Elo para Bayer Leverkusen 2023/24
==========================================

Este modulo junta cuatro ideas del trabajo:
1. EPV: valor esperado de posesion por zonas del campo.
2. Probabilidad de tiro: calidad del tiro segun distancia, angulo y presion.
3. Probabilidad/riesgo de pase: carrera temporal entre balon y jugadores visibles.
4. Notas tipo ajedrez: compara la accion real con la mejor accion disponible.

El codigo NO depende de mplsoccer. Dibuja el campo con matplotlib para que sea
facil de ejecutar en cualquier entorno.

Estructura de datos esperada para el analisis completo:

data/
  competitions.json
  matches/9/281.json
  events/<match_id>.json
  three-sixty/<match_id>.json
  lineups/<match_id>.json  (opcional)

El codigo busca por defecto una carpeta llamada data en el mismo directorio
desde el que ejecutes Python/Jupyter. Aunque data/events y data/three-sixty
contengan miles de JSON de otros partidos, NO los recorre todos: primero lee
data/matches/9/281.json, identifica los match_id de Bayer Leverkusen y despues
carga exclusivamente data/events/<match_id>.json y data/three-sixty/<match_id>.json.

Por defecto la ejecucion directa analiza SOLO 1 partido de Bayer para que no
reviente el ordenador. Si quieres toda la temporada, llama a ejecutar_bayer(...,
max_matches=None). Tambien acepta rutas absolutas de Windows.

Actualizacion solicitada:
- Analisis principal: pases completados, rasos/bajos y en juego; tiros totales.
  Los pases incompletos se conservan en el JSON de revision, pero no entran en
  el Decision-Elo principal para evitar mezclar decision y ejecucion.
- En los pases candidatos se controla el fuera de juego con una regla simple
  orientada a x creciente. El portero no cuenta para esa comprobacion.
- Los diagramas 360 usan Bayer en azul, rivales en rojo, actor con estrella azul,
  accion real negra continua y mejor accion morada discontinua. Solo se dibujan
  pases si actor y receptor aparecen en el 360, y tiros si la porteria aparece
  dentro del area visible 360.
- Se generan diagramas de tiros y graficos extra: riesgo vs DeltaEPV, ELO por
  posicion, score por tipo de accion, Q_real vs Q_best y fuentes de valor.
- Los ejemplos visuales de mala decision se eligen con criterios de claridad,
  evitando casos con Q_real casi nulo y regret pequeno que puedan resultar confusos.
- La clasificacion de posiciones distingue correctamente los centrales tipo
  Left/Right Center Back de los laterales.
"""

from __future__ import annotations

import json
import math
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Polygon, Rectangle

# -----------------------------------------------------------------------------
# 0. Constantes generales
# -----------------------------------------------------------------------------
PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0
GOAL_CENTER = np.array([120.0, 40.0])
LEFT_POST = np.array([120.0, 36.0])
RIGHT_POST = np.array([120.0, 44.0])

BAYER_TEAM_ID = 904
BUNDESLIGA_COMPETITION_ID = 9
BUNDESLIGA_2023_24_SEASON_ID = 281

# Por defecto trabajamos con 1 partido para que la ejecucion sea ligera.
# Usa max_matches=None si quieres procesar la temporada completa.
DEFAULT_MAX_MATCHES = 1
DEFAULT_MATCH_INDEX = 0

# Filtros del MODELO PRINCIPAL. El JSON de revision conserva todos los pases y
# todos los tiros, pero el ranking Decision-Elo principal usa pases completados
# y tiros totales.
USE_ONLY_COMPLETED_PASSES_MAIN = True
# Filtro extra de seguridad: el analisis principal y los diagramas solo
# aceptan pases con pass.outcome ausente y con receptor registrado.
REQUIRE_RECIPIENT_FOR_COMPLETED_PASS = True
EXCLUDE_HIGH_PASSES_MAIN = True
EXCLUDE_SET_PIECES_MAIN = True
USE_ALL_SHOTS_MAIN = True

# Para elegir ejemplos visuales evitamos acciones de impacto minimo.
MIN_QBEST_FOR_DIAGRAM = 0.015
MIN_REGRET_FOR_BAD_DIAGRAM = 0.007
MIN_QREAL_FOR_GOOD_DIAGRAM = 0.015

# Criterios especificos para elegir ejemplos visuales de mala decision.
# Evitan casos confusos con score=0 por Q_real casi nulo y regret pequeno/moderado.
MIN_QBEST_FOR_CLEAR_BAD_DIAGRAM = 0.030
MIN_REGRET_FOR_CLEAR_BAD_DIAGRAM = 0.020
MIN_QREAL_FOR_CLEAR_BAD_DIAGRAM = 0.003
MAX_SCORE_FOR_CLEAR_BAD_DIAGRAM = 0.750
MIN_SCORE_FOR_CLEAR_BAD_DIAGRAM = 0.030

# Criterios para que el grafico metodologico de pase sea explicativo.
# Buscamos una linea de pase larga y con al menos una interseccion visible
# entre la trayectoria discretizada del balon y la zona de alcance de un rival.
MIN_PASS_LENGTH_FOR_METHOD_GRID = 22.0
MIN_PASS_INTERSECTIONS_FOR_METHOD_GRID = 1
METHODOLOGY_GRID_RADIUS = 2.8

# Regla simplificada de fuera de juego usada en este trabajo.
# Se asume ataque hacia la derecha en StatsBomb: mayor x = mas cerca de la porteria rival.
# Un pase adelantado a un companero se considera no valido si no hay ningun rival
# de campo, excluido el portero, con x mayor que la x del receptor.
OFFSIDE_X_MARGIN = 0.25
OFFSIDE_REQUIRE_OPPONENT_BEHIND = True

# Tolerancias para decidir si un receptor/candidato esta visible en el freeze-frame.
# En StatsBomb 360 no siempre hay identificador nominal del jugador visible, por
# eso se permite una busqueda espacial alrededor del destino del pase.
VISIBLE_RECEIVER_TOLERANCE = 5.0
VISIBLE_BEST_RECEIVER_TOLERANCE = 1.75

NOTE_LABELS = {
    "!!": "Brillante",
    "!": "Mejor movimiento",
    "✓": "Excelente",
    "=": "Buena",
    "?!": "Imprecision",
    "?": "Error",
    "??": "Pifia"
}


# -----------------------------------------------------------------------------
# 1. Utilidades de lectura y validacion
# -----------------------------------------------------------------------------
def load_json(path: Path) -> Any:
    """Carga un archivo JSON con codificacion UTF-8."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Path) -> None:
    """Guarda un objeto en JSON bonito y legible."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def looks_like_statsbomb_data_root(data_root: Path) -> bool:
    """Comprueba si una carpeta parece la raiz data/ de StatsBomb Open Data.

    La estructura que se espera en tu caso es exactamente la de la captura:
        data/competitions.json
        data/matches/9/281.json
        data/events/<match_id>.json
        data/three-sixty/<match_id>.json
        data/lineups/<match_id>.json
    """
    data_root = Path(data_root)
    return (
        (data_root / "competitions.json").exists()
        and (data_root / "matches" / str(BUNDESLIGA_COMPETITION_ID) / f"{BUNDESLIGA_2023_24_SEASON_ID}.json").exists()
        and (data_root / "events").exists()
    )


def resolve_data_root(data_root: str | Path | None = None) -> Path:
    """Devuelve la raiz real de los datos.

    Esta funcion permite trabajar desde Python/Jupyter sin cambiar codigo:
    - si pasas "data", usa data/;
    - si pasas la carpeta del proyecto que contiene data/, entra automaticamente;
    - si no pasas nada, prueba data/ y open-data-master/data/.

    Ejemplos:
        resolve_data_root("data")
        resolve_data_root(r"C:/Users/.../Trabajo final/data")
        resolve_data_root(r"C:/Users/.../Trabajo final")  # contiene data/
    """
    candidates: List[Path] = []

    if data_root is not None:
        root = Path(data_root)
        candidates.extend([
            root,
            root / "data",
            root / "open-data-master" / "data",
        ])
    else:
        cwd = Path.cwd()
        here = Path(__file__).resolve().parent
        candidates.extend([
            cwd / "data",
            cwd / "open-data-master" / "data",
            cwd,
            here / "data",
            here / "open-data-master" / "data",
        ])

    seen = set()
    for cand in candidates:
        cand = cand.expanduser().resolve()
        if cand in seen:
            continue
        seen.add(cand)
        if looks_like_statsbomb_data_root(cand):
            return cand

    checked = "\n".join(f"  - {c}" for c in candidates)
    raise FileNotFoundError(
        "No encuentro la carpeta data con estructura StatsBomb. He buscado en:\n"
        f"{checked}\n\n"
        "Debe existir algo como:\n"
        "  data/competitions.json\n"
        "  data/matches/9/281.json\n"
        "  data/events/<match_id>.json\n"
        "  data/three-sixty/<match_id>.json"
    )


def has_full_bayer_data(data_root: Path) -> bool:
    """Comprueba si existe la estructura minima para Bayer 2023/24."""
    try:
        root = resolve_data_root(data_root)
    except FileNotFoundError:
        return False
    return looks_like_statsbomb_data_root(root) and (root / "three-sixty").exists()


def validate_or_demo(data_root: Path, demo_zip: Optional[Path] = None) -> str:
    """
    Devuelve 'bayer' si estan los datos completos; si no, prepara modo demo.

    El ZIP que se adjunto en la conversacion contiene un partido de ejemplo,
    no los eventos/360 de Bayer. Por eso esta funcion hace que el proyecto sea
    reproducible aunque aun falten los datos finales.
    """
    if has_full_bayer_data(data_root):
        return "bayer"
    if demo_zip and demo_zip.exists():
        return "demo"
    return "missing"


# -----------------------------------------------------------------------------
# 2. Dibujo del campo y diagramas 360
# -----------------------------------------------------------------------------
def draw_pitch(ax: plt.Axes, *, dark: bool = False) -> None:
    """Dibuja un campo StatsBomb 120x80 en un eje matplotlib."""
    ax.set_xlim(0, PITCH_LENGTH)
    ax.set_ylim(0, PITCH_WIDTH)
    ax.set_aspect("equal")
    ax.axis("off")

    line_color = "white" if dark else "black"
    ax.set_facecolor("#2b2b2b" if dark else "#f7f7f7")

    # Contorno y medio campo
    ax.add_patch(Rectangle((0, 0), 120, 80, fill=False, ec=line_color, lw=1.2))
    ax.plot([60, 60], [0, 80], color=line_color, lw=1.0)
    ax.add_patch(Arc((60, 40), 20, 20, theta1=0, theta2=360, color=line_color, lw=1.0))

    # Areas
    ax.add_patch(Rectangle((0, 18), 18, 44, fill=False, ec=line_color, lw=1.0))
    ax.add_patch(Rectangle((102, 18), 18, 44, fill=False, ec=line_color, lw=1.0))
    ax.add_patch(Rectangle((0, 30), 6, 20, fill=False, ec=line_color, lw=1.0))
    ax.add_patch(Rectangle((114, 30), 6, 20, fill=False, ec=line_color, lw=1.0))

    # Puntos de penalti y porterias
    ax.scatter([12, 60, 108], [40, 40, 40], s=8, color=line_color, zorder=2)
    ax.add_patch(Rectangle((-2, 36), 2, 8, fill=False, ec=line_color, lw=1.0))
    ax.add_patch(Rectangle((120, 36), 2, 8, fill=False, ec=line_color, lw=1.0))


def add_current_note_box(ax: plt.Axes, selected_symbol: Optional[str] = None, label: Optional[str] = None, loc: str = "upper right") -> None:
    """Muestra solo la nota de ESTA accion, no toda la escala.

    La escala completa se guarda aparte en 00b_escala_notas_ajedrez.png. En los
    diagramas de eventos dejamos una caja pequena con el icono y su significado
    para no tapar el campo.
    """
    if not selected_symbol:
        return
    label = label or NOTE_LABELS.get(selected_symbol, "")
    text = f"{selected_symbol}  {label}"
    if loc == "upper left":
        x, y, ha, va = 0.01, 0.99, "left", "top"
    elif loc == "lower right":
        x, y, ha, va = 0.99, 0.01, "right", "bottom"
    else:
        x, y, ha, va = 0.99, 0.99, "right", "top"
    ax.text(
        x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="black", alpha=0.90),
        zorder=20, weight="bold"
    )


def add_action_legend(
    ax: plt.Axes,
    *,
    has_actual: bool = True,
    has_best: bool = True,
    is_shot: bool = False,
    best_is_pass: bool = False,
    loc: str = "lower left",
) -> None:
    """Leyenda compacta de los planos 360.

    Convencion visual final:
    - Bayer / companeros del actor: azul.
    - Rivales: rojo.
    - Portero: amarillo circular.
    - Jugador que realiza la accion: estrella azul.
    - Accion real: linea negra continua.
    - Mejor accion: linea morada discontinua.
    - Si la mejor accion es un pase a un companero visible, ese companero se
      pinta morado en vez de azul. No se crean circulos morados artificiales.
    """
    handles = [
        plt.Line2D(
            [0], [0], marker="*", color="w", markerfacecolor="tab:blue",
            markeredgecolor="black", markersize=14, label="Jugador con balon"
        ),
    ]
    if has_actual:
        label = "Tiro real" if is_shot else "Pase real"
        handles.append(plt.Line2D([0], [0], color="black", lw=3, ls="-", label=label))
    if has_best:
        handles.append(plt.Line2D([0], [0], color="purple", lw=3, ls="--", label="Mejor accion"))
        if best_is_pass:
            handles.append(
                plt.Line2D(
                    [0], [0], marker="o", color="w", markerfacecolor="purple",
                    markeredgecolor="black", markersize=9, label="Mejor receptor"
                )
            )
    if is_shot:
        handles.append(plt.Line2D([0], [0], color="green", lw=1.5, ls=":", label="Triangulo de tiro"))
    ax.legend(handles=handles, loc=loc, frameon=True)



def _visible_area_points(frame: Optional[Dict[str, Any]]) -> Optional[np.ndarray]:
    """Devuelve el poligono visible de StatsBomb 360 como matriz Nx2."""
    if not frame:
        return None
    visible_area = frame.get("visible_area")
    if not visible_area or len(visible_area) < 6:
        return None
    try:
        return np.array(visible_area, dtype=float).reshape(-1, 2)
    except Exception:
        return None


def _point_on_segment(point: np.ndarray, a: np.ndarray, b: np.ndarray, tol: float = 1e-6) -> bool:
    """Comprueba si un punto cae sobre un segmento, usado para bordes del 360."""
    ab = b - a
    ap = point - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-12:
        return float(np.linalg.norm(point - a)) <= tol
    t = float(np.dot(ap, ab) / denom)
    if t < -tol or t > 1.0 + tol:
        return False
    closest = a + np.clip(t, 0.0, 1.0) * ab
    return float(np.linalg.norm(point - closest)) <= tol


def point_in_visible_area(point: List[float] | np.ndarray, frame: Optional[Dict[str, Any]], margin: float = 0.50) -> bool:
    """Comprueba si un punto esta dentro del area visible 360.

    Para los puntos en la linea de banda/porteria se anade una tolerancia porque
    muchos poligonos 360 cortan exactamente en x=120 y el test geometrico puede
    clasificar el punto como borde.
    """
    pts = _visible_area_points(frame)
    if pts is None:
        return False
    p = np.array(point[:2], dtype=float)

    # Filtro rapido con margen.
    if p[0] < float(np.min(pts[:, 0])) - margin or p[0] > float(np.max(pts[:, 0])) + margin:
        return False
    if p[1] < float(np.min(pts[:, 1])) - margin or p[1] > float(np.max(pts[:, 1])) + margin:
        return False

    # Borde del poligono.
    for i in range(len(pts)):
        if _point_on_segment(p, pts[i], pts[(i + 1) % len(pts)], tol=margin):
            return True

    # Ray casting.
    inside = False
    x, y = float(p[0]), float(p[1])
    n = len(pts)
    j = n - 1
    for i in range(n):
        xi, yi = float(pts[i, 0]), float(pts[i, 1])
        xj, yj = float(pts[j, 0]), float(pts[j, 1])
        intersects = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi)
        if intersects:
            inside = not inside
        j = i
    return inside


def goal_visible_in_360(frame: Optional[Dict[str, Any]]) -> bool:
    """Un diagrama de tiro solo se dibuja si la porteria derecha aparece en el 360."""
    pts = _visible_area_points(frame)
    if pts is None:
        return False
    goal_points = [GOAL_CENTER, LEFT_POST, RIGHT_POST, np.array([119.0, 40.0])]
    if any(point_in_visible_area(g, frame, margin=1.0) for g in goal_points):
        return True
    # Fallback: el poligono llega practicamente a la porteria y cubre su altura.
    return bool(float(np.max(pts[:, 0])) >= 119.0 and float(np.min(pts[:, 1])) <= 44.5 and float(np.max(pts[:, 1])) >= 35.5)


def actor_visible_in_360(frame: Optional[Dict[str, Any]], origin: Optional[List[float]] = None, tolerance: float = 2.0) -> bool:
    """Comprueba si el jugador que realiza la accion aparece como actor en el 360."""
    if not frame or not frame.get("freeze_frame"):
        return False
    for player in frame.get("freeze_frame", []):
        loc = player.get("location")
        if not loc or len(loc) < 2:
            continue
        if player.get("actor"):
            if origin is None:
                return True
            return float(np.hypot(float(loc[0]) - float(origin[0]), float(loc[1]) - float(origin[1]))) <= tolerance
    return False


def visible_teammate_near_target(
    frame: Optional[Dict[str, Any]],
    target: Optional[List[float]],
    tolerance: float = VISIBLE_RECEIVER_TOLERANCE,
) -> Optional[Dict[str, Any]]:
    """Busca un companero visible cerca del destino/candidato de pase.

    La funcion no crea un receptor artificial: solo devuelve un jugador real del
    freeze-frame. Si no hay companero visible cerca, el diagrama de pase no se
    genera para evitar figuras confusas.
    """
    if not frame or not frame.get("freeze_frame") or not target or len(target) < 2:
        return None
    tx, ty = float(target[0]), float(target[1])
    best_player = None
    best_dist = float("inf")
    for player in frame.get("freeze_frame", []):
        if not player.get("teammate") or player.get("actor") or is_keeper_player(player):
            continue
        loc = player.get("location")
        if not loc or len(loc) < 2:
            continue
        dist = float(np.hypot(float(loc[0]) - tx, float(loc[1]) - ty))
        if dist < best_dist:
            best_dist = dist
            best_player = player
    if best_player is not None and best_dist <= tolerance:
        return best_player
    return None


def pass_visible_in_360(
    frame: Optional[Dict[str, Any]],
    origin: Optional[List[float]],
    target: Optional[List[float]],
    tolerance: float = VISIBLE_RECEIVER_TOLERANCE,
) -> bool:
    """Un pase se dibuja solo si actor y receptor/candidato aparecen en el 360."""
    return bool(actor_visible_in_360(frame, origin) and visible_teammate_near_target(frame, target, tolerance=tolerance))


def last_outfield_rival_x(frame: Optional[Dict[str, Any]]) -> float:
    """Max x de los rivales de campo visibles, excluyendo portero y actor."""
    if not frame or not frame.get("freeze_frame"):
        return float("nan")
    xs: List[float] = []
    for player in frame.get("freeze_frame", []):
        if player.get("teammate") or player.get("actor") or is_keeper_player(player):
            continue
        loc = player.get("location")
        if loc and len(loc) >= 2:
            xs.append(float(loc[0]))
    return max(xs) if xs else float("nan")


def offside_line_for_plot(origin: List[float], frame: Optional[Dict[str, Any]]) -> float:
    """Linea visual de fuera de juego: la mas adelantada entre balon y ultimo rival.

    Para la simplificacion del trabajo, se mira hacia x creciente. Dibujar una
    unica linea evita la confusion de pintar lineas distintas de balon/receptor.
    """
    ball_x = float(origin[0]) if origin and len(origin) >= 1 else float("nan")
    last_x = last_outfield_rival_x(frame)
    if not np.isfinite(last_x):
        return ball_x
    if not np.isfinite(ball_x):
        return last_x
    return float(max(ball_x, last_x))


def plot_freeze_frame(
    event: Dict[str, Any],
    frame: Optional[Dict[str, Any]],
    save_path: Optional[Path] = None,
    title: Optional[str] = None,
    actual_target: Optional[List[float]] = None,
    best_target: Optional[List[float]] = None,
    best_action: Optional[str] = None,
    note: Optional[str] = None,
    note_symbol: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Diagrama 360 de un evento con la convencion visual final.

    - Bayer/companeros del actor: azul.
    - Rivales: rojo.
    - Portero: amarillo circular.
    - Actor/jugador con balon: estrella azul.
    - Accion real: linea negra continua, sin X ni cambio de color del destino.
    - Mejor accion: linea morada discontinua.
    - Si la mejor accion es un pase y el receptor esta visible en el 360, ese
      companero se pinta morado en vez de azul. No se crea un circulo morado
      desde cero en tiros ni en destinos no visibles.
    """
    loc = event.get("location")
    freeze_frame = frame.get("freeze_frame") if frame else None
    visible_area = frame.get("visible_area") if frame else None
    if not loc or not freeze_frame:
        return None

    event_type = event.get("type", {}).get("name")
    # Pases: solo se dibuja si actor y receptor real aparecen en el 360.
    if event_type == "Pass" and actual_target and not pass_visible_in_360(frame, loc, actual_target):
        return None
    # Tiros: solo se dibuja si la porteria aparece en el area visible 360.
    if event_type == "Shot" and not goal_visible_in_360(frame):
        return None
    # Si la mejor alternativa es un pase, el receptor morado debe ser un jugador visible.
    if best_action == "Pass" and best_target and not visible_teammate_near_target(frame, best_target, tolerance=VISIBLE_BEST_RECEIVER_TOLERANCE):
        return None

    fig, ax = plt.subplots(figsize=(12, 8))
    draw_pitch(ax)

    # Area visible de StatsBomb 360, si existe.
    if visible_area and len(visible_area) >= 6:
        pts = np.array(visible_area, dtype=float).reshape(-1, 2)
        ax.add_patch(Polygon(pts, closed=True, alpha=0.08, ec="black", fc="gray", lw=1.0, zorder=0))

    best_is_pass = (best_action == "Pass") and bool(best_target)

    def is_same_as_best(p_loc: Any, tolerance: float = VISIBLE_BEST_RECEIVER_TOLERANCE) -> bool:
        """Detecta si un jugador visible coincide con el receptor de la mejor opcion."""
        if not best_is_pass or not best_target or not p_loc or len(p_loc) < 2:
            return False
        return float(np.hypot(float(p_loc[0]) - float(best_target[0]), float(p_loc[1]) - float(best_target[1]))) <= tolerance

    # Jugadores visibles. Los companeros del actor son Bayer en estos partidos.
    for p in freeze_frame:
        p_loc = p.get("location")
        if not p_loc or len(p_loc) < 2:
            continue
        x, y = p_loc[:2]
        if p.get("actor"):
            continue

        is_team = bool(p.get("teammate"))
        is_keeper = bool(p.get("keeper")) or p.get("position", {}).get("id") == 1
        is_best_receiver = is_team and is_same_as_best(p_loc)

        if is_keeper:
            color, size, marker, zorder = "gold", 120, "o", 4
        elif is_best_receiver:
            color, size, marker, zorder = "purple", 150, "o", 6
        elif is_team:
            color, size, marker, zorder = "tab:blue", 90, "o", 3
        else:
            color, size, marker, zorder = "tab:red", 90, "o", 3

        ax.scatter(x, y, s=size, marker=marker, c=color, ec="black", zorder=zorder)

    # Jugador con balon / actor: estrella azul grande.
    ax.scatter(loc[0], loc[1], s=340, marker="*", c="tab:blue", ec="black", lw=0.9, zorder=7)
    # Accion real: linea negra continua. No se dibuja X ni circulo en destino.
    if actual_target and len(actual_target) >= 2:
        ax.plot(
            [loc[0], actual_target[0]], [loc[1], actual_target[1]],
            lw=3, ls="-", c="black", zorder=2
        )
    # Mejor accion: solo linea morada. Si es pase y el receptor esta visible,
    # el jugador ya se ha pintado morado arriba.
    if best_target and len(best_target) >= 2:
        ax.plot([loc[0], best_target[0]], [loc[1], best_target[1]], lw=3, ls="--", c="purple", zorder=3)
    # Si es tiro, dibujamos triangulo de tiro. No se anade ningun circulo extra.
    if event.get("type", {}).get("name") == "Shot":
        triangle = np.array([loc[:2], LEFT_POST, RIGHT_POST], dtype=float)
        ax.fill(triangle[:, 0], triangle[:, 1], alpha=0.15, color="green", zorder=1)
        ax.plot([loc[0], LEFT_POST[0]], [loc[1], LEFT_POST[1]], ls=":", c="green", lw=1.5)
        ax.plot([loc[0], RIGHT_POST[0]], [loc[1], RIGHT_POST[1]], ls=":", c="green", lw=1.5)

    if title is None:
        title = f"{event.get('minute', '?')}:{str(event.get('second', '?')).zfill(2)} - {event.get('type', {}).get('name')}"
    if note:
        title += f"\nNota: {note}"
    ax.set_title(title, fontsize=14, weight="bold")
    add_current_note_box(ax, selected_symbol=note_symbol, label=NOTE_LABELS.get(note_symbol or ""), loc="upper right")
    add_action_legend(
        ax,
        has_actual=bool(actual_target),
        has_best=bool(best_target),
        best_is_pass=best_is_pass,
        is_shot=event.get("type", {}).get("name") == "Shot",
        loc="lower left",
    )

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_shot_diagram(
    event: Dict[str, Any],
    frame: Optional[Dict[str, Any]],
    decision: Optional[Dict[str, Any]],
    save_path: Path,
) -> Optional[plt.Figure]:
    """Diagrama especifico para tiros.

    Muestra el triangulo de tiro, la accion real hacia porteria y, si el motor
    encuentra una alternativa mejor de pase, la linea discontinua hacia ese
    companero. Tambien incluye la escala de notas tipo ajedrez dentro del campo.
    """
    if event.get("type", {}).get("name") != "Shot":
        return None
    if decision is None:
        note_text = None
        note_symbol = None
        best_target = None
        title = f"Tiro - {event.get('player', {}).get('name', '')}"
    else:
        note_symbol = decision.get("note_symbol")
        note_text = f"score={decision.get('decision_score', 0):.2f}, regret={decision.get('regret', 0):.3f}"
        best_target = decision.get("best_target") if decision.get("best_action") == "Pass" else None
        title = (
            f"Diagrama de tiro - {decision.get('player_name')} - "
            f"{decision.get('note_symbol')} {decision.get('note_label')}"
        )
    return plot_freeze_frame(
        event=event,
        frame=frame,
        save_path=save_path,
        title=title,
        actual_target=[120.0, 40.0],
        best_target=best_target,
        best_action="Pass" if best_target else None,
        note=note_text,
        note_symbol=note_symbol,
    )


def get_line_cells(origin: List[float], target: List[float], cell_size: float = 1.0) -> set:
    """Celdas de una trayectoria de pase discretizada."""
    if not origin or not target:
        return set()
    x1, y1 = float(origin[0]), float(origin[1])
    x2, y2 = float(target[0]), float(target[1])
    dist = float(np.hypot(x2 - x1, y2 - y1))
    n = max(int(dist / max(cell_size, 1e-6) * 2), 2)
    cells = set()
    for t in np.linspace(0.0, 1.0, n):
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        cells.add((int(x // cell_size), int(y // cell_size)))
    return cells


def get_player_cells(location: List[float], radius: float = 2.0, cell_size: float = 1.0) -> set:
    """Celdas ocupadas por un jugador con un radio dado."""
    if not location or len(location) < 2:
        return set()
    x, y = float(location[0]), float(location[1])
    cells = set()
    for cx in range(int((x - radius) // cell_size), int((x + radius) // cell_size) + 1):
        for cy in range(int((y - radius) // cell_size), int((y + radius) // cell_size) + 1):
            center_x = (cx + 0.5) * cell_size
            center_y = (cy + 0.5) * cell_size
            if np.hypot(center_x - x, center_y - y) <= radius:
                cells.add((cx, cy))
    return cells


def plot_pass_grid(
    origin: List[float],
    target: List[float],
    frame: Optional[Dict[str, Any]],
    save_path: Path,
    cell_size: float = 1.0,
    radius: float = METHODOLOGY_GRID_RADIUS,
    title: Optional[str] = None,
    note_symbol: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Figura metodologica de pase con grid defensivo y fuera de juego.

    Esta figura no marca una mejor alternativa, sino que explica como se evalua
    una linea de pase concreta: trayectoria del balon, celdas ocupadas por
    defensores rivales, intersecciones con la trayectoria y lineas de fuera de
    juego. Se mantiene la misma convencion cromatica que los planos 360:
    Bayer/companeros en azul, rivales en rojo y portero en amarillo.
    """
    if not origin or not target or not frame or not frame.get("freeze_frame"):
        return None
    # Para evitar diagramas confusos, solo se dibuja si el actor y el receptor
    # del pase aparecen como jugadores visibles en el freeze-frame 360.
    if not pass_visible_in_360(frame, origin, target):
        return None

    freeze_frame = frame.get("freeze_frame", [])
    path_cells = get_line_cells(origin, target, cell_size)
    defender_cells_all = set()
    intersection_cells = set()

    for player in freeze_frame:
        if player.get("teammate") or is_keeper_player(player) or player.get("actor"):
            continue
        loc = player.get("location")
        if not loc or len(loc) < 2:
            continue
        def_cells = get_player_cells(loc, radius, cell_size)
        defender_cells_all |= def_cells
        intersection_cells |= def_cells & path_cells

    offside_info = offside_components(origin, target, frame)

    fig, ax = plt.subplots(figsize=(12, 8))
    draw_pitch(ax)

    # Trayectoria discretizada y celdas defensivas.
    for cx, cy in path_cells:
        ax.add_patch(Rectangle((cx * cell_size, cy * cell_size), cell_size, cell_size, color="green", alpha=0.22, zorder=1))
    for cx, cy in defender_cells_all:
        ax.add_patch(Rectangle((cx * cell_size, cy * cell_size), cell_size, cell_size, color="tab:red", alpha=0.18, zorder=2))
    for cx, cy in intersection_cells:
        ax.add_patch(Rectangle((cx * cell_size, cy * cell_size), cell_size, cell_size, color="purple", alpha=0.68, zorder=7))

    # Jugadores: Bayer azul, rivales rojo, portero amarillo, actor estrella azul.
    for player in freeze_frame:
        loc = player.get("location")
        if not loc or len(loc) < 2:
            continue
        x, y = loc[:2]
        if player.get("keeper") or player.get("position", {}).get("id") == 1:
            ax.scatter(x, y, c="gold", s=120, edgecolors="black", marker="o", zorder=8)
        elif player.get("actor"):
            ax.scatter(x, y, c="tab:blue", s=300, edgecolors="black", marker="*", zorder=9)
        elif player.get("teammate"):
            ax.scatter(x, y, c="tab:blue", s=85, edgecolors="black", marker="o", zorder=8)
        else:
            ax.scatter(x, y, c="tab:red", s=85, edgecolors="black", marker="o", zorder=8)

    # Accion real/evaluada: linea negra continua. No se crea circulo morado.
    ax.scatter(origin[0], origin[1], c="tab:blue", s=340, marker="*", edgecolors="black", zorder=10)
    # Halo blanco + linea negra para que la trayectoria se vea clara incluso
    # cuando cruza celdas de riesgo o jugadores.
    ax.plot([origin[0], target[0]], [origin[1], target[1]], linestyle="-", color="white", linewidth=6.0, alpha=0.98, zorder=4, solid_capstyle="round")
    ax.plot([origin[0], target[0]], [origin[1], target[1]], linestyle="-", color="black", linewidth=3.5, zorder=5, solid_capstyle="round")

    # Linea visual de fuera de juego: una unica linea en la referencia valida.
    # Para x creciente, es la mas adelantada entre el balon y el ultimo rival de campo.
    offside_line_x = offside_line_for_plot(origin, frame)
    if np.isfinite(offside_line_x):
        ax.axvline(offside_line_x, color="orange", linestyle="--", linewidth=2.2, zorder=2)

    blocked = len(intersection_cells) > 0
    if offside_info.get("is_offside"):
        ax.text(2, 76, "FUERA DE JUEGO", fontsize=12, weight="bold", color="red",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.9), zorder=30)
    if title is None:
        estado = 'BLOQUEADO' if blocked else 'LIBRE'
        fj = ' | FUERA DE JUEGO' if offside_info.get("is_offside") else ''
        title = f"Metodologia de pase | trayectoria {estado}{fj}"
    ax.set_title(title, fontsize=13, weight="bold")
    add_current_note_box(ax, selected_symbol=note_symbol, label=NOTE_LABELS.get(note_symbol or ""), loc="upper right")
    ax.legend(handles=[
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="tab:blue", markeredgecolor="black", markersize=14, label="Jugador con balon"),
        plt.Line2D([0], [0], color="black", lw=3.2, ls="-", label="Linea de pase"),
        plt.Rectangle((0, 0), 1, 1, color="tab:red", alpha=0.18, label="Zona alcance rival"),
        plt.Rectangle((0, 0), 1, 1, color="purple", alpha=0.60, label="Interseccion"),
        plt.Line2D([0], [0], color="orange", lw=2, ls="--", label="Linea fuera de juego"),
    ], loc="lower left", frameon=True)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return fig


# -----------------------------------------------------------------------------
# 3. Carga de eventos StatsBomb
# -----------------------------------------------------------------------------
def get_bayer_match_ids(data_root: str | Path, verbose: bool = False) -> List[int]:
    """Devuelve SOLO los match_id de Bayer Leverkusen en Bundesliga 2023/24.

    Esta funcion sigue exactamente la logica que quieres usar para no mezclar
    partidos de otros equipos:

    1. Lee data/competitions.json.
    2. Filtra competition_id=9 y season_id=281.
    3. Abre unicamente data/matches/9/281.json.
    4. Dentro de ese archivo obtiene los partidos donde home_team_id o
       away_team_id es 904, el id de Bayer Leverkusen en StatsBomb.

    Importante: aunque data/events/ y data/three-sixty/ tengan miles de JSON de
    otros partidos, aqui NO se escanean. Mas adelante solo se cargan los archivos
    cuyo nombre coincide con estos match_id.
    """
    data_root = resolve_data_root(data_root)

    competitions_path = data_root / "competitions.json"
    matches_path = data_root / "matches"

    competitions = load_json(competitions_path)
    filtered_comp = [
        c for c in competitions
        if c.get("competition_id") == BUNDESLIGA_COMPETITION_ID
        and c.get("season_id") == BUNDESLIGA_2023_24_SEASON_ID
    ]

    if verbose:
        print(f"Competiciones filtradas: {len(filtered_comp)}")

    if not filtered_comp:
        raise ValueError(
            "No se ha encontrado competition_id=9 y season_id=281 en competitions.json. "
            "Comprueba que la carpeta data corresponde a StatsBomb Open Data."
        )

    matches_file = matches_path / str(BUNDESLIGA_COMPETITION_ID) / f"{BUNDESLIGA_2023_24_SEASON_ID}.json"
    matches = load_json(matches_file)

    match_ids: List[int] = []
    for m in matches:
        home_id = m.get("home_team", {}).get("home_team_id")
        away_id = m.get("away_team", {}).get("away_team_id")
        if home_id == BAYER_TEAM_ID or away_id == BAYER_TEAM_ID:
            match_ids.append(int(m["match_id"]))

    # Eliminar duplicados manteniendo orden, por seguridad.
    match_ids = list(dict.fromkeys(match_ids))

    if verbose:
        print(f"Partidos del Leverkusen: {len(match_ids)}")
        print("Match IDs Bayer:", match_ids)

    return match_ids


def select_bayer_match_ids(
    match_ids: List[int],
    max_matches: Optional[int] = DEFAULT_MAX_MATCHES,
    match_id: Optional[int] = None,
    match_index: int = DEFAULT_MATCH_INDEX,
    verbose: bool = True,
) -> List[int]:
    """Selecciona los partidos que se van a analizar.

    Por defecto devuelve UN solo partido de Bayer para que el analisis no sea
    pesado. Si quieres toda la temporada, pasa max_matches=None. Si quieres un
    partido concreto, pasa match_id=<id>.
    """
    match_ids = list(dict.fromkeys(int(m) for m in match_ids))
    if not match_ids:
        return []

    if match_id is not None:
        match_id = int(match_id)
        if match_id not in match_ids:
            raise ValueError(
                f"El match_id {match_id} no pertenece a Bayer Leverkusen 2023/24. "
                f"Match IDs disponibles: {match_ids}"
            )
        selected = [match_id]
    elif max_matches is None:
        selected = match_ids
    else:
        n = max(1, int(max_matches))
        start = min(max(int(match_index), 0), max(len(match_ids) - 1, 0))
        selected = match_ids[start:start + n]

    if verbose:
        print(f"Partidos disponibles de Bayer: {len(match_ids)}")
        print(f"Partidos que se analizaran ahora: {len(selected)}")
        print("Match IDs analizados:", selected)
        if len(selected) < len(match_ids):
            print("Nota: ejecucion limitada para que no sea pesada. Para toda la temporada usa max_matches=None.")
    return selected


def load_events_for_matches(data_root: str | Path, match_ids: Iterable[int]) -> Dict[int, List[Dict[str, Any]]]:
    """Carga los archivos data/events/<match_id>.json disponibles."""
    data_root = resolve_data_root(data_root)
    out: Dict[int, List[Dict[str, Any]]] = {}
    for mid in match_ids:
        path = data_root / "events" / f"{mid}.json"
        if path.exists():
            out[int(mid)] = load_json(path)
    return out


def load_360_for_matches(data_root: str | Path, match_ids: Iterable[int]) -> Dict[int, Dict[str, Dict[str, Any]]]:
    """Carga data/three-sixty y crea un mapa match_id -> event_uuid -> frame."""
    data_root = resolve_data_root(data_root)
    out: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for mid in match_ids:
        path = data_root / "three-sixty" / f"{mid}.json"
        if not path.exists():
            out[int(mid)] = {}
            continue
        frames = load_json(path)
        out[int(mid)] = {f.get("event_uuid"): f for f in frames if f.get("event_uuid")}
    return out


def comprobar_datos_bayer(
    data_root: str | Path = "data",
    max_matches: Optional[int] = DEFAULT_MAX_MATCHES,
    match_id: Optional[int] = None,
    match_index: int = DEFAULT_MATCH_INDEX,
) -> Dict[str, Any]:
    """Comprueba que los JSON de la carpeta data corresponden a Bayer 2023/24.

    No ejecuta el modelo completo: solo identifica los partidos de Bayer, cuenta
    eventos, pases, tiros y frames 360 disponibles. Es la primera funcion que te
    recomiendo llamar desde Python/Jupyter.

    Uso:
        import decision_elo_leverkusen_unico as de
        resumen = de.comprobar_datos_bayer("data")
        resumen
    """
    root = resolve_data_root(data_root)

    # Confirmar que la competicion/temporada existe en competitions.json.
    competitions = load_json(root / "competitions.json")
    comp_ok = [
        c for c in competitions
        if c.get("competition_id") == BUNDESLIGA_COMPETITION_ID
        and c.get("season_id") == BUNDESLIGA_2023_24_SEASON_ID
    ]

    all_match_ids = get_bayer_match_ids(root, verbose=True)
    match_ids = select_bayer_match_ids(
        all_match_ids,
        max_matches=max_matches,
        match_id=match_id,
        match_index=match_index,
        verbose=True,
    )
    events_found = 0
    frames_found = 0
    lineups_found = 0
    passes = 0
    shots = 0
    bayer_actions = 0
    bayer_actions_with_360 = 0
    missing_events: List[int] = []
    missing_360: List[int] = []

    for mid in match_ids:
        event_file = root / "events" / f"{mid}.json"
        frame_file = root / "three-sixty" / f"{mid}.json"
        lineup_file = root / "lineups" / f"{mid}.json"

        if lineup_file.exists():
            lineups_found += 1

        if not event_file.exists():
            missing_events.append(mid)
            continue
        events_found += 1
        events = load_json(event_file)

        frame_map: Dict[str, Dict[str, Any]] = {}
        if frame_file.exists():
            frames_found += 1
            frames = load_json(frame_file)
            frame_map = {f.get("event_uuid"): f for f in frames if f.get("event_uuid")}
        else:
            missing_360.append(mid)

        for ev in events:
            if ev.get("team", {}).get("id") != BAYER_TEAM_ID:
                continue
            typ = ev.get("type", {}).get("name")
            if typ not in {"Pass", "Shot"}:
                continue
            bayer_actions += 1
            if typ == "Pass":
                passes += 1
            elif typ == "Shot":
                shots += 1
            if ev.get("id") in frame_map:
                bayer_actions_with_360 += 1

    resumen = {
        "data_root_detectado": str(root),
        "competition_id": BUNDESLIGA_COMPETITION_ID,
        "season_id": BUNDESLIGA_2023_24_SEASON_ID,
        "team_id": BAYER_TEAM_ID,
        "competicion_encontrada": bool(comp_ok),
        "partidos_bayer_total": len(all_match_ids),
        "partidos_bayer_analizados": len(match_ids),
        "partidos_bayer": len(match_ids),
        "match_ids_bayer_total": all_match_ids,
        "match_ids_bayer": match_ids,
        "archivos_events_encontrados": events_found,
        "archivos_360_encontrados": frames_found,
        "archivos_lineups_encontrados": lineups_found,
        "pases_bayer": passes,
        "tiros_bayer": shots,
        "acciones_bayer_pase_tiro": bayer_actions,
        "acciones_bayer_con_360": bayer_actions_with_360,
        "missing_events": missing_events,
        "missing_360": missing_360,
    }

    print("Resumen Bayer Leverkusen 2023/24")
    print(f"  data_root: {resumen['data_root_detectado']}")
    print(f"  partidos Bayer totales: {resumen['partidos_bayer_total']}")
    print(f"  partidos analizados: {resumen['partidos_bayer_analizados']}")
    print(f"  events encontrados: {events_found}/{len(match_ids)}")
    print(f"  360 encontrados: {frames_found}/{len(match_ids)}")
    print(f"  pases Bayer: {passes}")
    print(f"  tiros Bayer: {shots}")
    print(f"  acciones con 360: {bayer_actions_with_360}/{bayer_actions}")
    return resumen


def event_record_with_360(event: Dict[str, Any], match_id: int, frame_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Registro compacto Pass/Shot + 360 para inspeccion y guardado.

    Esta funcion es la version ampliada del codigo de lectura propuesto: incluye
    TODOS los tiros, no solo tiros a puerta, y tambien TODOS los pases de Bayer.
    """
    typ = event.get("type", {}).get("name")
    frame = frame_map.get(event.get("id"))
    record: Dict[str, Any] = {
        "match_id": match_id,
        "event_id": event.get("id"),
        "event_type": typ,
        "period": event.get("period"),
        "minute": event.get("minute"),
        "second": event.get("second"),
        "team": event.get("team", {}).get("name"),
        "team_id": event.get("team", {}).get("id"),
        "player": event.get("player", {}).get("name"),
        "player_id": event.get("player", {}).get("id"),
        "location": event.get("location"),
        "freeze_frame": frame.get("freeze_frame") if frame else None,
        "visible_area": frame.get("visible_area") if frame else None,
    }
    if typ == "Shot":
        shot = event.get("shot", {})
        record.update({
            "outcome": shot.get("outcome", {}).get("name"),
            "outcome_id": shot.get("outcome", {}).get("id"),
            "xg": shot.get("statsbomb_xg"),
            "body_part": shot.get("body_part", {}).get("name"),
            "technique": shot.get("technique", {}).get("name"),
        })
    elif typ == "Pass":
        pas = event.get("pass", {})
        record.update({
            "end_location": pas.get("end_location"),
            "recipient": pas.get("recipient", {}).get("name"),
            "recipient_id": pas.get("recipient", {}).get("id"),
            "outcome": pas.get("outcome", {}).get("name"),
            "outcome_id": pas.get("outcome", {}).get("id"),
            "height": pas.get("height", {}).get("name"),
            "pass_type": pas.get("type", {}).get("name"),
            "length": pas.get("length"),
            "angle": pas.get("angle"),
        })
    return record



# -----------------------------------------------------------------------------
# 3b. Filtros del analisis principal
# -----------------------------------------------------------------------------
def is_completed_pass(event: Dict[str, Any], require_recipient: bool = REQUIRE_RECIPIENT_FOR_COMPLETED_PASS) -> bool:
    """Devuelve True solo para pases completados del dato StatsBomb.

    Regla StatsBomb: un pase completado no trae ``pass.outcome``.
    Para evitar que entren pases dudosos en los diagramas, tambien exigimos
    receptor registrado cuando ``require_recipient=True``. Esto no usa el
    resultado visual del 360, porque el freeze-frame es el instante del pase y
    el receptor puede moverse hasta el punto de recepcion.
    """
    if event.get("type", {}).get("name") != "Pass":
        return False
    pas = event.get("pass", {}) or {}
    if pas.get("outcome") is not None:
        return False
    if require_recipient and not pas.get("recipient"):
        return False
    if not pas.get("end_location"):
        return False
    return True


def is_clean_pass_for_main(event: Dict[str, Any]) -> bool:
    """Filtro principal de pases: completados, no altos y en juego abierto.

    El objetivo es medir decision, no ejecucion. Por eso los pases incompletos
    quedan en el JSON de revision, pero no entran en el ranking principal.
    """
    if event.get("type", {}).get("name") != "Pass":
        return False
    pas = event.get("pass", {})
    if USE_ONLY_COMPLETED_PASSES_MAIN and not is_completed_pass(event):
        return False
    height = pas.get("height", {}).get("name")
    if EXCLUDE_HIGH_PASSES_MAIN and height == "High Pass":
        return False
    pass_type = pas.get("type", {}).get("name")
    if EXCLUDE_SET_PIECES_MAIN and pass_type in {"Corner", "Free Kick", "Throw-in", "Goal Kick", "Kick Off"}:
        return False
    return True


def is_open_play_shot_for_main(event: Dict[str, Any]) -> bool:
    """Filtro principal de tiros: todos los tiros, excluyendo solo penaltis.

    No filtramos por tiros a puerta: ir a puerta es ejecucion posterior a la
    decision. Un tiro fuera o bloqueado tambien puede ser buena o mala decision.
    """
    if event.get("type", {}).get("name") != "Shot":
        return False
    shot_type = event.get("shot", {}).get("type", {}).get("name")
    if shot_type == "Penalty":
        return False
    return True


def should_analyze_event_main(event: Dict[str, Any]) -> bool:
    typ = event.get("type", {}).get("name")
    if typ == "Pass":
        return is_clean_pass_for_main(event)
    if typ == "Shot":
        return is_open_play_shot_for_main(event)
    return False


def position_group_from_name(position_name: Optional[str]) -> str:
    """
    Agrupa posiciones StatsBomb en:
    - Portero
    - Defensa
    - Lateral
    - Mediocentro
    - Extremo
    - Delantero

    Compatible con posiciones como:
    - Right Center Back
    - Left Wing
    - Center Attacking Midfield
    - Left Center Forward
    - etc.
    """

    if not position_name or str(position_name).lower() in {"nan", "none"}:
        return "Sin posicion"

    n = str(position_name).lower().replace("-", " ").strip()

    # -------------------------
    # PORTERO
    # -------------------------
    if "goalkeeper" in n or "keeper" in n:
        return "Portero"

    # -------------------------
    # CENTRALES
    # -------------------------
    center_back_terms = [
        "center back",
        "centre back",
        "central back",
        "left center back",
        "right center back",
        "left centre back",
        "right centre back",
    ]

    if any(term in n for term in center_back_terms):
        return "Defensa"

    # -------------------------
    # LATERALES / CARRILEROS
    # -------------------------
    lateral_terms = [
        "left back",
        "right back",
        "full back",
        "fullback",
        "wing back",
        "left wing back",
        "right wing back",
    ]

    if any(term in n for term in lateral_terms):
        return "Lateral"

    # -------------------------
    # DELANTEROS
    # -------------------------
    forward_terms = [
        "center forward",
        "centre forward",
        "left center forward",
        "right center forward",
        "striker",
        "forward",
    ]

    if any(term in n for term in forward_terms):
        return "Delantero"

    # -------------------------
    # EXTREMOS / BANDAS
    # -------------------------
    winger_terms = [
        "left wing",
        "right wing",
        "wing",
        "wide",
        "left midfield",
        "right midfield",
    ]

    if any(term in n for term in winger_terms):
        return "Extremo"

    # -------------------------
    # MEDIOCAMPISTAS
    # -------------------------
    midfield_terms = [
        "midfield",
        "midfielder",
        "defensive midfield",
        "attacking midfield",
        "center midfield",
        "centre midfield",
        "center attacking midfield",
        "center defensive midfield",
        "left center midfield",
        "right center midfield",
        "left defensive midfield",
        "right defensive midfield",
        "left attacking midfield",
        "right attacking midfield",
    ]

    if any(term in n for term in midfield_terms):
        return "Mediocentro"

    # -------------------------
    # FALLBACK
    # -------------------------
    return "Mediocentro"


def export_bayer_passes_shots_360(
    events_by_match: Dict[int, List[Dict[str, Any]]],
    frames_by_match: Dict[int, Dict[str, Dict[str, Any]]],
    output_file: Path,
    team_id: int = BAYER_TEAM_ID,
    require_360: bool = False,
) -> List[Dict[str, Any]]:
    """Guarda un JSON con pases y tiros de Bayer unidos a su freeze-frame 360."""
    rows: List[Dict[str, Any]] = []
    for mid, events in events_by_match.items():
        frame_map = frames_by_match.get(mid, {})
        for ev in events:
            if ev.get("team", {}).get("id") != team_id:
                continue
            if ev.get("type", {}).get("name") not in {"Pass", "Shot"}:
                continue
            rec = event_record_with_360(ev, mid, frame_map)
            if require_360 and not rec.get("freeze_frame"):
                continue
            rows.append(rec)
    save_json(rows, output_file)
    return rows


def extract_demo_events(demo_zip: Path, output_dir: Path) -> Path:
    """Extrae el ZIP de ejemplo y devuelve la ruta al partido 7298.json."""
    output_dir.mkdir(parents=True, exist_ok=True)
    marker = output_dir / "Nueva carpeta" / "7298.json"
    if not marker.exists():
        with zipfile.ZipFile(demo_zip, "r") as zf:
            zf.extractall(output_dir)
    return marker


def make_inline_shot_frame(event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Convierte el freeze_frame que viene dentro de algunos eventos de tiro en una
    estructura parecida a StatsBomb 360.
    """
    shot = event.get("shot", {})
    ff = shot.get("freeze_frame")
    if not ff:
        return None
    players = []
    for p in ff:
        q = dict(p)
        # En el freeze-frame de tiros el portero no siempre viene con keeper=True.
        if q.get("position", {}).get("id") == 1:
            q["keeper"] = True
        q.setdefault("actor", False)
        players.append(q)
    return {"event_uuid": event.get("id"), "freeze_frame": players, "visible_area": None}


# -----------------------------------------------------------------------------
# 4. EPV: malla, transiciones y valor esperado
# -----------------------------------------------------------------------------
def cell_id(location: List[float], rows: int = 20, cols: int = 20) -> Optional[int]:
    """Convierte una coordenada StatsBomb [x,y] en id de celda 0..rows*cols-1."""
    if not location or len(location) < 2:
        return None
    x, y = float(location[0]), float(location[1])
    if not np.isfinite(x) or not np.isfinite(y):
        return None
    col = min(max(int(x / PITCH_LENGTH * cols), 0), cols - 1)
    row = min(max(int(y / PITCH_WIDTH * rows), 0), rows - 1)
    return row * cols + col


def cell_center(cid: int, rows: int = 20, cols: int = 20) -> Tuple[float, float]:
    """Centro geometrico de una celda."""
    row, col = divmod(cid, cols)
    return ((col + 0.5) * PITCH_LENGTH / cols, (row + 0.5) * PITCH_WIDTH / rows)


def flatten_event_for_epv(event: Dict[str, Any], match_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """Transforma un evento Pass/Shot en una fila simple para el calculo EPV."""
    typ = event.get("type", {}).get("name")
    if typ not in {"Pass", "Shot"}:
        return None
    # Segunda barrera de seguridad: aunque otra funcion llame directamente al
    # evaluador, ningun pase incompleto entra en el analisis principal.
    if typ == "Pass" and not is_clean_pass_for_main(event):
        return None
    loc = event.get("location")
    if not loc or len(loc) < 2:
        return None

    row: Dict[str, Any] = {
        "match_id": match_id,
        "event_id": event.get("id"),
        "type": typ,
        "team_id": event.get("team", {}).get("id"),
        "team_name": event.get("team", {}).get("name"),
        "player_id": event.get("player", {}).get("id"),
        "player_name": event.get("player", {}).get("name"),
        "x": float(loc[0]),
        "y": float(loc[1]),
        "start_cell": cell_id(loc),
        "end_x": None,
        "end_y": None,
        "end_cell": None,
        "is_completed": None,
        "xg": None,
    }

    if typ == "Pass":
        p = event.get("pass", {})
        end = p.get("end_location")
        if not end or len(end) < 2:
            return None
        row.update({
            "end_x": float(end[0]),
            "end_y": float(end[1]),
            "end_cell": cell_id(end),
            # En StatsBomb, si pass.outcome no existe, el pase se completo.
            "is_completed": p.get("outcome") is None,
            "pass_height": p.get("height", {}).get("name"),
            "pass_length": p.get("length"),
            "pass_angle": p.get("angle"),
        })
    else:
        s = event.get("shot", {})
        row.update({
            "xg": s.get("statsbomb_xg"),
            "shot_outcome": s.get("outcome", {}).get("name"),
        })
    return row


@dataclass
class EPVModel:
    """Modelo EPV discreto por malla."""
    rows: int = 20
    cols: int = 20
    values: Optional[np.ndarray] = None
    iterations: int = 0

    def value_at(self, location: List[float]) -> float:
        cid = cell_id(location, self.rows, self.cols)
        if cid is None or self.values is None:
            return 0.0
        return float(self.values[cid])

    def heatmap(self) -> np.ndarray:
        if self.values is None:
            return np.zeros((self.rows, self.cols))
        return self.values.reshape(self.rows, self.cols)


def build_epv_model(rows_df: pd.DataFrame, rows: int = 20, cols: int = 20, max_iter: int = 5000, tol: float = 1e-8) -> EPVModel:
    """
    Calcula EPV por iteracion de Bellman/Markov.

    La ecuacion implementada es:
        V_i = p_shot(i) * xG_medio(i) + sum_j P_move(i,j) * V_j

    donde P_move incluye solo pases completados y queda normalizado por TODAS
    las acciones observadas desde la celda: pases completados, pases fallidos y
    tiros. Las perdidas tienen valor 0, por eso no aparecen explicitamente.
    """
    n_cells = rows * cols
    P = np.zeros((n_cells, n_cells), dtype=float)
    reward = np.zeros(n_cells, dtype=float)
    denom = np.zeros(n_cells, dtype=float)

    if rows_df.empty:
        return EPVModel(rows=rows, cols=cols, values=np.zeros(n_cells), iterations=0)

    for _, r in rows_df.iterrows():
        i = r.get("start_cell")
        if pd.isna(i):
            continue
        i = int(i)
        denom[i] += 1.0
        if r["type"] == "Pass":
            if bool(r.get("is_completed")) and not pd.isna(r.get("end_cell")):
                j = int(r["end_cell"])
                P[i, j] += 1.0
            # pase incompleto: se cuenta en denom y su valor futuro es cero.
        elif r["type"] == "Shot":
            xg = r.get("xg")
            if xg is not None and not pd.isna(xg):
                reward[i] += float(xg)
            else:
                # Fallback si no hay xG: valor heuristico segun la posicion.
                reward[i] += heuristic_shot_probability([r["x"], r["y"]], None)

    for i in range(n_cells):
        if denom[i] > 0:
            P[i, :] /= denom[i]
            reward[i] /= denom[i]

    V = np.zeros(n_cells, dtype=float)
    for it in range(max_iter):
        V_new = reward + P.dot(V)
        if np.linalg.norm(V_new - V, ord=1) < tol:
            return EPVModel(rows=rows, cols=cols, values=V_new, iterations=it + 1)
        V = V_new
    return EPVModel(rows=rows, cols=cols, values=V, iterations=max_iter)


def plot_epv_heatmap(epv: EPVModel, save_path: Path, title: str = "EPV - Bayer Leverkusen 2023/24") -> None:
    """Guarda un mapa de calor EPV sobre el campo."""
    fig, ax = plt.subplots(figsize=(12, 8))
    draw_pitch(ax)
    heat = epv.heatmap()
    im = ax.imshow(
        heat,
        extent=[0, PITCH_LENGTH, 0, PITCH_WIDTH],
        origin="lower",
        alpha=0.78,
        interpolation="nearest",
        aspect="auto",
    )
    ax.set_title(title, fontsize=15, weight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("EPV")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# 5. Probabilidad de tiro
# -----------------------------------------------------------------------------
def shot_angle(location: List[float]) -> float:
    """Angulo de tiro en radianes entre los dos postes."""
    ball = np.array(location[:2], dtype=float)
    v1 = LEFT_POST - ball
    v2 = RIGHT_POST - ball
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return 0.0
    cosv = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
    return float(np.arccos(cosv))


def point_in_triangle(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    """Comprueba si p esta dentro del triangulo a-b-c con coordenadas baricentricas."""
    v0 = c - a
    v1 = b - a
    v2 = p - a
    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)
    denom = dot00 * dot11 - dot01 * dot01
    if denom == 0:
        return False
    inv = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv
    v = (dot00 * dot12 - dot01 * dot02) * inv
    return bool((u >= 0) and (v >= 0) and (u + v <= 1))


def defensive_pressure_for_shot(location: List[float], frame: Optional[Dict[str, Any]]) -> Tuple[int, float]:
    """Cuenta rivales en triangulo de tiro y distancia al defensor mas cercano."""
    if not frame:
        return 0, 20.0
    ball = np.array(location[:2], dtype=float)
    defenders_in_triangle = 0
    min_dist = 20.0
    for p in frame.get("freeze_frame", []):
        if p.get("teammate") or p.get("actor"):
            continue
        loc = p.get("location")
        if not loc or len(loc) < 2:
            continue
        q = np.array(loc[:2], dtype=float)
        if point_in_triangle(q, ball, LEFT_POST, RIGHT_POST):
            defenders_in_triangle += 1
        min_dist = min(min_dist, float(np.linalg.norm(q - ball)))
    return defenders_in_triangle, min_dist


def heuristic_shot_probability(location: List[float], frame: Optional[Dict[str, Any]]) -> float:
    """
    Probabilidad de gol propia para tiros o receptores potenciales.

    Combina la idea del codigo del usuario (distancia, defensores dentro del
    triangulo y distancia al defensor mas cercano) con el angulo de tiro.
    No pretende sustituir a un xG entrenado, sino generar una escala comparable
    para alternativas que no tienen xG observado.
    """
    ball = np.array(location[:2], dtype=float)
    distance = float(np.linalg.norm(GOAL_CENTER - ball))
    angle = shot_angle(location)  # radianes
    defenders, closest = defensive_pressure_for_shot(location, frame)
    closest = max(float(closest), 0.75)

    # Formula exponencial, parecida al codigo adjunto, con un ajuste por angulo.
    # La probabilidad se acota para evitar valores extremos en alternativas.
    base = math.exp(-(0.075 * distance + 0.50 * defenders + 0.35 / closest))
    angle_factor = min(max(angle / 0.55, 0.12), 1.35)
    prob = base * angle_factor
    return float(np.clip(prob, 0.001, 0.65))


def shot_value(event: Dict[str, Any], frame: Optional[Dict[str, Any]]) -> float:
    """Valor Q de un tiro: xG StatsBomb si existe; si no, modelo heuristico."""
    xg = event.get("shot", {}).get("statsbomb_xg")
    if xg is not None:
        try:
            return float(xg)
        except (TypeError, ValueError):
            pass
    return heuristic_shot_probability(event.get("location"), frame)


# -----------------------------------------------------------------------------
# 6. Probabilidad y riesgo de pase
# -----------------------------------------------------------------------------
def sigmoid(x: float) -> float:
    """Sigmoide estable."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def ball_time(start: np.ndarray, point: np.ndarray) -> float:
    """Tiempo esperado del balon al punto: T = 0.075 * distancia."""
    return 0.075 * float(np.linalg.norm(point - start))


def ball_sigma(start: np.ndarray, point: np.ndarray) -> float:
    """Incertidumbre del tiempo de viaje del balon segun la distancia."""
    d = float(np.linalg.norm(point - start))
    return float(math.exp(-1.009 + 0.011 * d))


def max_player_distance(t: float, d0: float = 1.0, alpha: float = 1.3, vmax: float = 7.8) -> float:
    """
    Distancia maxima aproximada que puede cubrir un jugador en t segundos.

    r(t) = max{d0, vmax * (t - (1-exp(-alpha*t))/alpha)}
    """
    if t <= 0:
        return d0
    return max(d0, vmax * (t - (1.0 - math.exp(-alpha * t)) / alpha))


def player_time_to_distance(distance: float, d0: float = 1.0, alpha: float = 1.3, vmax: float = 7.8) -> float:
    """Invierte r(t) por busqueda binaria para obtener el tiempo minimo."""
    if distance <= d0:
        return 0.0
    lo, hi = 0.0, 5.0
    while max_player_distance(hi, d0=d0, alpha=alpha, vmax=vmax) < distance:
        hi *= 2.0
        if hi > 20.0:
            break
    for _ in range(40):
        mid = (lo + hi) / 2.0
        if max_player_distance(mid, d0=d0, alpha=alpha, vmax=vmax) >= distance:
            hi = mid
        else:
            lo = mid
    return hi


def sample_pass_line(start: List[float], end: List[float], n: int = 25) -> np.ndarray:
    """Puntos de la trayectoria del pase, sin incluir exactamente el origen."""
    a = np.array(start[:2], dtype=float)
    b = np.array(end[:2], dtype=float)
    ts = np.linspace(0.05, 1.0, n)
    return np.array([a + t * (b - a) for t in ts])


def point_to_segment_distance(point: List[float], start: List[float], end: List[float]) -> float:
    """Distancia euclidea de un defensor a la linea real del pase.

    1. Proyectamos el defensor sobre el segmento start-end.
    2. Acotamos la proyeccion a [0,1] para no salirnos del pase.
    3. Calculamos la distancia al punto mas cercano de ese segmento.
    """
    p = np.array(point[:2], dtype=float)
    a = np.array(start[:2], dtype=float)
    b = np.array(end[:2], dtype=float)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom == 0:
        return float(np.linalg.norm(p - a))
    t = float(np.dot(p - a, ab) / denom)
    t = float(np.clip(t, 0.0, 1.0))
    closest = a + t * ab
    return float(np.linalg.norm(p - closest))


def is_keeper_player(player: Dict[str, Any]) -> bool:
    """Identifica porteros en el freeze-frame de StatsBomb 360."""
    return bool(player.get("keeper")) or player.get("position", {}).get("id") == 1


def offside_components(start: List[float], end: List[float], frame: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Regla simplificada de fuera de juego para pases hacia x creciente.

    Se aplica solo a pases adelantados y a receptores situados en campo rival
    (x > 60). Siguiendo la condicion solicitada, el pase se considera valido
    si existe al menos un rival de campo, sin contar el portero, con x mayor que
    la x del receptor. Si no existe ese rival por detras, se marca como fuera de
    juego y el valor Q del pase se anula en pass_value().

    Nota: es una aproximacion operativa para este trabajo. No sustituye a una
    decision arbitral completa porque no usa momento exacto del golpeo, parte
    del cuerpo habilitada, participacion activa ni todos los jugadores si no
    aparecen en el 360.
    """
    result = {
        "offside_checked": False,
        "is_offside": False,
        "offside_valid": True,
        "defenders_behind_receiver": float("nan"),
        "last_outfield_defender_x": float("nan"),
        "receiver_x": float("nan"),
        "ball_x": float("nan"),
    }
    if not start or not end or len(start) < 2 or len(end) < 2:
        return result

    ball_x = float(start[0])
    receiver_x = float(end[0])
    result["receiver_x"] = receiver_x
    result["ball_x"] = ball_x

    # No se comprueba fuera de juego en pases hacia atras/laterales ni en campo propio.
    if receiver_x <= ball_x + OFFSIDE_X_MARGIN or receiver_x <= 60.0:
        return result

    result["offside_checked"] = True
    if not frame or not frame.get("freeze_frame"):
        # Sin 360 no podemos sancionar fuera de juego; dejamos pasar la accion.
        return result

    outfield_rival_xs: List[float] = []
    for player in frame.get("freeze_frame", []):
        if player.get("teammate") or player.get("actor") or is_keeper_player(player):
            continue
        loc = player.get("location")
        if not loc or len(loc) < 2:
            continue
        outfield_rival_xs.append(float(loc[0]))

    if outfield_rival_xs:
        last_x = max(outfield_rival_xs)
        defenders_behind = sum(1 for x in outfield_rival_xs if x > receiver_x + OFFSIDE_X_MARGIN)
    else:
        last_x = float("nan")
        defenders_behind = 0

    result["last_outfield_defender_x"] = float(last_x)
    result["defenders_behind_receiver"] = int(defenders_behind)
    result["is_offside"] = bool(defenders_behind == 0)
    result["offside_valid"] = not result["is_offside"]
    return result


def apply_offside_penalty(value: Dict[str, Any], offside: Dict[str, Any]) -> Dict[str, Any]:
    """Anula el valor de un pase si el receptor esta en fuera de juego."""
    value.update(offside)
    if offside.get("is_offside"):
        value["Q_before_offside"] = float(value.get("Q", 0.0))
        value["p_success_before_offside"] = float(value.get("p_success", 0.0))
        value["Q"] = 0.0
        value["p_success"] = 0.0
        value["q_pass_epv"] = 0.0
        value["q_pass_shot"] = 0.0
        value["receiver_value_source"] = "offside"
    return value


def pass_risk_components(start: List[float], end: List[float], frame: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calcula I, P, riesgo y probabilidad de exito del pase.

    I: probabilidad agregada de interceptacion rival.
    P: probabilidad agregada de control por el propio equipo.
    riesgo = I - P en [-1, 1].
    p_success = (1 - riesgo)/2, una transformacion simple para valorar Q_pass.
    """
    pass_length = float(np.linalg.norm(np.array(end[:2], dtype=float) - np.array(start[:2], dtype=float)))
    if not frame or not frame.get("freeze_frame"):
        # Fallback sin 360: penaliza la distancia, pero no usa contexto defensivo.
        p_success = float(np.clip(1.0 - pass_length / 95.0, 0.25, 0.92))
        return {
            "I": 1.0 - p_success,
            "P": p_success,
            "risk": (1.0 - p_success) - p_success,
            "p_success": p_success,
            "pass_length": pass_length,
            "min_defender_lane_distance": float("inf"),
            "nearest_defender_target_distance": float("inf"),
        }

    start_np = np.array(start[:2], dtype=float)
    points = sample_pass_line(start, end)

    opp_probs: List[float] = []
    team_probs: List[float] = []
    min_defender_lane_distance = float("inf")
    nearest_defender_target_distance = float("inf")

    for p in frame.get("freeze_frame", []):
        loc = p.get("location")
        if not loc or len(loc) < 2 or p.get("actor"):
            continue
        player_pos = np.array(loc[:2], dtype=float)
        best_prob = 0.0
        for k in points:
            t_ball = ball_time(start_np, k)
            sigma = max(ball_sigma(start_np, k), 0.05)
            d_player = float(np.linalg.norm(player_pos - k))
            t_player = player_time_to_distance(d_player)
            # Si el jugador llega antes que el balon, t_ball - t_player > 0.
            rho = sigmoid((t_ball - t_player) / sigma)
            best_prob = max(best_prob, rho)
        if p.get("teammate"):
            team_probs.append(best_prob)
        else:
            opp_probs.append(best_prob)
            # Distancia del rival a la linea de pase y al receptor.
            min_defender_lane_distance = min(min_defender_lane_distance, point_to_segment_distance(loc, start, end))
            nearest_defender_target_distance = min(nearest_defender_target_distance, float(np.linalg.norm(player_pos - np.array(end[:2], dtype=float))))

    I = 1.0 - float(np.prod([1.0 - p for p in opp_probs])) if opp_probs else 0.0
    P = 1.0 - float(np.prod([1.0 - p for p in team_probs])) if team_probs else 0.0
    risk = float(np.clip(I - P, -1.0, 1.0))
    # Probabilidad practica de exito: controlar el pase y no ser interceptado.
    # Esta version es mas fiel al sentido I/P del PDF que convertir directamente
    # risk=(I-P) a probabilidad.
    p_success = float(np.clip(P * (1.0 - I), 0.02, 0.98))
    return {
        "I": I,
        "P": P,
        "risk": risk,
        "p_success": p_success,
        "pass_length": pass_length,
        "min_defender_lane_distance": float(min_defender_lane_distance),
        "nearest_defender_target_distance": float(nearest_defender_target_distance),
    }


def pass_value(
    start: List[float],
    end: List[float],
    frame: Optional[Dict[str, Any]],
    epv: EPVModel,
    force_offside: bool = False,
) -> Dict[str, Any]:
    """Valor esperado de un pase candidato.

    Version actualizada: cada receptor potencial se valora de dos maneras:

    1. Pase para continuar la posesion:
           q_pass_epv = P(pase completado) * EPV(destino)

    2. Pase para generar un tiro inmediato del receptor:
           q_pass_shot = P(pase completado) * P(gol desde la posicion del receptor)

    El valor final del pase es el maximo de ambas opciones. Asi, un companero
    visible en 360 puede ser valioso por progresar la posesion o porque queda
    en una posicion de remate mejor que la accion real.
    """
    comps = pass_risk_components(start, end, frame)
    epv_start = epv.value_at(start)
    epv_end = epv.value_at(end)

    # Probabilidad de gol del receptor si recibiera y tirara desde su posicion.
    # Se calcula para TODOS los companeros/destinos 360, no solo para el tirador.
    p_goal_receiver = heuristic_shot_probability(end, frame)

    q_pass_epv = comps["p_success"] * epv_end
    q_pass_shot = comps["p_success"] * p_goal_receiver
    q = max(q_pass_epv, q_pass_shot)
    receiver_value_source = "shot_probability" if q_pass_shot > q_pass_epv else "epv"

    value = {
        "Q": float(q),
        "epv_start": float(epv_start),
        "epv_end": float(epv_end),
        "delta_epv": float(epv_end - epv_start),
        "p_goal_receiver": float(p_goal_receiver),
        "q_pass_epv": float(q_pass_epv),
        "q_pass_shot": float(q_pass_shot),
        "receiver_value_source": receiver_value_source,
        **comps,
    }

    offside = offside_components(start, end, frame)
    if force_offside:
        offside["offside_checked"] = True
        offside["is_offside"] = True
        offside["offside_valid"] = False
    return apply_offside_penalty(value, offside)


# -----------------------------------------------------------------------------
# 7. Motor de decision: candidatos, mejor movimiento, notas tipo ajedrez
# -----------------------------------------------------------------------------
def teammates_from_frame(frame: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Devuelve companeros visibles en el freeze-frame, excluyendo el actor."""
    if not frame:
        return []
    out = []
    for p in frame.get("freeze_frame", []):
        if p.get("teammate") and not p.get("actor") and p.get("location"):
            out.append(p)
    return out


def candidate_actions(event: Dict[str, Any], frame: Optional[Dict[str, Any]], epv: EPVModel) -> List[Dict[str, Any]]:
    """
    Genera acciones candidatas:
    - pase a cada companero visible
    - tiro si la accion real es tiro o si el jugador esta en zona razonable de tiro
    """
    loc = event.get("location")
    if not loc:
        return []
    candidates: List[Dict[str, Any]] = []

    for idx, mate in enumerate(teammates_from_frame(frame)):
        target = mate.get("location")
        if not target or len(target) < 2:
            continue
        val = pass_value(loc, target, frame, epv)
        candidates.append({
            "action_type": "Pass",
            "target": target[:2],
            "target_player": mate.get("player", {}).get("name") or f"Companero visible {idx+1}",
            "Q": val["Q"],
            "details": val,
        })

    # Incluir tiro si el evento es tiro o si esta cerca de zona de remate.
    include_shot = event.get("type", {}).get("name") == "Shot" or (loc[0] >= 80 and heuristic_shot_probability(loc, frame) >= 0.015)
    if include_shot:
        q_shot = shot_value(event, frame) if event.get("type", {}).get("name") == "Shot" else heuristic_shot_probability(loc, frame)
        candidates.append({
            "action_type": "Shot",
            "target": [120.0, 40.0],
            "target_player": "Porteria",
            "Q": float(q_shot),
            "details": {
                "xg_or_model": float(q_shot),
                "angle_rad": shot_angle(loc),
                "defensive_pressure": defensive_pressure_for_shot(loc, frame),
            },
        })

    return candidates


def real_action(event: Dict[str, Any], frame: Optional[Dict[str, Any]], epv: EPVModel) -> Optional[Dict[str, Any]]:
    """Valora la accion real del evento."""
    typ = event.get("type", {}).get("name")
    loc = event.get("location")
    if typ == "Pass":
        end = event.get("pass", {}).get("end_location")
        if not loc or not end:
            return None
        outcome_name = event.get("pass", {}).get("outcome", {}).get("name")
        force_offside = bool(outcome_name and "Offside" in str(outcome_name))
        completed = is_completed_pass(event)
        # En el analisis principal no deberia llegar nunca un pase incompleto.
        # Si llega por llamada manual, no se valora.
        if USE_ONLY_COMPLETED_PASSES_MAIN and not completed:
            return None
        val = pass_value(loc, end, frame, epv, force_offside=force_offside)
        val["pass_completed"] = bool(completed)
        val["pass_outcome"] = outcome_name or "Complete"
        return {
            "action_type": "Pass",
            "target": end[:2],
            "target_player": event.get("pass", {}).get("recipient", {}).get("name"),
            "Q": val["Q"],
            "details": val,
            "completed": completed,
        }
    if typ == "Shot":
        q = shot_value(event, frame)
        return {
            "action_type": "Shot",
            "target": [120.0, 40.0],
            "target_player": "Porteria",
            "Q": float(q),
            "details": {"xg_or_model": float(q)},
            "completed": None,
        }
    return None


def chess_note(score: float, regret: float, q_best: float) -> Tuple[str, str]:
    """Clasifica la decision con una escala menos restrictiva y mas variada.

    La version anterior dejaba demasiadas acciones como "= Buena" porque
    neutralizaba muchas jugadas con regret bajo. Esta escala mantiene una
    proteccion contra acciones de impacto minimo, pero reparte mejor las notas:

    - mas facil recibir !!, ! y ✓ si la accion real esta cerca de la mejor;
    - mas facil recibir ?! cuando hay una alternativa mejor;
    - ? y ?? siguen reservadas para perdidas relevantes.
    """
    if q_best <= 0:
        return "-", "Sin valor"

    # Solo neutralizamos acciones casi sin valor y con diferencia minuscula.
    # Antes el corte era demasiado amplio y muchas acababan en "= Buena".
    if q_best < 0.006 and regret < 0.002:
        if score >= 0.92:
            return "!", "Mejor movimiento"
        return "=", "Buena"

    # Premios: umbrales mas abiertos para que aparezcan mas notas positivas.
    if score >= 0.970:
        if q_best >= 0.050:
            return "!!", "Brillante"
        return "!", "Mejor movimiento"
    if score >= 0.920:
        return "!", "Mejor movimiento"
    if score >= 0.800:
        return "✓", "Excelente"

    # Buena queda como banda intermedia mas estrecha.
    if score >= 0.580:
        return "=", "Buena"

    # Imprecisiones: ahora aparecen con regrets moderados, aunque no sean enormes.
    if score >= 0.400:
        if regret >= 0.006:
            return "?!", "Imprecision"
        return "=", "Buena"

    if score >= 0.250:
        if regret >= 0.035:
            return "?", "Error"
        if regret >= 0.005:
            return "?!", "Imprecision"
        return "=", "Buena"

    # Errores graves solo cuando la perdida absoluta tiene impacto real.
    if regret >= 0.070 and q_best >= 0.070:
        return "??", "Pifia"
    if regret >= 0.025:
        return "?", "Error"
    if regret >= 0.004:
        return "?!", "Imprecision"

    return "=", "Buena"


def evaluate_event_decision(event: Dict[str, Any], frame: Optional[Dict[str, Any]], epv: EPVModel, match_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """Evalua un evento Pass/Shot y devuelve Q_real, Q_best, regret, score y nota."""
    typ = event.get("type", {}).get("name")
    if typ not in {"Pass", "Shot"}:
        return None
    loc = event.get("location")
    if not loc:
        return None

    real = real_action(event, frame, epv)
    if real is None:
        return None
    candidates = candidate_actions(event, frame, epv)

    # Es importante incluir siempre la accion real como candidata, incluso si el
    # receptor real no aparece en el freeze-frame. Antes quitamos duplicados
    # equivalentes para que un tiro real no aparezca dos veces: una como
    # candidato teorico y otra como accion real.
    def same_target(a: Optional[List[float]], b: Optional[List[float]], tol: float = 1.0) -> bool:
        if not a or not b or len(a) < 2 or len(b) < 2:
            return False
        return float(np.linalg.norm(np.array(a[:2], dtype=float) - np.array(b[:2], dtype=float))) <= tol

    filtered_candidates = []
    for c in candidates:
        duplicate_real = (
            c.get("action_type") == real.get("action_type")
            and same_target(c.get("target"), real.get("target"))
        )
        if not duplicate_real:
            filtered_candidates.append(c)
    filtered_candidates.append({**real, "is_real": True})
    candidates_sorted = sorted(filtered_candidates, key=lambda x: x.get("Q", 0.0), reverse=True)
    best = candidates_sorted[0]
    worst = candidates_sorted[-1]

    q_real = float(real["Q"])
    q_best = max(float(best["Q"]), 1e-9)
    regret = max(q_best - q_real, 0.0)
    score = float(np.clip(q_real / q_best, 0.0, 1.0))
    symbol, label = chess_note(score, regret, q_best)

    # Top-3: si la accion real esta entre las tres mejores por valor.
    real_rank = None
    for i, c in enumerate(candidates_sorted, start=1):
        if c.get("is_real"):
            real_rank = i
            break
    if real_rank is None:
        # fallback por proximidad al Q real.
        qs = [c.get("Q", 0.0) for c in candidates_sorted]
        real_rank = sorted(qs, reverse=True).index(q_real) + 1 if q_real in qs else None

    action_span = max(float(best.get("Q", 0.0)) - float(worst.get("Q", 0.0)), 0.001)
    importance = max(action_span, q_best, 0.01)

    # Campos auxiliares para que el CSV muestre explicitamente como se valora
    # cada pase: por EPV o por probabilidad de gol del receptor.
    def detail_number(action: Dict[str, Any], key: str) -> float:
        value = action.get("details", {}).get(key)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    def detail_text(action: Dict[str, Any], key: str) -> Optional[str]:
        value = action.get("details", {}).get(key)
        return None if value is None else str(value)

    def detail_bool(action: Dict[str, Any], key: str) -> bool:
        return bool(action.get("details", {}).get(key, False))

    return {
        "match_id": match_id,
        "event_id": event.get("id"),
        "period": event.get("period"),
        "minute": event.get("minute"),
        "second": event.get("second"),
        "team_id": event.get("team", {}).get("id"),
        "team_name": event.get("team", {}).get("name"),
        "player_id": event.get("player", {}).get("id"),
        "player_name": event.get("player", {}).get("name"),
        "position_id": event.get("position", {}).get("id"),
        "position_name": event.get("position", {}).get("name"),
        "position_group": position_group_from_name(event.get("position", {}).get("name")),
        "event_type": typ,
        "x": loc[0],
        "y": loc[1],
        "real_action": real["action_type"],
        "real_target": real.get("target"),
        "real_target_player": real.get("target_player"),
        "real_pass_completed": bool(real.get("completed")) if real.get("action_type") == "Pass" else None,
        "real_pass_outcome": detail_text(real, "pass_outcome") if real.get("action_type") == "Pass" else None,
        "q_real": q_real,
        "best_action": best.get("action_type"),
        "best_target": best.get("target"),
        "best_target_player": best.get("target_player"),
        "q_best": q_best,
        "real_p_goal_receiver": detail_number(real, "p_goal_receiver"),
        "real_q_pass_epv": detail_number(real, "q_pass_epv"),
        "real_q_pass_shot": detail_number(real, "q_pass_shot"),
        "real_receiver_value_source": detail_text(real, "receiver_value_source"),
        "real_delta_epv": detail_number(real, "delta_epv"),
        "real_risk": detail_number(real, "risk"),
        "real_p_success": detail_number(real, "p_success"),
        "real_I": detail_number(real, "I"),
        "real_P": detail_number(real, "P"),
        "real_pass_length": detail_number(real, "pass_length"),
        "real_min_defender_lane_distance": detail_number(real, "min_defender_lane_distance"),
        "real_nearest_defender_target_distance": detail_number(real, "nearest_defender_target_distance"),
        "real_offside_checked": detail_bool(real, "offside_checked"),
        "real_is_offside": detail_bool(real, "is_offside"),
        "real_offside_valid": detail_bool(real, "offside_valid"),
        "real_defenders_behind_receiver": detail_number(real, "defenders_behind_receiver"),
        "real_last_outfield_defender_x": detail_number(real, "last_outfield_defender_x"),
        "best_p_goal_receiver": detail_number(best, "p_goal_receiver"),
        "best_q_pass_epv": detail_number(best, "q_pass_epv"),
        "best_q_pass_shot": detail_number(best, "q_pass_shot"),
        "best_receiver_value_source": detail_text(best, "receiver_value_source"),
        "best_delta_epv": detail_number(best, "delta_epv"),
        "best_risk": detail_number(best, "risk"),
        "best_p_success": detail_number(best, "p_success"),
        "best_I": detail_number(best, "I"),
        "best_P": detail_number(best, "P"),
        "best_pass_length": detail_number(best, "pass_length"),
        "best_min_defender_lane_distance": detail_number(best, "min_defender_lane_distance"),
        "best_nearest_defender_target_distance": detail_number(best, "nearest_defender_target_distance"),
        "best_offside_checked": detail_bool(best, "offside_checked"),
        "best_is_offside": detail_bool(best, "is_offside"),
        "best_offside_valid": detail_bool(best, "offside_valid"),
        "best_defenders_behind_receiver": detail_number(best, "defenders_behind_receiver"),
        "best_last_outfield_defender_x": detail_number(best, "last_outfield_defender_x"),
        "q_worst": float(worst.get("Q", 0.0)),
        "regret": regret,
        "decision_score": score,
        "note_symbol": symbol,
        "note_label": label,
        "real_rank": real_rank,
        "num_candidates": len(candidates_sorted),
        "importance": importance,
        "real_details": real.get("details", {}),
        "best_details": best.get("details", {}),
        "candidates": candidates_sorted,
    }


# -----------------------------------------------------------------------------
# 8. Agregacion por jugador y ELO
# -----------------------------------------------------------------------------
def aggregate_player_ratings(decisions: pd.DataFrame) -> pd.DataFrame:
    """Agrega notas por jugador y calcula un Elo-like de decision."""
    if decisions.empty:
        return pd.DataFrame()

    rows = []
    for player, g in decisions.groupby("player_name"):
        w = g["importance"].astype(float).clip(lower=0.01)
        s = g["decision_score"].astype(float).clip(0, 1)
        score_mean = float(np.average(s, weights=w))
        score_clip = float(np.clip(score_mean, 0.05, 0.95))
        elo = 1500.0 + 400.0 * math.log10(score_clip / (1.0 - score_clip))
        pos_name = None
        pos_group = None
        if "position_name" in g.columns and g["position_name"].notna().any():
            pos_name = g["position_name"].dropna().astype(str).mode().iloc[0]
        if "position_group" in g.columns and g["position_group"].notna().any():
            pos_group = g["position_group"].dropna().astype(str).mode().iloc[0]
        rows.append({
            "player_name": player,
            "position_name": pos_name,
            "position_group": pos_group or position_group_from_name(pos_name),
            "actions": int(len(g)),
            "passes": int((g["event_type"] == "Pass").sum()),
            "shots": int((g["event_type"] == "Shot").sum()),
            "decision_score": score_mean,
            "decision_elo": elo,
            "avg_regret": float(np.average(g["regret"].astype(float), weights=w)),
            "best_move_pct": float((g["real_rank"] == 1).mean() * 100.0),
            "top3_pct": float((g["real_rank"].astype(float) <= 3).mean() * 100.0),
            "brilliant_or_best_pct": float(g["note_symbol"].isin(["!!", "!"]).mean() * 100.0),
            "error_pct": float(g["note_symbol"].isin(["?", "??"]).mean() * 100.0),
        })
    out = pd.DataFrame(rows).sort_values("decision_elo", ascending=False).reset_index(drop=True)
    return out


def plot_rating_bar(ratings: pd.DataFrame, save_path: Path, min_actions: int = 5) -> None:
    """Grafico ranking Decision-Elo."""
    if ratings.empty:
        return
    df = ratings[ratings["actions"] >= min_actions].head(15).sort_values("decision_elo")
    if df.empty:
        df = ratings.head(15).sort_values("decision_elo")
    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(df))))
    ax.barh(df["player_name"], df["decision_elo"])
    ax.axvline(1500, lw=1.0, ls="--", c="black")
    ax.set_xlabel("Decision-Elo")
    ax.set_title("Ranking Decision-Elo por jugador")
    for i, v in enumerate(df["decision_elo"]):
        ax.text(v + 5, i, f"{v:.0f}", va="center", fontsize=9)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_note_distribution(decisions: pd.DataFrame, save_path: Path) -> None:
    """Grafico de distribucion de notas tipo ajedrez."""
    if decisions.empty:
        return
    order = ["!!", "!", "✓", "=", "?!", "?", "??"]
    counts = decisions["note_symbol"].value_counts().reindex(order).fillna(0)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_title("Distribucion de notas tipo ajedrez")
    ax.set_xlabel("Nota")
    ax.set_ylabel("Numero de acciones")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.5, str(int(v)), ha="center", fontsize=9)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_regret_histogram(decisions: pd.DataFrame, save_path: Path) -> None:
    """Histograma de regret."""
    if decisions.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(decisions["regret"].astype(float), bins=30)
    ax.set_title("Distribucion del regret: Q_best - Q_real")
    ax.set_xlabel("Regret")
    ax.set_ylabel("Acciones")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)



def plot_risk_delta_epv_scatter(decisions: pd.DataFrame, save_path: Path) -> None:
    """Grafico Riesgo vs incremento EPV de los pases evaluados.

    Replica la idea de los apuntes: bajo/alto riesgo frente a baja/alta ganancia.
    Usa pases reales del analisis principal; si no hay, usa mejores alternativas
    de pase para que la demo siga generando una figura util.
    """
    if decisions.empty:
        return
    points = []
    for _, r in decisions.iterrows():
        if r.get("real_action") == "Pass" and pd.notna(r.get("real_risk")) and pd.notna(r.get("real_delta_epv")):
            points.append((float(r["real_delta_epv"]), float(r["real_risk"]), "Pase real"))
        if r.get("best_action") == "Pass" and pd.notna(r.get("best_risk")) and pd.notna(r.get("best_delta_epv")):
            points.append((float(r["best_delta_epv"]), float(r["best_risk"]), "Mejor alternativa"))
    if not points:
        return
    df = pd.DataFrame(points, columns=["delta_epv", "risk", "kind"])
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    x_min = min(-0.04, float(df["delta_epv"].min()) - 0.01)
    x_max = max(0.10, float(df["delta_epv"].max()) + 0.01)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-1.02, 1.02)

    # Zonas conceptuales
    ax.add_patch(Rectangle((0, 0), x_max, 1.02, fc="green", alpha=0.12, ec="green", lw=1.0))
    ax.add_patch(Rectangle((x_min, -1.02), -x_min, 1.02, fc="green", alpha=0.12, ec="green", lw=1.0))
    ax.add_patch(Rectangle((0, -1.02), x_max, 1.02, fc="gold", alpha=0.10, ec="gold", lw=1.0))
    ax.add_patch(Rectangle((x_min, 0), -x_min, 1.02, fc="red", alpha=0.08, ec="red", lw=1.0))
    ax.axhline(0, color="black", ls="--", lw=1.0)
    ax.axvline(0, color="black", ls="--", lw=1.0)
    for kind, sub in df.groupby("kind"):
        marker = "o" if kind == "Pase real" else "x"
        ax.scatter(sub["delta_epv"], sub["risk"], alpha=0.55, label=kind, marker=marker)
    ax.text(x_max * 0.50, 0.86, "Alto riesgo - alta ganancia", fontsize=11, weight="bold", ha="center")
    ax.text(x_min * 0.50, -0.88, "Bajo riesgo - baja ganancia", fontsize=11, weight="bold", ha="center", rotation=90)
    ax.text(x_max * 0.55, -0.88, "Bajo riesgo - alta ganancia", fontsize=10, ha="center")
    ax.text(x_min * 0.55, 0.78, "Alto riesgo - baja ganancia", fontsize=10, ha="center")
    ax.set_xlabel("Incremento de EPV (DeltaEPV)")
    ax.set_ylabel("Riesgo del pase = I - P")
    ax.set_title("Riesgo e incremento del Valor Esperado de Posesion")
    ax.legend(loc="lower right", frameon=True)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_elo_by_position_group(
    ratings: pd.DataFrame,
    save_path: Path,
    min_actions: int = 2,
) -> None:
    """Grafico de Decision-Elo medio por grupo posicional."""

    if ratings.empty or "position_group" not in ratings.columns:
        return
    df = ratings[ratings["actions"] >= min_actions].copy()
    if df.empty:
        df = ratings.copy()
    order = [
        "Defensa",
        "Lateral",
        "Mediocentro",
        "Extremo",
        "Delantero",
        "Portero",
        "Sin posicion",
    ]
    summary = (
        df.groupby("position_group", dropna=False)
        .agg(
            decision_elo=("decision_elo", "mean"),
            players=("player_name", "count"),
            actions=("actions", "sum"),
        )
        .reset_index()
    )
    summary["order"] = summary["position_group"].apply(
        lambda x: order.index(x) if x in order else len(order)
    )
    summary = summary.sort_values("order")
    if summary.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        summary["position_group"].astype(str),
        summary["decision_elo"],
        color="steelblue",
    )
    ax.axhline(1500, color="black", lw=1.2, ls="--")
    ax.set_ylabel("Decision-Elo medio")
    ax.set_xlabel("Grupo posicional")
    ax.set_title("Decision-Elo por posiciones")
    ax.tick_params(axis="x", rotation=25)

    # Espacio superior para etiquetas
    ymax = summary["decision_elo"].max()
    ax.set_ylim(0, ymax + 160)

    # Etiquetas bien colocadas
    for bar, (_, r) in zip(bars, summary.iterrows()):
        height = bar.get_height()
        x = bar.get_x() + bar.get_width() / 2
        ax.annotate(
            f"{r['decision_elo']:.0f}\n{int(r['players'])} jug.",
            xy=(x, height),
            xytext=(0, 4),  # separacion vertical
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        save_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_score_by_action_type(decisions: pd.DataFrame, save_path: Path) -> None:
    """DecisionScore medio por tipo de evento."""
    if decisions.empty:
        return
    summary = decisions.groupby("event_type").agg(
        decision_score=("decision_score", "mean"),
        regret=("regret", "mean"),
        actions=("event_id", "count"),
    ).reset_index()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(summary["event_type"], summary["decision_score"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("DecisionScore medio")
    ax.set_title("Calidad media de decision por tipo de accion")
    for i, r in summary.iterrows():
        ax.text(i, r["decision_score"] + 0.02, f"{r['decision_score']:.2f}\nn={int(r['actions'])}", ha="center", fontsize=9)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_qreal_qbest(decisions: pd.DataFrame, save_path: Path) -> None:
    """Compara valor real contra mejor valor disponible."""
    if decisions.empty:
        return
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    for typ, g in decisions.groupby("event_type"):
        ax.scatter(g["q_best"], g["q_real"], alpha=0.55, label=typ)
    lim = max(float(decisions[["q_best", "q_real"]].max().max()), 0.05)
    ax.plot([0, lim], [0, lim], color="black", ls="--", lw=1)
    ax.set_xlim(0, lim * 1.05)
    ax.set_ylim(0, lim * 1.05)
    ax.set_xlabel("Q_best")
    ax.set_ylabel("Q_real")
    ax.set_title("Valor de la accion real vs mejor movimiento")
    ax.legend(frameon=True)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_pass_success_distribution(decisions: pd.DataFrame, save_path: Path) -> None:
    """Histograma de P(pase) para pases reales y mejores alternativas."""
    if decisions.empty:
        return
    vals = []
    labels = []
    if "real_p_success" in decisions.columns:
        x = decisions.loc[decisions["real_action"] == "Pass", "real_p_success"].dropna().astype(float)
        vals.extend(x[np.isfinite(x)].tolist())
        labels.extend(["Pase real"] * len(x[np.isfinite(x)]))
    if "best_p_success" in decisions.columns:
        x = decisions.loc[decisions["best_action"] == "Pass", "best_p_success"].dropna().astype(float)
        vals.extend(x[np.isfinite(x)].tolist())
        labels.extend(["Mejor alternativa"] * len(x[np.isfinite(x)]))
    if not vals:
        return
    df = pd.DataFrame({"p_success": vals, "tipo": labels})
    fig, ax = plt.subplots(figsize=(9, 5))
    for tipo, g in df.groupby("tipo"):
        ax.hist(g["p_success"], bins=20, alpha=0.55, label=tipo)
    ax.set_title("Distribución de probabilidad de pase")
    ax.set_xlabel("P(pase completado/controlado)")
    ax.set_ylabel("Acciones")
    ax.legend(frameon=True)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_delta_epv_distribution(decisions: pd.DataFrame, save_path: Path) -> None:
    """Distribución del incremento EPV en opciones de pase."""
    if decisions.empty:
        return
    vals = []
    labels = []
    if "real_delta_epv" in decisions.columns:
        x = decisions.loc[decisions["real_action"] == "Pass", "real_delta_epv"].dropna().astype(float)
        vals.extend(x[np.isfinite(x)].tolist())
        labels.extend(["Pase real"] * len(x[np.isfinite(x)]))
    if "best_delta_epv" in decisions.columns:
        x = decisions.loc[decisions["best_action"] == "Pass", "best_delta_epv"].dropna().astype(float)
        vals.extend(x[np.isfinite(x)].tolist())
        labels.extend(["Mejor alternativa"] * len(x[np.isfinite(x)]))
    if not vals:
        return
    df = pd.DataFrame({"delta_epv": vals, "tipo": labels})
    fig, ax = plt.subplots(figsize=(9, 5))
    for tipo, g in df.groupby("tipo"):
        ax.hist(g["delta_epv"], bins=25, alpha=0.55, label=tipo)
    ax.axvline(0, color="black", ls="--", lw=1)
    ax.set_title("Distribución del incremento de EPV")
    ax.set_xlabel("ΔEPV")
    ax.set_ylabel("Acciones")
    ax.legend(frameon=True)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_pipeline_diagram(save_path: Path) -> None:
    """Diagrama del pipeline metodologico."""
    fig, ax = plt.subplots(figsize=(14, 4.2))
    ax.axis("off")
    box_w = 0.135
    boxes = [
        ("Eventos\nPass + Shot", 0.025),
        ("360\nfreeze-frame\njugadores", 0.185),
        ("EPV\npor zonas\nMarkov", 0.345),
        ("Q(pase), Q(tiro)\nriesgo + xG", 0.505),
        ("Mejor\nmovimiento\nargmax Q", 0.665),
        ("Notas + ELO\npor jugador", 0.825),
    ]
    for text, x in boxes:
        ax.add_patch(Rectangle((x, 0.32), box_w, 0.34, ec="black", fc="#eeeeee", lw=1.4))
        ax.text(x + box_w / 2, 0.49, text, ha="center", va="center", fontsize=9, weight="bold")
    for (_, x1), (_, x2) in zip(boxes[:-1], boxes[1:]):
        ax.annotate("", xy=(x2, 0.49), xytext=(x1 + box_w, 0.49), arrowprops=dict(arrowstyle="->", lw=2))
    ax.set_title("Pipeline Decision-Elo: del evento al ranking", fontsize=14, weight="bold")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# 9. Ejecucion completa
# -----------------------------------------------------------------------------
def build_events_dataframe(events_by_match: Dict[int, List[Dict[str, Any]]]) -> pd.DataFrame:
    """Convierte los eventos de muchos partidos en DataFrame Pass/Shot para EPV."""
    rows = []
    for mid, events in events_by_match.items():
        for ev in events:
            row = flatten_event_for_epv(ev, mid)
            if row is not None:
                rows.append(row)
    return pd.DataFrame(rows)


def collect_bayer_decisions(
    events_by_match: Dict[int, List[Dict[str, Any]]],
    frames_by_match: Dict[int, Dict[str, Dict[str, Any]]],
    epv: EPVModel,
    team_id: int = BAYER_TEAM_ID,
    limit: Optional[int] = None,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Evalua pases y tiros del equipo elegido."""
    decisions: List[Dict[str, Any]] = []
    decision_events: List[Dict[str, Any]] = []
    for mid, events in events_by_match.items():
        frame_map = frames_by_match.get(mid, {})
        for ev in events:
            if ev.get("team", {}).get("id") != team_id:
                continue
            if not should_analyze_event_main(ev):
                continue
            frame = frame_map.get(ev.get("id"))
            d = evaluate_event_decision(ev, frame, epv, mid)
            if d is None:
                continue
            decisions.append(d)
            decision_events.append({"event": ev, "frame": frame, "decision": d})
            if limit and len(decisions) >= limit:
                break
        if limit and len(decisions) >= limit:
            break
    return pd.DataFrame(decisions), decision_events



def _diagram_badness_score(obj: Dict[str, Any]) -> float:
    """Puntuacion para elegir un ejemplo visual de mala decision claro.

    No buscamos simplemente el score mas bajo, porque puede salir una accion
    con Q_real = 0 pero regret pequeno, que visualmente es confusa. Priorizamos
    alto regret, alto Q_best y un score bajo pero no completamente nulo.
    """
    d = obj.get("decision", {})
    score = float(d.get("decision_score", 0.0) or 0.0)
    regret = float(d.get("regret", 0.0) or 0.0)
    q_best = float(d.get("q_best", 0.0) or 0.0)
    q_real = float(d.get("q_real", 0.0) or 0.0)
    # Bonus si el evento es tiro: suele ser mas facil de explicar en una presentacion.
    shot_bonus = 0.010 if d.get("event_type") == "Shot" else 0.0
    # Penalizacion si la accion real tiene valor casi cero: suele dar diagramas confusos.
    zero_penalty = 0.030 if q_real < MIN_QREAL_FOR_CLEAR_BAD_DIAGRAM else 0.0
    return regret + 0.35 * q_best + shot_bonus - zero_penalty - 0.015 * abs(score - 0.35)


def choose_bad_visual_example(valid: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Elige una mala decision clara para el diagrama 360.

    Criterios principales:
    - valor de la mejor alternativa relevante;
    - regret absoluto relevante;
    - evitar score=0 por Q_real casi nulo;
    - preferir ejemplos que se puedan explicar visualmente.
    """
    if not valid:
        return None

    clear = []
    negative_clear = []
    for obj in valid:
        d = obj.get("decision", {})
        q_best = float(d.get("q_best", 0.0) or 0.0)
        q_real = float(d.get("q_real", 0.0) or 0.0)
        regret = float(d.get("regret", 0.0) or 0.0)
        score = float(d.get("decision_score", 0.0) or 0.0)
        note = str(d.get("note_symbol", ""))
        if (
            q_best >= MIN_QBEST_FOR_CLEAR_BAD_DIAGRAM
            and regret >= MIN_REGRET_FOR_CLEAR_BAD_DIAGRAM
            and q_real >= MIN_QREAL_FOR_CLEAR_BAD_DIAGRAM
            and MIN_SCORE_FOR_CLEAR_BAD_DIAGRAM <= score <= MAX_SCORE_FOR_CLEAR_BAD_DIAGRAM
        ):
            clear.append(obj)
            if note in {"?!", "?", "??"}:
                negative_clear.append(obj)

    # Primero buscamos una accion con nota claramente negativa. Si no existe,
    # usamos la mejor candidata visual aunque su nota sea neutra.
    if negative_clear:
        return max(negative_clear, key=_diagram_badness_score)
    if clear:
        return max(clear, key=_diagram_badness_score)

    # Fallback menos estricto, pero seguimos evitando Q_real = 0 si hay alternativas.
    fallback = []
    for obj in valid:
        d = obj.get("decision", {})
        q_best = float(d.get("q_best", 0.0) or 0.0)
        q_real = float(d.get("q_real", 0.0) or 0.0)
        regret = float(d.get("regret", 0.0) or 0.0)
        if q_best >= MIN_QBEST_FOR_DIAGRAM and regret >= MIN_REGRET_FOR_BAD_DIAGRAM and q_real > 0:
            fallback.append(obj)
    if fallback:
        return max(fallback, key=_diagram_badness_score)

    # Ultimo recurso: mayor regret absoluto, no menor score.
    return max(valid, key=lambda obj: float(obj.get("decision", {}).get("regret", 0.0) or 0.0))


def choose_good_visual_example(valid: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Elige una buena decision clara para el diagrama 360."""
    if not valid:
        return None
    good = [
        x for x in valid
        if float(x.get("decision", {}).get("q_real", 0.0) or 0.0) >= MIN_QREAL_FOR_GOOD_DIAGRAM
    ]
    pool = good or valid
    return max(
        pool,
        key=lambda obj: (
            float(obj.get("decision", {}).get("decision_score", 0.0) or 0.0),
            float(obj.get("decision", {}).get("q_real", 0.0) or 0.0),
        ),
    )


# -----------------------------------------------------------------------------
# 9.b Seleccion y generacion de los siete planos 360 solicitados
# -----------------------------------------------------------------------------
def _num(d: Dict[str, Any], key: str, default: float = 0.0) -> float:
    """Convierte un campo numerico de decision a float de forma segura."""
    try:
        v = d.get(key, default)
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return default
        return float(v)
    except Exception:
        return default


def _valid_360_events(decision_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filtra eventos 360 para que los diagramas sean interpretables.

    - Pases: actor y receptor real deben aparecer en el freeze-frame.
    - Si la mejor accion es un pase, el receptor de esa alternativa tambien debe
      ser un companero visible. Asi el circulo morado siempre es un jugador real.
    - Tiros: la porteria debe estar dentro del area visible 360.
    - Si la mejor accion es tirar, tambien exigimos porteria visible.
    """
    out: List[Dict[str, Any]] = []
    for x in decision_events:
        frame = x.get("frame")
        event = x.get("event", {})
        d = x.get("decision", {})
        if not frame or not frame.get("freeze_frame"):
            continue
        if d.get("event_type") == "Pass" and d.get("real_pass_completed") is False:
            continue

        loc = event.get("location")
        event_type = d.get("event_type") or event.get("type", {}).get("name")
        real_target = d.get("real_target")
        best_target = d.get("best_target")
        best_action = d.get("best_action")

        if event_type == "Pass":
            if not pass_visible_in_360(frame, loc, real_target):
                continue
        if event_type == "Shot":
            if not goal_visible_in_360(frame):
                continue
        if best_action == "Shot" and not goal_visible_in_360(frame):
            continue
        if best_action == "Pass" and best_target and not visible_teammate_near_target(frame, best_target, tolerance=VISIBLE_BEST_RECEIVER_TOLERANCE):
            continue
        out.append(x)
    return out


def _pick_best(
    valid: List[Dict[str, Any]],
    predicate,
    key,
    *,
    reverse: bool = True,
) -> Optional[Dict[str, Any]]:
    """Selecciona el mejor ejemplo segun un predicado y una funcion de puntuacion."""
    pool = [x for x in valid if predicate(x)]
    if not pool:
        return None
    return sorted(pool, key=key, reverse=reverse)[0]


def _event_label(d: Dict[str, Any], prefix: str) -> str:
    """Titulo compacto para los ejemplos 360."""
    return f"{prefix} - {d.get('player_name')} - {d.get('note_symbol')} {d.get('note_label')}"


def _plot_decision_360(obj: Dict[str, Any], save_path: Path, title_prefix: str) -> None:
    """Dibuja un evento concreto con accion real y mejor accion."""
    d = obj["decision"]
    plot_freeze_frame(
        obj["event"], obj["frame"], save_path,
        title=_event_label(d, title_prefix),
        actual_target=d.get("real_target"),
        best_target=d.get("best_target"),
        best_action=d.get("best_action"),
        note=f"score={d.get('decision_score', 0):.2f}, regret={d.get('regret', 0):.3f}",
        note_symbol=d.get("note_symbol"),
    )


def _xy(value: Any) -> Optional[np.ndarray]:
    """Convierte una coordenada StatsBomb en array [x, y]."""
    if value is None or len(value) < 2:
        return None
    try:
        return np.array([float(value[0]), float(value[1])], dtype=float)
    except Exception:
        return None


def _point_distance(a: Any, b: Any) -> float:
    """Distancia euclidea segura entre dos puntos."""
    aa, bb = _xy(a), _xy(b)
    if aa is None or bb is None:
        return 0.0
    return float(np.linalg.norm(aa - bb))


def _angle_between_targets(origin: Any, target_a: Any, target_b: Any) -> float:
    """Angulo, en grados, entre dos posibles acciones desde el mismo origen."""
    o, a, b = _xy(origin), _xy(target_a), _xy(target_b)
    if o is None or a is None or b is None:
        return 0.0
    va = a - o
    vb = b - o
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    cosang = float(np.dot(va, vb) / (na * nb))
    cosang = max(-1.0, min(1.0, cosang))
    return float(np.degrees(np.arccos(cosang)))


def _clear_other_pass_score(obj: Dict[str, Any]) -> float:
    """Puntuacion visual para elegir un ejemplo claro de 'otro pase era mejor'.

    No basta con que el regret sea alto: si el pase real y la alternativa son
    muy cortos, casi superpuestos o dentro de una zona muy congestionada, el
    grafico no se entiende. Por eso premiamos que el destino real y el destino
    sugerido esten separados y que formen un angulo visible desde el pasador.
    """
    d = obj.get("decision", {})
    origin = obj.get("event", {}).get("location")
    real_target = d.get("real_target")
    best_target = d.get("best_target")
    real_len = _point_distance(origin, real_target)
    best_len = _point_distance(origin, best_target)
    sep = _point_distance(real_target, best_target)
    angle = _angle_between_targets(origin, real_target, best_target)
    regret = _num(d, "regret")
    q_best = _num(d, "q_best")
    score = _num(d, "decision_score")
    p_success = _num(d, "best_p_success")
    # Preferimos acciones explicables: alternativa clara, separada y valiosa.
    return (
        2.5 * regret
        + 0.7 * q_best
        + 0.015 * sep
        + 0.004 * angle
        + 0.004 * min(real_len, best_len)
        + 0.05 * p_success
        - 0.04 * abs(score - 0.45)
    )


def _is_clear_other_pass_candidate(obj: Dict[str, Any], strict: bool = True) -> bool:
    """Filtro para que el grafico de mala decision por otro pase sea legible."""
    d = obj.get("decision", {})
    if d.get("event_type") != "Pass" or d.get("best_action") != "Pass":
        return False
    origin = obj.get("event", {}).get("location")
    real_target = d.get("real_target")
    best_target = d.get("best_target")
    if not real_target or not best_target:
        return False
    real_len = _point_distance(origin, real_target)
    best_len = _point_distance(origin, best_target)
    sep = _point_distance(real_target, best_target)
    angle = _angle_between_targets(origin, real_target, best_target)

    # En modo estricto buscamos que se vea claramente que habia otro pase.
    if strict:
        return (
            _num(d, "regret") >= max(MIN_REGRET_FOR_BAD_DIAGRAM, 0.015)
            and _num(d, "q_best") >= max(MIN_QBEST_FOR_DIAGRAM, 0.025)
            and _num(d, "decision_score") < 0.75
            and real_len >= 7.0
            and best_len >= 9.0
            and sep >= 10.0
            and angle >= 22.0
        )

    # Fallback: aun exigimos separacion visual minima para no repetir ejemplos
    # casi superpuestos como el que quedaba confuso en el area.
    return (
        _num(d, "regret") >= 0.010
        and _num(d, "q_best") >= 0.015
        and real_len >= 5.0
        and best_len >= 7.0
        and sep >= 7.0
        and angle >= 15.0
    )



def pass_grid_intersection_count(
    origin: Optional[List[float]],
    target: Optional[List[float]],
    frame: Optional[Dict[str, Any]],
    cell_size: float = 1.0,
    radius: float = METHODOLOGY_GRID_RADIUS,
) -> int:
    """Numero de celdas de interseccion entre la linea de pase y zonas rivales.

    Se usa para elegir automaticamente un ejemplo metodologico que ensene de
    forma visible como el modelo detecta riesgo sobre la trayectoria del pase.
    No modifica el calculo del riesgo: solo ayuda a seleccionar una figura mas
    clara para el informe.
    """
    if not origin or not target or not frame or not frame.get("freeze_frame"):
        return 0
    path_cells = get_line_cells(origin, target, cell_size)
    intersection_cells: set = set()
    for player in frame.get("freeze_frame", []):
        if player.get("teammate") or is_keeper_player(player) or player.get("actor"):
            continue
        loc = player.get("location")
        if not loc or len(loc) < 2:
            continue
        defender_cells = get_player_cells(loc, radius, cell_size)
        intersection_cells |= defender_cells & path_cells
    return len(intersection_cells)


def pass_methodology_visual_ok(obj: Dict[str, Any], min_length: float = MIN_PASS_LENGTH_FOR_METHOD_GRID, min_intersections: int = MIN_PASS_INTERSECTIONS_FOR_METHOD_GRID) -> bool:
    """Filtro visual para el grafico metodologico de pase.

    Exige que el pase sea suficientemente largo, que el pasador y receptor sean
    visibles en el 360 y que exista al menos una interseccion con la zona de
    alcance de un rival. Asi evitamos ejemplos donde la metodologia no se ve.
    """
    d = obj.get("decision", {})
    event = obj.get("event", {})
    frame = obj.get("frame")
    origin = event.get("location")
    target = d.get("real_target")
    if d.get("event_type") != "Pass" or not origin or not target:
        return False
    if _point_distance(origin, target) < min_length:
        return False
    if not pass_visible_in_360(frame, origin, target):
        return False
    return pass_grid_intersection_count(origin, target, frame) >= min_intersections

def _pass_methodology_score(obj: Dict[str, Any]) -> float:
    """Puntuacion para elegir un pase metodologico claro.

    Para esta figura damos prioridad a que se vea la metodologia: linea negra
    larga, interseccion morada con zona de alcance rival y trayectoria sin
    quedar completamente tapada por jugadores. No se usa para valorar al
    jugador; solo para elegir un ejemplo pedagogico.
    """
    d = obj.get("decision", {})
    origin = obj.get("event", {}).get("location")
    target = d.get("real_target")
    frame = obj.get("frame")
    length = _point_distance(origin, target)
    intersections = pass_grid_intersection_count(origin, target, frame)
    risk = max(0.0, _num(d, "real_risk"))
    delta = abs(_num(d, "real_delta_epv"))
    # Intersecciones y longitud pesan mas que el valor futbolistico porque el
    # objetivo de esta figura es explicar el mecanismo de riesgo del pase.
    return 3.00 * min(intersections, 8) + 0.10 * length + 0.40 * risk + 0.20 * delta


def select_pass_methodology_example(valid: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Ejemplo para explicar metodologia de pase.

    Prioridad:
    1) pase largo, pasador y receptor visibles, con interseccion defensa-linea;
    2) si no existe, se relajan los umbrales;
    3) solo como ultimo recurso se usa cualquier pase visible.
    """
    # Caso ideal: linea muy larga y al menos una interseccion visible.
    obj = _pick_best(
        valid,
        lambda x: pass_methodology_visual_ok(x, min_length=30.0, min_intersections=1),
        lambda x: _pass_methodology_score(x),
    )
    if obj is not None:
        return obj

    # Segundo intento: linea larga con interseccion.
    obj = _pick_best(
        valid,
        lambda x: pass_methodology_visual_ok(x, min_length=MIN_PASS_LENGTH_FOR_METHOD_GRID, min_intersections=1),
        lambda x: _pass_methodology_score(x),
    )
    if obj is not None:
        return obj

    # Tercer intento: algo menos largo, pero siempre con interseccion visible.
    obj = _pick_best(
        valid,
        lambda x: pass_methodology_visual_ok(x, min_length=16.0, min_intersections=1),
        lambda x: _pass_methodology_score(x),
    )
    if obj is not None:
        return obj

    # Si no existe ningun pase con interseccion en el partido analizado, no
    # generamos esta figura. Es preferible no forzar un ejemplo que no explique
    # bien la metodologia.
    return None


def select_shot_methodology_example(valid: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Ejemplo para explicar metodologia de tiro: preferimos un tiro con valor alto."""
    return _pick_best(
        valid,
        lambda x: x.get("decision", {}).get("event_type") == "Shot",
        lambda x: (_num(x["decision"], "q_real"), _num(x["decision"], "q_best")),
    )


def select_good_pass_example(valid: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Buena decision de pase: pase real cercano a la mejor opcion."""
    obj = _pick_best(
        valid,
        lambda x: x.get("decision", {}).get("event_type") == "Pass"
        and _num(x["decision"], "decision_score") >= 0.80
        and _num(x["decision"], "q_real") >= MIN_QREAL_FOR_GOOD_DIAGRAM,
        lambda x: (_num(x["decision"], "decision_score"), _num(x["decision"], "q_real")),
    )
    if obj is None:
        obj = _pick_best(
            valid,
            lambda x: x.get("decision", {}).get("event_type") == "Pass",
            lambda x: (_num(x["decision"], "decision_score"), _num(x["decision"], "q_real")),
        )
    return obj


def select_bad_pass_other_pass_example(valid: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Mala decision de pase porque habia otro pase mejor.

    La seleccion prioriza que el ejemplo sea visualmente claro: el pase real y
    la alternativa deben tener destinos separados, angulo apreciable y longitud
    suficiente. Asi se evita elegir acciones muy cortas o congestionadas que no
    sirven bien para explicar el concepto en el informe.
    """
    obj = _pick_best(
        valid,
        lambda x: _is_clear_other_pass_candidate(x, strict=True),
        lambda x: _clear_other_pass_score(x),
    )
    if obj is None:
        obj = _pick_best(
            valid,
            lambda x: _is_clear_other_pass_candidate(x, strict=False),
            lambda x: _clear_other_pass_score(x),
        )
    # Ultimo recurso: exigimos que al menos sea otro pase mejor; si no hay un
    # caso claro en ese partido, no generamos la figura para evitar confusiones.
    return obj


def select_bad_pass_should_shoot_example(valid: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Mala decision de pase porque el motor preferia tirar."""
    obj = _pick_best(
        valid,
        lambda x: x.get("decision", {}).get("event_type") == "Pass"
        and x.get("decision", {}).get("best_action") == "Shot"
        and _num(x["decision"], "regret") >= MIN_REGRET_FOR_BAD_DIAGRAM,
        lambda x: (_num(x["decision"], "regret"), _num(x["decision"], "q_best")),
    )
    return obj


def select_good_shot_example(valid: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Buena decision de tiro: tirar era la mejor opcion o una opcion muy cercana."""
    obj = _pick_best(
        valid,
        lambda x: x.get("decision", {}).get("event_type") == "Shot"
        and x.get("decision", {}).get("best_action") == "Shot"
        and _num(x["decision"], "decision_score") >= 0.80,
        lambda x: (_num(x["decision"], "decision_score"), _num(x["decision"], "q_real")),
    )
    if obj is None:
        obj = _pick_best(
            valid,
            lambda x: x.get("decision", {}).get("event_type") == "Shot",
            lambda x: (_num(x["decision"], "decision_score"), _num(x["decision"], "q_real")),
        )
    return obj


def select_bad_shot_should_pass_example(valid: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Mala decision de tiro porque habia un pase mejor."""
    obj = _pick_best(
        valid,
        lambda x: x.get("decision", {}).get("event_type") == "Shot"
        and x.get("decision", {}).get("best_action") == "Pass"
        and _num(x["decision"], "regret") >= MIN_REGRET_FOR_BAD_DIAGRAM,
        lambda x: (_num(x["decision"], "regret"), _num(x["decision"], "q_best")),
    )
    if obj is None:
        obj = _pick_best(
            valid,
            lambda x: x.get("decision", {}).get("event_type") == "Shot" and x.get("decision", {}).get("best_action") == "Pass",
            lambda x: (_num(x["decision"], "regret"), _num(x["decision"], "q_best")),
        )
    return obj


def generate_requested_360_figures(decision_events: List[Dict[str, Any]], figures: Path, prefix: str = "") -> Dict[str, Optional[str]]:
    """Genera exactamente los siete graficos 360 solicitados.

    Devuelve un diccionario con el nombre de cada figura y el event_id usado. Si
    en un partido no existe una categoria concreta, el valor queda en None.
    """
    figures.mkdir(parents=True, exist_ok=True)
    valid = _valid_360_events(decision_events)
    used: Dict[str, Optional[str]] = {}

    # 1. Grafico metodologico de pase con linea negra, fuera de juego y grid.
    obj = select_pass_methodology_example(valid)
    if obj is not None:
        d = obj["decision"]
        plot_pass_grid(
            obj["event"].get("location"), d.get("real_target"), obj["frame"],
            figures / f"{prefix}10_grafico_01_metodologia_pase_grid.png",
            title=f"1. Metodologia de pase - {d.get('player_name')}",
            note_symbol=d.get("note_symbol"),
        )
        used["01_metodologia_pase"] = d.get("event_id")
    else:
        used["01_metodologia_pase"] = None

    # 2. Grafico metodologico de tiro con linea negra y triangulo de tiro.
    # No mostramos mejor alternativa aqui para que sea una figura puramente
    # metodologica del tiro.
    obj = select_shot_methodology_example(valid)
    if obj is not None:
        d = obj["decision"]
        plot_freeze_frame(
            obj["event"], obj["frame"], figures / f"{prefix}11_grafico_02_metodologia_tiro_360.png",
            title=f"2. Metodologia de tiro - {d.get('player_name')}",
            actual_target=[120.0, 40.0],
            best_target=None,
            best_action=None,
            note=f"P(gol)={d.get('q_real', 0):.3f}",
            note_symbol=d.get("note_symbol"),
        )
        used["02_metodologia_tiro"] = d.get("event_id")
    else:
        used["02_metodologia_tiro"] = None

    # 3. Ejemplo de buena decision de pase.
    obj = select_good_pass_example(valid)
    if obj is not None:
        _plot_decision_360(obj, figures / f"{prefix}12_grafico_03_buena_decision_pase_360.png", "3. Buena decision de pase")
        used["03_buena_decision_pase"] = obj["decision"].get("event_id")
    else:
        used["03_buena_decision_pase"] = None

    # 4. Mala decision de pase porque habia otro pase mejor.
    obj = select_bad_pass_other_pass_example(valid)
    if obj is not None:
        _plot_decision_360(obj, figures / f"{prefix}13_grafico_04_mala_decision_pase_otro_pase_360.png", "4. Mala decision: otro pase era mejor")
        used["04_mala_decision_pase_otro_pase"] = obj["decision"].get("event_id")
    else:
        used["04_mala_decision_pase_otro_pase"] = None

    # 5. Mala decision de pase porque deberia tirar.
    obj = select_bad_pass_should_shoot_example(valid)
    if obj is not None:
        _plot_decision_360(obj, figures / f"{prefix}14_grafico_05_mala_decision_pase_deberia_tirar_360.png", "5. Mala decision: debia tirar")
        used["05_mala_decision_pase_deberia_tirar"] = obj["decision"].get("event_id")
    else:
        used["05_mala_decision_pase_deberia_tirar"] = None

    # 6. Buena decision de tiro.
    obj = select_good_shot_example(valid)
    if obj is not None:
        _plot_decision_360(obj, figures / f"{prefix}15_grafico_06_buena_decision_tiro_360.png", "6. Buena decision de tiro")
        used["06_buena_decision_tiro"] = obj["decision"].get("event_id")
    else:
        used["06_buena_decision_tiro"] = None

    # 7. Mala decision de tiro porque deberia pasar.
    obj = select_bad_shot_should_pass_example(valid)
    if obj is not None:
        _plot_decision_360(obj, figures / f"{prefix}16_grafico_07_mala_decision_tiro_deberia_pasar_360.png", "7. Mala decision: debia pasar")
        used["07_mala_decision_tiro_deberia_pasar"] = obj["decision"].get("event_id")
    else:
        used["07_mala_decision_tiro_deberia_pasar"] = None

    return used

def run_full_analysis(
    data_root: Path,
    output_root: Path,
    max_matches: Optional[int] = DEFAULT_MAX_MATCHES,
    match_id: Optional[int] = None,
    match_index: int = DEFAULT_MATCH_INDEX,
    max_actions: Optional[int] = None,
) -> Dict[str, Any]:
    """Ejecuta el analisis final de Bayer si estan los datos completos."""
    all_match_ids = get_bayer_match_ids(data_root, verbose=True)
    match_ids = select_bayer_match_ids(
        all_match_ids,
        max_matches=max_matches,
        match_id=match_id,
        match_index=match_index,
        verbose=True,
    )
    events_by_match = load_events_for_matches(data_root, match_ids)
    frames_by_match = load_360_for_matches(data_root, match_ids)

    epv_df = build_events_dataframe(events_by_match)
    epv = build_epv_model(epv_df)

    decisions, decision_events = collect_bayer_decisions(events_by_match, frames_by_match, epv, limit=max_actions)
    ratings = aggregate_player_ratings(decisions)

    figures = output_root / "figures"
    tables = output_root / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    export_bayer_passes_shots_360(events_by_match, frames_by_match, tables / "leverkusen_passes_shots_360.json")

    plot_pipeline_diagram(figures / "00_pipeline_decision_elo.png")
    plot_note_scale(figures / "00b_escala_notas_ajedrez.png")
    plot_epv_heatmap(epv, figures / "01_epv_heatmap.png")
    plot_note_distribution(decisions, figures / "02_note_distribution.png")
    plot_regret_histogram(decisions, figures / "03_regret_histogram.png")
    plot_rating_bar(ratings, figures / "04_decision_elo_ranking.png", min_actions=20)
    plot_risk_delta_epv_scatter(decisions, figures / "05_riesgo_vs_delta_epv.png")
    plot_pass_success_distribution(decisions, figures / "05b_distribucion_probabilidad_pase.png")
    plot_delta_epv_distribution(decisions, figures / "05c_distribucion_delta_epv.png")
    plot_elo_by_position_group(ratings, figures / "06_elo_por_posicion.png", min_actions=1)
    plot_score_by_action_type(decisions, figures / "07_score_por_tipo_accion.png")
    plot_qreal_qbest(decisions, figures / "08_qreal_vs_qbest.png")

    # Siete planos 360 solicitados para explicar la metodologia y ejemplos.
    figuras_360_generadas = generate_requested_360_figures(decision_events, figures)

    decisions.drop(columns=["candidates"], errors="ignore").to_csv(tables / "decisions.csv", index=False)
    ratings.to_csv(tables / "player_ratings.csv", index=False)
    save_json({
        "match_ids_total_bayer": all_match_ids,
        "match_ids_analizados": match_ids,
        "max_matches": max_matches,
        "match_id_forzado": match_id,
        "match_index": match_index,
        "max_actions": max_actions,
        "epv_iterations": epv.iterations,
        "num_decisions": len(decisions),
        "pass_filter_main": "completed_only_no_high_open_play",
        "require_recipient_for_completed_pass": REQUIRE_RECIPIENT_FOR_COMPLETED_PASS,
        "figuras_360_generadas": figuras_360_generadas,
    }, tables / "run_summary.json")

    return {"mode": "bayer", "decisions": decisions, "ratings": ratings, "epv": epv, "decision_events": decision_events}


def run_demo_analysis(demo_zip: Path, output_root: Path) -> Dict[str, Any]:
    """
    Modo demo con el partido 7298.json incluido en el ZIP del usuario.

    Sirve para comprobar el pipeline de tiros y diagramas. No es el analisis de
    Bayer porque el ZIP no contiene los eventos/360 de Bayer 2023/24.
    """
    demo_event_path = extract_demo_events(demo_zip, output_root / "demo_extracted")
    events = load_json(demo_event_path)
    events_by_match = {7298: events}

    # Para EPV usamos passes + shots del partido de ejemplo.
    epv_df = build_events_dataframe(events_by_match)
    epv = build_epv_model(epv_df)

    # Creamos frames solo para tiros con freeze_frame incrustado.
    frames = {ev.get("id"): make_inline_shot_frame(ev) for ev in events if ev.get("type", {}).get("name") == "Shot"}
    frames = {k: v for k, v in frames.items() if v is not None}
    frames_by_match = {7298: frames}

    # En demo evaluamos tiros de ambos equipos porque no es Bayer.
    decisions: List[Dict[str, Any]] = []
    decision_events: List[Dict[str, Any]] = []
    for ev in events:
        if ev.get("type", {}).get("name") != "Shot":
            continue
        frame = frames.get(ev.get("id"))
        d = evaluate_event_decision(ev, frame, epv, 7298)
        if d is None:
            continue
        decisions.append(d)
        decision_events.append({"event": ev, "frame": frame, "decision": d})
    decisions_df = pd.DataFrame(decisions)
    ratings = aggregate_player_ratings(decisions_df)

    figures = output_root / "figures"
    tables = output_root / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    plot_pipeline_diagram(figures / "00_pipeline_decision_elo.png")
    plot_note_scale(figures / "00b_escala_notas_ajedrez.png")
    plot_epv_heatmap(epv, figures / "01_epv_heatmap_demo.png", title="EPV - demo con partido adjunto")
    plot_note_distribution(decisions_df, figures / "02_note_distribution_demo.png")
    plot_regret_histogram(decisions_df, figures / "03_regret_histogram_demo.png")
    plot_rating_bar(ratings, figures / "04_decision_elo_ranking_demo.png", min_actions=1)
    plot_risk_delta_epv_scatter(decisions_df, figures / "05_riesgo_vs_delta_epv_demo.png")
    plot_pass_success_distribution(decisions_df, figures / "05b_distribucion_probabilidad_pase_demo.png")
    plot_delta_epv_distribution(decisions_df, figures / "05c_distribucion_delta_epv_demo.png")
    plot_elo_by_position_group(ratings, figures / "06_elo_por_posicion_demo.png", min_actions=1)
    plot_score_by_action_type(decisions_df, figures / "07_score_por_tipo_accion_demo.png")
    plot_qreal_qbest(decisions_df, figures / "08_qreal_vs_qbest_demo.png")

    if decision_events:
        valid = [x for x in decision_events if x["frame"] and x["frame"].get("freeze_frame")]
        # No dibujamos pases reales incompletos en el estudio principal.
        # Si en el futuro se activa un modo de todos los pases, seguiran fuera
        # de los diagramas para evitar confundir decision y ejecucion.
        valid = [
            x for x in valid
            if not (x["decision"].get("event_type") == "Pass" and x["decision"].get("real_pass_completed") is False)
        ]
        examples = []
        bad_obj = choose_bad_visual_example(valid)
        good_obj = choose_good_visual_example(valid)
        if bad_obj is not None:
            examples.append((bad_obj, "10_demo_mala_decision_360.png"))
        if good_obj is not None:
            examples.append((good_obj, "11_demo_buena_decision_360.png"))
        for obj, fname in examples:
            d = obj["decision"]
            title = f"DEMO - {d['player_name']} - {d['event_type']} - {d['note_symbol']} {d['note_label']}"
            plot_freeze_frame(
                obj["event"], obj["frame"], figures / fname,
                title=title, actual_target=d.get("real_target"), best_target=d.get("best_target"),
                note=f"score={d['decision_score']:.2f}, regret={d['regret']:.3f}",
                note_symbol=d.get("note_symbol"),
            )
            # Figura adicional tipo grid para la linea de pase si la mejor o la real son pase.
            grid_target = d.get("best_target") if d.get("best_action") == "Pass" else d.get("real_target") if d.get("real_action") == "Pass" else None
            if grid_target:
                plot_pass_grid(
                    obj["event"].get("location"), grid_target, obj["frame"],
                    figures / fname.replace("360", "pass_grid"),
                    title=f"Grid de pase - {d['player_name']} - {d['note_symbol']} {d['note_label']}",
                    note_symbol=d.get("note_symbol"),
                )

        # Diagramas especificos de tiros para la demo.
        shot_valid = [x for x in valid if x["decision"].get("event_type") == "Shot"]
        shot_valid_sorted = sorted(shot_valid, key=lambda x: x["decision"]["decision_score"])
        shot_examples = []
        if shot_valid_sorted:
            shot_examples.append((shot_valid_sorted[0], "12_demo_tiro_mala_decision_360.png"))
            if len(shot_valid_sorted) > 1:
                shot_examples.append((shot_valid_sorted[-1], "13_demo_tiro_buena_decision_360.png"))
        for obj, fname in shot_examples:
            d = obj["decision"]
            plot_shot_diagram(obj["event"], obj["frame"], d, figures / fname)
            if d.get("best_action") == "Pass" and d.get("best_target"):
                plot_pass_grid(
                    obj["event"].get("location"), d.get("best_target"), obj["frame"],
                    figures / fname.replace("360", "mejor_pase_grid"),
                    title=f"Tiro: mejor alternativa de pase - {d['player_name']} - {d['note_symbol']} {d['note_label']}",
                    note_symbol=d.get("note_symbol"),
                )

    decisions_df.drop(columns=["candidates"], errors="ignore").to_csv(tables / "decisions_demo.csv", index=False)
    ratings.to_csv(tables / "player_ratings_demo.csv", index=False)
    save_json({"mode": "demo", "num_events": len(events), "num_decisions": len(decisions_df), "epv_iterations": epv.iterations}, tables / "run_summary_demo.json")

    return {"mode": "demo", "decisions": decisions_df, "ratings": ratings, "epv": epv, "decision_events": decision_events}


def plot_note_scale(save_path: Path) -> None:
    """Escala de notas tipo ajedrez usada para clasificar cada decision."""
    labels = [
        ("!!", "Brillante", "score >= 0.97 y Q_best >= 0.05"),
        ("!", "Mejor movimiento", "score >= 0.92"),
        ("✓", "Excelente", "score >= 0.80"),
        ("=", "Buena", "score >= 0.58"),
        ("?!", "Imprecision", "score bajo con regret >= 0.004-0.006"),
        ("?", "Error", "regret >= 0.025 y score bajo"),
        ("??", "Pifia", "regret >= 0.070 y Q_best >= 0.070"),
    ]
    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    ax.axis("off")
    ax.set_title("Escala de notas tipo ajedrez: version menos restrictiva", fontsize=15, weight="bold")
    for i, (sym, name, rule) in enumerate(labels):
        y = 0.86 - i * 0.12
        ax.add_patch(Rectangle((0.06, y - 0.04), 0.08, 0.08, ec="black", fc="#eeeeee"))
        ax.text(0.10, y, sym, ha="center", va="center", fontsize=14, weight="bold")
        ax.text(0.18, y, name, ha="left", va="center", fontsize=12, weight="bold")
        ax.text(0.50, y, rule, ha="left", va="center", fontsize=10.7)
    ax.text(0.06, 0.04, "Escala menos restrictiva: mas variedad de notas, sin convertir acciones de impacto minimo en pifias.", fontsize=10)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 10. FUNCIONES SIMPLES PARA USAR DESDE PYTHON / JUPYTER
# =============================================================================
# Estas funciones son la entrada recomendada. No necesitas terminal ni otros
# archivos .py: importa este archivo y llama a ejecutar_bayer(), ejecutar_demo()
# o ejecutar_auto().

DEFAULT_DATA_ROOT = Path("data")
DEFAULT_DEMO_ZIP = Path(__file__).resolve().parent / "data" / "demo" / "Nueva_carpeta.zip"
DEFAULT_DEMO_OUTPUT = Path("resultados_demo")
DEFAULT_BAYER_OUTPUT = Path("resultados_bayer")


def extraer_leverkusen_json(
    data_root: str | Path = DEFAULT_DATA_ROOT,
    output_file: str | Path = "leverkusen_passes_shots_360.json",
    require_360: bool = False,
    max_matches: Optional[int] = DEFAULT_MAX_MATCHES,
    match_id: Optional[int] = None,
    match_index: int = DEFAULT_MATCH_INDEX,
) -> Path:
    """
    Crea un JSON de inspeccion con TODOS los pases y TODOS los tiros de Bayer
    Leverkusen 2023/24 unidos a su freeze-frame 360.

    Sustituye al script separado extract_leverkusen_events_360.py.

    Uso en Python/Jupyter:
        import decision_elo_leverkusen_unico as de
        de.extraer_leverkusen_json("data")  # por defecto 1 partido
    """
    data_root = Path(data_root)
    output_file = Path(output_file)

    all_match_ids = get_bayer_match_ids(data_root, verbose=True)
    match_ids = select_bayer_match_ids(
        all_match_ids,
        max_matches=max_matches,
        match_id=match_id,
        match_index=match_index,
        verbose=True,
    )
    events_by_match = load_events_for_matches(data_root, match_ids)
    frames_by_match = load_360_for_matches(data_root, match_ids)
    export_bayer_passes_shots_360(
        events_by_match=events_by_match,
        frames_by_match=frames_by_match,
        output_file=output_file,
        team_id=BAYER_TEAM_ID,
        require_360=require_360,
    )
    print(f"JSON guardado en: {output_file.resolve()}")
    return output_file


def ejecutar_bayer(
    data_root: str | Path = DEFAULT_DATA_ROOT,
    output_root: str | Path = DEFAULT_BAYER_OUTPUT,
    max_matches: Optional[int] = DEFAULT_MAX_MATCHES,
    match_id: Optional[int] = None,
    match_index: int = DEFAULT_MATCH_INDEX,
    max_actions: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Ejecuta el analisis real de Bayer Leverkusen 2023/24.

    Estructura esperada:
        open-data-master/data/
          competitions.json
          matches/9/281.json
          events/<match_id>.json
          three-sixty/<match_id>.json

    Uso:
        import decision_elo_leverkusen_unico as de
        result = de.ejecutar_bayer("data", "resultados_bayer")  # por defecto 1 partido
        # Para toda la temporada: de.ejecutar_bayer("data", "resultados_bayer_full", max_matches=None)
    """
    data_root = Path(data_root)
    output_root = Path(output_root)
    mode = validate_or_demo(data_root=data_root, demo_zip=None)
    if mode != "bayer":
        raise FileNotFoundError(
            "No se detecta la estructura StatsBomb de Bayer 2023/24. "
            "Comprueba que data_root contiene matches/9/281.json, events/ y three-sixty/."
        )

    result = run_full_analysis(
        data_root=data_root,
        output_root=output_root,
        max_matches=max_matches,
        match_id=match_id,
        match_index=match_index,
        max_actions=max_actions,
    )
    print("Analisis de Bayer ejecutado correctamente.")
    print(f"Graficos: {output_root / 'figures'}")
    print(f"Tablas:   {output_root / 'tables'}")
    return result


def ejecutar_demo(
    demo_zip: str | Path = DEFAULT_DEMO_ZIP,
    output_root: str | Path = DEFAULT_DEMO_OUTPUT,
) -> Dict[str, Any]:
    """
    Ejecuta la demo incluida para verificar graficos/tablas sin tener que cargar
    todos los JSON reales de Bayer. La demo NO es Bayer; solo valida el pipeline.

    Uso:
        import decision_elo_leverkusen_unico as de
        result = de.ejecutar_demo(output_root="resultados_demo")
    """
    demo_zip = Path(demo_zip)
    output_root = Path(output_root)
    if not demo_zip.exists():
        raise FileNotFoundError(f"No encuentro el ZIP demo: {demo_zip}")

    result = run_demo_analysis(demo_zip=demo_zip, output_root=output_root)
    print("Demo ejecutada correctamente.")
    print(f"Graficos: {output_root / 'figures'}")
    print(f"Tablas:   {output_root / 'tables'}")
    return result


def ejecutar_auto(
    data_root: str | Path = DEFAULT_DATA_ROOT,
    output_root: str | Path = DEFAULT_BAYER_OUTPUT,
    demo_zip: str | Path = DEFAULT_DEMO_ZIP,
    max_matches: Optional[int] = DEFAULT_MAX_MATCHES,
    match_id: Optional[int] = None,
    match_index: int = DEFAULT_MATCH_INDEX,
    max_actions: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Detecta automaticamente si existen los JSON reales de Bayer.

    - Si encuentra Bayer, ejecuta el analisis real.
    - Si no encuentra Bayer pero existe el ZIP demo, ejecuta la demo.

    Uso:
        import decision_elo_leverkusen_unico as de
        result = de.ejecutar_auto("data", "resultados")
    """
    data_root = Path(data_root)
    output_root = Path(output_root)
    demo_zip = Path(demo_zip)
    mode = validate_or_demo(data_root=data_root, demo_zip=demo_zip)
    print(f"Modo detectado: {mode}")

    if mode == "bayer":
        return ejecutar_bayer(
            data_root=data_root,
            output_root=output_root,
            max_matches=max_matches,
            match_id=match_id,
            match_index=match_index,
            max_actions=max_actions,
        )
    if mode == "demo":
        return ejecutar_demo(demo_zip=demo_zip, output_root=output_root)

    raise FileNotFoundError("No se han encontrado datos de Bayer ni ZIP demo.")


def cargar_tablas(output_root: str | Path = DEFAULT_DEMO_OUTPUT) -> Dict[str, pd.DataFrame]:
    """
    Carga las tablas generadas por el analisis en DataFrames de pandas.
    Busca automaticamente nombres demo y nombres Bayer.
    """
    output_root = Path(output_root)
    tables_dir = output_root / "tables"
    paths = {
        "decisiones": [tables_dir / "decisions.csv", tables_dir / "decisions_demo.csv"],
        "ratings": [tables_dir / "player_ratings.csv", tables_dir / "player_ratings_demo.csv"],
    }
    out: Dict[str, pd.DataFrame] = {}
    for name, candidates in paths.items():
        for path in candidates:
            if path.exists():
                out[name] = pd.read_csv(path)
                break
    if not out:
        raise FileNotFoundError(f"No hay tablas CSV en {tables_dir}")
    return out


def ver_resumen(output_root: str | Path = DEFAULT_DEMO_OUTPUT, n: int = 8) -> None:
    """Imprime primeras filas de las tablas generadas."""
    tablas = cargar_tablas(output_root)
    for name, df in tablas.items():
        print("=" * 80)
        print(name)
        print("filas:", len(df), "columnas:", len(df.columns))
        print(df.head(n).to_string(index=False))


def ejecutar_directo_desde_python(
    data_root: str | Path | None = None,
    output_root: str | Path = DEFAULT_BAYER_OUTPUT,
    generar_json_revision: bool = True,
    max_matches: Optional[int] = DEFAULT_MAX_MATCHES,
    match_id: Optional[int] = None,
    match_index: int = DEFAULT_MATCH_INDEX,
    max_actions: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Funcion pensada para usarla con el boton Run/F5 de Python, Spyder, VS Code
    o Jupyter. Hace el flujo completo sin terminal:

    1. Detecta la carpeta data/ de StatsBomb.
    2. Comprueba Bayer Leverkusen 2023/24.
    3. Por defecto selecciona SOLO 1 partido para que no sea pesado.
    4. Genera un JSON de revision con pases + tiros + 360.
    4. Ejecuta EPV, calidad de tiro, riesgo/probabilidad de pase, Decision-Elo.
    5. Guarda graficos y tablas en output_root.

    Uso desde una celda:
        import decision_elo_leverkusen_unico as de
        result = de.ejecutar_directo_desde_python()

    Si el archivo .py esta en la misma carpeta que data/, basta con darle a Run/F5.
    """
    print("=" * 88)
    print("DECISION-ELO BAYER LEVERKUSEN 2023/24 - EJECUCION AUTOMATICA")
    print("=" * 88)

    # Si data_root es None, resolve_data_root busca en el directorio actual y junto al script.
    root = resolve_data_root(data_root)
    output_root = Path(output_root)

    print(f"Carpeta de datos detectada: {root}")
    print(f"Carpeta de salida:          {output_root.resolve()}")
    print(f"Limite de partidos:         {max_matches if max_matches is not None else 'toda la temporada'}")
    if match_id is not None:
        print(f"Match ID forzado:           {match_id}")

    print("\n[1/4] Comprobando partidos, eventos y 360 de Bayer...")
    resumen = comprobar_datos_bayer(
        root,
        max_matches=max_matches,
        match_id=match_id,
        match_index=match_index,
    )

    if generar_json_revision:
        print("\n[2/4] Generando JSON de revision con pases + tiros + 360...")
        json_path = output_root / "tables" / "leverkusen_passes_shots_360.json"
        extraer_leverkusen_json(
            data_root=root,
            output_file=json_path,
            require_360=False,
            max_matches=max_matches,
            match_id=match_id,
            match_index=match_index,
        )
    else:
        print("\n[2/4] Saltando JSON de revision...")

    print("\n[3/4] Ejecutando analisis completo...")
    result = ejecutar_bayer(
        data_root=root,
        output_root=output_root,
        max_matches=max_matches,
        match_id=match_id,
        match_index=match_index,
        max_actions=max_actions,
    )

    print("\n[4/4] Primeras filas de las tablas generadas...")
    try:
        ver_resumen(output_root, n=8)
    except Exception as exc:
        print(f"No he podido imprimir el resumen de tablas: {exc}")

    print("\nEjecucion terminada.")
    print(f"Graficos: {output_root.resolve() / 'figures'}")
    print(f"Tablas:   {output_root.resolve() / 'tables'}")
    print("=" * 88)

    return {"resumen_datos": resumen, **result}


if __name__ == "__main__":
    # Al pulsar Run/F5 sobre este archivo, se ejecuta todo automaticamente.

    try:
        entrada = input(
            "Numero maximo de partidos "
        ).strip().lower()

        # Todos los partidos
        if entrada in {"max_matches", "all"}:
            max_matches = None
        else:
            max_matches = int(entrada)

        ejecutar_directo_desde_python(
            data_root=None,
            output_root=DEFAULT_BAYER_OUTPUT,
            generar_json_revision=True,
            max_matches=max_matches,
            match_id=None,
            match_index=DEFAULT_MATCH_INDEX,
            max_actions=None,
        )

    except Exception as exc:
        print("\nERROR DURANTE LA EJECUCION AUTOMATICA")
        print(str(exc))

        print("\nComprueba que el archivo decision_elo_leverkusen_unico.py esta en la carpeta")
        print("que contiene data/, o ejecuta desde Python:")

        print("    import decision_elo_leverkusen_unico as de")
        print("    de.ejecutar_bayer(r'C:/ruta/a/tu/data', 'resultados_bayer')")

        raise
