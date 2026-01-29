"""
map_generation.py (ROBUST v3)

Easy difficulty (5 slots only). Goal:
- Long intuitive path to goal (many tile-moves), but solvable with 5 slots:
  loop tokens (2) + 3 actions.
- Ladders are VISUAL ONLY (Up/Down allowed anywhere; no ladder restriction).
- The program's route must be the SHORTEST PATH on that map (tile-moves).
- The program must be locally minimal: removing any 1 action breaks solvability.
- Keep only ONE path (path-only corridor; everything else is WALL).
- Avoid boring extremes:
  - No "R,U,R,U,..." for a long stretch (cap consecutive gated columns).
  - No "R,R,R,..." for 20 tiles in a row (cap consecutive straight columns).

Fixed test program (for now):
  f0 {
    →                 (unconditional Right)
    if_green + ↓       (Down if on GREEN tile)
    if_blue  + ↑       (Up if on BLUE tile)
  } f1

Map strategy (guaranteed by validation):
- Build a single 1-tile-wide corridor only (everything else is WALL).
- Progress is mostly RIGHT; occasional "gates" force a vertical move:
  - On a gate step, the current tile is colored GREEN (forces ↓) or BLUE (forces ↑).
  - The corridor then continues on the new row.
  - The tile to the right on the OLD row is never carved, so skipping the gate
    causes the next loop’s → to hit a WALL.
- Prevent BFS shortcuts by forbidding adjacent gate direction flips (no DOWN then UP
  on the next column, etc.).
- Enforce “forced gate” property and local minimality via validation + retry attempts.

Public API:
  generate_level(seed=None, width=12, height=12, max_steps=500, target_route_tiles=22) -> LevelData
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Set, Deque
from collections import deque
import random


# ----------------------------
# Tile / program structures
# ----------------------------

class TileKind(str, Enum):
    WALL = "wall"
    FLOOR = "floor"
    LADDER = "ladder"   # visual only; not used for movement rules


class Color(str, Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


class Dir(str, Enum):
    RIGHT = "R"
    LEFT = "L"
    UP = "U"
    DOWN = "D"


@dataclass(frozen=True)
class Tile:
    kind: TileKind
    color: Optional[Color] = None
    goal: bool = False


@dataclass(frozen=True)
class Token:
    pass


@dataclass(frozen=True)
class LoopStart(Token):
    pass


@dataclass(frozen=True)
class LoopEnd(Token):
    pass


@dataclass(frozen=True)
class Move(Token):
    direction: Dir


@dataclass(frozen=True)
class CondMove(Token):
    color: Color
    direction: Dir


@dataclass(frozen=True)
class LevelData:
    width: int
    height: int
    grid: List[List[Tile]]           # grid[y][x]
    start: Tuple[int, int]
    goal: Tuple[int, int]
    program: List[Token]             # always 5 slots here
    max_steps: int = 500


@dataclass
class SimResult:
    solved: bool
    lost: bool
    steps: int      # instruction steps executed
    moves: int      # tile moves executed
    reason: str


# ----------------------------
# Helpers
# ----------------------------

def _dir_delta(d: Dir) -> Tuple[int, int]:
    return {
        Dir.RIGHT: (1, 0),
        Dir.LEFT: (-1, 0),
        Dir.UP: (0, -1),
        Dir.DOWN: (0, 1),
    }[d]


def _find_loop_indices(program: List[Token]) -> Tuple[int, int]:
    starts = [i for i, t in enumerate(program) if isinstance(t, LoopStart)]
    ends = [i for i, t in enumerate(program) if isinstance(t, LoopEnd)]
    if len(starts) != 1 or len(ends) != 1:
        raise ValueError("Program must contain exactly one LoopStart and one LoopEnd.")
    if starts[0] >= ends[0]:
        raise ValueError("LoopStart must appear before LoopEnd.")
    return starts[0], ends[0]


def in_bounds(level: LevelData, x: int, y: int) -> bool:
    return 0 <= x < level.width and 0 <= y < level.height


def get_tile(level: LevelData, x: int, y: int) -> Tile:
    return level.grid[y][x]


# ----------------------------
# Simulation (ladder ignored)
# ----------------------------

def simulate(level: LevelData,
             program: Optional[List[Token]] = None,
             max_steps: Optional[int] = None) -> SimResult:
    prog = program if program is not None else level.program
    limit = max_steps if max_steps is not None else level.max_steps

    try:
        loop_s, loop_e = _find_loop_indices(prog)
    except ValueError as e:
        return SimResult(False, True, 0, 0, f"invalid_program: {e}")

    if loop_e - loop_s <= 1:
        return SimResult(False, True, 0, 0, "invalid_program: empty_loop_body")

    x, y = level.start
    ip = 0
    visited: Set[Tuple[int, int, int]] = set()
    move_count = 0

    for step in range(limit):
        if (x, y) == level.goal:
            return SimResult(True, False, step, move_count, "reached_goal")

        state = (x, y, ip)
        if state in visited:
            return SimResult(False, False, step, move_count, "cycle_detected")
        visited.add(state)

        tok = prog[ip]

        if isinstance(tok, LoopStart):
            ip = (ip + 1) % len(prog)
            continue

        if isinstance(tok, LoopEnd):
            ip = loop_s
            continue

        move_dir: Optional[Dir] = None
        if isinstance(tok, Move):
            move_dir = tok.direction
        elif isinstance(tok, CondMove):
            cur = get_tile(level, x, y)
            move_dir = tok.direction if cur.color == tok.color else None
        else:
            return SimResult(False, True, step, move_count, "invalid_program: unknown_token")

        if move_dir is not None:
            dx, dy = _dir_delta(move_dir)
            nx, ny = x + dx, y + dy

            if not in_bounds(level, nx, ny):
                return SimResult(False, True, step, move_count, "lose: out_of_bounds")

            dest = get_tile(level, nx, ny)
            if dest.kind == TileKind.WALL:
                return SimResult(False, True, step, move_count, "lose: hit_wall")

            x, y = nx, ny
            move_count += 1

            if (x, y) == level.goal:
                return SimResult(True, False, step + 1, move_count, "reached_goal")

        ip = (ip + 1) % len(prog)

    return SimResult(False, False, limit, move_count, "step_limit")


# ----------------------------
# Program (fixed 5 slots for now)
# ----------------------------

def _fixed_easy_program() -> List[Token]:
    return [
        LoopStart(),
        Move(Dir.RIGHT),
        CondMove(Color.GREEN, Dir.DOWN),
        CondMove(Color.BLUE, Dir.UP),
        LoopEnd(),
    ]


def program_to_labels(program: List[Token]) -> List[str]:
    out: List[str] = []
    for t in program:
        if isinstance(t, LoopStart):
            out.append("f0 {")
        elif isinstance(t, LoopEnd):
            out.append("} f1")
        elif isinstance(t, Move):
            out.append("→")
        elif isinstance(t, CondMove):
            arrow = {"R": "→", "L": "←", "U": "↑", "D": "↓"}[t.direction.value]
            out.append(f"if_{t.color.value}+{arrow}")
        else:
            out.append("?")
    return out


# ----------------------------
# BFS shortest path distance (tile moves)
# ----------------------------

def bfs_shortest_distance(level: LevelData) -> Optional[int]:
    sx, sy = level.start
    gx, gy = level.goal

    q: Deque[Tuple[int, int, int]] = deque()
    q.append((sx, sy, 0))
    seen = {(sx, sy)}

    while q:
        x, y, d = q.popleft()
        if (x, y) == (gx, gy):
            return d

        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < level.width and 0 <= ny < level.height:
                if (nx, ny) in seen:
                    continue
                if level.grid[ny][nx].kind == TileKind.WALL:
                    continue
                seen.add((nx, ny))
                q.append((nx, ny, d + 1))

    return None


# ----------------------------
# Minimality checks
# ----------------------------

def local_minimality_check(level: LevelData) -> bool:
    """Removing any one movement/conditional action must break solvability."""
    if not simulate(level).solved:
        return False

    prog = level.program
    for i, t in enumerate(prog):
        if not isinstance(t, (Move, CondMove)):
            continue
        shorter = prog[:i] + prog[i + 1:]
        try:
            _find_loop_indices(shorter)
        except Exception:
            continue
        if simulate(level, shorter, level.max_steps).solved:
            return False
    return True


def shorter_program_rejection(level: LevelData) -> bool:
    """Reject if any 3-slot or 4-slot unconditional-move loop can solve."""
    for a in [Move(Dir.RIGHT), Move(Dir.LEFT), Move(Dir.UP), Move(Dir.DOWN)]:
        if simulate(level, [LoopStart(), a, LoopEnd()], level.max_steps).solved:
            return False

    basic = [Move(Dir.RIGHT), Move(Dir.LEFT), Move(Dir.UP), Move(Dir.DOWN)]
    for a in basic:
        for b in basic:
            if simulate(level, [LoopStart(), a, b, LoopEnd()], level.max_steps).solved:
                return False

    return True


# ----------------------------
# Forced-gate validation
# ----------------------------

def gates_are_forced(level: LevelData, gate_sources: List[Tuple[int, int, Color]]) -> bool:
    """
    A gate is 'forced' iff:
    - the gate source tile is colored correctly
    - the tile immediately to the RIGHT of the gate source (same row) is WALL (or OOB)
      so skipping the conditional causes next loop's → to crash
    - the carved vertical destination tile exists and is FLOOR

    Also enforces: at least one GREEN gate and one BLUE gate exist.
    """
    if not gate_sources:
        return False

    has_green = any(c == Color.GREEN for _, _, c in gate_sources)
    has_blue = any(c == Color.BLUE for _, _, c in gate_sources)
    if not (has_green and has_blue):
        return False

    for gx, gy, col in gate_sources:
        t = level.grid[gy][gx]
        if t.kind != TileKind.FLOOR or t.color != col:
            return False

        # Right neighbor must be WALL or out-of-bounds.
        if gx + 1 < level.width:
            if level.grid[gy][gx + 1].kind != TileKind.WALL:
                return False

        # Vertical destination must exist and be FLOOR.
        if col == Color.GREEN:
            ny = gy + 1
            if ny >= level.height or level.grid[ny][gx].kind != TileKind.FLOOR:
                return False
        elif col == Color.BLUE:
            ny = gy - 1
            if ny < 0 or level.grid[ny][gx].kind != TileKind.FLOOR:
                return False

    return True


# ----------------------------
# Generation (retry loop)
# ----------------------------

def generate_level(seed: Optional[int] = None,
                   width: int = 12,
                   height: int = 12,
                   max_steps: int = 500,
                   target_route_tiles: int = 22,
                   max_attempts: int = 400) -> LevelData:
    """
    Generate a valid Easy level.

    Guarantees (by validation):
    - solvable by the fixed 5-slot program
    - BFS shortest path distance == solver move count
    - gates are forced (so conditionals are needed)
    - local minimality holds
    - shorter unconditional 3/4-slot loops do not solve
    """
    rng = random.Random(seed) if seed is not None else random.Random()
    program = _fixed_easy_program()

    if width < 6 or height < 6:
        raise ValueError("Grid too small; use at least 6x6.")

    # With monotonic RIGHT and at most one vertical per advance:
    # corridor tiles = 1 + A + G, with G<=A and A<=width-1 => max tiles = 1 + 2*(width-1)
    max_route_tiles_by_width = 1 + 2 * (width - 1)
    desired = max(10, min(target_route_tiles, width * height, max_route_tiles_by_width))

    for _ in range(max_attempts):
        level, gate_sources = _build_path_only_snake(program, width, height, max_steps, desired, rng)

        sim = simulate(level)
        if not sim.solved:
            continue

        bfs_d = bfs_shortest_distance(level)
        if bfs_d is None or bfs_d != sim.moves:
            continue

        if not gates_are_forced(level, gate_sources):
            continue

        if not local_minimality_check(level):
            continue

        if not shorter_program_rejection(level):
            continue

        return level

    raise RuntimeError("Failed to generate a valid Easy level within attempts.")


# ----------------------------
# Corridor builder (varied + caps, no adjacent flip)
# ----------------------------

def _build_path_only_snake(program: List[Token],
                           width: int,
                           height: int,
                           max_steps: int,
                           desired_route_tiles: int,
                           rng: random.Random) -> Tuple[LevelData, List[Tuple[int, int, Color]]]:
    """
    Path-only corridor with vertical variation.

    Anti-boring caps:
    - MAX_GATE_STREAK caps consecutive gated columns (prevents long R,U,R,U,... runs)
    - MAX_STRAIGHT_STREAK caps consecutive non-gated columns (prevents long straight highways)

    Critical constraints:
    - Path-only grid => no alternate routes
    - No adjacent gate direction flip (prevents 2x2 cycles that create BFS shortcuts)
    - Gates are forced by construction (we never carve right tile on old row after a gate)
    """
    # ----------------------------
    # Control knobs (tune here)
    # ----------------------------
    MAX_GATE_STREAK = 4        # prevents very long R,U,R,U,... patterns
    MAX_STRAIGHT_STREAK = 7    # prevents 15-20 straight rights
    GATE_PROB = 0.60           # base chance to try a gate when allowed

    # Choose A (right advances).
    max_A = width - 1
    max_route_tiles_by_width = 1 + 2 * max_A
    desired = max(10, min(desired_route_tiles, width * height, max_route_tiles_by_width))

    min_A = max(3, (desired - 1 + 1) // 2)  # ceil((desired-1)/2)
    if min_A > max_A:
        min_A = max_A

    # Favor long corridors (more interesting) but still random
    A = max_A if rng.random() < 0.70 else rng.randrange(min_A, max_A + 1)

    # Choose start x so sx + A <= width-1
    sx = rng.randrange(0, (width - 1) - A + 1)

    # Wider vertical band for more height variety
    min_row = 1
    max_row = height - 2
    sy = rng.randrange(min_row, max_row + 1)

    # Decide how many gates G we want: tiles = 1 + A + G >= desired
    needed_G = max(0, desired - 1 - A)
    base_G = max(2, needed_G if needed_G > 0 else max(2, A // 2))
    G = min(A, base_G)

    # Build path-only grid
    grid = [[Tile(TileKind.WALL) for _ in range(width)] for _ in range(height)]
    carved: Set[Tuple[int, int]] = set()
    gate_sources: List[Tuple[int, int, Color]] = []

    def carve(x: int, y: int, color: Optional[Color] = None, goal: bool = False) -> None:
        grid[y][x] = Tile(TileKind.FLOOR, color=color, goal=goal)
        carved.add((x, y))

    x, y = sx, sy
    carve(x, y)

    # Ensure at least one GREEN and one BLUE gate attempt at two spaced steps
    forced_green_i = 2 if A >= 4 else 1
    forced_blue_i = (A - 2) if A >= 6 else (A - 1)
    forced_green_i = max(1, min(A, forced_green_i))
    forced_blue_i = max(1, min(A, forced_blue_i))
    if forced_blue_i == forced_green_i:
        forced_blue_i = min(A, forced_green_i + 2) if forced_green_i + 2 <= A else max(1, forced_green_i - 2)

    green_used = 0
    blue_used = 0
    gates_used = 0

    last_gate_dir: Optional[Color] = None   # GREEN=down, BLUE=up
    last_gate_step: Optional[int] = None

    gate_streak = 0
    straight_streak = 0

    def can_gate_down(cur_x: int, cur_y: int) -> bool:
        ny = cur_y + 1
        return ny <= max_row and (cur_x, ny) not in carved

    def can_gate_up(cur_x: int, cur_y: int) -> bool:
        ny = cur_y - 1
        return ny >= min_row and (cur_x, ny) not in carved

    def place_gate(color: Color, step_i: int) -> bool:
        """
        Place a gate on current (x,y) and carve the vertical destination.
        Record gate source as (x, old_y, color).
        """
        nonlocal y, green_used, blue_used, last_gate_dir, last_gate_step, gates_used

        old_y = y

        if color == Color.GREEN:
            if not can_gate_down(x, old_y):
                return False
            grid[old_y][x] = Tile(TileKind.FLOOR, color=Color.GREEN, goal=False)
            gate_sources.append((x, old_y, Color.GREEN))
            carve(x, old_y + 1)
            y = old_y + 1
            green_used += 1
        else:
            if not can_gate_up(x, old_y):
                return False
            grid[old_y][x] = Tile(TileKind.FLOOR, color=Color.BLUE, goal=False)
            gate_sources.append((x, old_y, Color.BLUE))
            carve(x, old_y - 1)
            y = old_y - 1
            blue_used += 1

        last_gate_dir = color
        last_gate_step = step_i
        gates_used += 1
        return True

    for i in range(1, A + 1):
        # Always carve RIGHT one tile (unconditional →)
        nx = x + 1
        if (nx, y) in carved:
            break
        carve(nx, y)
        x = nx

        # Decide if we *attempt* a gate here
        steps_left = A - i
        gates_left_needed = max(0, G - gates_used)

        must_force_green = (i == forced_green_i and green_used == 0)
        must_force_blue = (i == forced_blue_i and blue_used == 0)

        # Core desire for gates (count + randomness)
        must_gate_for_count = (gates_left_needed > 0 and steps_left < gates_left_needed)
        want_gate = (rng.random() < GATE_PROB)
        do_gate = must_force_green or must_force_blue or must_gate_for_count or (gates_left_needed > 0 and want_gate)

        # Anti-boring caps (do NOT override forced gates)
        if not (must_force_green or must_force_blue):
            if gate_streak >= MAX_GATE_STREAK:
                do_gate = False
            if straight_streak >= MAX_STRAIGHT_STREAK and gates_left_needed > 0:
                do_gate = True

        placed_gate = False

        if do_gate:
            # Forced gates: do NOT substitute the other color if it fails
            if must_force_green:
                placed_gate = place_gate(Color.GREEN, i)
            elif must_force_blue:
                placed_gate = place_gate(Color.BLUE, i)
            else:
                # Choose direction; sometimes continue same direction (stairs),
                # but bounded by MAX_GATE_STREAK so it won't go forever.
                if last_gate_dir is not None and rng.random() < 0.55:
                    desired_color = last_gate_dir
                else:
                    desired_color = Color.GREEN if rng.random() < 0.5 else Color.BLUE

                # Prevent adjacent gate direction flip (avoids 2x2 cycles)
                if last_gate_step == i - 1 and last_gate_dir is not None:
                    desired_color = last_gate_dir

                placed_gate = place_gate(desired_color, i)
                if not placed_gate:
                    other = Color.BLUE if desired_color == Color.GREEN else Color.GREEN

                    # If adjacent gating, switching would be a flip -> disallow
                    if not (last_gate_step == i - 1 and last_gate_dir is not None and other != last_gate_dir):
                        placed_gate = place_gate(other, i)

        # Update streak counters based on what *actually happened*
        if placed_gate:
            gate_streak += 1
            straight_streak = 0
        else:
            straight_streak += 1
            gate_streak = 0

    # Place goal at end of corridor
    grid[y][x] = Tile(TileKind.FLOOR, color=grid[y][x].color, goal=True)
    level = LevelData(width, height, grid, (sx, sy), (x, y), program, max_steps=max_steps)
    return level, gate_sources 