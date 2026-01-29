"""map_generation.py (ROBUST v4)

Program-first Easy generator (5 slots: loop + 3 actions).

This version removes the old snake-corridor strategy and replaces it with
motif-driven corridor compilation. The generator:
  1) Randomizes the 3 actions (distribution 60/40 as per spec)
  2) Picks ONE motif using weights, then generates only for that motif
  3) Carves a single 1-tile-wide corridor compatible with the program
  4) Validates: solvable, shortest-path, forced gates, local minimality,
     shorter-program rejection, and ≥2 distinct colored tiles.

No decoy tiles are generated yet (only required gate tiles; an extra "safe"
colored tile is added only when needed to meet the ≥2-color rule).

Public API (used by app.py):
  - generate_level(...)
  - program_to_labels(program)
  - TileKind, Color
"""

from __future__ import annotations

from enum import Enum
from collections import deque
from typing import List, Optional, Tuple, Set, Dict
import random


# ----------------------------
# Core structures
# ----------------------------


class TileKind(str, Enum):
    WALL = "wall"
    FLOOR = "floor"
    LADDER = "ladder"  # visual only


class Color(str, Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


class Dir(str, Enum):
    RIGHT = "R"
    LEFT = "L"
    UP = "U"
    DOWN = "D"


class Tile:
    __slots__ = ("kind", "color", "goal")

    def __init__(self, kind: TileKind, color: Optional[Color] = None, goal: bool = False):
        self.kind = kind
        self.color = color
        self.goal = goal


class Token:
    __slots__ = ()


class LoopStart(Token):
    __slots__ = ()


class LoopEnd(Token):
    __slots__ = ()


class Move(Token):
    __slots__ = ("direction",)

    def __init__(self, direction: Dir):
        self.direction = direction


class CondMove(Token):
    __slots__ = ("color", "direction")

    def __init__(self, color: Color, direction: Dir):
        self.color = color
        self.direction = direction


class LevelData:
    __slots__ = ("width", "height", "grid", "start", "goal", "program", "max_steps")

    def __init__(
        self,
        width: int,
        height: int,
        grid: List[List[Tile]],
        start: Tuple[int, int],
        goal: Tuple[int, int],
        program: List[Token],
        max_steps: int = 500,
    ):
        self.width = width
        self.height = height
        self.grid = grid
        self.start = start
        self.goal = goal
        self.program = program
        self.max_steps = max_steps


# ----------------------------
# Small helpers
# ----------------------------


_ARROW = {"R": "→", "L": "←", "U": "↑", "D": "↓"}


def program_to_labels(program: List[Token]) -> List[str]:
    out: List[str] = []
    for t in program:
        if isinstance(t, LoopStart):
            out.append("f0 {")
        elif isinstance(t, LoopEnd):
            out.append("} f1")
        elif isinstance(t, Move):
            out.append(_ARROW[t.direction.value])
        elif isinstance(t, CondMove):
            out.append(f"if_{t.color.value}+{_ARROW[t.direction.value]}")
        else:
            out.append("?")
    return out


def _delta(d: Dir) -> Tuple[int, int]:
    return {Dir.RIGHT: (1, 0), Dir.LEFT: (-1, 0), Dir.UP: (0, -1), Dir.DOWN: (0, 1)}[d]


def _opposite(d: Dir) -> Dir:
    return {Dir.RIGHT: Dir.LEFT, Dir.LEFT: Dir.RIGHT, Dir.UP: Dir.DOWN, Dir.DOWN: Dir.UP}[d]


def _is_vertical(d: Dir) -> bool:
    return d in (Dir.UP, Dir.DOWN)


def _find_loop(program: List[Token]) -> Tuple[int, int]:
    s = next((i for i, t in enumerate(program) if isinstance(t, LoopStart)), None)
    e = next((i for i, t in enumerate(program) if isinstance(t, LoopEnd)), None)
    if s is None or e is None or s >= e:
        raise ValueError("invalid loop")
    return s, e


# ----------------------------
# Simulation + BFS
# ----------------------------


def simulate(level: LevelData, program: Optional[List[Token]] = None, max_steps: Optional[int] = None) -> Tuple[bool, int]:
    """Return (solved, move_count)."""
    prog = program or level.program
    limit = max_steps or level.max_steps
    try:
        loop_s, _loop_e = _find_loop(prog)
    except Exception:
        return False, 0

    x, y = level.start
    ip = 0
    seen: Set[Tuple[int, int, int]] = set()
    moves = 0

    for _ in range(limit):
        if (x, y) == level.goal:
            return True, moves
        st = (x, y, ip)
        if st in seen:
            return False, moves
        seen.add(st)

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
            if level.grid[y][x].color == tok.color:
                move_dir = tok.direction

        if move_dir is not None:
            dx, dy = _delta(move_dir)
            nx, ny = x + dx, y + dy
            if not (0 <= nx < level.width and 0 <= ny < level.height):
                return False, moves
            if level.grid[ny][nx].kind == TileKind.WALL:
                return False, moves
            x, y = nx, ny
            moves += 1

        ip = (ip + 1) % len(prog)

    return False, moves


def bfs_shortest_distance(level: LevelData) -> Optional[int]:
    sx, sy = level.start
    gx, gy = level.goal
    q = deque([(sx, sy, 0)])
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


def local_minimality_ok(level: LevelData, step_cap: Optional[int] = None) -> bool:
    """Removing any single action must break solvability."""
    if not simulate(level, max_steps=step_cap)[0]:
        return False
    prog = level.program
    for i, t in enumerate(prog):
        if not isinstance(t, (Move, CondMove)):
            continue
        shorter = prog[:i] + prog[i + 1 :]
        try:
            _find_loop(shorter)
        except Exception:
            continue
        if simulate(level, shorter, max_steps=step_cap)[0]:
            return False
    return True


def shorter_program_rejection(level: LevelData, step_cap: Optional[int] = None) -> bool:
    """Reject if any 1- or 2-action loop program can solve."""
    basic = [Move(Dir.RIGHT), Move(Dir.LEFT), Move(Dir.UP), Move(Dir.DOWN)]
    for a in basic:
        if simulate(level, [LoopStart(), a, LoopEnd()], max_steps=step_cap)[0]:
            return False
    for a in basic:
        for b in basic:
            if simulate(level, [LoopStart(), a, b, LoopEnd()], max_steps=step_cap)[0]:
                return False
    return True


# ----------------------------
# Motifs
# ----------------------------


class Motif(str, Enum):
    MEANDER = "meander"                 # 35%
    BAND_TRANSFER = "three_band"        # 20%
    SWITCHBACKS = "switchbacks"         # 15%
    ZIGZAG_LADDER = "zigzag_ladder"     # 10%
    SYMMETRY = "symmetry"               # 10%


_MOTIF_CHOICES = [
    Motif.MEANDER,
    Motif.BAND_TRANSFER,
    Motif.SWITCHBACKS,
    Motif.ZIGZAG_LADDER,
    Motif.SYMMETRY,
]
_MOTIF_WEIGHTS = [35, 20, 15, 10, 10]  # Spur is optional later (not used)


# ----------------------------
# Program generation (3 actions)
# ----------------------------


def _action_dirs(actions: List[Token]) -> List[Dir]:
    out: List[Dir] = []
    for t in actions:
        if isinstance(t, (Move, CondMove)):
            out.append(t.direction)
    return out


def _reducible(actions: List[Token]) -> bool:
    if len(actions) != 3:
        return True
    dirs = _action_dirs(actions)
    if len(set(dirs)) == 1:
        return True
    # immediate unconditional cancel
    for i in range(2):
        a, b = actions[i], actions[i + 1]
        if isinstance(a, Move) and isinstance(b, Move) and b.direction == _opposite(a.direction):
            return True
    # detour collapse a,b,opposite(a) when b is perpendicular
    a, b, c = dirs
    if c == _opposite(a) and (_is_vertical(a) != _is_vertical(b)):
        return True
    # duplicate conditional colors
    conds = [t for t in actions if isinstance(t, CondMove)]
    if len(conds) >= 2 and len({t.color for t in conds}) != len(conds):
        return True
    return False


def _gate_crash_dir(actions: List[Token], cond_idx: int) -> Optional[Dir]:
    """First upcoming unconditional direction != conditional direction (cyclic)."""
    cond = actions[cond_idx]
    if not isinstance(cond, CondMove):
        return None
    n = len(actions)
    for k in range(1, n + 1):
        j = (cond_idx + k) % n
        t = actions[j]
        if isinstance(t, Move) and t.direction != cond.direction:
            return t.direction
    return None


def _gateable(actions: List[Token]) -> bool:
    for i, t in enumerate(actions):
        if isinstance(t, CondMove) and _gate_crash_dir(actions, i) is None:
            return False
    return True


def _motif_action_constraints(motif: Motif, actions: List[Token]) -> bool:
    dirs = _action_dirs(actions)
    unconds = [t for t in actions if isinstance(t, Move)]
    conds = [t for t in actions if isinstance(t, CondMove)]
    has_h = any(d in (Dir.LEFT, Dir.RIGHT) for d in dirs)
    has_v = any(d in (Dir.UP, Dir.DOWN) for d in dirs)

    if motif == Motif.MEANDER:
        return True

    if motif == Motif.BAND_TRANSFER:
        if not (has_h and has_v) or not conds:
            return False
        uncond_dirs = [u.direction for u in unconds]
        uncond_h = sum(d in (Dir.LEFT, Dir.RIGHT) for d in uncond_dirs)
        uncond_v = sum(d in (Dir.UP, Dir.DOWN) for d in uncond_dirs)
        minor_is_v = uncond_h > uncond_v
        if minor_is_v:
            return any(_is_vertical(c.direction) for c in conds)
        return any(not _is_vertical(c.direction) for c in conds)

    if motif == Motif.SWITCHBACKS:
        # Compatible "switchbacks" for a 3-action loop:
        # 1 unconditional advance; 2 conditionals are opposites on the perpendicular axis.
        if len(unconds) != 1 or len(conds) != 2:
            return False
        adv = unconds[0].direction
        c1, c2 = conds[0].direction, conds[1].direction
        if _is_vertical(c1) == _is_vertical(adv):
            return False
        if _is_vertical(c2) == _is_vertical(adv):
            return False
        return c1 == _opposite(c2)

    if motif == Motif.ZIGZAG_LADDER:
        return has_h and has_v and len(conds) >= 1

    if motif == Motif.SYMMETRY:
        return len(conds) == 2

    return True


def _biased_dir(rng: random.Random, preferred: Dir) -> Dir:
    dirs = list(Dir)
    w = []
    for d in dirs:
        if d == preferred:
            w.append(0.55)
        elif d == _opposite(preferred):
            w.append(0.05)
        else:
            w.append(0.20)
    return rng.choices(dirs, weights=w, k=1)[0]


def _random_program(rng: random.Random, motif: Motif) -> List[Token]:
    """Generate 5-slot program with 3 randomized actions."""
    if motif == Motif.BAND_TRANSFER:
        for _ in range(1500):
            major = rng.choice(list(Dir))
            minor = rng.choice([d for d in Dir if _is_vertical(d) != _is_vertical(major)])
            if rng.random() < 0.60:
                c1, c2 = rng.sample(list(Color), 2)
                actions = [Move(_biased_dir(rng, major)), CondMove(c1, minor), CondMove(c2, minor)]
            else:
                c = rng.choice(list(Color))
                actions = [Move(_biased_dir(rng, major)), Move(_biased_dir(rng, major)), CondMove(c, minor)]
            rng.shuffle(actions)
            if _reducible(actions) or not _gateable(actions):
                continue
            return [LoopStart(), actions[0], actions[1], actions[2], LoopEnd()]

    if motif == Motif.SWITCHBACKS:
        # Advance along one axis; the two conditionals are opposite on the perpendicular axis.
        for _ in range(2000):
            adv = rng.choice(list(Dir))
            if _is_vertical(adv):
                s1, s2 = Dir.LEFT, Dir.RIGHT
            else:
                s1, s2 = Dir.UP, Dir.DOWN
            c1, c2 = rng.sample(list(Color), 2)
            actions = [Move(adv), CondMove(c1, s1), CondMove(c2, s2)]
            rng.shuffle(actions)
            if _reducible(actions) or not _gateable(actions):
                continue
            return [LoopStart(), actions[0], actions[1], actions[2], LoopEnd()]

    if motif == Motif.ZIGZAG_LADDER:
        for _ in range(1500):
            major = rng.choice(list(Dir))
            perp = rng.choice([d for d in Dir if _is_vertical(d) != _is_vertical(major)])
            if rng.random() < 0.60:
                c1, c2 = rng.sample(list(Color), 2)
                actions = [Move(major), CondMove(c1, perp), CondMove(c2, perp)]
            else:
                c = rng.choice(list(Color))
                actions = [Move(major), Move(perp), CondMove(c, perp)]
            rng.shuffle(actions)
            if _reducible(actions) or not _gateable(actions):
                continue
            return [LoopStart(), actions[0], actions[1], actions[2], LoopEnd()]

    if motif == Motif.SYMMETRY:
        for _ in range(1500):
            major = rng.choice(list(Dir))
            axis = "h" if major in (Dir.LEFT, Dir.RIGHT) else "v"
            d1, d2 = (Dir.LEFT, Dir.RIGHT) if axis == "h" else (Dir.UP, Dir.DOWN)
            c1, c2 = rng.sample(list(Color), 2)
            actions = [Move(major), CondMove(c1, d1), CondMove(c2, d2)]
            rng.shuffle(actions)
            if _reducible(actions) or not _gateable(actions):
                continue
            return [LoopStart(), actions[0], actions[1], actions[2], LoopEnd()]

    for _ in range(1200):
        num_uncond = 1 if rng.random() < 0.60 else 2
        num_cond = 3 - num_uncond

        drift = rng.choice(list(Dir))
        unconds = [Move(_biased_dir(rng, drift)) for _ in range(num_uncond)]

        if num_cond == 2:
            c1, c2 = rng.sample(list(Color), 2)
            conds: List[Token] = [CondMove(c1, _biased_dir(rng, drift)), CondMove(c2, _biased_dir(rng, drift))]
        else:
            conds = [CondMove(rng.choice(list(Color)), _biased_dir(rng, drift))]

        actions = unconds + conds
        rng.shuffle(actions)

        if _reducible(actions) or not _gateable(actions) or not _motif_action_constraints(motif, actions):
            continue
        return [LoopStart(), actions[0], actions[1], actions[2], LoopEnd()]

    return [LoopStart(), Move(Dir.RIGHT), CondMove(Color.GREEN, Dir.DOWN), CondMove(Color.BLUE, Dir.UP), LoopEnd()]


# ----------------------------
# Forced gates validation
# ----------------------------


def _forced_gates_ok(level: LevelData, actions: List[Token], gates: List[Tuple[int, int, int, Dir]]) -> bool:
    # gates: (x,y,cond_idx,crash_dir)
    cond_idxs = [i for i, t in enumerate(actions) if isinstance(t, CondMove)]
    if cond_idxs and not gates:
        return False
    seen: Set[int] = set()

    for x, y, ci, crash_dir in gates:
        if not (0 <= x < level.width and 0 <= y < level.height):
            return False
        cond = actions[ci]
        if not isinstance(cond, CondMove):
            return False
        src = level.grid[y][x]
        if src.kind != TileKind.FLOOR or src.color != cond.color:
            return False

        # conditional destination must be floor
        dx, dy = _delta(cond.direction)
        nx, ny = x + dx, y + dy
        if not (0 <= nx < level.width and 0 <= ny < level.height):
            return False
        if level.grid[ny][nx].kind != TileKind.FLOOR:
            return False

        # crash target must be wall or OOB
        cdx, cdy = _delta(crash_dir)
        cx, cy = x + cdx, y + cdy
        if 0 <= cx < level.width and 0 <= cy < level.height:
            if level.grid[cy][cx].kind != TileKind.WALL:
                return False

        seen.add(ci)

    return all(ci in seen for ci in cond_idxs)


def _ensure_two_colors(level: LevelData, program: List[Token], rng: random.Random) -> None:
    used: Set[Color] = set()
    for row in level.grid:
        for t in row:
            if t.kind == TileKind.FLOOR and t.color is not None:
                used.add(t.color)
    if len(used) >= 2:
        return

    cond_colors = {t.color for t in program if isinstance(t, CondMove)}
    safe = [c for c in Color if c not in cond_colors and c not in used]
    if not safe:
        safe = [c for c in Color if c not in used]
    if not safe:
        return
    add_color = rng.choice(safe)

    candidates: List[Tuple[int, int]] = []
    for y in range(level.height):
        for x in range(level.width):
            t = level.grid[y][x]
            if t.kind == TileKind.FLOOR and not t.goal and t.color is None:
                candidates.append((x, y))
    if candidates:
        x, y = rng.choice(candidates)
        level.grid[y][x].color = add_color


# ----------------------------
# Corridor compilation (motif-driven)
# ----------------------------


def _build_corridor(
    program: List[Token],
    motif: Motif,
    width: int,
    height: int,
    max_steps: int,
    desired_tiles: int,
    rng: random.Random,
) -> Optional[Tuple[LevelData, List[Tuple[int, int, int, Dir]]]]:
    actions = program[1:4]
    cond_idxs = [i for i, t in enumerate(actions) if isinstance(t, CondMove)]
    cond_hits: Dict[int, int] = {i: 0 for i in cond_idxs}

    desired_moves = max(9, min(desired_tiles - 1, width * height - 1))

    # global caps (tuned for 12x12-ish)
    MAX_STRAIGHT = 8
    MAX_ALT = 4
    MAX_VERT = 2

    # grid init
    grid = [[Tile(TileKind.WALL) for _ in range(width)] for _ in range(height)]
    carved: Set[Tuple[int, int]] = set()
    forbidden: Set[Tuple[int, int]] = set()  # cells that must remain WALL (crash targets)
    gates: List[Tuple[int, int, int, Dir]] = []

    def carve(x: int, y: int) -> None:
        grid[y][x].kind = TileKind.FLOOR
        carved.add((x, y))

    def can_step(x: int, y: int, d: Dir) -> bool:
        dx, dy = _delta(d)
        nx, ny = x + dx, y + dy
        if not (0 <= nx < width and 0 <= ny < height):
            return False
        if (nx, ny) in carved or (nx, ny) in forbidden:
            return False
        # no touching older corridor tiles (prevents shortcuts)
        for ax, ay in ((nx + 1, ny), (nx - 1, ny), (nx, ny + 1), (nx, ny - 1)):
            if (ax, ay) in carved and (ax, ay) != (x, y):
                return False
        return True

    def do_step(x: int, y: int, d: Dir) -> Optional[Tuple[int, int]]:
        if not can_step(x, y, d):
            return None
        dx, dy = _delta(d)
        nx, ny = x + dx, y + dy
        carve(nx, ny)
        return nx, ny

    # motif setup
    dirs = _action_dirs(actions)
    heading = rng.choice(dirs) if dirs else Dir.RIGHT

    # band transfer setup
    band_axis = "y" if heading in (Dir.LEFT, Dir.RIGHT) else "x"
    transfer_dir: Optional[Dir] = None
    if motif == Motif.BAND_TRANSFER:
        candidates = [d for d in dirs if (_is_vertical(d) if band_axis == "y" else (not _is_vertical(d)))]
        if candidates:
            preferred = Dir.DOWN if band_axis == "y" else Dir.RIGHT
            transfer_dir = preferred if preferred in candidates else candidates[0]

    def band_ranges(axis: str, direction: Dir) -> List[Tuple[int, int]]:
        if axis == "y":
            lo, hi = 1, height - 2
        else:
            lo, hi = 1, width - 2
        span = hi - lo + 1
        a = lo
        b = lo + max(1, span // 3) - 1
        c = b + 1
        d = c + max(1, span // 3) - 1
        e = d + 1
        f = hi
        bands = [(a, b), (c, d), (e, f)]
        return bands if direction in (Dir.DOWN, Dir.RIGHT) else list(reversed(bands))

    bands = band_ranges(band_axis, transfer_dir or Dir.DOWN)
    phase_targets = [bands[0], bands[1], bands[2]]

    # start coordinate
    if motif == Motif.BAND_TRANSFER:
        if band_axis == "y":
            sy = rng.randrange(phase_targets[0][0], phase_targets[0][1] + 1)
            sx = rng.randrange(1, width - 1)
        else:
            sx = rng.randrange(phase_targets[0][0], phase_targets[0][1] + 1)
            sy = rng.randrange(1, height - 1)
    else:
        sx = rng.randrange(1, width - 1)
        sy = rng.randrange(1, height - 1)

    carve(sx, sy)
    x, y = sx, sy

    # shape counters
    last_dir: Optional[Dir] = None
    prev_dir: Optional[Dir] = None
    straight = 0
    alt = 0
    vert_dir: Optional[Dir] = None
    vert_run = 0

    def violates_caps(d: Dir) -> bool:
        nonlocal last_dir, prev_dir, straight, alt, vert_dir, vert_run
        if last_dir == d and straight >= MAX_STRAIGHT:
            return True
        if prev_dir is not None and last_dir is not None:
            if d == prev_dir and d != last_dir and alt >= MAX_ALT:
                return True
        if _is_vertical(d):
            if vert_dir == d and vert_run >= MAX_VERT:
                return True
        return False

    def update_caps(d: Dir) -> None:
        nonlocal last_dir, prev_dir, straight, alt, vert_dir, vert_run
        if last_dir == d:
            straight += 1
        else:
            straight = 1
        if prev_dir is not None and last_dir is not None and d == prev_dir and d != last_dir:
            alt += 1
        else:
            alt = 0
        if _is_vertical(d):
            if vert_dir == d:
                vert_run += 1
            else:
                vert_dir = d
                vert_run = 1
        else:
            vert_dir = None
            vert_run = 0
        prev_dir, last_dir = last_dir, d

    # motif state
    half = desired_moves // 2
    symm_record: List[Tuple[int, bool]] = []
    symm_replay_i = 0

    cond_streak_ci: Optional[int] = None
    cond_streak = 0

    # switchbacks state: burst side-steps on one conditional, then swap to the other
    sb_conds = [i for i, t in enumerate(actions) if isinstance(t, CondMove)]
    sb_active_ci: Optional[int] = None
    sb_burst_left = 0
    sb_cooldown_left = 0
    sb_switches = 0
    if motif == Motif.SWITCHBACKS and len(sb_conds) >= 1:
        sb_active_ci = rng.choice(sb_conds)
        sb_burst_left = rng.randint(2, 4)
        sb_cooldown_left = 0
        sb_switches = 0

    def in_band(val: int, band: Tuple[int, int]) -> bool:
        return band[0] <= val <= band[1]

    def want_transfer(move_i: int) -> bool:
        if motif != Motif.BAND_TRANSFER or transfer_dir is None:
            return False
        return move_i in (desired_moves // 3, (2 * desired_moves) // 3)

    # main compilation loop
    ap = 0
    moves = 0
    max_moves = min(width * height - 1, desired_moves + 12)

    for _ in range(max_steps):
        if moves >= desired_moves and all(v > 0 for v in cond_hits.values()):
            grid[y][x].goal = True
            level = LevelData(width, height, grid, (sx, sy), (x, y), program, max_steps=max_steps)
            return level, gates

        tok = actions[ap]

        if isinstance(tok, Move):
            d = tok.direction
            if violates_caps(d):
                return None
            nxt = do_step(x, y, d)
            if nxt is None:
                return None
            x, y = nxt
            moves += 1
            update_caps(d)

        elif isinstance(tok, CondMove):
            ci = ap
            need = (ci in cond_hits and cond_hits[ci] == 0)
            late_force = (ci in cond_hits and cond_hits[ci] == 0 and moves >= int(desired_moves * 0.60))

            # base take probability by motif
            p = 0.18
            if motif == Motif.MEANDER:
                offers_turn = (last_dir is None) or (tok.direction != last_dir)
                p = 0.25 if offers_turn else 0.08

            elif motif == Motif.BAND_TRANSFER:
                offers_turn = (last_dir is None) or (tok.direction != last_dir)
                p = 0.20 if offers_turn else 0.06
                if transfer_dir is not None and tok.direction == transfer_dir and want_transfer(moves):
                    p = 0.95

            elif motif == Motif.SWITCHBACKS:
                # bursts of side steps, then a forward-only cooldown; swap side 1–2 times.
                if sb_cooldown_left > 0:
                    sb_cooldown_left -= 1
                    p = 0.05
                else:
                    p = 0.85 if (sb_active_ci is not None and ci == sb_active_ci) else 0.05

            elif motif == Motif.ZIGZAG_LADDER:
                p = 0.70
                if cond_streak_ci == ci and cond_streak >= 2:
                    p = 0.05

            elif motif == Motif.SYMMETRY:
                p = 0.30 if moves < half else 0.0

            take = False
            if motif == Motif.SYMMETRY and moves >= half and symm_record:
                idx = len(symm_record) - 1 - symm_replay_i
                if idx >= 0:
                    src_ci, src_take = symm_record[idx]
                    other = None
                    conds_here = [j for j, t in enumerate(actions) if isinstance(t, CondMove)]
                    if len(conds_here) == 2:
                        a, b = conds_here
                        ta, tb = actions[a], actions[b]
                        if isinstance(ta, CondMove) and isinstance(tb, CondMove):
                            if ta.direction == _opposite(tb.direction):
                                other = b if src_ci == a else (a if src_ci == b else None)
                    want_ci = other if other is not None else src_ci
                    want = bool(src_take and want_ci == ci)
                    p = 0.85 if want else 0.10
                take = (rng.random() < p)
            else:
                take = (rng.random() < p)

            take = need or late_force or take

            if motif == Motif.SYMMETRY and moves < half:
                symm_record.append((ci, bool(take)))

            if not take:
                if grid[y][x].color == tok.color:
                    return None
            else:
                crash_dir = _gate_crash_dir(actions, ci)
                if crash_dir is None:
                    return None
                cdx, cdy = _delta(crash_dir)
                crash_cell = (x + cdx, y + cdy)
                if 0 <= crash_cell[0] < width and 0 <= crash_cell[1] < height:
                    if crash_cell in carved:
                        return None
                    forbidden.add(crash_cell)

                if grid[y][x].color is not None and grid[y][x].color != tok.color:
                    return None
                grid[y][x].color = tok.color

                d = tok.direction
                if violates_caps(d):
                    return None
                nxt = do_step(x, y, d)
                if nxt is None:
                    return None
                gates.append((x, y, ci, crash_dir))
                x, y = nxt
                moves += 1
                update_caps(d)

                if ci in cond_hits:
                    cond_hits[ci] += 1

                if motif == Motif.ZIGZAG_LADDER:
                    if cond_streak_ci == ci:
                        cond_streak += 1
                    else:
                        cond_streak_ci, cond_streak = ci, 1

                if motif == Motif.SWITCHBACKS and sb_active_ci is not None and ci == sb_active_ci:
                    sb_burst_left -= 1
                    if sb_burst_left <= 0:
                        sb_cooldown_left = rng.randint(3, 7)
                        sb_burst_left = rng.randint(2, 4)
                        if len(sb_conds) == 2 and sb_switches < 2:
                            sb_active_ci = sb_conds[0] if sb_active_ci == sb_conds[1] else sb_conds[1]
                            sb_switches += 1

            if motif == Motif.SYMMETRY and moves >= half and symm_record:
                symm_replay_i += 1

            if motif == Motif.BAND_TRANSFER and transfer_dir is not None:
                band_i = 0 if moves < desired_moves // 3 else (1 if moves < (2 * desired_moves) // 3 else 2)
                band = phase_targets[band_i]
                axis_val = y if band_axis == "y" else x
                if not in_band(axis_val, band):
                    if axis_val < band[0] - 1 or axis_val > band[1] + 1:
                        return None

        else:
            return None

        ap = (ap + 1) % 3
        if moves > max_moves:
            return None

    return None


# ----------------------------
# Generation
# ----------------------------


def generate_level(
    seed: Optional[int] = None,
    width: int = 12,
    height: int = 12,
    max_steps: int = 500,
    target_route_tiles: int = 22,
    max_attempts: int = 2200,
) -> LevelData:
    if width < 6 or height < 6:
        raise ValueError("Grid too small; use at least 6x6.")

    base_seed = seed if seed is not None else random.randrange(1 << 30)
    rng0 = random.Random(base_seed)

    # Pick ONE motif per level, by weight.
    motif = rng0.choices(_MOTIF_CHOICES, weights=_MOTIF_WEIGHTS, k=1)[0]
    
    print(motif)

    # Relax route length gradually if needed (within the same motif).
    desired0 = max(10, min(target_route_tiles, width * height))
    desired_list = [d for d in range(desired0, 11, -2)]  # e.g., 22,20,18,16,14,12

    # A few deterministic salts avoid unlucky RNG sequences without exploding runtime.
    salts = [0, 0x9E3779B9, 0xBB67AE85, 0x243F6A88, 0xB7E15162]

    remaining = max_attempts
    blocks = [(salt, desired) for salt in salts for desired in desired_list]

    for bi, (salt, desired) in enumerate(blocks):
        if remaining <= 0:
            break
        rng = random.Random((base_seed ^ salt) & 0xFFFFFFFF)
        blocks_left = max(1, len(blocks) - bi)
        per_block = max(1, remaining // blocks_left)

        for _ in range(per_block):
            if remaining <= 0:
                break
            remaining -= 1

            program = _random_program(rng, motif)
            built = _build_corridor(program, motif, width, height, max_steps, desired, rng)
            if built is None:
                continue
            level, gates = built
            actions = program[1:4]

            _ensure_two_colors(level, program, rng)

            solved, prog_moves = simulate(level)
            if not solved:
                continue
            bfs_d = bfs_shortest_distance(level)
            if bfs_d is None or bfs_d != prog_moves:
                continue

            step_cap = max(200, bfs_d * 12)
            if not _forced_gates_ok(level, actions, gates):
                continue
            if not local_minimality_ok(level, step_cap=step_cap):
                continue
            if not shorter_program_rejection(level, step_cap=step_cap):
                continue

            return level

    raise RuntimeError(
        f"Failed to generate a valid Easy level. seed={seed}, base_seed={base_seed}, motif={motif}, "
        f"size={width}x{height}, target_route_tiles={target_route_tiles}"
    )
