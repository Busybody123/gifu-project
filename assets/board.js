function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function clamp(n, lo, hi) {
  return Math.max(lo, Math.min(hi, n));
}

function parseGrid(gridLines) {
  const h = gridLines.length;
  const w = gridLines[0].length;

  let start = null;
  let goal = null;

  const walls = new Set();
  const ladders = new Set();
  const colors = new Map(); // key "x,y" -> "red|green|blue"

  for (let y = 0; y < h; y++) {
    const row = gridLines[y];
    for (let x = 0; x < w; x++) {
      const ch = row[x];
      const k = `${x},${y}`;

      if (ch === "#") walls.add(k);
      else if (ch === "L") ladders.add(k);
      else if (ch === "S") start = { x, y };
      else if (ch === "G") goal = { x, y };
      else if (ch === "r") colors.set(k, "red");
      else if (ch === "g") colors.set(k, "green");
      else if (ch === "b") colors.set(k, "blue");
      // "." is just floor
    }
  }

  return { w, h, start, goal, walls, ladders, colors };
}

function dirDelta(d) {
  switch (d) {
    case "R": return { dx: 1, dy: 0 };
    case "L": return { dx: -1, dy: 0 };
    case "U": return { dx: 0, dy: -1 };
    case "D": return { dx: 0, dy: 1 };
    default: return { dx: 0, dy: 0 };
  }
}

function prettyMain(base) {
  const arrows = { up: "↑", down: "↓", left: "←", right: "→" };
  if (base === "f0{") return "f0 {";
  if (base === "}f0") return "} f0";
  if (arrows[base]) return arrows[base];
  return base || "";
}

function baseToDir(base) {
  switch (base) {
    case "up": return "U";
    case "down": return "D";
    case "left": return "L";
    case "right": return "R";
    default: return null;
  }
}

/* NEW: map base -> python function-ish token (no parentheses to match your example) */
function baseToPyMove(base) {
  switch (base) {
    case "up": return "move_up";
    case "down": return "move_down";
    case "left": return "move_left";
    case "right": return "move_right";
    default: return null;
  }
}

(function main() {
  const cfg = window.BOARD_CONFIG;
  const gridLines = cfg.grid;

  const board = document.getElementById("board");
  const palette = document.getElementById("palette");
  const slotsEl = document.getElementById("slots");
  const statusEl = document.getElementById("status");

  const codeView = document.getElementById("codeView"); // NEW

  const runBtn = document.getElementById("runBtn");
  const restartBtn = document.getElementById("restartBtn");
  const resetSlotsBtn = document.getElementById("resetSlotsBtn");

  document.documentElement.style.setProperty("--cell", `${cfg.cellPx}px`);

  const parsed = parseGrid(gridLines);
  const { w, h, start, goal, walls, ladders, colors } = parsed;

  board.style.gridTemplateColumns = `repeat(${w}, var(--cell))`;
  board.style.gridTemplateRows = `repeat(${h}, var(--cell))`;

  // ----- Render grid -----
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const cell = document.createElement("div");
      cell.className = "cell";
      const k = `${x},${y}`;

      if (walls.has(k)) cell.classList.add("wall");
      if (ladders.has(k)) cell.classList.add("ladder");

      const c = colors.get(k);
      if (c === "red") cell.classList.add("red");
      if (c === "green") cell.classList.add("green");
      if (c === "blue") cell.classList.add("blue");

      if (goal && x === goal.x && y === goal.y) {
        cell.classList.add("goal");
        const icon = document.createElement("div");
        icon.className = "goalIcon";
        icon.textContent = "🍎";
        cell.appendChild(icon);
      }

      board.appendChild(cell);
    }
  }

  // ----- Avatar -----
  const avatar = document.createElement("img");
  avatar.id = "avatar";

  const hasIdle = !!(window.SPRITES && window.SPRITES.idle && window.SPRITES.idle.length > 0);
  const hasWalk = !!(window.SPRITES && window.SPRITES.walk && window.SPRITES.walk.length > 0);

  function spriteIdle() {
    if (hasIdle) avatar.src = `data:image/gif;base64,${window.SPRITES.idle}`;
    else avatar.removeAttribute("src");
  }
  function spriteWalk() {
    if (hasWalk) avatar.src = `data:image/gif;base64,${window.SPRITES.walk}`;
    else avatar.removeAttribute("src");
  }

  board.appendChild(avatar);

  function toPixels(x, y) {
    const gap = 4;
    const pad = cfg.cellPx * 0.09;
    return {
      px: x * (cfg.cellPx + gap) + pad,
      py: y * (cfg.cellPx + gap) + pad
    };
  }

  function setAvatarPos(x, y, instant = false) {
    const { px, py } = toPixels(x, y);
    if (instant) {
      const prev = avatar.style.transition;
      avatar.style.transition = "none";
      avatar.style.transform = `translate(${px}px, ${py}px)`;
      avatar.offsetHeight;
      avatar.style.transition = prev;
    } else {
      avatar.style.transform = `translate(${px}px, ${py}px)`;
    }
  }

  let player = { x: start.x, y: start.y };
  setAvatarPos(player.x, player.y, true);
  spriteIdle();

  // ----- Palette blocks -----
  const paletteItems = [
    { kind: "base", value: "up", label: "↑" },
    { kind: "base", value: "down", label: "↓" },
    { kind: "base", value: "left", label: "←" },
    { kind: "base", value: "right", label: "→" },
    { kind: "base", value: "f0{", label: "f0 {" },
    { kind: "base", value: "}f0", label: "} f0" },

    { kind: "cond", value: "red", label: "if_red" },
    { kind: "cond", value: "green", label: "if_green" },
    { kind: "cond", value: "blue", label: "if_blue" },
  ];

  for (const item of paletteItems) {
    const el = document.createElement("div");
    el.className = "block";
    if (item.kind === "cond") el.classList.add("cond", item.value);
    el.textContent = item.label;
    el.draggable = true;
    el.dataset.kind = item.kind;
    el.dataset.value = item.value;

    el.addEventListener("dragstart", (e) => {
      e.dataTransfer.setData("text/plain", JSON.stringify({ kind: item.kind, value: item.value }));
    });

    palette.appendChild(el);
  }

  // ----- Slots -----
  const slots = [];
  const slotCount = clamp(cfg.slotCount || 5, 1, 30);

  /* NEW: generate python code from current slots */
  function findLoopBoundsFromSlots() {
    let ls = -1;
    let le = -1;

    for (let i = 0; i < slots.length; i++) {
      if (slots[i].base === "f0{") { ls = i; break; }
    }
    if (ls >= 0) {
      for (let i = ls + 1; i < slots.length; i++) {
        if (slots[i].base === "}f0") { le = i; break; }
      }
    }
    return { ls, le, hasLoop: ls >= 0 && le > ls };
  }

  function slotToPyLines(slot, indent) {
    if (!slot.base) return [];

    // loop tokens handled at higher level
    if (slot.base === "f0{" || slot.base === "}f0") return [];

    const move = baseToPyMove(slot.base);
    if (!move) return [];

    if (slot.cond) {
      return [
        `${indent}if is_${slot.cond}:`,
        `${indent}  ${move}`
      ];
    }
    return [`${indent}${move}`];
  }

  function updateCodeView() {
    const indent = "  "; // matches your sample
    const lines = [];

    // if nothing placed
    const any = slots.some(s => !!s.base);
    if (!any) {
      codeView.textContent = "# Drag blocks into slots to generate code\n";
      return;
    }

    const { ls, le, hasLoop } = findLoopBoundsFromSlots();

    if (hasLoop) {
      // pre-loop statements (run once)
      for (let i = 0; i < ls; i++) {
        lines.push(...slotToPyLines(slots[i], ""));
      }

      // loop body
      lines.push("while True:");
      let bodyHasAnything = false;
      for (let i = ls + 1; i < le; i++) {
        const s = slots[i];
        const py = slotToPyLines(s, indent);
        if (py.length > 0) bodyHasAnything = true;
        lines.push(...py);
      }
      if (!bodyHasAnything) {
        lines.push(`${indent}pass`);
      }

      // post-loop (technically unreachable)
      const postLines = [];
      for (let i = le + 1; i < slots.length; i++) {
        postLines.push(...slotToPyLines(slots[i], ""));
      }
      if (postLines.length > 0) {
        lines.push("");
        lines.push("# Note: statements after an infinite loop do not run");
        lines.push(...postLines);
      }
    } else {
      // no loop tokens -> straight-line code
      for (let i = 0; i < slots.length; i++) {
        lines.push(...slotToPyLines(slots[i], ""));
      }
    }

    codeView.textContent = lines.join("\n") + "\n";
  }

  function renderSlot(idx) {
    const slot = slots[idx];
    const el = slot.el;

    el.classList.toggle("filled", !!slot.base);

    el.classList.remove("tint-red", "tint-green", "tint-blue");
    if (slot.cond === "red") el.classList.add("tint-red");
    if (slot.cond === "green") el.classList.add("tint-green");
    if (slot.cond === "blue") el.classList.add("tint-blue");

    el.innerHTML = "";

    const main = document.createElement("div");
    main.className = "main";
    main.textContent = slot.base ? prettyMain(slot.base) : `slot ${idx + 1}`;
    el.appendChild(main);

    if (slot.base && slot.cond && baseToDir(slot.base)) {
      const tag = document.createElement("div");
      tag.className = "tag";
      tag.textContent = `if_${slot.cond}`;
      el.appendChild(tag);
    }

    // NEW: refresh code whenever a slot re-renders
    updateCodeView();
  }

  function clearSlot(idx) {
    slots[idx].base = null;
    slots[idx].cond = null;
    renderSlot(idx);
  }

  for (let i = 0; i < slotCount; i++) {
    const el = document.createElement("div");
    el.className = "slot";
    el.dataset.idx = String(i);

    el.addEventListener("dragover", (e) => e.preventDefault());
    el.addEventListener("drop", (e) => {
      e.preventDefault();
      let payload = null;
      try {
        payload = JSON.parse(e.dataTransfer.getData("text/plain"));
      } catch {
        return;
      }
      if (!payload || !payload.kind) return;

      const idx = Number(el.dataset.idx);
      const slot = slots[idx];

      if (payload.kind === "base") {
        slot.base = payload.value;
        // If base isn't a move, drop any condition (since it wouldn't apply)
        if (!baseToDir(slot.base)) slot.cond = null;
        renderSlot(idx);
        return;
      }

      if (payload.kind === "cond") {
        // Only apply condition if slot already has a move
        if (slot.base && baseToDir(slot.base)) {
          slot.cond = payload.value;
          renderSlot(idx);
        }
        return;
      }
    });

    el.addEventListener("dblclick", () => clearSlot(i));

    slots.push({ el, base: null, cond: null });
    slotsEl.appendChild(el);
    renderSlot(i);
  }

  // ----- Program compile & execution -----
  function compileProgram() {
    // {t:"loopStart"} {t:"loopEnd"} {t:"move", dir:"U|D|L|R", cond:null|"red|green|blue"}
    const prog = [];
    for (const s of slots) {
      if (!s.base) continue;

      if (s.base === "f0{") {
        prog.push({ t: "loopStart" });
        continue;
      }
      if (s.base === "}f0") {
        prog.push({ t: "loopEnd" });
        continue;
      }

      const dir = baseToDir(s.base);
      if (!dir) continue;

      prog.push({ t: "move", dir, cond: s.cond || null });
    }
    return prog;
  }

  function findLoopBounds(prog) {
    let ls = -1;
    let le = -1;
    for (let i = 0; i < prog.length; i++) {
      if (prog[i].t === "loopStart") { ls = i; break; }
    }
    if (ls >= 0) {
      for (let i = ls + 1; i < prog.length; i++) {
        if (prog[i].t === "loopEnd") { le = i; break; }
      }
    }
    return { ls, le, hasLoop: ls >= 0 && le > ls };
  }

  function tileColorAt(x, y) {
    const k = `${x},${y}`;
    return colors.get(k) || null;
  }

  function isWall(x, y) {
    if (x < 0 || x >= w || y < 0 || y >= h) return true;
    return walls.has(`${x},${y}`);
  }

  let running = false;
  let cancelRun = false;

  async function runProgram() {
    const prog = compileProgram();
    if (prog.length === 0) {
      statusEl.textContent = "Add some blocks first.";
      return;
    }

    const { ls, hasLoop } = findLoopBounds(prog);

    cancelRun = false;
    running = true;
    runBtn.disabled = true;

    avatar.style.transition = `transform ${cfg.speedMs}ms linear`;
    statusEl.textContent = "Running...";

    let ip = 0;
    let steps = 0;

    while (steps < (cfg.maxSteps || 500)) {
      if (cancelRun) break;

      if (goal && player.x === goal.x && player.y === goal.y) {
        statusEl.textContent = "Reached goal. ✅";
        break;
      }

      if (ip >= prog.length) {
        if (hasLoop) ip = ls + 1;
        else {
          statusEl.textContent = "Finished program.";
          break;
        }
      }

      const tok = prog[ip];

      if (tok.t === "loopStart") { ip += 1; continue; }
      if (tok.t === "loopEnd") {
        if (hasLoop) ip = ls + 1;
        else ip += 1;
        continue;
      }

      if (tok.t === "move") {
        if (tok.cond) {
          const c = tileColorAt(player.x, player.y);
          if (c !== tok.cond) {
            ip += 1;
            steps += 1;
            continue;
          }
        }

        const { dx, dy } = dirDelta(tok.dir);
        const nx = player.x + dx;
        const ny = player.y + dy;

        if (isWall(nx, ny)) {
          spriteIdle();
          statusEl.textContent = "Stopped: hit wall / out of bounds.";
          break;
        }

        spriteWalk();
        player = { x: nx, y: ny };
        setAvatarPos(player.x, player.y, false);
        await sleep(cfg.speedMs);

        if (goal && player.x === goal.x && player.y === goal.y) {
          spriteIdle();
          statusEl.textContent = "Reached goal. ✅";
          break;
        }

        spriteIdle();
        ip += 1;
        steps += 1;
        continue;
      }

      ip += 1;
      steps += 1;
    }

    if (steps >= (cfg.maxSteps || 500) && !(goal && player.x === goal.x && player.y === goal.y)) {
      statusEl.textContent = "Stopped: step limit.";
    }

    spriteIdle();
    running = false;
    runBtn.disabled = false;
  }

  function restart() {
    cancelRun = true;
    player = { x: start.x, y: start.y };
    setAvatarPos(player.x, player.y, true);
    spriteIdle();
    statusEl.textContent = "Restarted.";
    runBtn.disabled = false;
    running = false;
  }

  function resetSlots() {
    cancelRun = true;
    for (let i = 0; i < slots.length; i++) {
      slots[i].base = null;
      slots[i].cond = null;
      renderSlot(i);
    }
    statusEl.textContent = "Slots cleared.";
    runBtn.disabled = false;
    running = false;
    spriteIdle();
    updateCodeView();
  }

  runBtn.addEventListener("click", () => {
    if (running) return;
    runProgram();
  });

  restartBtn.addEventListener("click", restart);
  resetSlotsBtn.addEventListener("click", resetSlots);

  statusEl.textContent = "Ready. Drag blocks into slots, then Run.";
  updateCodeView(); // initial
})();
