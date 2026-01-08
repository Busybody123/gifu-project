import streamlit as st

GRID_W, GRID_H = 8, 8

def new_state():
    return {
        "x": 0,
        "y": 0,
        "goal": (5, 3),
        "blocked": {(2, 2), (2, 3), (2, 4)},
        "log": [],
        "last_error": None,
    }

print(st.session_state)
if "state" not in st.session_state:
    st.session_state.state = new_state()

def S():
    """Convenience accessor for current state dict."""
    return st.session_state.state

# -----------------------
# 4) Grid rendering (emoji grid)
# -----------------------
def render_grid(state) -> str:
    x, y = state["x"], state["y"]
    gx, gy = state["goal"]
    blocked = state["blocked"]

    rows = []
    for r in range(GRID_H):
        row = []
        for c in range(GRID_W):
            if (c, r) == (x, y):
                row.append("🤖")
            elif (c, r) == (gx, gy):
                row.append("🎯")
            elif (c, r) in blocked:
                row.append("🧱")
            else:
                row.append("⬜")
        rows.append(" ".join(row))
    return "\n".join(rows)



# API for student code
def _move(dx, dy):
    state = S()
    nx, ny = state["x"] + dx, state["y"] + dy

    if not (0 <= nx < GRID_W and 0 <= ny < GRID_H):
        state["log"].append("❌ Bumped into boundary.")
        return

    if (nx, ny) in state["blocked"]:
        state["log"].append("❌ Hit a wall.")
        return

    state["x"], state["y"] = nx, ny
    state["log"].append(f"✅ Moved to ({state['x']},{state['y']}).")

def move_up(): _move(0, -1)
def move_down(): _move(0, 1)
def move_left(): _move(-1, 0)
def move_right(): _move(1, 0)

def at_goal():
    state = S()
    return (state["x"], state["y"]) == state["goal"]



# Safe code exec
BANNED = ["import", "__", "open(", "eval(", "exec(", "os.", "sys.", "subprocess", "socket"]

def run_student_code(code: str):
    lowered = code.lower()
    for token in BANNED:
        if token in lowered:
            raise ValueError(f"Blocked for safety: '{token}' is not allowed.")

    safe_globals = {
        "__builtins__": {
            "range": range,
            "len": len,
            "print": print,
        },
        "move_up": move_up,
        "move_down": move_down,
        "move_left": move_left,
        "move_right": move_right,
        "at_goal": at_goal,
    }

    exec(code, safe_globals, {})


# UI
st.title("Code Courier (Beta)")

with st.form("code_form"):
    st.subheader("Write Python")
    starter = """# Get the courier to the 🎯
for _ in range(5):
    move_right()
for _ in range(3):
    move_down()
"""
    code = st.text_area("Code editor", value=starter, height=220)
    run_clicked = st.form_submit_button("Run")
    reset_clicked = st.form_submit_button("Reset board")

# ---- Handle actions BEFORE rendering the board ----
state = S()

if reset_clicked:
    st.session_state.state = new_state()
    st.rerun()

if run_clicked:
    state["log"].clear()
    state["last_error"] = None
    try:
        run_student_code(code)
    except Exception as e:
        state["last_error"] = str(e)

    # Force a rerun so the board is rendered using the updated state immediately
    st.rerun()

# ---- Now render board + output (always reflects latest state) ----
st.subheader("Board")
st.code(render_grid(S()), language="text")

if at_goal():
    st.success("Delivery complete! 🎉")

st.subheader("Log")
if state["last_error"]:
    st.error(f"Error: {state['last_error']}")

if state["log"]:
    st.write("\n".join(state["log"]))
else:
    st.write("(no actions yet)")
