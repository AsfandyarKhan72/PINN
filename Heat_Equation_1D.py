# --- Standard library ---
import os
import glob
import shutil    
import re       

# Set DeepXDE backend BEFORE importing deepxde
os.environ["DDE_BACKEND"] = "pytorch"
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D  
import imageio.v2 as imageio
import deepxde as dde

# DeepXDE config
dde.config.set_random_seed(42)


# --- Problem params 
alpha_true = 0.4   # thermal diffusivity
L = 1.0   # bar length
TMAX = 1.0 # max time
n = 1     # sinusoidal mode

# --- Geometry and time (x∈[0,1], t∈[0,1]) ---
geom = dde.geometry.Interval(0, L)
timedomain = dde.geometry.TimeDomain(0, TMAX)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# --- PDE residual: u_t - a u_xx = 0 ---
def pde(x, y):
    dy_t  = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_t - alpha_true * dy_xx

# --- Initial condition: u(x,0) = sin(nπx/L) ---
ic = dde.icbc.IC(
    geomtime,
    lambda X: np.sin(n * np.pi * X[:, 0:1] / L),
    lambda _, on_initial: on_initial,
)

# --- Dirichlet BC: u(0,t)=u(1,t)=0 
bc = dde.icbc.DirichletBC(
    geomtime, lambda X: 0.0, lambda _, on_boundary: on_boundary
)

# --- Data config 
data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc, ic],
    num_domain=2540,
    num_boundary=80,
    num_initial=160,
    num_test=2540,
)

# --- Network (FNN, 3 hidden layers of 20, tanh) ---
net_inver = dde.nn.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")

model = dde.Model(data, net_inver)

# --- Train: Adam then L-BFGS ---
initial_weights = [1, 1, 100]
model.compile("adam", lr=1e-3, loss='MSE', loss_weights=initial_weights)
losshistory, train_state = model.train(iterations=20000, batch_size=32, display_every=1000, model_save_path=f"./time_models/model_time_step_adam_best")

model.compile("L-BFGS")
losshistory, train_state = model.train(model_save_path=f"./time_models/model_time_step_adam_best")

# --- Exact solution
def exact(x, t):
    return np.exp(-(n**2 * np.pi**2 * alpha_true * t) / (L**2)) * np.sin(n * np.pi * x / L)


# ----- Build dense space–time fields -----
nx, nt = 256, 256
x = np.linspace(0.0, L, nx)
t = np.linspace(0.0, TMAX, nt)
Xg, Tg = np.meshgrid(x, t)           # shape (nt, nx)

XT = np.column_stack([Xg.ravel(), Tg.ravel()])
U_pred = model.predict(XT).reshape(nt, nx)
U_true = exact(Xg, Tg)

# Consistent color limits across frames
vmin = float(min(U_true.min(), U_pred.min()))
vmax = float(max(U_true.max(), U_pred.max()))

# -----GIF -----
frames_dir = "frames_heat_contour"
os.makedirs(frames_dir, exist_ok=True)

num_frames = 60
time_indices = np.linspace(0, nt-1, num_frames).astype(int)

levels = 50  # number of contour levels

for k, ti in enumerate(time_indices):
    Up = U_pred.copy()
    Ut = U_true.copy()
    Up[ti+1:, :] = np.nan
    Ut[ti+1:, :] = np.nan

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), dpi=160, constrained_layout=True)
    for ax, Z, title in zip(
        axes,
        (Up, Ut),
        (f"Forward PINN prediction (t ≤ {t[ti]:.3f})", "Exact solution"),
    ):
        cf = ax.contourf(x, t, Z, levels=levels, vmin=vmin, vmax=vmax)
        ax.contour(x, t, Z, levels=levels, linewidths=0.3, colors="k", alpha=0.25)
        ax.axhline(t[ti], linestyle="--", linewidth=1.0, color="k", alpha=0.6)
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        ax.set_title(title)

    # single shared colorbar
    cbar = fig.colorbar(cf, ax=axes.ravel().tolist(), shrink=0.92, pad=0.02)
    cbar.set_label("u(x,t)")

    outpng = os.path.join(frames_dir, f"frame_{k:03d}.png")
    plt.savefig(outpng)
    plt.close(fig)

# Stitch frames → GIF (~10 fps)
pngs = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
images = [imageio.imread(p) for p in pngs]
imageio.mimsave("heat_space_time_contour.gif", images, duration=0.10, loop=0)
print("Saved GIF: heat_space_time_contour.gif")



# -----------------------------
# Generate observations at t=1 *from the forward model* for Inverse PINN modeling
# -----------------------------
x_obs = np.linspace(0.0, L, 101)[:, None]
t_obs = np.ones_like(x_obs) * TMAX
X_obs = np.hstack([x_obs, t_obs])      # (101, 2)
y_obs = model.predict(X_obs)       

# "Measurement" constraint at t=1 
meas_bc = dde.icbc.PointSetBC(X_obs, y_obs)

# Learnable alpha
alpha = dde.Variable(0.1)  # initial guess
def pde_inverse(X, u):
    ut  = dde.grad.jacobian(u, X, i=0, j=1)
    uxx = dde.grad.hessian(u, X, i=0, j=0)
    return ut - alpha * uxx


# Use the same IC + spatial BC; add measurements; and set ANCHORS = X_obs
data_inv = dde.data.TimePDE(
    geomtime,
    pde_inverse,
    [bc, ic, meas_bc],
    num_domain=2540,
    num_boundary=80,
    num_initial=160,
    anchors=X_obs,            
)


# --- Network (FNN, 3 hidden layers of 20, tanh) ---
net_inver = dde.nn.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")
model_inv = dde.Model(data_inv, net_inver)


# Track alpha over time
class AlphaLogger(dde.callbacks.Callback):
    def __init__(self, var, period=100):
        super().__init__()
        self.var = var
        self.period = period
        self.step = 0
        self.steps, self.vals = [], []
        self._begun = False 

    def on_train_begin(self):
        if not self._begun:
            try:
                v = self.var.detach().cpu().item()  # PyTorch
            except Exception:
                v = float(self.var)
            self.steps.append(0)
            self.vals.append(v)
            self._begun = True

    def on_epoch_end(self):
        # Called once per training iteration (epoch) by DeepXDE
        self.step += 1
        if self.step % self.period == 0:
            try:
                v = self.var.detach().cpu().item()
            except Exception:
                v = float(self.var)
            self.steps.append(self.step)
            self.vals.append(v)


alpha_cb = AlphaLogger(alpha, period=100)

# IMPORTANT: keep α trainable in both compiles
loss_weights = [1, 1, 100, 100]
model_inv.compile(
    "adam",
    lr=1e-3,
    loss_weights=loss_weights,
    external_trainable_variables=[alpha],
)
model_inv.train(iterations=20000, callbacks=[alpha_cb], model_save_path=f"./time_models/model_time_step_adam_inve")

model_inv.compile(
    "L-BFGS",
    loss_weights=loss_weights,
    external_trainable_variables=[alpha],
)
model_inv.train(callbacks=[alpha_cb], model_save_path=f"./time_models/model_time_step_adam_inve")

alpha_hat = alpha.detach().cpu().item()
print(f"Estimated α (final) = {alpha_hat:.6f}")

frames_dir = "frames_alpha_dots"
os.makedirs(frames_dir, exist_ok=True)

steps = np.array(alpha_cb.steps, dtype=float)
vals  = np.array(alpha_cb.vals, dtype=float)

# quick sanity check
print("First few α logs:", list(zip(steps[:5], vals[:5])))

# y-limits that include the reference α=0.4
ref_alpha = 0.4
ymin = min(vals.min(), ref_alpha) - 0.05
ymax = max(vals.max(), ref_alpha) + 0.05


frames_dir = "frames_alpha_dots"
os.makedirs(frames_dir, exist_ok=True)

# Clean out any old frames so glob finds only fresh ones
for _f in glob.glob(os.path.join(frames_dir, "frame_*.png")):
    try: os.remove(_f)
    except: pass

# Pull logs (fallback to one point if empty so loop runs)
steps = np.array(getattr(alpha_cb, "steps", []), dtype=float)
vals  = np.array(getattr(alpha_cb, "vals", []), dtype=float)
if steps.size == 0 or vals.size == 0:
    steps = np.array([0.0])
    vals  = np.array([alpha.detach().cpu().item()])

# ---- visual constants ----
ref_alpha     = 0.4
eq_text       = r"$u_t=\alpha\,u_{xx}$   with   $u(x,0)=\sin(\pi x)$,   $u(0,t)=0=u(1,t)$"
est_color     = "C0"   # estimated α dots
true_color    = "C3"   # true α line/dot
hilite_color  = "C2"   # hollow circle highlight
ymin          = min(vals.min(), ref_alpha) - 0.05
ymax          = max(vals.max(), ref_alpha) + 0.05
xmax_fixed    = float(steps.max()) * 1.05 if steps.size else 1.0

# Legend handles
est_handle  = mlines.Line2D([], [], color=est_color, marker='o', linestyle='None',
                            markersize=5, label='Estimated α')
curr_handle = mlines.Line2D([], [], color=hilite_color, marker='o', linestyle='None',
                            markerfacecolor='none', markersize=9, label='Current estimate')
true_handle = mlines.Line2D([], [], color=true_color, linestyle='--',
                            label=f'True α = {ref_alpha}')

for k in range(1, len(vals) + 1):
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=160)

    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.16, top=0.78)

    fig.suptitle("Inverse PINN: α convergence (every 100 iters)", fontsize=12, y=0.96)
    fig.text(0.5, 0.88, eq_text, ha="center", va="top", fontsize=10)

    ax.scatter(steps[:k], vals[:k], s=22, color=est_color, zorder=2)

    ax.scatter(steps[k-1], vals[k-1], s=90, facecolors="none",
               edgecolors=hilite_color, linewidths=1.6, zorder=3)

    ax.axhline(ref_alpha, linestyle="--", linewidth=1.6, color=true_color, zorder=1)
    ax.scatter(steps[k-1], ref_alpha, s=36, color=true_color, zorder=2)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("α")
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(0, xmax_fixed)

    ax.legend(handles=[est_handle, curr_handle, true_handle],
              loc="upper right", frameon=False)

    outpng = os.path.join(frames_dir, f"frame_{k:04d}.png")
    fig.savefig(outpng)  
    plt.close(fig)

# ---------- STITCH TO GIF ----------
pngs = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
print(f"Wrote {len(pngs)} frames to {frames_dir}")
if not pngs:
    raise RuntimeError("No frames were generated—check that alpha_cb collected points.")

images = [imageio.imread(p) for p in pngs]
imageio.mimsave("alpha_convergence_dots.gif", images, duration=0.10, loop=0)
print("Saved GIF: alpha_convergence_dots.gif")
