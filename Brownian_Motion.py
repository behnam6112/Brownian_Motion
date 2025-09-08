import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ---- Parameters ----
np.random.seed(42)
dt = 0.01
T = 60
t = np.arange(0, T + dt, dt)
D = 1e-6
Ns = [1, 10, 100, 1000]

def brownian_increments(D, dt, steps, n_particles):
    return np.sqrt(2 * D * dt) * np.random.randn(steps, n_particles)

# ---- Simulate ----
results = {}
steps = len(t) - 1
for n in Ns:
    dWx = brownian_increments(D, dt, steps, n)
    dWy = brownian_increments(D, dt, steps, n)
    x = np.vstack([np.zeros((1, n)), np.cumsum(dWx, axis=0)])
    y = np.vstack([np.zeros((1, n)), np.cumsum(dWy, axis=0)])
    results[n] = {"x": x, "y": y}

# =========================================================
# Figure 1: XY trajectories + RMS circle in each panel
# =========================================================
fig1, axes1 = plt.subplots(2, 2, figsize=(10, 10))
axes1 = axes1.ravel()
r_rms = np.sqrt(4 * D * T)         # RMS radius for 2D Brownian motion

for ax, n in zip(axes1, Ns):
    x = results[n]["x"]; y = results[n]["y"]

    k = n if n <= 100 else 100     # limit lines drawn for clarity
    for j in range(k):
        ax.plot(x[:, j], y[:, j], color="blue", alpha=0.2, linewidth=0.8)

    # circle (same in all 4 panels)
    circ = Circle((0, 0), r_rms, fill=False, color="red", linestyle="--", linewidth=1.8)
    ax.add_patch(circ)
    ax.annotate("RMS radius √(4DT)", xy=(r_rms, 0), xytext=(5, 5),
                textcoords="offset points", fontsize=8, color="red")

    # view / cosmetics
    r = r_rms * 1.4                 # margin so circle fits nicely
    ax.set_xlim(-r, r); ax.set_ylim(-r, r)
    ax.set_aspect('equal', 'box')
    ax.set_title(f'{n} Brownian particle{"s" if n>1 else ""}')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.grid(True, alpha=0.3)

fig1.suptitle('2D Brownian Trajectories (x vs y) with RMS circle', fontsize=14, y=0.95)
fig1.tight_layout(rect=[0, 0, 1, 0.94])

# ==================================================================
# Figure 2: X(t) and Y(t) for each N (single-color trajectories)
# ==================================================================
fig2, axes2 = plt.subplots(len(Ns), 2, figsize=(12, 12), sharex=True)

for i, n in enumerate(Ns):
    x = results[n]["x"]; y = results[n]["y"]
    k = n if n <= 50 else 50

    mean_x = x.mean(axis=1); mean_y = y.mean(axis=1)
    std_x = x.std(axis=1);   std_y = y.std(axis=1)
    theory = np.sqrt(2 * D * t)

    axx = axes2[i, 0]
    for j in range(k):
        axx.plot(t, x[:, j], color="blue", alpha=0.2, linewidth=0.8)
    axx.plot(t, mean_x, color="black", linewidth=2, label='mean')
    axx.fill_between(t, mean_x - std_x, mean_x + std_x, color="gray", alpha=0.3, label='±1σ')
    axx.plot(t, theory, color="red", linestyle='--', linewidth=1.5, label='±√(2Dt)')
    axx.plot(t, -theory, color="red", linestyle='--', linewidth=1.5)
    axx.set_ylabel('x(t)'); axx.set_title(f'{n} particle{"s" if n>1 else ""} — X over time')
    axx.grid(True, alpha=0.3)
    if i == 0: axx.legend(loc='upper left')

    axy = axes2[i, 1]
    for j in range(k):
        axy.plot(t, y[:, j], color="blue", alpha=0.2, linewidth=0.8)
    axy.plot(t, mean_y, color="black", linewidth=2, label='mean')
    axy.fill_between(t, mean_y - std_y, mean_y + std_y, color="gray", alpha=0.3, label='±1σ')
    axy.plot(t, theory, color="red", linestyle='--', linewidth=1.5, label='±√(2Dt)')
    axy.plot(t, -theory, color="red", linestyle='--', linewidth=1.5)
    axy.set_ylabel('y(t)'); axy.set_title(f'{n} particle{"s" if n>1 else ""} — Y over time')
    axy.grid(True, alpha=0.3)
    if i == 0: axy.legend(loc='upper left')

axes2[-1, 0].set_xlabel('time (s)')
axes2[-1, 1].set_xlabel('time (s)')
fig2.suptitle('Brownian Motion: X(t) and Y(t) with mean, ±std, theory', fontsize=14, y=0.95)
fig2.tight_layout(rect=[0, 0, 1, 0.94])

plt.show()
