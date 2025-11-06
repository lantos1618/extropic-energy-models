#!/usr/bin/env python3
"""
XY Model with THRML: Vortex Formation in 2D

The XY model describes spins that can point in any direction in a 2D plane
(like compass needles). At low temperature, beautiful topological vortices form!

Energy: E = -J Σ_{<i,j>} cos(θ_i - θ_j)

where θ_i ∈ [0, 2π) is the angle of spin i

Key Features:
- Kosterlitz-Thouless transition (exotic phase transition!)
- Vortex-antivortex pairs at intermediate temperatures
- Long-range order at low temperatures
- Topological defects (vortices with winding number ±1)

This is REAL energy-based computing - vortices emerge from energy minimization!
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import hsv_to_rgb
import networkx as nx
from typing import Tuple

from thrml.block_management import Block
from thrml.block_sampling import BlockGibbsSpec, sample_states, SamplingSchedule
from thrml.models.discrete_ebm import CategoricalEBMFactor, CategoricalGibbsConditional
from thrml.factor import FactorSamplingProgram
from thrml.pgm import CategoricalNode


def discretize_angles(n_angles: int = 16):
    """
    Discretize continuous angles [0, 2π) into n_angles discrete states.

    For THRML, we need discrete states. More angles = better approximation.
    """
    return np.linspace(0, 2*np.pi, n_angles, endpoint=False)


def create_xy_model(grid_size: int, n_angles: int, beta: float):
    """
    Create XY model on 2D grid with discretized angles.

    Energy: E = -J Σ cos(θ_i - θ_j) = -J Σ [cos(θ_i)cos(θ_j) + sin(θ_i)sin(θ_j)]

    This is trickier than Ising/Potts because interactions involve cosine!
    """
    print(f"Creating {grid_size}x{grid_size} XY model with {n_angles} angles, β={beta:.3f}")

    # Create grid
    G = nx.grid_graph(dim=(grid_size, grid_size))
    coord_to_node = {coord: CategoricalNode() for coord in G.nodes}
    nx.relabel_nodes(G, coord_to_node, copy=False)
    nodes = list(G.nodes)

    # Get edges
    u, v = map(list, zip(*G.edges()))

    # Bipartite coloring for block Gibbs
    bicol = nx.bipartite.color(G)
    color0 = [n for n, c in bicol.items() if c == 0]
    color1 = [n for n, c in bicol.items() if c == 1]
    blocks = [Block(color0), Block(color1)]

    # Compute interaction matrix
    # For XY model: w[a,b] = β * cos(θ_a - θ_b)
    angles = discretize_angles(n_angles)
    interaction_matrix = np.zeros((n_angles, n_angles))

    for i, theta_i in enumerate(angles):
        for j, theta_j in enumerate(angles):
            # Energy contribution: -J cos(θ_i - θ_j)
            # For EBM, we want high weight = low energy
            interaction_matrix[i, j] = beta * np.cos(theta_i - theta_j)

    # Replicate for all edges
    weights = jnp.array([interaction_matrix for _ in range(len(u))])

    coupling_interaction = CategoricalEBMFactor([Block(u), Block(v)], weights)

    # Create sampling program
    spec = BlockGibbsSpec(blocks, [])
    sampler = CategoricalGibbsConditional(n_angles)
    prog = FactorSamplingProgram(
        spec,
        [sampler for _ in spec.free_blocks],
        [coupling_interaction],
        []
    )

    return prog, nodes, G, blocks, angles


def detect_vortices(angle_grid: np.ndarray, angles: np.ndarray) -> Tuple[list, list]:
    """
    Detect topological vortices in the angle field.

    A vortex has winding number ±1 when going around a plaquette.
    Winding number = (1/2π) * Σ Δθ around square

    Returns:
        vortex_positions: List of (i, j) with winding +1
        antivortex_positions: List of (i, j) with winding -1
    """
    h, w = angle_grid.shape
    vortices = []
    antivortices = []

    # Check each plaquette (2x2 square)
    for i in range(h-1):
        for j in range(w-1):
            # Get angles at corners (clockwise from top-left)
            theta_tl = angles[angle_grid[i, j]]
            theta_tr = angles[angle_grid[i, j+1]]
            theta_br = angles[angle_grid[i+1, j+1]]
            theta_bl = angles[angle_grid[i+1, j]]

            # Compute phase differences (handle 2π wrapping)
            def phase_diff(a, b):
                diff = a - b
                # Wrap to [-π, π]
                while diff > np.pi:
                    diff -= 2*np.pi
                while diff < -np.pi:
                    diff += 2*np.pi
                return diff

            # Sum phase differences around plaquette
            d1 = phase_diff(theta_tr, theta_tl)
            d2 = phase_diff(theta_br, theta_tr)
            d3 = phase_diff(theta_bl, theta_br)
            d4 = phase_diff(theta_tl, theta_bl)

            winding = (d1 + d2 + d3 + d4) / (2*np.pi)

            # Vortex if |winding| > 0.5
            if winding > 0.5:
                vortices.append((i, j))
            elif winding < -0.5:
                antivortices.append((i, j))

    return vortices, antivortices


def visualize_xy_spins(angle_grid: np.ndarray, angles: np.ndarray,
                       beta: float, output_file: str, show_vortices: bool = True):
    """
    Visualize XY model spins with arrows and vortex detection.
    """
    h, w = angle_grid.shape

    # Convert angle indices to actual angles
    theta_field = angles[angle_grid]

    # Detect vortices
    vortices, antivortices = detect_vortices(angle_grid, angles)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), facecolor='black')

    # 1. Arrow field (quiver plot)
    ax1 = axes[0]
    ax1.set_facecolor('black')

    # Create meshgrid
    x, y = np.meshgrid(np.arange(w), np.arange(h))

    # Arrow components
    u = np.cos(theta_field)
    v = np.sin(theta_field)

    # Color by angle
    colors = theta_field / (2*np.pi)

    # Quiver plot (subsample for clarity)
    stride = max(1, w // 32)
    quiv = ax1.quiver(x[::stride, ::stride], y[::stride, ::stride],
                     u[::stride, ::stride], v[::stride, ::stride],
                     colors[::stride, ::stride],
                     cmap='hsv', scale=20, width=0.003, headwidth=4)

    # Mark vortices
    if show_vortices and vortices:
        vx, vy = zip(*vortices)
        ax1.scatter(np.array(vy)+0.5, np.array(vx)+0.5,
                   s=100, c='red', marker='o', edgecolors='white', linewidths=2,
                   label=f'Vortices ({len(vortices)})')

    if show_vortices and antivortices:
        ax, ay = zip(*antivortices)
        ax1.scatter(np.array(ay)+0.5, np.array(ax)+0.5,
                   s=100, c='blue', marker='x', linewidths=3,
                   label=f'Antivortices ({len(antivortices)})')

    ax1.set_xlim(-0.5, w-0.5)
    ax1.set_ylim(-0.5, h-0.5)
    ax1.set_aspect('equal')
    ax1.set_title(f'Spin Field (Arrows) - β={beta:.3f}',
                 color='white', fontsize=14, fontweight='bold')
    ax1.tick_params(colors='white')
    if show_vortices:
        ax1.legend(facecolor='black', edgecolor='white', labelcolor='white', fontsize=10)

    # 2. Color wheel visualization
    ax2 = axes[1]
    ax2.set_facecolor('black')

    # Use HSV coloring (hue = angle)
    hsv_image = np.zeros((h, w, 3))
    hsv_image[:, :, 0] = theta_field / (2*np.pi)  # Hue from angle
    hsv_image[:, :, 1] = 1.0  # Full saturation
    hsv_image[:, :, 2] = 1.0  # Full brightness
    rgb_image = hsv_to_rgb(hsv_image)

    ax2.imshow(rgb_image, interpolation='nearest')

    # Mark vortices
    if show_vortices and vortices:
        vx, vy = zip(*vortices)
        ax2.scatter(np.array(vy)+0.5, np.array(vx)+0.5,
                   s=80, c='white', marker='o', edgecolors='black', linewidths=2)

    if show_vortices and antivortices:
        ax, ay = zip(*antivortices)
        ax2.scatter(np.array(ay)+0.5, np.array(ax)+0.5,
                   s=80, c='black', marker='x', linewidths=2)

    ax2.set_xlim(-0.5, w-0.5)
    ax2.set_ylim(-0.5, h-0.5)
    ax2.set_title('Color Wheel (Hue = Angle)',
                 color='white', fontsize=14, fontweight='bold')
    ax2.axis('off')

    # 3. Magnetization and energy
    ax3 = axes[2]
    ax3.set_facecolor('black')
    ax3.axis('off')

    # Compute order parameters
    mx = np.mean(np.cos(theta_field))
    my = np.mean(np.sin(theta_field))
    magnetization = np.sqrt(mx**2 + my**2)

    # Compute energy
    energy = 0
    for i in range(h):
        for j in range(w):
            # Right neighbor
            if j < w-1:
                energy -= np.cos(theta_field[i,j] - theta_field[i,j+1])
            # Down neighbor
            if i < h-1:
                energy -= np.cos(theta_field[i,j] - theta_field[i+1,j])

    energy_per_spin = energy / (h*w)

    # Statistics text
    stats = f"""
    XY MODEL STATISTICS

    Temperature: T = {1/beta:.3f}
    Inverse Temp: β = {beta:.3f}

    Order Parameter:
      Magnetization: {magnetization:.4f}
      M_x: {mx:.4f}
      M_y: {my:.4f}

    Energy:
      Total: {energy:.1f}
      Per Spin: {energy_per_spin:.4f}

    Topological Defects:
      Vortices (+1): {len(vortices)}
      Antivortices (-1): {len(antivortices)}
      Net charge: {len(vortices) - len(antivortices)}

    Phase:
      {_classify_phase(beta, magnetization, len(vortices))}
    """

    ax3.text(0.1, 0.5, stats, transform=ax3.transAxes,
            fontsize=11, color='white', fontfamily='monospace',
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='black',
                     edgecolor='cyan', linewidth=2, alpha=0.9))

    # Add color wheel legend
    ax3_inset = fig.add_axes([0.72, 0.15, 0.15, 0.15])
    theta_wheel = np.linspace(0, 2*np.pi, 100)
    x_wheel = np.cos(theta_wheel)
    y_wheel = np.sin(theta_wheel)
    colors_wheel = theta_wheel / (2*np.pi)

    ax3_inset.scatter(x_wheel, y_wheel, c=colors_wheel, cmap='hsv', s=50)
    ax3_inset.set_xlim(-1.2, 1.2)
    ax3_inset.set_ylim(-1.2, 1.2)
    ax3_inset.set_aspect('equal')
    ax3_inset.axis('off')
    ax3_inset.text(0, 0, '0°→', ha='center', va='center',
                  color='white', fontsize=10, fontweight='bold')

    plt.suptitle(f'XY Model: Topological Vortices in 2D Spins',
                color='white', fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_file, dpi=150, facecolor='black')
    print(f"Saved: {output_file}")
    plt.close()

    return magnetization, energy_per_spin, len(vortices), len(antivortices)


def _classify_phase(beta, magnetization, n_vortices):
    """Classify which phase the system is in."""
    if beta < 0.5:
        return "DISORDERED\n      (Paramagnetic, many vortices)"
    elif beta < 1.2:
        return "KOSTERLITZ-THOULESS\n      (Vortex-antivortex pairs)"
    else:
        return "QUASI-ORDERED\n      (Few vortices, aligned spins)"


def run_xy_sampling(prog, blocks, nodes, grid_size, n_angles, angles,
                   n_chains=8, n_warmup=500, n_samples=1000, seed=42):
    """Sample from XY model using THRML."""
    print(f"\nSampling with {n_chains} parallel chains...")

    key = jax.random.key(seed)

    # Initialize random angles
    init_state = []
    for block in blocks:
        key, subkey = jax.random.split(key, 2)
        init_state.append(
            jax.random.randint(subkey, (n_chains, len(block.nodes)),
                              minval=0, maxval=n_angles, dtype=jnp.uint8)
        )

    schedule = SamplingSchedule(n_warmup=n_warmup, n_samples=n_samples, steps_per_sample=2)

    print("  Running THRML block Gibbs sampling...")
    keys = jax.random.split(key, n_chains)
    samples = jax.jit(
        jax.vmap(lambda k, i: sample_states(k, prog, schedule, i, [], [Block(nodes)]))
    )(keys, init_state)

    samples_array = np.array(samples[0])
    samples_grid = samples_array.reshape(n_chains, -1, grid_size, grid_size)

    print(f"  Generated {samples_array.shape[1]} samples from {n_chains} chains")

    return samples_grid


def temperature_sweep():
    """Run XY model at different temperatures to see vortex formation."""
    print("="*80)
    print("XY MODEL: VORTEX FORMATION IN 2D")
    print("="*80)

    grid_size = 32
    n_angles = 16  # Discretize circle into 16 directions
    angles = discretize_angles(n_angles)

    # Temperature range (β = 1/T)
    betas = [0.3, 0.6, 0.9, 1.2, 1.5, 2.0]

    results = []

    for beta in betas:
        print(f"\n{'='*80}")
        print(f"Temperature: T = {1/beta:.3f} (β = {beta:.3f})")
        print(f"{'='*80}")

        # Create model
        prog, nodes, G, blocks, angles = create_xy_model(grid_size, n_angles, beta)

        # Sample
        samples_grid = run_xy_sampling(
            prog, blocks, nodes, grid_size, n_angles, angles,
            n_chains=6, n_warmup=400, n_samples=400, seed=42+int(beta*10)
        )

        # Visualize final sample from first chain
        final_sample = samples_grid[0, -1]

        output_file = f"xy_model_beta_{beta:.1f}.png"
        mag, energy, n_v, n_av = visualize_xy_spins(
            final_sample, angles, beta, output_file, show_vortices=True
        )

        results.append({
            'beta': beta,
            'temperature': 1/beta,
            'magnetization': mag,
            'energy': energy,
            'vortices': n_v,
            'antivortices': n_av
        })

        print(f"\n  Results:")
        print(f"    Magnetization: {mag:.4f}")
        print(f"    Energy/spin: {energy:.4f}")
        print(f"    Vortices: {n_v}, Antivortices: {n_av}")

    # Create summary
    create_xy_phase_diagram(results)

    print(f"\n{'='*80}")
    print("COMPLETE!")
    print("Generated 6 visualizations + phase diagram")
    print(f"{'='*80}")


def create_xy_phase_diagram(results):
    """Plot phase diagram for XY model."""
    print("\nCreating XY phase diagram...")

    temps = [r['temperature'] for r in results]
    mags = [r['magnetization'] for r in results]
    energies = [r['energy'] for r in results]
    vortices = [r['vortices'] + r['antivortices'] for r in results]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5), facecolor='black')

    # Magnetization
    ax1.set_facecolor('black')
    ax1.plot(temps, mags, 'o-', color='cyan', linewidth=2, markersize=8)
    ax1.axvline(0.89, color='red', linestyle='--', alpha=0.5,
               label='KT transition (~0.89)')
    ax1.set_xlabel('Temperature (T)', color='white', fontsize=12)
    ax1.set_ylabel('Magnetization', color='white', fontsize=12)
    ax1.set_title('Order Parameter', color='white', fontsize=14, fontweight='bold')
    ax1.tick_params(colors='white')
    ax1.spines['bottom'].set_color('white')
    ax1.spines['left'].set_color('white')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(True, alpha=0.2, color='gray')
    ax1.legend(facecolor='black', edgecolor='white', labelcolor='white')

    # Energy
    ax2.set_facecolor('black')
    ax2.plot(temps, energies, 's-', color='orange', linewidth=2, markersize=8)
    ax2.set_xlabel('Temperature (T)', color='white', fontsize=12)
    ax2.set_ylabel('Energy per spin', color='white', fontsize=12)
    ax2.set_title('Average Energy', color='white', fontsize=14, fontweight='bold')
    ax2.tick_params(colors='white')
    ax2.spines['bottom'].set_color('white')
    ax2.spines['left'].set_color('white')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(True, alpha=0.2, color='gray')

    # Vortex density
    ax3.set_facecolor('black')
    ax3.plot(temps, vortices, '^-', color='magenta', linewidth=2, markersize=8)
    ax3.axvline(0.89, color='red', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Temperature (T)', color='white', fontsize=12)
    ax3.set_ylabel('Total Vortices', color='white', fontsize=12)
    ax3.set_title('Topological Defects', color='white', fontsize=14, fontweight='bold')
    ax3.tick_params(colors='white')
    ax3.spines['bottom'].set_color('white')
    ax3.spines['left'].set_color('white')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.grid(True, alpha=0.2, color='gray')

    plt.suptitle('XY Model Phase Diagram: Kosterlitz-Thouless Transition',
                color='white', fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('xy_model_phase_diagram.png', dpi=150, facecolor='black')
    print("Saved: xy_model_phase_diagram.png")
    plt.close()


if __name__ == "__main__":
    print("="*80)
    print("XY MODEL WITH THRML: TOPOLOGICAL VORTICES")
    print("="*80)
    print("\nThis demonstrates:")
    print("  • Continuous spins discretized for THRML")
    print("  • Vortex-antivortex pair formation")
    print("  • Kosterlitz-Thouless phase transition")
    print("  • Topological defects emerging from energy!")
    print("="*80)

    temperature_sweep()
