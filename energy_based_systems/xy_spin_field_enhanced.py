#!/usr/bin/env python3
"""
Enhanced XY Spin Field Visualization
Shows spin directions with arrows + color field for maximum visual impact!
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import hsv_to_rgb
import networkx as nx

from thrml.block_management import Block
from thrml.block_sampling import BlockGibbsSpec, sample_states, SamplingSchedule
from thrml.models.discrete_ebm import CategoricalEBMFactor, CategoricalGibbsConditional
from thrml.factor import FactorSamplingProgram
from thrml.pgm import CategoricalNode


def create_xy_spin_field_video(size=48, n_angles=16, beta=1.2, fps=30, duration=15):
    """
    Create enhanced spin field video with arrows showing directions.

    Lower beta = MORE VORTICES and MORE CHAOS!
    """
    print(f"\nCreating ENHANCED XY spin field video ({duration}s @ {fps}fps)...")
    print(f"  Grid size: {size}x{size}")
    print(f"  Beta: {beta} (lower = more vortices!)")
    print(f"  Angles: {n_angles} discrete states")

    n_frames = fps * duration
    angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)

    # Create grid
    G = nx.grid_graph(dim=(size, size))
    coord_to_node = {coord: CategoricalNode() for coord in G.nodes}
    nx.relabel_nodes(G, coord_to_node, copy=False)
    nodes = list(G.nodes)

    u, v = map(list, zip(*G.edges()))

    # Bipartite coloring
    bicol = nx.bipartite.color(G)
    color0 = [n for n, c in bicol.items() if c == 0]
    color1 = [n for n, c in bicol.items() if c == 1]
    blocks = [Block(color0), Block(color1)]

    # XY interaction
    interaction_matrix = np.zeros((n_angles, n_angles))
    for i, theta_i in enumerate(angles):
        for j, theta_j in enumerate(angles):
            interaction_matrix[i, j] = beta * np.cos(theta_i - theta_j)

    weights = jnp.array([interaction_matrix for _ in range(len(u))])
    coupling_interaction = CategoricalEBMFactor([Block(u), Block(v)], weights)

    spec = BlockGibbsSpec(blocks, [])
    sampler = CategoricalGibbsConditional(n_angles)
    prog = FactorSamplingProgram(spec, [sampler for _ in spec.free_blocks],
                                 [coupling_interaction], [])

    # Initialize and sample
    key = jax.random.key(42)
    init_state = []
    for block in blocks:
        key, subkey = jax.random.split(key, 2)
        init_state.append(jax.random.randint(subkey, (len(block.nodes),),
                                             minval=0, maxval=n_angles, dtype=jnp.uint8))

    schedule = SamplingSchedule(n_warmup=0, n_samples=n_frames, steps_per_sample=1)

    print("  Sampling evolution...")
    key, subkey = jax.random.split(key)
    samples = sample_states(subkey, prog, schedule, init_state, [], [Block(nodes)])
    samples_array = np.array(samples[0])
    grids = samples_array.reshape(n_frames, size, size)

    # Create animation with ARROWS
    print("  Creating enhanced animation with spin arrows...")
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='black')
    ax.set_facecolor('black')
    ax.set_xlim(-0.5, size-0.5)
    ax.set_ylim(-0.5, size-0.5)
    ax.axis('off')
    ax.set_aspect('equal')

    # Convert to HSV coloring
    def grid_to_hsv(grid):
        theta_field = angles[grid]
        hsv = np.zeros((size, size, 3))
        hsv[:, :, 0] = theta_field / (2*np.pi)  # Hue
        hsv[:, :, 1] = 1.0  # Saturation
        hsv[:, :, 2] = 1.0  # Value
        return hsv_to_rgb(hsv)

    # Initial background
    rgb_image = grid_to_hsv(grids[0])
    im = ax.imshow(rgb_image, interpolation='bilinear', extent=[-0.5, size-0.5, size-0.5, -0.5])

    # Arrow subsample (too dense otherwise)
    arrow_step = max(1, size // 24)
    arrow_indices = np.arange(0, size, arrow_step)
    X, Y = np.meshgrid(arrow_indices, arrow_indices)

    # Initialize quiver
    theta_field = angles[grids[0]]
    U = np.cos(theta_field[::arrow_step, ::arrow_step])
    V = np.sin(theta_field[::arrow_step, ::arrow_step])
    quiver = ax.quiver(X, Y, U, -V, color='white', alpha=0.8, width=0.003,
                       headwidth=3, headlength=4, scale=30)

    title = ax.text(0.5, 0.98, f'XY Spin Field: Vortex Dynamics (beta={beta:.1f})',
                   transform=fig.transFigure, ha='center', va='top',
                   color='white', fontsize=18, fontweight='bold')

    frame_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                        color='cyan', fontsize=14, fontfamily='monospace',
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))

    def update(frame):
        # Update background color field
        rgb_image = grid_to_hsv(grids[frame])
        im.set_data(rgb_image)

        # Update arrows
        theta_field = angles[grids[frame]]
        U = np.cos(theta_field[::arrow_step, ::arrow_step])
        V = np.sin(theta_field[::arrow_step, ::arrow_step])
        quiver.set_UVC(U, -V)

        # Compute magnetization
        mx = np.mean(np.cos(theta_field))
        my = np.mean(np.sin(theta_field))
        mag = np.sqrt(mx**2 + my**2)

        # Compute local alignment (how ordered)
        energy = 0
        for i in range(size):
            for j in range(size):
                if j < size-1:
                    energy += np.cos(theta_field[i,j] - theta_field[i,j+1])
                if i < size-1:
                    energy += np.cos(theta_field[i,j] - theta_field[i+1,j])
        energy_per_spin = energy / (2 * size * size)

        frame_text.set_text(f'Step: {frame}/{n_frames}\nMag: {mag:.3f}\nEnergy/N: {energy_per_spin:.3f}')

        return [im, quiver, frame_text]

    anim = animation.FuncAnimation(fig, update, frames=n_frames,
                                  interval=1000/fps, blit=True)

    output_file = f'xy_spin_field_enhanced_{duration}s_beta{beta:.1f}.mp4'
    print(f"  Saving to {output_file}...")
    writer = animation.FFMpegWriter(fps=fps, bitrate=4000, codec='libx264')
    anim.save(output_file, writer=writer, dpi=100)

    print(f"  Done! Saved {output_file}")
    plt.close()

    return output_file


if __name__ == "__main__":
    print("="*80)
    print("ENHANCED XY SPIN FIELD VISUALIZATION")
    print("="*80)
    print("\nShowing spin directions with ARROWS + color field!")
    print("Lower beta = MORE vortices and chaos!")
    print("="*80)

    # Create enhanced video with MORE vortices (lower beta)
    video = create_xy_spin_field_video(size=48, n_angles=16, beta=1.2, fps=30, duration=15)

    print("\n" + "="*80)
    print("ENHANCED XY SPIN FIELD COMPLETE!")
    print("="*80)
    print(f"\nGenerated: {video}")
    print("\n15 seconds of mesmerizing spin field dynamics!")
    print("Watch the arrows dance as vortices form and annihilate!")
    print("="*80)
