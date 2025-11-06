#!/usr/bin/env python3
"""
HIGH RESOLUTION XY Model - 20 seconds
Bigger grid, higher quality for the ultimate spin field visualization!
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


def create_xy_hires_video(size=64, n_angles=16, beta=1.2, fps=30, duration=20):
    """
    Create HIGH RESOLUTION 20-second XY model video.

    size=64 gives 4x more spins than size=32!
    """
    print(f"\nCreating HIGH-RES XY model video ({duration}s @ {fps}fps)...")
    print(f"  Grid size: {size}x{size} = {size*size} spins!")
    print(f"  Beta: {beta}")
    print(f"  Total frames: {fps * duration}")

    n_frames = fps * duration
    angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)

    # Create grid
    print("  Building graph...")
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
    print("  Setting up interactions...")
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
    print("  Initializing random state...")
    key = jax.random.key(42)
    init_state = []
    for block in blocks:
        key, subkey = jax.random.split(key, 2)
        init_state.append(jax.random.randint(subkey, (len(block.nodes),),
                                             minval=0, maxval=n_angles, dtype=jnp.uint8))

    schedule = SamplingSchedule(n_warmup=0, n_samples=n_frames, steps_per_sample=1)

    print(f"  Sampling {n_frames} frames (this may take a minute)...")
    key, subkey = jax.random.split(key)
    samples = sample_states(subkey, prog, schedule, init_state, [], [Block(nodes)])
    samples_array = np.array(samples[0])
    grids = samples_array.reshape(n_frames, size, size)

    # Create HIGH-RES animation
    print("  Creating high-resolution animation...")
    fig, ax = plt.subplots(figsize=(12, 12), facecolor='black')
    ax.set_facecolor('black')
    ax.axis('off')

    # Convert to HSV coloring
    def grid_to_hsv(grid):
        theta_field = angles[grid]
        hsv = np.zeros((size, size, 3))
        hsv[:, :, 0] = theta_field / (2*np.pi)  # Hue
        hsv[:, :, 1] = 1.0  # Saturation
        hsv[:, :, 2] = 1.0  # Value
        return hsv_to_rgb(hsv)

    rgb_image = grid_to_hsv(grids[0])
    im = ax.imshow(rgb_image, interpolation='bilinear')

    title = ax.text(0.5, 0.98, f'XY Model: Vortex Dynamics (beta={beta:.1f})',
                   transform=fig.transFigure, ha='center', va='top',
                   color='white', fontsize=20, fontweight='bold')

    frame_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                        color='cyan', fontsize=14, fontfamily='monospace',
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))

    def update(frame):
        rgb_image = grid_to_hsv(grids[frame])
        im.set_data(rgb_image)

        # Compute magnetization
        theta_field = angles[grids[frame]]
        mx = np.mean(np.cos(theta_field))
        my = np.mean(np.sin(theta_field))
        mag = np.sqrt(mx**2 + my**2)

        # Energy
        energy = 0
        for i in range(size):
            for j in range(size):
                if j < size-1:
                    energy += np.cos(theta_field[i,j] - theta_field[i,j+1])
                if i < size-1:
                    energy += np.cos(theta_field[i,j] - theta_field[i+1,j])
        energy_per_spin = energy / (2 * size * size)

        frame_text.set_text(f'Step: {frame}/{n_frames}\nMag: {mag:.3f}\nE/N: {energy_per_spin:.3f}')

        return [im, frame_text]

    anim = animation.FuncAnimation(fig, update, frames=n_frames,
                                  interval=1000/fps, blit=True)

    output_file = f'xy_hires_{size}x{size}_{duration}s.mp4'
    print(f"  Encoding video to {output_file}...")
    writer = animation.FFMpegWriter(fps=fps, bitrate=5000, codec='libx264')
    anim.save(output_file, writer=writer, dpi=150)  # Higher DPI!

    print(f"  Done! Saved {output_file}")
    plt.close()

    return output_file


if __name__ == "__main__":
    print("="*80)
    print("HIGH-RESOLUTION XY MODEL - 20 SECONDS")
    print("="*80)

    video = create_xy_hires_video(size=64, n_angles=16, beta=1.2, fps=30, duration=20)

    print("\n" + "="*80)
    print("HIGH-RES VIDEO COMPLETE!")
    print("="*80)
    print(f"\nGenerated: {video}")
    print("\n20 seconds of high-resolution vortex dynamics!")
    print("="*80)
