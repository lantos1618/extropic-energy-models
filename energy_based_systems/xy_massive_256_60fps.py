#!/usr/bin/env python3
"""
MASSIVE XY Model - 256x256 grid @ 60fps
Ultimate resolution spin field with buttery smooth playback!
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import hsv_to_rgb
import networkx as nx
import os
from pathlib import Path

from thrml.block_management import Block
from thrml.block_sampling import BlockGibbsSpec, sample_states, SamplingSchedule
from thrml.models.discrete_ebm import CategoricalEBMFactor, CategoricalGibbsConditional
from thrml.factor import FactorSamplingProgram
from thrml.pgm import CategoricalNode


def create_xy_massive_video(size=256, n_angles=16, beta=1.2, fps=60, duration=20):
    """
    Create MASSIVE 256x256 XY model video at 60fps.

    256x256 = 65,536 spins!
    60fps = ultra smooth
    More frames = slower evolution (easier to watch)
    """
    print(f"\nCreating MASSIVE XY model video ({duration}s @ {fps}fps)...")
    print(f"  Grid size: {size}x{size} = {size*size:,} spins!!!")
    print(f"  Beta: {beta}")
    print(f"  Total frames: {fps * duration}")
    print(f"  This is HUGE - may take a few minutes...")

    n_frames = fps * duration
    angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)

    # Create grid
    print("  Building massive graph...")
    G = nx.grid_graph(dim=(size, size))
    coord_to_node = {coord: CategoricalNode() for coord in G.nodes}
    nx.relabel_nodes(G, coord_to_node, copy=False)
    nodes = list(G.nodes)

    u, v = map(list, zip(*G.edges()))

    # Bipartite coloring
    print("  Computing bipartite coloring...")
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

    print(f"  Sampling {n_frames} frames...")
    print(f"  (Processing {size*size:,} spins per frame - this will take a while!)")
    key, subkey = jax.random.split(key)
    samples = sample_states(subkey, prog, schedule, init_state, [], [Block(nodes)])
    samples_array = np.array(samples[0])
    grids = samples_array.reshape(n_frames, size, size)

    # Create animation with reasonable canvas size
    print("  Creating 60fps animation...")
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='black')
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

    title = ax.text(0.5, 0.98, f'XY Model: 256x256 @ 60fps (beta={beta:.1f})',
                   transform=fig.transFigure, ha='center', va='top',
                   color='white', fontsize=18, fontweight='bold')

    frame_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                        color='cyan', fontsize=13, fontfamily='monospace',
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

        frame_text.set_text(f'Frame: {frame}/{n_frames}\nMag: {mag:.3f}\nSpins: {size*size:,}')

        return [im, frame_text]

    anim = animation.FuncAnimation(fig, update, frames=n_frames,
                                  interval=1000/fps, blit=True)

    # Ensure output directory exists
    output_dir = Path(__file__).parent.parent / 'outputs' / 'energy_based'
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f'xy_massive_256x256_60fps_{duration}s.mp4'
    print(f"  Encoding to {output_file}...")
    print(f"  (Encoding {n_frames} frames at 60fps with HIGH quality...)")
    # Use H.265 with very high quality settings
    writer = animation.FFMpegWriter(
        fps=fps,
        codec='libx265',
        bitrate=-1,  # Let CRF control quality
        extra_args=['-crf', '15', '-preset', 'slow', '-pix_fmt', 'yuv420p']
    )
    anim.save(output_file, writer=writer, dpi=150)

    print(f"  Done! Saved {output_file}")
    plt.close()

    return output_file


if __name__ == "__main__":
    print("="*80)
    print("MASSIVE XY MODEL - 256x256 @ 60fps")
    print("="*80)
    print("\n65,536 spins evolving in buttery smooth 60fps!")
    print("This is going to be EPIC but will take several minutes...")
    print("="*80)

    video = create_xy_massive_video(size=256, n_angles=16, beta=1.2, fps=60, duration=20)

    print("\n" + "="*80)
    print("MASSIVE VIDEO COMPLETE!")
    print("="*80)
    print(f"\nGenerated: {video}")
    print("\n256x256 spins @ 60fps = ULTIMATE spin field visualization!")
    print("="*80)
