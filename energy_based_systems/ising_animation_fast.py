#!/usr/bin/env python3
"""
FAST 2D Ising Model Phase Transition Animation

Optimized version:
- Smaller lattice (64x64 instead of 128x128)
- Fewer frames (60 instead of 180)
- Continuous sampling at one temperature showing equilibration
- Much faster, still looks great for Twitter
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import networkx as nx
import numpy as np

from thrml.block_management import Block
from thrml.block_sampling import sample_states, SamplingSchedule
from thrml.models.ising import (
    hinton_init,
    IsingEBM,
    IsingSamplingProgram,
)
from thrml.pgm import SpinNode


def create_2d_ising_lattice(width, height, coupling_strength=1.0, beta=1.0):
    """Create a 2D Ising model on a square lattice with periodic boundaries."""
    graph = nx.grid_2d_graph(height, width, periodic=True)

    coord_to_node = {}
    nodes_list = []

    for i in range(height):
        for j in range(width):
            node = SpinNode()
            coord_to_node[(i, j)] = node
            nodes_list.append(node)

    nx.relabel_nodes(graph, coord_to_node, copy=False)

    biases = jnp.zeros(len(nodes_list))  # No external field
    edges = list(graph.edges)
    weights = jnp.ones(len(edges)) * coupling_strength

    beta_param = jnp.array(beta)
    model = IsingEBM(nodes_list, edges, biases, weights, beta_param)

    return model, nodes_list, graph


def sample_continuous_trajectory(model, nodes, graph, n_frames, seed=42):
    """
    Sample a continuous trajectory to show real-time dynamics.
    Much faster than recreating model at each temperature.
    """
    # Graph coloring
    coloring = nx.coloring.greedy_color(graph, strategy="DSATUR")
    n_colors = max(coloring.values()) + 1

    color_groups = [[] for _ in range(n_colors)]
    for node in graph.nodes:
        color_groups[coloring[node]].append(node)

    free_blocks = [Block(group) for group in color_groups]
    program = IsingSamplingProgram(model, free_blocks, clamped_blocks=[])

    # Initialize
    key = jax.random.key(seed)
    k_init, k_samp = jax.random.split(key, 2)
    init_state = hinton_init(k_init, model, free_blocks, ())

    # Sample trajectory - collect every few steps
    schedule = SamplingSchedule(
        n_warmup=0,
        n_samples=n_frames,
        steps_per_sample=5  # Advance 5 steps between frames
    )

    samples = sample_states(
        k_samp,
        program,
        schedule,
        init_state,
        [],
        [Block(nodes)]
    )

    return samples


def create_fast_animation(width=64, height=64, n_frames=60, fps=30):
    """
    Create fast animation showing phase transition.

    Strategy: Show 3 temperatures sequentially
    - High T (random) for 1 second
    - Critical T (fluctuating) for 2 seconds
    - Low T (ordered) for 1 second

    Total: 4 seconds, 120 frames
    """
    print("=" * 80)
    print("CREATING FAST ISING PHASE TRANSITION ANIMATION")
    print("=" * 80)
    print(f"\nLattice: {width}x{height}")
    print(f"Total frames: {n_frames*3} ({n_frames*3/fps:.1f} seconds at {fps} fps)")

    temperatures = [4.0, 2.3, 1.2]  # High, Critical, Low
    phase_names = ["PARAMAGNETIC (Disordered)", "CRITICAL POINT (Fluctuating)", "FERROMAGNETIC (Ordered)"]

    print(f"Temperatures: {temperatures}")

    # Setup figure
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3,
                  height_ratios=[3, 1])

    # Main plot: spin lattice
    ax_main = fig.add_subplot(gs[0, :])
    ax_main.set_aspect('equal')
    ax_main.axis('off')

    # Bottom left: magnetization over time
    ax_mag = fig.add_subplot(gs[1, 0])
    ax_mag.set_xlabel('Frame', fontsize=10)
    ax_mag.set_ylabel('|M|', fontsize=10)
    ax_mag.set_title('Magnetization', fontsize=11, fontweight='bold')
    ax_mag.grid(True, alpha=0.3)
    ax_mag.set_xlim(0, n_frames * 3)
    ax_mag.set_ylim(0, 1.1)

    # Bottom right: current phase info
    ax_info = fig.add_subplot(gs[1, 1])
    ax_info.axis('off')

    # Initialize empty image
    im = ax_main.imshow(
        np.zeros((height, width)),
        cmap='RdBu_r',
        vmin=-1,
        vmax=1,
        interpolation='nearest',
        origin='lower'
    )

    title = ax_main.text(
        0.5, 1.02, '',
        transform=ax_main.transAxes,
        ha='center',
        fontsize=16,
        fontweight='bold'
    )

    # Magnetization line
    mag_line, = ax_mag.plot([], [], 'b-', linewidth=2)
    current_marker, = ax_mag.plot([], [], 'ro', markersize=8)

    # Info text
    info_text = ax_info.text(
        0.5, 0.5, '',
        transform=ax_info.transAxes,
        ha='center',
        va='center',
        fontsize=12,
        fontfamily='monospace'
    )

    print("\n" + "=" * 80)
    print("RUNNING SIMULATIONS")
    print("=" * 80)

    all_configs = []
    magnetizations = []
    temp_labels = []

    # Run simulation for each temperature
    for temp_idx, (T, phase_name) in enumerate(zip(temperatures, phase_names)):
        beta = 1.0 / T
        print(f"\n[{temp_idx+1}/3] {phase_name}")
        print(f"  Temperature T = {T:.1f} (β = {beta:.3f})")
        print(f"  Computing {n_frames} frames...")

        # Create model
        model, nodes, graph = create_2d_ising_lattice(width, height, beta=beta)

        # Sample trajectory
        samples = sample_continuous_trajectory(model, nodes, graph, n_frames, seed=42 + temp_idx)

        # Extract configs
        if isinstance(samples, list):
            sample_array = samples[0]
        else:
            sample_array = samples

        print(f"  Generated {sample_array.shape[0]} samples")

        # Process each frame
        for frame_idx in range(sample_array.shape[0]):
            config = sample_array[frame_idx]
            spins = config.astype(jnp.float32) * 2 - 1
            config_2d = spins.reshape(height, width)

            all_configs.append(np.array(config_2d))

            mag = float(jnp.abs(jnp.mean(spins)))
            magnetizations.append(mag)
            temp_labels.append(f"T={T:.1f}")

    total_frames = len(all_configs)
    print(f"\n✓ Total frames generated: {total_frames}")

    print("\n" + "=" * 80)
    print("GENERATING ANIMATION")
    print("=" * 80)

    # Animation update function
    def update(frame):
        config = all_configs[frame]
        mag = magnetizations[frame]
        temp_label = temp_labels[frame]

        # Determine which phase we're in
        phase_idx = frame // n_frames
        if phase_idx >= len(phase_names):
            phase_idx = len(phase_names) - 1
        phase_name = phase_names[phase_idx]
        T = temperatures[phase_idx]

        # Update spin lattice
        im.set_array(config)

        # Update title
        title.set_text(f'{phase_name}  |  T = {T:.1f}  |  M = {mag:.3f}')

        # Update magnetization plot
        frames_so_far = list(range(frame + 1))
        mag_line.set_data(frames_so_far, magnetizations[:frame + 1])
        current_marker.set_data([frame], [mag])

        # Update info
        info_lines = [
            f"Frame: {frame+1}/{total_frames}",
            f"",
            f"Temperature: {T:.1f}",
            f"Phase: {phase_name.split('(')[0].strip()}",
            f"",
            f"Magnetization: {mag:.3f}",
        ]
        info_text.set_text('\n'.join(info_lines))

        return im, title, mag_line, current_marker, info_text

    # Create animation
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=total_frames,
        interval=1000/fps,
        blit=True
    )

    # Save
    output_file = "ising_phase_transition.mp4"
    print(f"\nSaving animation to {output_file}...")

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, bitrate=2000, metadata=dict(artist='THRML'))

    anim.save(output_file, writer=writer, dpi=100)

    print(f"\n✓ Animation saved: {output_file}")

    import os
    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"  Duration: {total_frames/fps:.1f} seconds")
    print(f"  Resolution: {width}x{height} lattice")
    print(f"  Filesize: {size_mb:.1f} MB")

    return anim


def main():
    """Create fast animation for Twitter."""
    width = 64
    height = 64
    n_frames = 60  # 60 frames per temperature = 2 seconds each at 30fps
    fps = 30

    anim = create_fast_animation(width, height, n_frames, fps)

    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print("\nAnimation shows:")
    print("  1. High T (4.0): Random paramagnetic phase")
    print("  2. Critical T (2.3): Fluctuations at phase transition")
    print("  3. Low T (1.2): Ordered ferromagnetic domains")
    print("\nThis is REAL thermodynamic computing - watching spontaneous")
    print("symmetry breaking happen in real-time!")
    print("=" * 80)


if __name__ == "__main__":
    main()
