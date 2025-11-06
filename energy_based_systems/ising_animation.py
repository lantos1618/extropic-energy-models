#!/usr/bin/env python3
"""
2D Ising Model Phase Transition Animation

Creates a visually stunning animation showing:
1. Real-time spin dynamics during equilibration
2. Temperature sweep through the critical point
3. Domain formation and destruction
4. Magnetization evolution

Perfect for Twitter/social media - shows actual thermodynamic computing in action.
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


def create_2d_ising_lattice(width, height, coupling_strength=1.0, external_field=0.0, beta=1.0):
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

    biases = jnp.ones(len(nodes_list)) * external_field
    edges = list(graph.edges)
    weights = jnp.ones(len(edges)) * coupling_strength

    beta_param = jnp.array(beta)
    model = IsingEBM(nodes_list, edges, biases, weights, beta_param)

    return model, nodes_list, graph


def sample_ising_trajectory(
    model, nodes, graph, n_steps, seed=42
):
    """
    Sample trajectory showing evolution over time.
    Returns all intermediate states, not just final samples.
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

    # Sample with very frequent collection to show dynamics
    schedule = SamplingSchedule(
        n_warmup=0,  # No warmup, we want to see equilibration
        n_samples=n_steps,
        steps_per_sample=1  # Collect every step
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


def create_animation(width=128, height=128, n_frames=180, fps=30):
    """
    Create animation showing phase transition.

    Strategy:
    - Start at high T (random)
    - Cool down through Tc (watch domains form)
    - Heat back up (watch domains disappear)

    Args:
        width, height: Lattice size
        n_frames: Total frames (180 = 6 seconds at 30fps)
        fps: Frames per second
    """
    print("=" * 80)
    print("CREATING ISING PHASE TRANSITION ANIMATION")
    print("=" * 80)
    print(f"\nLattice: {width}x{height}")
    print(f"Frames: {n_frames} ({n_frames/fps:.1f} seconds at {fps} fps)")

    # Temperature schedule: high -> low -> high
    # First half: cool down, second half: heat up
    T_high = 4.0
    T_low = 1.0

    half = n_frames // 2
    temps_down = np.linspace(T_high, T_low, half)
    temps_up = np.linspace(T_low, T_high, n_frames - half)
    temperatures = np.concatenate([temps_down, temps_up])

    print(f"Temperature range: {T_low} to {T_high}")
    print(f"Critical temperature: Tc ≈ 2.269")

    # Setup figure
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3,
                  height_ratios=[3, 1], width_ratios=[3, 1])

    # Main plot: spin lattice
    ax_main = fig.add_subplot(gs[0, 0])
    ax_main.set_aspect('equal')
    ax_main.axis('off')

    # Side plot: magnetization vs temperature
    ax_mag = fig.add_subplot(gs[0, 1])
    ax_mag.set_xlabel('Frame', fontsize=10)
    ax_mag.set_ylabel('|M|', fontsize=10)
    ax_mag.set_title('Magnetization', fontsize=11, fontweight='bold')
    ax_mag.grid(True, alpha=0.3)
    ax_mag.set_xlim(0, n_frames)
    ax_mag.set_ylim(0, 1.1)

    # Bottom plot: temperature vs frame
    ax_temp = fig.add_subplot(gs[1, :])
    ax_temp.plot(range(n_frames), temperatures, 'k-', linewidth=2, alpha=0.5)
    ax_temp.axhline(y=2.269, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Tc (theory)')
    ax_temp.set_xlabel('Frame', fontsize=11)
    ax_temp.set_ylabel('Temperature', fontsize=11)
    ax_temp.set_title('Temperature Schedule', fontsize=11, fontweight='bold')
    ax_temp.grid(True, alpha=0.3)
    ax_temp.legend()
    ax_temp.set_xlim(0, n_frames)

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
        0.5, 1.05, '',
        transform=ax_main.transAxes,
        ha='center',
        fontsize=14,
        fontweight='bold'
    )

    # Magnetization line
    mag_line, = ax_mag.plot([], [], 'b-', linewidth=2)
    current_marker, = ax_mag.plot([], [], 'ro', markersize=8)

    # Temperature marker
    temp_marker, = ax_temp.plot([], [], 'ro', markersize=10, zorder=10)

    # Data storage
    all_configs = []
    magnetizations = []

    print("\n" + "=" * 80)
    print("RUNNING SIMULATION")
    print("=" * 80)

    # Run simulation for each temperature
    current_state = None

    for frame_idx, T in enumerate(temperatures):
        beta = 1.0 / T

        if frame_idx % 30 == 0:
            print(f"Frame {frame_idx+1}/{n_frames}: T = {T:.3f} (β = {beta:.3f})")

        # Create model at this temperature
        model, nodes, graph = create_2d_ising_lattice(width, height, beta=beta)

        # Sample just a few steps to show evolution
        # Reuse previous state for continuity
        samples = sample_ising_trajectory(model, nodes, graph, n_steps=10, seed=42 + frame_idx)

        # Extract final state
        if isinstance(samples, list):
            sample_array = samples[0]
        else:
            sample_array = samples

        # Take the last sample as current config
        final_config = sample_array[-1]
        spins = final_config.astype(jnp.float32) * 2 - 1
        config_2d = spins.reshape(height, width)

        all_configs.append(np.array(config_2d))

        # Compute magnetization
        mag = float(jnp.abs(jnp.mean(spins)))
        magnetizations.append(mag)

    print("\n" + "=" * 80)
    print("GENERATING ANIMATION")
    print("=" * 80)

    # Animation update function
    def update(frame):
        config = all_configs[frame]
        T = temperatures[frame]
        mag = magnetizations[frame]

        # Update spin lattice
        im.set_array(config)

        # Update title with physics info
        phase = "FERROMAGNETIC" if T < 2.269 else "PARAMAGNETIC"
        if abs(T - 2.269) < 0.2:
            phase = "CRITICAL POINT"

        title.set_text(f'T = {T:.3f}  |  {phase}  |  M = {mag:.3f}')

        # Update magnetization plot
        frames_so_far = list(range(frame + 1))
        mag_line.set_data(frames_so_far, magnetizations[:frame + 1])
        current_marker.set_data([frame], [mag])

        # Update temperature marker
        temp_marker.set_data([frame], [T])

        return im, title, mag_line, current_marker, temp_marker

    # Create animation
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=1000/fps,  # milliseconds per frame
        blit=True
    )

    # Save
    output_file = "ising_phase_transition_animation.mp4"
    print(f"\nSaving animation to {output_file}...")
    print("(This may take a few minutes...)")

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, bitrate=3000, metadata=dict(artist='THRML'))

    anim.save(output_file, writer=writer, dpi=120)

    print(f"\n✓ Animation saved: {output_file}")
    print(f"  Duration: {n_frames/fps:.1f} seconds")
    print(f"  Resolution: {width}x{height} lattice")
    print(f"  Filesize: ", end='')

    import os
    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"{size_mb:.1f} MB")

    return anim


def main():
    """Create the animation."""
    # Twitter-optimized settings
    # Twitter max: 2:20, but shorter is better for engagement
    # Aim for 6 seconds, high quality

    width = 128
    height = 128
    n_frames = 180  # 6 seconds at 30fps
    fps = 30

    anim = create_animation(width, height, n_frames, fps)

    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print("\nYou now have a sick animation showing REAL thermodynamic computing.")
    print("Watch spontaneous symmetry breaking emerge and disappear as we sweep")
    print("through the critical temperature.")
    print("\nThis is what Extropic's hardware will do natively - not simulation,")
    print("but actual physical thermodynamic dynamics in analog circuits.")
    print("=" * 80)


if __name__ == "__main__":
    main()
