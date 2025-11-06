#!/usr/bin/env python3
"""
Energy Evolution Videos: Watch Systems Find Equilibrium!

Creates 3-5 second videos showing:
- Random initial state
- Energy minimization dynamics
- Convergence to equilibrium
- Domain formation emerging

This is what energy-based computing LOOKS like in action!
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap, hsv_to_rgb
import networkx as nx

from thrml.block_management import Block
from thrml.block_sampling import BlockGibbsSpec, sample_states, SamplingSchedule
from thrml.models.discrete_ebm import CategoricalEBMFactor, CategoricalGibbsConditional
from thrml.factor import FactorSamplingProgram
from thrml.pgm import CategoricalNode
from thrml.models.ising import IsingEBM, IsingSamplingProgram, hinton_init


def create_ising_evolution_video(size=64, beta=2.0, fps=30, duration=4):
    """
    Create video of Ising model finding equilibrium.

    Shows: Random → Clusters → Domains → Ordered
    """
    print(f"\nCreating Ising evolution video ({duration}s @ {fps}fps)...")

    n_frames = fps * duration

    # Create Ising model
    graph = nx.grid_2d_graph(size, size)
    coord_to_node = {coord: SpinNode() for coord in graph.nodes}
    nx.relabel_nodes(graph, coord_to_node, copy=False)
    nodes = list(graph.nodes)

    biases = jnp.zeros(len(nodes))
    edges = list(graph.edges)
    weights = jnp.ones(len(edges)) * 0.5
    beta_param = jnp.array(beta)

    model = IsingEBM(nodes, edges, biases, weights, beta_param)

    # Coloring for block Gibbs
    coloring = nx.coloring.greedy_color(graph, strategy="DSATUR")
    n_colors = max(coloring.values()) + 1
    color_groups = [[] for _ in range(n_colors)]
    for node in graph.nodes:
        color_groups[coloring[node]].append(node)

    free_blocks = [Block(group) for group in color_groups]
    program = IsingSamplingProgram(model, free_blocks, clamped_blocks=[])

    # Sample with recording
    key = jax.random.key(42)
    k_init, k_samp = jax.random.split(key, 2)
    init_state = hinton_init(k_init, model, free_blocks, ())

    # Sample MORE steps to capture evolution
    schedule = SamplingSchedule(
        n_warmup=0,
        n_samples=n_frames,
        steps_per_sample=1  # Record every step!
    )

    print("  Sampling evolution...")
    samples = sample_states(k_samp, program, schedule, init_state, [], [Block(nodes)])
    samples_array = np.array(samples[0])

    # Reshape to grid
    grids = samples_array.reshape(n_frames, size, size)

    # Create animation
    print("  Creating animation...")
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='black')
    ax.set_facecolor('black')
    ax.axis('off')

    # Convert binary to -1/+1 for display
    grid_display = grids[0].astype(float) * 2 - 1
    im = ax.imshow(grid_display, cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest')

    title = ax.text(0.5, 0.98, f'Ising Model: Finding Equilibrium (β={beta:.1f})',
                   transform=fig.transFigure, ha='center', va='top',
                   color='white', fontsize=16, fontweight='bold')

    frame_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                        color='cyan', fontsize=12, fontfamily='monospace',
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    def update(frame):
        grid_display = grids[frame].astype(float) * 2 - 1
        im.set_data(grid_display)

        # Compute magnetization
        mag = np.abs(np.mean(grid_display))

        # Compute energy (approximate)
        energy = 0
        for i in range(size):
            for j in range(size):
                if j < size-1:
                    energy -= grid_display[i,j] * grid_display[i,j+1]
                if i < size-1:
                    energy -= grid_display[i,j] * grid_display[i+1,j]
        energy_per_spin = energy / (size * size)

        frame_text.set_text(f'Step: {frame}\nMag: {mag:.3f}\nE/N: {energy_per_spin:.3f}')

        return [im, frame_text]

    anim = animation.FuncAnimation(fig, update, frames=n_frames,
                                  interval=1000/fps, blit=True)

    output_file = f'ising_evolution_{duration}s.mp4'
    print(f"  Saving to {output_file}...")
    writer = animation.FFMpegWriter(fps=fps, bitrate=3000, codec='libx264')
    anim.save(output_file, writer=writer, dpi=100)

    print(f"  Done! Saved {output_file}")
    plt.close()

    return output_file


def create_potts_evolution_video(size=48, n_states=5, beta=2.0, fps=30, duration=4):
    """
    Create video of Potts model finding equilibrium.

    Shows: Random colors → Clusters → Domains → One color dominates
    """
    print(f"\nCreating Potts evolution video ({duration}s @ {fps}fps)...")

    n_frames = fps * duration

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

    # Potts interaction
    id_mat = jnp.eye(n_states)
    weights = beta * jnp.broadcast_to(jnp.expand_dims(id_mat, 0), (len(u), *id_mat.shape))

    coupling_interaction = CategoricalEBMFactor([Block(u), Block(v)], weights)

    spec = BlockGibbsSpec(blocks, [])
    sampler = CategoricalGibbsConditional(n_states)
    prog = FactorSamplingProgram(spec, [sampler for _ in spec.free_blocks],
                                 [coupling_interaction], [])

    # Initialize and sample
    key = jax.random.key(42)
    init_state = []
    for block in blocks:
        key, subkey = jax.random.split(key, 2)
        init_state.append(jax.random.randint(subkey, (len(block.nodes),),
                                             minval=0, maxval=n_states, dtype=jnp.uint8))

    schedule = SamplingSchedule(n_warmup=0, n_samples=n_frames, steps_per_sample=1)

    print("  Sampling evolution...")
    key, subkey = jax.random.split(key)
    samples = sample_states(subkey, prog, schedule, init_state, [], [Block(nodes)])
    samples_array = np.array(samples[0])
    grids = samples_array.reshape(n_frames, size, size)

    # Create animation
    print("  Creating animation...")
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='black')
    ax.set_facecolor('black')
    ax.axis('off')

    # Color map
    colors = plt.cm.tab10(np.linspace(0, 1, n_states))
    cmap = ListedColormap(colors)

    im = ax.imshow(grids[0], cmap=cmap, vmin=0, vmax=n_states-1, interpolation='nearest')

    title = ax.text(0.5, 0.98, f'Potts Model: Domain Formation (q={n_states}, β={beta:.1f})',
                   transform=fig.transFigure, ha='center', va='top',
                   color='white', fontsize=16, fontweight='bold')

    frame_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                        color='cyan', fontsize=12, fontfamily='monospace',
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    def update(frame):
        im.set_data(grids[frame])

        # Compute order parameter (fraction in most common state)
        counts = np.bincount(grids[frame].flatten(), minlength=n_states)
        magnetization = counts.max() / (size * size)

        frame_text.set_text(f'Step: {frame}\nOrder: {magnetization:.3f}')

        return [im, frame_text]

    anim = animation.FuncAnimation(fig, update, frames=n_frames,
                                  interval=1000/fps, blit=True)

    output_file = f'potts_evolution_{duration}s.mp4'
    print(f"  Saving to {output_file}...")
    writer = animation.FFMpegWriter(fps=fps, bitrate=3000, codec='libx264')
    anim.save(output_file, writer=writer, dpi=100)

    print(f"  Done! Saved {output_file}")
    plt.close()

    return output_file


def create_xy_evolution_video(size=48, n_angles=16, beta=1.5, fps=30, duration=4):
    """
    Create video of XY model finding equilibrium.

    Shows: Random spins → Vortices form → Vortices annihilate → Aligned
    """
    print(f"\nCreating XY evolution video ({duration}s @ {fps}fps)...")

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

    # Create animation
    print("  Creating animation...")
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='black')
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
    im = ax.imshow(rgb_image, interpolation='nearest')

    title = ax.text(0.5, 0.98, f'XY Model: Vortex Dynamics (β={beta:.1f})',
                   transform=fig.transFigure, ha='center', va='top',
                   color='white', fontsize=16, fontweight='bold')

    frame_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                        color='cyan', fontsize=12, fontfamily='monospace',
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    def update(frame):
        rgb_image = grid_to_hsv(grids[frame])
        im.set_data(rgb_image)

        # Compute magnetization
        theta_field = angles[grids[frame]]
        mx = np.mean(np.cos(theta_field))
        my = np.mean(np.sin(theta_field))
        mag = np.sqrt(mx**2 + my**2)

        frame_text.set_text(f'Step: {frame}\nMag: {mag:.3f}')

        return [im, frame_text]

    anim = animation.FuncAnimation(fig, update, frames=n_frames,
                                  interval=1000/fps, blit=True)

    output_file = f'xy_evolution_{duration}s.mp4'
    print(f"  Saving to {output_file}...")
    writer = animation.FFMpegWriter(fps=fps, bitrate=3000, codec='libx264')
    anim.save(output_file, writer=writer, dpi=100)

    print(f"  Done! Saved {output_file}")
    plt.close()

    return output_file


if __name__ == "__main__":
    import sys

    print("="*80)
    print("🎬 CREATING EVOLUTION VIDEOS")
    print("="*80)

    # Import here to avoid issues
    from thrml.pgm import SpinNode

    # Check if we want just the 15-second XY model
    if len(sys.argv) > 1 and sys.argv[1] == "xy_15s":
        print("\nCreating EXTENDED 15-second XY model video!")
        print("Watch vortices dance for longer! 🌀")
        print("="*80)

        video = create_xy_evolution_video(size=48, n_angles=16, beta=1.5, fps=30, duration=15)

        print("\n" + "="*80)
        print("🎉 EXTENDED XY VIDEO COMPLETE!")
        print("="*80)
        print(f"\nGenerated: {video}")
        print("\n15 seconds of beautiful vortex dynamics!")
        print("="*80)
    else:
        print("\nShowing systems finding equilibrium from random initial states!")
        print("Duration: 3-5 seconds @ 30fps")
        print("="*80)

        # Create all three videos
        videos = []

        # Ising: 4 seconds
        videos.append(create_ising_evolution_video(size=64, beta=2.0, fps=30, duration=4))

        # Potts: 4 seconds
        videos.append(create_potts_evolution_video(size=48, n_states=5, beta=2.0, fps=30, duration=4))

        # XY: 5 seconds
        videos.append(create_xy_evolution_video(size=48, n_angles=16, beta=1.5, fps=30, duration=5))

        print("\n" + "="*80)
        print("🎉 ALL EVOLUTION VIDEOS COMPLETE!")
        print("="*80)
        print("\nGenerated:")
        for v in videos:
            print(f"  • {v}")
        print("\nThese show REAL energy-based dynamics in action!")
        print("Watch systems evolve from chaos → order!")
        print("="*80)
