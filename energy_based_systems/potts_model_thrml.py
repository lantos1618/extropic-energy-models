#!/usr/bin/env python3
"""
Potts Model with THRML: ACTUAL Energy-Based Computing

This is what THRML is designed for - not Mandelbrot!

The q-state Potts model is a generalization of Ising (q=2).
It exhibits rich phase transitions with domain formation.

Energy: H = -J Σ_{<i,j>} δ(s_i, s_j)
where δ(a,b) = 1 if a=b, 0 otherwise

At low temperature: Ordered domains (one state dominates)
At high temperature: Disordered (all states mixed)
Critical point: Phase transition with fluctuations

This is LEGITIMATE energy-based computing:
- Energy IS the problem definition
- THRML samples low-energy states
- No pre-computation or circular logic
"""

import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap
from typing import List

from thrml.block_management import Block
from thrml.block_sampling import BlockGibbsSpec, sample_states, SamplingSchedule
from thrml.models.discrete_ebm import CategoricalEBMFactor, CategoricalGibbsConditional
from thrml.factor import FactorSamplingProgram
from thrml.pgm import CategoricalNode


def create_potts_model(grid_size: int, n_states: int, beta: float):
    """
    Create a q-state Potts model on a 2D grid.

    Args:
        grid_size: Side length of square grid
        n_states: Number of states (q) - like colors in graph coloring
        beta: Inverse temperature (β = 1/T)

    Returns:
        prog: Sampling program
        nodes: List of nodes
        grid_graph: NetworkX graph
        blocks: Free blocks for sampling
    """
    print(f"Creating {grid_size}x{grid_size} Potts model with q={n_states} states, β={beta:.3f}")

    # Create 2D grid graph
    G = nx.grid_graph(dim=(grid_size, grid_size))

    # Create categorical nodes
    coord_to_node = {coord: CategoricalNode() for coord in G.nodes}
    nx.relabel_nodes(G, coord_to_node, copy=False)
    nodes = list(G.nodes)

    # Get edges
    u, v = map(list, zip(*G.edges()))

    # Bipartite coloring for efficient block Gibbs
    bicol = nx.bipartite.color(G)
    color0 = [n for n, c in bicol.items() if c == 0]
    color1 = [n for n, c in bicol.items() if c == 1]
    blocks = [Block(color0), Block(color1)]

    # Potts interaction: Energy = -β Σ δ(s_i, s_j)
    # Represented as identity matrix (reward same states)
    id_mat = jnp.eye(n_states)
    weights = beta * jnp.broadcast_to(
        jnp.expand_dims(id_mat, 0),
        (len(u), *id_mat.shape)
    )

    coupling_interaction = CategoricalEBMFactor([Block(u), Block(v)], weights)

    # Create sampling program
    spec = BlockGibbsSpec(blocks, [])
    sampler = CategoricalGibbsConditional(n_states)
    prog = FactorSamplingProgram(
        spec,
        [sampler for _ in spec.free_blocks],
        [coupling_interaction],
        []
    )

    return prog, nodes, G, blocks


def run_potts_sampling(prog, blocks, nodes, grid_size, n_states, n_chains=10,
                       n_warmup=500, n_samples=1000, seed=42):
    """
    Sample from Potts model using THRML block Gibbs sampling.

    This is REAL energy-based computation:
    - System explores state space
    - Settles into low-energy configurations
    - Domain formation emerges from energy minimization
    """
    print(f"\nSampling with {n_chains} parallel chains...")
    print(f"  Warmup: {n_warmup} steps")
    print(f"  Samples: {n_samples} steps")

    key = jax.random.key(seed)

    # Initialize random states
    init_state = []
    for block in blocks:
        key, subkey = jax.random.split(key, 2)
        init_state.append(
            jax.random.randint(
                subkey,
                (n_chains, len(block.nodes)),
                minval=0,
                maxval=n_states,
                dtype=jnp.uint8
            )
        )

    # Sampling schedule
    schedule = SamplingSchedule(
        n_warmup=n_warmup,
        n_samples=n_samples,
        steps_per_sample=2
    )

    # Run sampling (vectorized over chains)
    keys = jax.random.split(key, n_chains)

    print("  Running THRML block Gibbs sampling...")
    samples = jax.jit(
        jax.vmap(
            lambda k, i: sample_states(k, prog, schedule, i, [], [Block(nodes)])
        )
    )(keys, init_state)

    # Extract samples (chains, samples, nodes)
    samples_array = np.array(samples[0])

    print(f"  Generated {samples_array.shape[1]} samples from {n_chains} chains")

    # Reshape to grid for visualization
    samples_grid = samples_array.reshape(n_chains, -1, grid_size, grid_size)

    return samples_grid


def compute_energy_and_magnetization(samples_grid, beta, n_states):
    """
    Compute thermodynamic observables from samples.

    Energy: Average number of aligned neighbors
    Magnetization: Fraction of sites in dominant state
    """
    n_chains, n_samples, h, w = samples_grid.shape

    energies = []
    magnetizations = []

    for chain_idx in range(n_chains):
        for sample_idx in range(n_samples):
            grid = samples_grid[chain_idx, sample_idx]

            # Compute energy (count aligned neighbors)
            aligned = 0
            total = 0
            for i in range(h):
                for j in range(w):
                    # Right neighbor
                    if j < w - 1:
                        aligned += (grid[i, j] == grid[i, j+1])
                        total += 1
                    # Down neighbor
                    if i < h - 1:
                        aligned += (grid[i, j] == grid[i+1, j])
                        total += 1

            energy = -beta * aligned  # Negative because we reward alignment
            energies.append(energy)

            # Compute magnetization (order parameter)
            # Fraction in most common state
            counts = np.bincount(grid.flatten(), minlength=n_states)
            magnetization = counts.max() / (h * w)
            magnetizations.append(magnetization)

    return np.array(energies), np.array(magnetizations)


def visualize_potts_states(samples_grid, beta, n_states, output_file):
    """
    Visualize samples from Potts model showing domain formation.
    """
    print(f"\nCreating visualization...")

    n_chains, n_samples, h, w = samples_grid.shape

    # Compute observables
    energies, magnetizations = compute_energy_and_magnetization(
        samples_grid, beta, n_states
    )

    # Create figure
    fig = plt.figure(figsize=(18, 12), facecolor='black')
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

    # Title
    fig.text(0.5, 0.97, f'Potts Model (q={n_states}) at β={beta:.3f} - THRML Block Gibbs Sampling',
            ha='center', va='top', color='white', fontsize=18, fontweight='bold')

    # Colormap for states
    colors = plt.cm.tab10(np.linspace(0, 1, n_states))
    cmap = ListedColormap(colors)

    # Show sample configurations
    sample_indices = [0, n_samples//4, n_samples//2, -1]
    for idx, sample_idx in enumerate(sample_indices):
        ax = fig.add_subplot(gs[0, idx])
        ax.set_facecolor('black')

        grid = samples_grid[0, sample_idx]  # First chain

        im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=n_states-1,
                      interpolation='nearest')
        ax.set_title(f'Sample {sample_idx + (n_samples if sample_idx < 0 else 0)}',
                    color='white', fontsize=12)
        ax.axis('off')

    # Show multiple chains (final state)
    for idx in range(4):
        ax = fig.add_subplot(gs[1, idx])
        ax.set_facecolor('black')

        if idx < n_chains:
            grid = samples_grid[idx, -1]  # Final sample from chain
            ax.imshow(grid, cmap=cmap, vmin=0, vmax=n_states-1,
                     interpolation='nearest')
            ax.set_title(f'Chain {idx+1} (final state)',
                        color='white', fontsize=12)
        ax.axis('off')

    # Energy histogram
    ax_energy = fig.add_subplot(gs[2, 0:2])
    ax_energy.set_facecolor('black')
    ax_energy.hist(energies, bins=50, color='orange', alpha=0.7, edgecolor='white')
    ax_energy.set_xlabel('Energy', color='white', fontsize=11)
    ax_energy.set_ylabel('Count', color='white', fontsize=11)
    ax_energy.set_title('Energy Distribution', color='white', fontsize=12, fontweight='bold')
    ax_energy.tick_params(colors='white')
    ax_energy.spines['bottom'].set_color('white')
    ax_energy.spines['left'].set_color('white')
    ax_energy.spines['top'].set_visible(False)
    ax_energy.spines['right'].set_visible(False)
    ax_energy.grid(True, alpha=0.2, color='gray')

    # Magnetization (order parameter)
    ax_mag = fig.add_subplot(gs[2, 2:4])
    ax_mag.set_facecolor('black')
    ax_mag.hist(magnetizations, bins=50, color='cyan', alpha=0.7, edgecolor='white')
    ax_mag.set_xlabel('Magnetization (order parameter)', color='white', fontsize=11)
    ax_mag.set_ylabel('Count', color='white', fontsize=11)
    ax_mag.set_title('Order Parameter Distribution', color='white', fontsize=12, fontweight='bold')
    ax_mag.tick_params(colors='white')
    ax_mag.spines['bottom'].set_color('white')
    ax_mag.spines['left'].set_color('white')
    ax_mag.spines['top'].set_visible(False)
    ax_mag.spines['right'].set_visible(False)
    ax_mag.grid(True, alpha=0.2, color='gray')

    # Statistics
    stats_text = f"""
    Statistics:
    • Avg Energy: {energies.mean():.2f} ± {energies.std():.2f}
    • Avg Magnetization: {magnetizations.mean():.3f} ± {magnetizations.std():.3f}
    • Phase: {'ORDERED' if magnetizations.mean() > 0.5 else 'DISORDERED'}
    """
    fig.text(0.5, 0.02, stats_text, ha='center', va='bottom',
            color='white', fontsize=11, fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.8, edgecolor='cyan'))

    plt.savefig(output_file, dpi=150, facecolor='black')
    print(f"Saved: {output_file}")
    plt.close()


def temperature_sweep(grid_size=32, n_states=5):
    """
    Run Potts model at different temperatures to observe phase transition.
    """
    print("="*80)
    print("POTTS MODEL TEMPERATURE SWEEP")
    print("="*80)

    # Temperature range (β = 1/T)
    betas = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    results = []

    for beta in betas:
        print(f"\n{'='*80}")
        print(f"Temperature: T = {1/beta:.3f} (β = {beta:.3f})")
        print(f"{'='*80}")

        # Create model
        prog, nodes, G, blocks = create_potts_model(grid_size, n_states, beta)

        # Sample
        samples_grid = run_potts_sampling(
            prog, blocks, nodes, grid_size, n_states,
            n_chains=8,
            n_warmup=300,
            n_samples=500,
            seed=42
        )

        # Compute observables
        energies, magnetizations = compute_energy_and_magnetization(
            samples_grid, beta, n_states
        )

        results.append({
            'beta': beta,
            'temperature': 1/beta,
            'samples': samples_grid,
            'energy_mean': energies.mean(),
            'energy_std': energies.std(),
            'magnetization_mean': magnetizations.mean(),
            'magnetization_std': magnetizations.std(),
        })

        print(f"\n  Results:")
        print(f"    Energy: {energies.mean():.2f} ± {energies.std():.2f}")
        print(f"    Magnetization: {magnetizations.mean():.3f} ± {magnetizations.std():.3f}")
        print(f"    Phase: {'ORDERED (domains)' if magnetizations.mean() > 0.5 else 'DISORDERED (mixed)'}")

        # Visualize
        output_file = f"potts_beta_{beta:.1f}.png"
        visualize_potts_states(samples_grid, beta, n_states, output_file)

    # Create summary plot
    create_phase_diagram(results, grid_size, n_states)

    return results


def create_phase_diagram(results, grid_size, n_states):
    """
    Plot phase diagram showing transition.
    """
    print("\nCreating phase diagram...")

    temperatures = [r['temperature'] for r in results]
    magnetizations = [r['magnetization_mean'] for r in results]
    mag_errors = [r['magnetization_std'] for r in results]
    energies = [r['energy_mean'] for r in results]
    energy_errors = [r['energy_std'] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), facecolor='black')

    # Order parameter vs temperature
    ax1.set_facecolor('black')
    ax1.errorbar(temperatures, magnetizations, yerr=mag_errors,
                fmt='o-', color='cyan', linewidth=2, markersize=8,
                capsize=5, capthick=2)
    ax1.axhline(1/n_states, color='red', linestyle='--', alpha=0.5,
               label=f'Random baseline (1/{n_states})')
    ax1.set_xlabel('Temperature (T)', color='white', fontsize=13)
    ax1.set_ylabel('Magnetization (order parameter)', color='white', fontsize=13)
    ax1.set_title('Phase Transition: Order Parameter vs Temperature',
                 color='white', fontsize=14, fontweight='bold')
    ax1.tick_params(colors='white')
    ax1.spines['bottom'].set_color('white')
    ax1.spines['left'].set_color('white')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(True, alpha=0.2, color='gray')
    ax1.legend(facecolor='black', edgecolor='white', fontsize=10,
              labelcolor='white')

    # Energy vs temperature
    ax2.set_facecolor('black')
    ax2.errorbar(temperatures, energies, yerr=energy_errors,
                fmt='s-', color='orange', linewidth=2, markersize=8,
                capsize=5, capthick=2)
    ax2.set_xlabel('Temperature (T)', color='white', fontsize=13)
    ax2.set_ylabel('Average Energy', color='white', fontsize=13)
    ax2.set_title('Energy vs Temperature', color='white', fontsize=14, fontweight='bold')
    ax2.tick_params(colors='white')
    ax2.spines['bottom'].set_color('white')
    ax2.spines['left'].set_color('white')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(True, alpha=0.2, color='gray')

    plt.suptitle(f'Potts Model (q={n_states}, {grid_size}x{grid_size}) - THRML Energy-Based Sampling',
                color='white', fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig('potts_phase_diagram.png', dpi=150, facecolor='black')
    print("Saved: potts_phase_diagram.png")
    plt.close()


def main():
    print("="*80)
    print("POTTS MODEL WITH THRML: REAL ENERGY-BASED COMPUTING")
    print("="*80)
    print("\nThis demonstrates what THRML is ACTUALLY designed for:")
    print("  • Energy minimization IS the computation")
    print("  • No pre-computation or circular logic")
    print("  • Native energy-based problem")
    print("  • Block Gibbs sampling finds low-energy states")
    print("\nWe'll run the q-state Potts model at different temperatures")
    print("and observe the phase transition from ordered → disordered.")
    print("="*80)

    # Run temperature sweep
    results = temperature_sweep(grid_size=32, n_states=5)

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  • potts_beta_*.png - Configurations at different temperatures")
    print("  • potts_phase_diagram.png - Phase transition summary")
    print("\nThis is LEGITIMATE energy-based computing with THRML!")
    print("Domain formation emerges from energy minimization.")
    print("="*80)


if __name__ == "__main__":
    main()
