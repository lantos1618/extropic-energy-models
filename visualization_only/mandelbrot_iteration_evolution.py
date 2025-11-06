#!/usr/bin/env python3
"""
Mandelbrot Energy Landscape: Iteration Evolution

Show how the exterior potential energy field EMERGES from the
iteration dynamics. This demonstrates the actual energy-based
computation happening.

Instead of zooming, we:
1. Pick an interesting region near the boundary
2. Show iterations from 1 to N, watching the energy field form
3. Visualize how the potential "crystallizes" at the boundary
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
from matplotlib.gridspec import GridSpec

def compute_mandelbrot_by_iteration(c_grid, target_iter):
    """
    Compute Mandelbrot for exactly target_iter iterations,
    returning the energy landscape at that iteration depth.

    This shows how energy emerges from dynamics!
    """
    z = np.zeros_like(c_grid, dtype=complex)
    potential = np.zeros(c_grid.shape, dtype=float)
    escaped_mask = np.zeros(c_grid.shape, dtype=bool)
    escape_iter = np.full(c_grid.shape, target_iter, dtype=int)

    # Track the trajectory for all points
    for n in range(target_iter):
        # Iterate
        z = z**2 + c_grid

        # Check escape
        abs_z = np.abs(z)
        just_escaped = (abs_z > 1000.0) & (~escaped_mask)

        if np.any(just_escaped):
            # Record when they escaped
            escape_iter[just_escaped] = n

            # Compute potential for escaped points
            log_zn = np.log(abs_z[just_escaped] + 1e-10)
            potential[just_escaped] = log_zn / (2.0 ** n)

            escaped_mask[just_escaped] = True

    # Points still inside have zero potential
    still_inside = ~escaped_mask
    potential[still_inside] = 0

    return potential, escape_iter, escaped_mask

def create_iteration_evolution_animation(
    num_frames=120,
    output_file='mandelbrot_iteration_evolution.mp4'
):
    """
    Animate the energy landscape as iterations increase.
    Watch the potential field crystallize!
    """
    print(f"\nGenerating {num_frames} frame iteration evolution...")

    # Focus on an interesting region near the boundary
    # "Seahorse valley" - great for seeing detail emerge
    center_x, center_y = -0.75, 0.1
    width, height_span = 0.2, 0.15

    xmin = center_x - width/2
    xmax = center_x + width/2
    ymin = center_y - height_span/2
    ymax = center_y + height_span/2

    # High resolution for detail
    res_x, res_y = 1200, 900

    x = np.linspace(xmin, xmax, res_x)
    y = np.linspace(ymin, ymax, res_y)
    c_grid = x[None, :] + 1j * y[:, None]

    # Iteration schedule - start slow, then speed up
    max_iterations = 500

    # Exponential ramp up
    iter_schedule = []
    for i in range(num_frames):
        t = i / (num_frames - 1)
        # Smooth S-curve
        t_smooth = (np.tanh((t - 0.5) * 5) + 1) / 2
        target_iter = int(2 + (max_iterations - 2) * t_smooth)
        iter_schedule.append(target_iter)

    # Create figure with multiple views
    fig = plt.figure(figsize=(16, 10), facecolor='black')
    gs = GridSpec(2, 2, figure=fig, hspace=0.25, wspace=0.15)

    # Main potential view
    ax_main = fig.add_subplot(gs[:, 0])
    ax_main.set_facecolor('black')
    ax_main.set_aspect('equal')
    ax_main.axis('off')

    # Energy profile (slice)
    ax_slice = fig.add_subplot(gs[0, 1])
    ax_slice.set_facecolor('black')
    ax_slice.tick_params(colors='white', labelsize=9)
    ax_slice.spines['bottom'].set_color('white')
    ax_slice.spines['left'].set_color('white')
    ax_slice.spines['top'].set_visible(False)
    ax_slice.spines['right'].set_visible(False)

    # Escape time histogram
    ax_hist = fig.add_subplot(gs[1, 1])
    ax_hist.set_facecolor('black')
    ax_hist.tick_params(colors='white', labelsize=9)
    ax_hist.spines['bottom'].set_color('white')
    ax_hist.spines['left'].set_color('white')
    ax_hist.spines['top'].set_visible(False)
    ax_hist.spines['right'].set_visible(False)

    # Compute first frame
    target_iter = iter_schedule[0]
    potential, escape_iter, escaped = compute_mandelbrot_by_iteration(c_grid, target_iter)

    # Normalize
    pot_norm = potential.copy()
    if np.any(escaped):
        pmin, pmax = potential[escaped].min(), potential[escaped].max()
        if pmax > pmin:
            pot_norm[escaped] = (potential[escaped] - pmin) / (pmax - pmin)

    im_main = ax_main.imshow(pot_norm, extent=[xmin, xmax, ymin, ymax],
                            cmap='plasma', origin='lower', interpolation='bilinear')

    # Title
    title = fig.text(0.5, 0.96, 'Energy Landscape Evolution: Watching Potential Field Crystallize',
                    ha='center', va='top', color='white', fontsize=18, fontweight='bold')

    def update_frame(frame):
        target_iter = iter_schedule[frame]

        print(f"Frame {frame+1}/{num_frames} - Iterations: {target_iter:4d}    ", end='\r')

        # Compute for this iteration depth
        potential, escape_iter, escaped = compute_mandelbrot_by_iteration(c_grid, target_iter)

        # Normalize
        pot_norm = potential.copy()
        if np.any(escaped):
            pmin, pmax = potential[escaped].min(), potential[escaped].max()
            if pmax > pmin:
                pot_norm[escaped] = (potential[escaped] - pmin) / (pmax - pmin)

        # Update main view
        ax_main.clear()
        ax_main.set_facecolor('black')
        ax_main.axis('off')
        ax_main.set_aspect('equal')

        im = ax_main.imshow(pot_norm, extent=[xmin, xmax, ymin, ymax],
                           cmap='plasma', origin='lower', interpolation='bilinear')

        # Add boundary contour
        ax_main.contour(x, y, escaped.astype(float), levels=[0.5],
                       colors='cyan', linewidths=1.5, alpha=0.8)

        # Add info text
        percent_escaped = 100 * escaped.sum() / escaped.size
        ax_main.text(0.02, 0.98, f'Iterations: {target_iter}\n{percent_escaped:.1f}% escaped',
                    transform=ax_main.transAxes, ha='left', va='top',
                    color='cyan', fontsize=11, fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))

        # Energy slice (horizontal through middle)
        ax_slice.clear()
        ax_slice.set_facecolor('black')
        slice_idx = res_y // 2
        energy_slice = pot_norm[slice_idx, :]
        ax_slice.plot(x, energy_slice, color='cyan', linewidth=2)
        ax_slice.fill_between(x, energy_slice, alpha=0.3, color='cyan')
        ax_slice.set_xlim(xmin, xmax)
        ax_slice.set_ylim(0, 1.1)
        ax_slice.set_xlabel('Re(c)', color='white', fontsize=10)
        ax_slice.set_ylabel('Potential Energy', color='white', fontsize=10)
        ax_slice.set_title('Energy Profile (horizontal slice)', color='white', fontsize=11)
        ax_slice.tick_params(colors='white', labelsize=8)
        ax_slice.spines['bottom'].set_color('white')
        ax_slice.spines['left'].set_color('white')
        ax_slice.grid(True, alpha=0.2, color='gray')

        # Escape time histogram
        ax_hist.clear()
        ax_hist.set_facecolor('black')

        if np.any(escaped):
            escape_times = escape_iter[escaped]
            bins = min(50, target_iter)
            ax_hist.hist(escape_times, bins=bins, color='orange', alpha=0.7, edgecolor='white', linewidth=0.5)
            ax_hist.set_xlabel('Escape Iteration', color='white', fontsize=10)
            ax_hist.set_ylabel('Count', color='white', fontsize=10)
            ax_hist.set_title('Escape Time Distribution', color='white', fontsize=11)
            ax_hist.tick_params(colors='white', labelsize=8)
            ax_hist.spines['bottom'].set_color('white')
            ax_hist.spines['left'].set_color('white')
            ax_hist.grid(True, alpha=0.2, color='gray', axis='y')

        # Subtitle
        subtitle = f'Max Iterations: {target_iter}/{max_iterations} | Resolution: {res_x}x{res_y} | Region: Re[{xmin:.3f}, {xmax:.3f}] x Im[{ymin:.3f}, {ymax:.3f}]'
        fig.text(0.5, 0.92, subtitle, ha='center', va='top',
                color='white', fontsize=10, alpha=0.7)

    anim = animation.FuncAnimation(fig, update_frame, frames=num_frames,
                                  interval=50, blit=False)

    print(f"\nSaving to {output_file}...")
    writer = animation.FFMpegWriter(fps=30, bitrate=5000, codec='libx264')
    anim.save(output_file, writer=writer, dpi=100)

    print("Done!")
    plt.close()

def create_side_by_side_comparison(output_file='mandelbrot_iteration_comparison.png'):
    """
    Show multiple iteration depths side-by-side to illustrate
    how the energy field refines.
    """
    print("\nCreating iteration depth comparison...")

    # Same region as animation
    center_x, center_y = -0.75, 0.1
    width, height_span = 0.2, 0.15

    xmin = center_x - width/2
    xmax = center_x + width/2
    ymin = center_y - height_span/2
    ymax = center_y + height_span/2

    res_x, res_y = 800, 600

    x = np.linspace(xmin, xmax, res_x)
    y = np.linspace(ymin, ymax, res_y)
    c_grid = x[None, :] + 1j * y[:, None]

    # Different iteration depths
    iter_depths = [5, 10, 25, 50, 100, 200, 400, 800]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10), facecolor='black')
    fig.suptitle('Energy Landscape Evolution: Increasing Iteration Depth',
                color='white', fontsize=18, fontweight='bold', y=0.98)

    for idx, (ax, target_iter) in enumerate(zip(axes.flat, iter_depths)):
        print(f"  Computing iteration depth {target_iter}...")

        potential, escape_iter, escaped = compute_mandelbrot_by_iteration(c_grid, target_iter)

        # Normalize
        pot_norm = potential.copy()
        if np.any(escaped):
            pmin, pmax = potential[escaped].min(), potential[escaped].max()
            if pmax > pmin:
                pot_norm[escaped] = (potential[escaped] - pmin) / (pmax - pmin)

        ax.set_facecolor('black')
        ax.imshow(pot_norm, extent=[xmin, xmax, ymin, ymax],
                 cmap='plasma', origin='lower', interpolation='bilinear')

        # Boundary
        ax.contour(x, y, escaped.astype(float), levels=[0.5],
                  colors='cyan', linewidths=1, alpha=0.7)

        percent_escaped = 100 * escaped.sum() / escaped.size
        ax.set_title(f'n = {target_iter} ({percent_escaped:.1f}% escaped)',
                    color='white', fontsize=12, fontweight='bold')
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_file, dpi=150, facecolor='black')
    print(f"Saved: {output_file}")
    plt.close()

def main():
    print("=" * 80)
    print("MANDELBROT ENERGY LANDSCAPE: ITERATION EVOLUTION")
    print("=" * 80)
    print("\nShowing how the exterior potential EMERGES from iteration dynamics.")
    print("This demonstrates the energy-based computation in action!")
    print("\nKey insight:")
    print("  The energy landscape isn't pre-programmed - it crystallizes as we")
    print("  iterate the map z → z² + c. Points that escape early form the outer")
    print("  high-energy shell. Points near the boundary emerge only at high iterations.")
    print("=" * 80)

    # Static comparison
    create_side_by_side_comparison('mandelbrot_iteration_comparison.png')

    print("\n" + "=" * 80)

    # Dynamic animation
    create_iteration_evolution_animation(num_frames=120,
                                        output_file='mandelbrot_iteration_evolution.mp4')

    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("\nGenerated files:")
    print("  • mandelbrot_iteration_comparison.png - Side-by-side iteration depths")
    print("  • mandelbrot_iteration_evolution.mp4 - Animated evolution of energy field")
    print("\nThis shows REAL energy-based dynamics - the potential emerges from iteration!")
    print("=" * 80)

if __name__ == "__main__":
    main()
