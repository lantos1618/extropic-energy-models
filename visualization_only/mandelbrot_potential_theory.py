#!/usr/bin/env python3
"""
Mandelbrot Set: LEGITIMATE Potential Theory Energy-Based System

This is NOT the performative Ising encoding. This is the actual
Douady-Hubbard exterior potential from complex dynamics.

Key difference:
- Old (BS): Pre-compute → encode as biases → sample Ising
- New (Real): Directly compute φ(c) = lim 2^{-n} log|z_n| from dynamics

The potential is the ACTUAL Green's function for the exterior domain.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource

def compute_exterior_potential(c_grid, max_iter=256, escape_radius=1000.0):
    """
    Compute the Douady-Hubbard exterior potential.

    φ(c) = lim_{n→∞} 2^{-n} log|z_n|

    This is the Green's function for the complement of the Mandelbrot set.
    Satisfies Laplace's equation ∇²φ = 0 in the exterior.

    Returns:
        potential: Exterior potential values
        continuous_escape: Smooth escape time (for coloring)
        inside_mask: Boolean mask of points in set
    """
    z = np.zeros_like(c_grid, dtype=complex)
    potential = np.zeros(c_grid.shape, dtype=float)
    continuous_escape = np.zeros(c_grid.shape, dtype=float)
    inside_mask = np.ones(c_grid.shape, dtype=bool)

    # Also track derivative for distance estimation
    dz = np.ones_like(c_grid, dtype=complex)

    for n in range(max_iter):
        # Update derivative: dz/dc
        dz = 2 * z * dz + 1

        # Mandelbrot iteration
        z = z**2 + c_grid

        # Check escape
        abs_z = np.abs(z)
        just_escaped = (abs_z > escape_radius) & inside_mask

        if np.any(just_escaped):
            # Compute smooth/continuous iteration count
            log_zn = np.log(abs_z[just_escaped])
            nu = n + 1 - np.log(np.log(abs_z[just_escaped])) / np.log(2)
            continuous_escape[just_escaped] = nu

            # Compute exterior potential (Green's function)
            # φ(c) = log|z_n| / 2^n
            potential[just_escaped] = log_zn / (2.0 ** n)

            inside_mask[just_escaped] = False

    # Points still inside after max_iter have zero potential
    potential[inside_mask] = 0.0
    continuous_escape[inside_mask] = max_iter

    return potential, continuous_escape, inside_mask, dz

def compute_potential_gradient(potential, extent):
    """
    Compute gradient of potential field ∇φ
    This represents the "force" or flow direction in the energy landscape
    """
    xmin, xmax, ymin, ymax = extent
    height, width = potential.shape

    dy, dx = np.gradient(potential)

    # Scale by coordinate system
    dx_scale = (xmax - xmin) / width
    dy_scale = (ymax - ymin) / height

    dx /= dx_scale
    dy /= dy_scale

    return dx, dy

def create_3d_energy_landscape(output_file='mandelbrot_3d_energy.png'):
    """
    Create a 3D visualization of the Mandelbrot exterior potential as a
    literal energy landscape - like a topographic map or potential well.
    """
    print("Generating 3D energy landscape...")

    # High resolution for nice visualization
    width, height = 1200, 1000
    xmin, xmax = -2.5, 1.0
    ymin, ymax = -1.25, 1.25

    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    c_grid = x[None, :] + 1j * y[:, None]

    potential, _, inside_mask, _ = compute_exterior_potential(c_grid, max_iter=512)

    # Normalize potential for better visualization
    pot_masked = np.ma.masked_array(potential, mask=inside_mask)
    pot_min, pot_max = pot_masked.min(), pot_masked.max()
    potential_norm = (potential - pot_min) / (pot_max - pot_min)
    potential_norm[inside_mask] = 0  # Set inside to zero energy

    # Subsample for 3D plot (too dense is slow)
    subsample = 4
    x_sub = x[::subsample]
    y_sub = y[::subsample]
    potential_sub = potential_norm[::subsample, ::subsample]
    inside_sub = inside_mask[::subsample, ::subsample]

    X, Y = np.meshgrid(x_sub, y_sub)

    # Create figure with 2 subplots
    fig = plt.figure(figsize=(20, 10), facecolor='black')

    # 3D landscape
    ax1 = fig.add_subplot(121, projection='3d', facecolor='black')
    ax1.set_facecolor('black')

    # Use lighting for better depth perception
    ls = LightSource(azdeg=315, altdeg=45)

    # Color by height
    rgb = cm.plasma(potential_sub)
    illuminated = ls.shade_rgb(rgb, potential_sub)

    surf = ax1.plot_surface(X, Y, potential_sub,
                           facecolors=illuminated,
                           rstride=1, cstride=1,
                           linewidth=0, antialiased=True,
                           shade=True, alpha=0.95)

    ax1.set_xlabel('Re(c)', color='white', fontsize=12)
    ax1.set_ylabel('Im(c)', color='white', fontsize=12)
    ax1.set_zlabel('Potential Energy', color='white', fontsize=12)
    ax1.set_title('Exterior Potential: Douady-Hubbard Green\'s Function',
                 color='white', fontsize=14, fontweight='bold', pad=20)

    # Style
    ax1.tick_params(colors='white', labelsize=9)
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False
    ax1.grid(True, alpha=0.3, color='gray')
    ax1.view_init(elev=25, azim=45)

    # 2D contour map with equipotentials
    ax2 = fig.add_subplot(122, facecolor='black')
    ax2.set_facecolor('black')

    # Full resolution contour
    potential_display = potential_norm.copy()
    potential_display[inside_mask] = np.nan  # NaN for interior

    # Main visualization
    im = ax2.imshow(potential_display,
                   extent=[xmin, xmax, ymin, ymax],
                   cmap='plasma', origin='lower',
                   interpolation='bilinear')

    # Equipotential lines (energy levels)
    levels = np.linspace(0.05, 0.8, 15)
    contours = ax2.contour(x, y, potential_norm,
                          levels=levels,
                          colors='cyan', linewidths=0.8,
                          alpha=0.6)
    ax2.clabel(contours, inline=True, fontsize=7, fmt='%.2f', colors='white')

    # Mark the set boundary (zero potential)
    ax2.contour(x, y, inside_mask.astype(float),
               levels=[0.5], colors='white', linewidths=2)

    ax2.set_xlabel('Re(c)', color='white', fontsize=12)
    ax2.set_ylabel('Im(c)', color='white', fontsize=12)
    ax2.set_title('Equipotential Lines (Energy Contours)',
                 color='white', fontsize=14, fontweight='bold')
    ax2.set_aspect('equal')
    ax2.tick_params(colors='white')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('Potential Energy (Normalized)', color='white', fontsize=11)
    cbar.ax.tick_params(colors='white')

    plt.suptitle('Mandelbrot Set as Energy-Based System: Potential Theory Formulation\nPotential(c) = lim 2^(-n) * log|z_n|  as  n -> infinity  [Douady-Hubbard, 1982]',
                color='white', fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_file, dpi=150, facecolor='black')
    print(f"Saved: {output_file}")
    plt.close()

def create_potential_flow_animation(num_frames=180, output_file='mandelbrot_potential_flow.mp4'):
    """
    Animate the energy landscape showing:
    1. Potential field coloring
    2. Equipotential lines forming
    3. Gradient vectors (flow field)
    4. Zoom into interesting region
    """
    print(f"\nGenerating {num_frames} frame potential flow animation...")

    fig, ax = plt.subplots(figsize=(14, 10), facecolor='black')
    ax.set_facecolor('black')
    ax.set_aspect('equal')

    # Zoom sequence - spiral into boundary
    zoom_frames = num_frames // 3
    hold_frames = num_frames // 3
    feature_frames = num_frames - zoom_frames - hold_frames

    coords_sequence = []

    # Phase 1: Zoom in
    for i in range(zoom_frames):
        t = i / (zoom_frames - 1)
        t_smooth = (np.tanh((t - 0.5) * 4) + 1) / 2

        width = 3.5 * (0.01 ** t_smooth)
        center_x = -0.5 + (-0.7 + 0.5) * t_smooth
        center_y = 0.0 + (0.0 - 0.0) * t_smooth

        coords_sequence.append({
            'xmin': center_x - width/2,
            'xmax': center_x + width/2,
            'ymin': center_y - width*0.8/2,
            'ymax': center_y + width*0.8/2,
            'show_flow': False,
            'show_contours': i > zoom_frames // 2
        })

    # Phase 2: Hold and show features
    final_coord = coords_sequence[-1].copy()
    for i in range(hold_frames):
        coord = final_coord.copy()
        coord['show_flow'] = i > hold_frames // 3
        coord['show_contours'] = True
        coord['flow_density'] = min(1.0, i / (hold_frames // 2))
        coords_sequence.append(coord)

    # Phase 3: Pan to another feature
    start_coord = coords_sequence[-1]
    for i in range(feature_frames):
        t = i / (feature_frames - 1)

        # Pan to different region
        center_x = start_coord['xmin'] + (start_coord['xmax'] - start_coord['xmin'])/2
        center_y = start_coord['ymin'] + (start_coord['ymax'] - start_coord['ymin'])/2

        new_center_x = center_x + 0.1 * np.sin(t * 2 * np.pi)
        new_center_y = center_y + 0.05 * np.cos(t * 2 * np.pi)

        width = start_coord['xmax'] - start_coord['xmin']
        height = start_coord['ymax'] - start_coord['ymin']

        coords_sequence.append({
            'xmin': new_center_x - width/2,
            'xmax': new_center_x + width/2,
            'ymin': new_center_y - height/2,
            'ymax': new_center_y + height/2,
            'show_flow': True,
            'show_contours': True,
            'flow_density': 1.0
        })

    # Initialize first frame
    coord = coords_sequence[0]
    width, height = 800, 640
    x = np.linspace(coord['xmin'], coord['xmax'], width)
    y = np.linspace(coord['ymin'], coord['ymax'], height)
    c_grid = x[None, :] + 1j * y[:, None]

    potential, cont_escape, inside_mask, _ = compute_exterior_potential(c_grid, max_iter=512)

    # Normalize
    pot_display = potential.copy()
    if np.any(~inside_mask):
        pmin, pmax = potential[~inside_mask].min(), potential[~inside_mask].max()
        if pmax > pmin:
            pot_display[~inside_mask] = (potential[~inside_mask] - pmin) / (pmax - pmin)
    pot_display[inside_mask] = 0

    im = ax.imshow(pot_display, extent=[coord['xmin'], coord['xmax'], coord['ymin'], coord['ymax']],
                  cmap='plasma', origin='lower', interpolation='bilinear')

    title = ax.text(0.5, 0.98, 'Exterior Potential Energy Landscape: Potential(c) = lim 2^(-n) * log|z_n|',
                   transform=fig.transFigure, ha='center', va='top',
                   color='white', fontsize=16, fontweight='bold')

    ax.axis('off')

    def update_frame(frame):

        coord = coords_sequence[frame]

        # Recompute for this view
        x = np.linspace(coord['xmin'], coord['xmax'], width)
        y = np.linspace(coord['ymin'], coord['ymax'], height)
        c_grid = x[None, :] + 1j * y[:, None]

        max_iter = min(512 + frame, 2000)
        potential, cont_escape, inside_mask, _ = compute_exterior_potential(c_grid, max_iter=max_iter)

        # Normalize potential
        pot_display = potential.copy()
        if np.any(~inside_mask):
            pmin, pmax = potential[~inside_mask].min(), potential[~inside_mask].max()
            if pmax > pmin:
                pot_display[~inside_mask] = (potential[~inside_mask] - pmin) / (pmax - pmin)
        pot_display[inside_mask] = 0

        # Clear axis and redraw
        ax.clear()
        ax.set_facecolor('black')
        ax.axis('off')

        # Redraw image
        im = ax.imshow(pot_display, extent=[coord['xmin'], coord['xmax'], coord['ymin'], coord['ymax']],
                      cmap='plasma', origin='lower', interpolation='bilinear')

        # Add equipotential contours
        if coord.get('show_contours', False):
            levels = np.linspace(0.1, 0.9, 12)
            contour_lines = ax.contour(x, y, pot_display, levels=levels,
                                      colors='cyan', linewidths=0.6, alpha=0.5)

            # Boundary
            boundary_line = ax.contour(x, y, inside_mask.astype(float),
                                      levels=[0.5], colors='white', linewidths=2)

        # Add flow field (gradient)
        if coord.get('show_flow', False):
            flow_density = coord.get('flow_density', 1.0)
            stride = max(20, int(40 * (1 - flow_density)))

            dx, dy = compute_potential_gradient(pot_display,
                                               [coord['xmin'], coord['xmax'],
                                                coord['ymin'], coord['ymax']])

            X_sub, Y_sub = np.meshgrid(x[::stride], y[::stride])
            U = dx[::stride, ::stride]
            V = dy[::stride, ::stride]

            magnitude = np.sqrt(U**2 + V**2)
            magnitude_norm = magnitude / (magnitude.max() + 1e-10)

            quiver_plot = ax.quiver(X_sub, Y_sub, U, V,
                                   magnitude_norm, cmap='cool',
                                   scale=15, scale_units='xy',
                                   width=0.003, alpha=0.7)

        # Update info (recreate since we cleared)
        zoom_level = 3.5 / (coord['xmax'] - coord['xmin'])
        phase = "Zooming..." if frame < zoom_frames else \
                "Analyzing..." if frame < zoom_frames + hold_frames else \
                "Exploring..."

        info_text = ax.text(0.02, 0.98, f'{phase}\nZoom: {zoom_level:.1e}x\nMax iter: {max_iter}',
                           transform=ax.transAxes, ha='left', va='top',
                           color='cyan', fontsize=10, fontfamily='monospace',
                           bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

        print(f"Frame {frame+1}/{num_frames} - {phase} - Zoom: {zoom_level:.1e}x          ", end='\r')

    anim = animation.FuncAnimation(fig, update_frame, frames=num_frames,
                                  interval=50, blit=False)

    print(f"\nSaving animation to {output_file}...")
    writer = animation.FFMpegWriter(fps=30, bitrate=4000, codec='libx264')
    anim.save(output_file, writer=writer, dpi=120)

    print(f"Animation saved!")
    plt.close()

def main():
    print("=" * 80)
    print("MANDELBROT SET: LEGITIMATE POTENTIAL THEORY ENERGY-BASED SYSTEM")
    print("=" * 80)
    print("\nThis is NOT the performative Ising encoding.")
    print("This is the actual Douady-Hubbard exterior potential φ(c).")
    print("\nKey mathematical properties:")
    print("  • φ(c) = lim_{n→∞} 2^{-n} log|z_n|  [Green's function]")
    print("  • Satisfies Laplace equation: ∇²φ = 0  [harmonic in exterior]")
    print("  • φ = 0 on ∂M (boundary), φ → ∞ at infinity")
    print("  • Equipotentials are closed curves around M")
    print("  • ∇φ points outward (escape direction)")
    print("\nReferences:")
    print("  - Douady & Hubbard (1982): Itération des polynômes quadratiques")
    print("  - Milnor (2006): Dynamics in One Complex Variable")
    print("=" * 80)

    # Generate visualizations
    create_3d_energy_landscape('mandelbrot_potential_theory_3d.png')

    print("\n" + "=" * 80)

    create_potential_flow_animation(num_frames=150,
                                   output_file='mandelbrot_potential_theory.mp4')

    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("\nGenerated files:")
    print("  • mandelbrot_potential_theory_3d.png - 3D energy landscape + equipotentials")
    print("  • mandelbrot_potential_theory.mp4 - Animated potential flow field")
    print("\nThis is LEGITIMATE energy-based mathematics from complex dynamics.")
    print("The potential emerges from the iteration, not pre-encoded.")
    print("=" * 80)

if __name__ == "__main__":
    main()
