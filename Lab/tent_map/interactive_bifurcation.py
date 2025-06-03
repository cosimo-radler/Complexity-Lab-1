import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RangeSlider
import ipywidgets as widgets
from IPython.display import display

def tent_map(x, mu):
    """Tent map function"""
    return np.where(x <= 0.5, mu * x, mu * (1 - x))

def iterate_tent_map(x0, mu, n_iterations):
    """Iterate the tent map n times"""
    trajectory = [x0]
    x = x0
    for _ in range(n_iterations):
        x = tent_map(x, mu)
        trajectory.append(x)
    return np.array(trajectory)

def explore_bifurcation_diagram(mu_min=0.1, mu_max=2.5, mu_points=2000, 
                               n_transient=500, n_plot=100, x0=0.7, auto_scale_y=True):
    """
    Create an interactive bifurcation diagram with zoom capability
    
    Parameters:
    -----------
    mu_min : float
        Minimum mu value to plot
    mu_max : float  
        Maximum mu value to plot
    mu_points : int
        Number of mu values to calculate (resolution)
    n_transient : int
        Number of initial iterations to skip (let system settle)
    n_plot : int
        Number of final iterations to plot
    x0 : float
        Initial condition
    auto_scale_y : bool
        If True, automatically scale y-axis to data range
    """
    
    # Create parameter array
    mu_values = np.linspace(mu_min, mu_max, mu_points)
    
    # Store results
    mu_plot = []
    x_plot = []
    
    print(f"Computing bifurcation diagram from μ={mu_min:.3f} to μ={mu_max:.3f}")
    print(f"Resolution: {mu_points} points")
    
    # Calculate bifurcation data
    for i, mu in enumerate(mu_values):
        if i % (mu_points // 10) == 0:
            print(f"Progress: {100*i/mu_points:.0f}%")
        
        # Generate trajectory
        trajectory = iterate_tent_map(x0, mu, n_transient + n_plot)
        
        # Take only the final points (after transients)
        final_points = trajectory[-n_plot:]
        
        # Add to plot data
        mu_plot.extend([mu] * len(final_points))
        x_plot.extend(final_points)
    
    # Calculate y-axis limits
    if auto_scale_y and len(x_plot) > 0:
        x_min, x_max = np.min(x_plot), np.max(x_plot)
        y_range = x_max - x_min
        y_padding = max(0.05, y_range * 0.1)  # 10% padding or minimum 0.05
        y_limits = [max(0, x_min - y_padding), min(1, x_max + y_padding)]
        print(f"Auto-scaled y-axis: [{y_limits[0]:.3f}, {y_limits[1]:.3f}]")
    else:
        y_limits = [0, 1]
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left plot: Full bifurcation diagram
    ax1.plot(mu_plot, x_plot, ',k', markersize=0.1, alpha=0.5)
    ax1.set_xlabel('Parameter μ')
    ax1.set_ylabel('x')
    ax1.set_title('Full Bifurcation Diagram')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 2.5)
    ax1.set_ylim(0, 1)  # Keep full scale for overview
    
    # Highlight the zoomed region
    ax1.axvspan(mu_min, mu_max, alpha=0.2, color='red', label=f'Zoom: [{mu_min:.2f}, {mu_max:.2f}]')
    ax1.legend()
    
    # Right plot: Zoomed bifurcation diagram with auto-scaled y-axis
    ax2.plot(mu_plot, x_plot, ',k', markersize=0.5, alpha=0.7)
    ax2.set_xlabel('Parameter μ')
    ax2.set_ylabel('x')
    ax2.set_title(f'Zoomed View: μ ∈ [{mu_min:.3f}, {mu_max:.3f}]')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(mu_min, mu_max)
    ax2.set_ylim(y_limits)  # Use auto-scaled limits
    
    plt.tight_layout()
    plt.show()
    
    # Print some analysis
    print(f"\nAnalysis for μ ∈ [{mu_min:.3f}, {mu_max:.3f}]:")
    print(f"- Total data points: {len(mu_plot):,}")
    print(f"- Parameter resolution: Δμ = {(mu_max-mu_min)/mu_points:.6f}")
    print(f"- Transient iterations: {n_transient}")
    print(f"- Plotted iterations per μ: {n_plot}")
    if auto_scale_y:
        print(f"- Data range: x ∈ [{x_min:.4f}, {x_max:.4f}]")
        print(f"- Y-axis scaled to: [{y_limits[0]:.3f}, {y_limits[1]:.3f}]")

def multi_resolution_comparison(mu_min=1.5, mu_max=2.0, x0=0.7):
    """
    Compare different resolutions for the same parameter range
    """
    resolutions = [500, 1000, 2000, 4000]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, res in enumerate(resolutions):
        mu_values = np.linspace(mu_min, mu_max, res)
        mu_plot = []
        x_plot = []
        
        for mu in mu_values:
            trajectory = iterate_tent_map(x0, mu, 550)  # 50 transient + 500 plot
            final_points = trajectory[-50:]  # Plot last 50 points
            
            mu_plot.extend([mu] * len(final_points))
            x_plot.extend(final_points)
        
        axes[i].plot(mu_plot, x_plot, ',k', markersize=0.3, alpha=0.6)
        axes[i].set_title(f'Resolution: {res} points')
        axes[i].set_xlabel('Parameter μ')
        axes[i].set_ylabel('x')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlim(mu_min, mu_max)
        axes[i].set_ylim(0, 1)
    
    plt.suptitle(f'Resolution Comparison: μ ∈ [{mu_min}, {mu_max}]', fontsize=16)
    plt.tight_layout()
    plt.show()

def custom_zoom_exploration():
    """
    Template function for exploring specific regions
    Modify the parameters below to explore different areas
    """
    
    # Example 1: Onset of chaos around μ = 1
    print("Example 1: Onset of chaos")
    explore_bifurcation_diagram(mu_min=0.8, mu_max=1.2, mu_points=1500,
                               n_transient=300, n_plot=80, x0=0.7)
    
    # Example 2: Deep into chaos
    print("\nExample 2: Chaotic regime") 
    explore_bifurcation_diagram(mu_min=1.8, mu_max=2.2, mu_points=2000,
                               n_transient=500, n_plot=150, x0=0.3)
    
    # Example 3: Very fine detail in a small region
    print("\nExample 3: Fine detail")
    explore_bifurcation_diagram(mu_min=1.95, mu_max=2.05, mu_points=1000,
                               n_transient=800, n_plot=200, x0=0.1)

# Interactive widget version (for Jupyter notebooks)
def interactive_bifurcation_widget():
    """
    Create interactive widgets for exploring the bifurcation diagram
    """
    
    # Create widgets
    mu_min_slider = widgets.FloatSlider(
        value=1.5, min=0.1, max=2.4, step=0.01,
        description='μ min:', style={'description_width': 'initial'}
    )
    
    mu_max_slider = widgets.FloatSlider(
        value=2.0, min=0.2, max=2.5, step=0.01,
        description='μ max:', style={'description_width': 'initial'}
    )
    
    resolution_slider = widgets.IntSlider(
        value=1000, min=100, max=5000, step=100,
        description='Resolution:', style={'description_width': 'initial'}
    )
    
    x0_slider = widgets.FloatSlider(
        value=0.7, min=0.01, max=0.99, step=0.01,
        description='Initial x₀:', style={'description_width': 'initial'}
    )
    
    # Interactive function
    def update_plot(mu_min, mu_max, resolution, x0):
        if mu_min >= mu_max:
            print("Error: μ_min must be less than μ_max")
            return
        
        explore_bifurcation_diagram(
            mu_min=mu_min, mu_max=mu_max, mu_points=resolution,
            n_transient=400, n_plot=100, x0=x0
        )
    
    # Create interactive widget
    interactive_plot = widgets.interactive(
        update_plot,
        mu_min=mu_min_slider,
        mu_max=mu_max_slider, 
        resolution=resolution_slider,
        x0=x0_slider
    )
    
    return interactive_plot

def bifurcation_with_fixed_points(mu_min=0.1, mu_max=2.5, mu_points=2000, 
                                 n_transient=500, n_plot=100, x0=0.7):
    """
    Create bifurcation diagram with fixed points overlaid
    
    For the tent map, the fixed points are:
    - x* = 0 (always exists)
    - x* = μ/(1+μ) (exists for all μ > 0)
    
    Stability:
    - Both fixed points are stable for μ < 1
    - Both become unstable for μ > 1
    """
    
    # Create parameter array
    mu_values = np.linspace(mu_min, mu_max, mu_points)
    
    # Store bifurcation results
    mu_plot = []
    x_plot = []
    
    print(f"Computing bifurcation diagram with fixed points from μ={mu_min:.3f} to μ={mu_max:.3f}")
    print(f"Resolution: {mu_points} points")
    
    # Calculate bifurcation data
    for i, mu in enumerate(mu_values):
        if i % (mu_points // 10) == 0:
            print(f"Progress: {100*i/mu_points:.0f}%")
        
        # Generate trajectory
        trajectory = iterate_tent_map(x0, mu, n_transient + n_plot)
        
        # Take only the final points (after transients)
        final_points = trajectory[-n_plot:]
        
        # Add to plot data
        mu_plot.extend([mu] * len(final_points))
        x_plot.extend(final_points)
    
    # Calculate fixed points
    mu_fp = np.linspace(mu_min, mu_max, 1000)
    
    # Fixed point 1: x* = 0 (always exists)
    x_fp1 = np.zeros_like(mu_fp)
    
    # Fixed point 2: x* = μ/(1+μ)
    x_fp2 = mu_fp / (1 + mu_fp)
    
    # Determine stability regions
    stable_mask = mu_fp < 1.0
    unstable_mask = mu_fp >= 1.0
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot bifurcation diagram
    ax.plot(mu_plot, x_plot, ',k', markersize=0.5, alpha=0.6, label='Attractors')
    
    # Plot stable fixed points (thick lines)
    if np.any(stable_mask):
        ax.plot(mu_fp[stable_mask], x_fp1[stable_mask], 'b-', linewidth=3, 
                label='Stable fixed points', alpha=0.8)
        ax.plot(mu_fp[stable_mask], x_fp2[stable_mask], 'b-', linewidth=3, alpha=0.8)
    
    # Plot unstable fixed points (dashed lines)
    if np.any(unstable_mask):
        ax.plot(mu_fp[unstable_mask], x_fp1[unstable_mask], 'r--', linewidth=2, 
                label='Unstable fixed points', alpha=0.8)
        ax.plot(mu_fp[unstable_mask], x_fp2[unstable_mask], 'r--', linewidth=2, alpha=0.8)
    
    # Mark the critical value μ = 1
    ax.axvline(x=1.0, color='orange', linestyle=':', linewidth=2, 
               label='μ = 1 (stability transition)', alpha=0.8)
    
    # Formatting
    ax.set_xlabel('Parameter μ', fontsize=12)
    ax.set_ylabel('x', fontsize=12)
    ax.set_title('Tent Map: Bifurcation Diagram with Fixed Points', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(mu_min, mu_max)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # Analysis
    print(f"\n=== Fixed Point Analysis ===")
    print(f"Fixed Points for the Tent Map:")
    print(f"1. x₁* = 0 (always exists)")
    print(f"2. x₂* = μ/(1+μ)")
    print(f"\nStability:")
    print(f"- For μ < 1: Both fixed points are stable")
    print(f"- For μ = 1: Transition to instability")  
    print(f"- For μ > 1: Both fixed points become unstable → Chaos")
    print(f"\nAt μ = 1:")
    print(f"- x₁* = 0")
    print(f"- x₂* = 1/2 = 0.5")

def analyze_specific_mu_values(mu_values=[0.5, 0.8, 1.0, 1.5, 2.0], x0=0.7, n_iterations=1000):
    """
    Analyze specific μ values and show fixed points vs. actual dynamics
    """
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, mu in enumerate(mu_values):
        if i >= len(axes):
            break
            
        # Calculate trajectory
        trajectory = iterate_tent_map(x0, mu, n_iterations)
        
        # Calculate fixed points
        fp1 = 0
        fp2 = mu / (1 + mu)
        
        # Determine stability
        stable = mu < 1.0
        
        # Plot time series
        axes[i].plot(trajectory, 'b-', alpha=0.7, linewidth=1)
        
        # Mark fixed points
        if stable:
            axes[i].axhline(y=fp1, color='green', linestyle='-', linewidth=3, 
                           label=f'Stable FP: x₁* = {fp1:.3f}', alpha=0.8)
            axes[i].axhline(y=fp2, color='green', linestyle='-', linewidth=3, 
                           label=f'Stable FP: x₂* = {fp2:.3f}', alpha=0.8)
        else:
            axes[i].axhline(y=fp1, color='red', linestyle='--', linewidth=2, 
                           label=f'Unstable FP: x₁* = {fp1:.3f}', alpha=0.8)
            axes[i].axhline(y=fp2, color='red', linestyle='--', linewidth=2, 
                           label=f'Unstable FP: x₂* = {fp2:.3f}', alpha=0.8)
        
        axes[i].set_title(f'μ = {mu:.1f} {"(Stable)" if stable else "(Chaotic)"}')
        axes[i].set_xlabel('Iteration n')
        axes[i].set_ylabel('x_n')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(fontsize=8)
        axes[i].set_ylim(0, 1)
    
    # Remove empty subplot
    if len(mu_values) < len(axes):
        fig.delaxes(axes[-1])
    
    plt.suptitle('Fixed Points vs. Actual Dynamics for Different μ Values', fontsize=16)
    plt.tight_layout()
    plt.show()

def theoretical_vs_numerical_comparison(mu_min=0.1, mu_max=2.5, mu_points=1000):
    """
    Compare theoretical fixed points with numerical attractors
    """
    
    mu_values = np.linspace(mu_min, mu_max, mu_points)
    
    # Theoretical fixed points
    fp1_theory = np.zeros_like(mu_values)  # x* = 0
    fp2_theory = mu_values / (1 + mu_values)  # x* = μ/(1+μ)
    
    # Numerical attractors (simplified - just final value after long iteration)
    attractors = []
    for mu in mu_values:
        traj = iterate_tent_map(0.7, mu, 2000)  # Long trajectory
        final_vals = traj[-100:]  # Last 100 points
        
        if mu < 1.0:
            # Should converge to one of the fixed points
            attractor = np.mean(final_vals)
        else:
            # Chaotic - take representative values
            attractor = final_vals
        
        attractors.append(attractor)
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Top plot: Theoretical fixed points
    stable_mask = mu_values < 1.0
    unstable_mask = mu_values >= 1.0
    
    ax1.plot(mu_values[stable_mask], fp1_theory[stable_mask], 'b-', linewidth=3, 
             label='Stable FP: x₁* = 0')
    ax1.plot(mu_values[stable_mask], fp2_theory[stable_mask], 'b-', linewidth=3, 
             label='Stable FP: x₂* = μ/(1+μ)')
    ax1.plot(mu_values[unstable_mask], fp1_theory[unstable_mask], 'r--', linewidth=2, 
             label='Unstable FP: x₁* = 0')
    ax1.plot(mu_values[unstable_mask], fp2_theory[unstable_mask], 'r--', linewidth=2, 
             label='Unstable FP: x₂* = μ/(1+μ)')
    
    ax1.axvline(x=1.0, color='orange', linestyle=':', linewidth=2, alpha=0.8)
    ax1.set_title('Theoretical Fixed Points')
    ax1.set_ylabel('x*')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(mu_min, mu_max)
    ax1.set_ylim(0, 1)
    
    # Bottom plot: Numerical simulation
    mu_plot = []
    x_plot = []
    
    for i, (mu, attr) in enumerate(zip(mu_values, attractors)):
        if isinstance(attr, (list, np.ndarray)):
            # Chaotic case
            mu_plot.extend([mu] * len(attr))
            x_plot.extend(attr)
        else:
            # Fixed point case
            mu_plot.append(mu)
            x_plot.append(attr)
    
    ax2.plot(mu_plot, x_plot, ',k', markersize=0.5, alpha=0.6)
    ax2.axvline(x=1.0, color='orange', linestyle=':', linewidth=2, alpha=0.8)
    ax2.set_title('Numerical Attractors')
    ax2.set_xlabel('Parameter μ')
    ax2.set_ylabel('x')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(mu_min, mu_max)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.show()
    
    print("=== Theoretical vs Numerical Comparison ===")
    print("Above μ = 1: Fixed points become unstable and chaos emerges")
    print("Below μ = 1: System converges to one of the stable fixed points")
    print("The transition at μ = 1 marks the onset of chaos")

if __name__ == "__main__":
    # Example usage
    print("Example: Zooming into the chaotic region")
    explore_bifurcation_diagram(mu_min=1.8, mu_max=2.2, mu_points=1500) 