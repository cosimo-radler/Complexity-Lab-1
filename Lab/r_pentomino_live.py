"""
R-Pentomino Live Animation - Complexity Lab

This script shows the R-pentomino evolution in real-time with live animation.
Watch the complex patterns emerge and evolve over 1000+ generations.

The R-pentomino pattern:
 XX
XX 
 X
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from game_of_life import GameOfLife


def create_r_pentomino_animation(generations: int = 1000, grid_size: int = 100, 
                                interval: int = 50, save_animation: bool = False):
    """
    Create live animation of R-pentomino evolution
    
    Args:
        generations: Number of generations to animate
        grid_size: Size of the grid
        interval: Animation speed in milliseconds (lower = faster)
        save_animation: Whether to save as gif
    """
    print("Starting R-Pentomino Live Animation")
    print("=" * 40)
    print(f"Grid size: {grid_size}x{grid_size}")
    print(f"Generations: {generations}")
    print(f"Animation speed: {interval}ms per frame")
    print("\nClose the window to stop the animation.")
    print()
    
    # Initialize game
    game = GameOfLife(grid_size, grid_size)
    game.set_initial_state("r_pentomino")
    
    # Setup the figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create initial display
    im = ax.imshow(game.grid, cmap='binary', interpolation='nearest', 
                   vmin=0, vmax=1, animated=True)
    
    # Add colorbar and styling
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f'R-Pentomino Evolution - Generation {game.generation}', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    
    # Add grid lines
    ax.set_xticks(np.arange(-0.5, grid_size, 10), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_size, 10), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', alpha=0.2)
    
    # Population counter text
    pop_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                      verticalalignment='top', fontsize=12,
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Track statistics
    population_history = [game.get_statistics()['population']]
    max_population = population_history[0]
    
    def animate(frame):
        nonlocal max_population
        
        # Advance one generation
        game.step()
        stats = game.get_statistics()
        population_history.append(stats['population'])
        
        # Update max population
        if stats['population'] > max_population:
            max_population = stats['population']
        
        # Update display
        im.set_array(game.grid)
        ax.set_title(f'R-Pentomino Evolution - Generation {game.generation}', 
                    fontsize=14, fontweight='bold')
        
        # Update population info
        pop_info = (f"Generation: {game.generation}\n"
                   f"Population: {stats['population']}\n"
                   f"Max Population: {max_population}\n"
                   f"Density: {stats['density']:.4f}")
        pop_text.set_text(pop_info)
        
        return [im, pop_text]
    
    # Create and run animation
    ani = animation.FuncAnimation(fig, animate, frames=generations, 
                                interval=interval, blit=True, repeat=False)
    
    # Save animation if requested
    if save_animation:
        print("Saving animation as r_pentomino_evolution.gif...")
        ani.save('r_pentomino_evolution.gif', writer='pillow', fps=10)
        print("Animation saved!")
    
    # Show the animation
    plt.tight_layout()
    plt.show()
    
    return ani, population_history


def create_dual_view_animation(generations: int = 500, grid_size: int = 80):
    """
    Create animation with both grid view and population graph
    """
    print("Starting Dual-View R-Pentomino Animation")
    print("Grid + Population Graph")
    print("=" * 40)
    
    # Initialize game
    game = GameOfLife(grid_size, grid_size)
    game.set_initial_state("r_pentomino")
    
    # Setup figure with subplots
    fig = plt.figure(figsize=(15, 6))
    
    # Grid view
    ax1 = plt.subplot(1, 2, 1)
    im = ax1.imshow(game.grid, cmap='binary', interpolation='nearest', 
                    vmin=0, vmax=1, animated=True)
    ax1.set_title('R-Pentomino Grid')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    
    # Population graph
    ax2 = plt.subplot(1, 2, 2)
    population_history = [game.get_statistics()['population']]
    line, = ax2.plot(population_history, 'b-', linewidth=2)
    ax2.set_title('Population Over Time')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Population')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, generations)
    ax2.set_ylim(0, 200)  # Will adjust dynamically
    
    def animate_dual(frame):
        # Advance generation
        game.step()
        stats = game.get_statistics()
        population_history.append(stats['population'])
        
        # Update grid
        im.set_array(game.grid)
        ax1.set_title(f'R-Pentomino Grid - Gen {game.generation}')
        
        # Update population graph
        line.set_data(range(len(population_history)), population_history)
        if len(population_history) > 0:
            ax2.set_ylim(0, max(population_history) * 1.1)
        
        return [im, line]
    
    # Create animation
    ani = animation.FuncAnimation(fig, animate_dual, frames=generations, 
                                interval=100, blit=True, repeat=False)
    
    plt.tight_layout()
    plt.show()
    
    return ani


def quick_live_demo():
    """Quick demo with fast animation"""
    print("Quick R-Pentomino Demo (200 generations)")
    return create_r_pentomino_animation(generations=200, grid_size=60, interval=30)


if __name__ == "__main__":
    print("R-Pentomino Live Animation Options:")
    print("1. Standard live view (1000 generations)")
    print("2. Quick demo (200 generations, faster)")
    print("3. Dual view (grid + population graph)")
    print("4. Custom parameters")
    
    choice = input("Enter choice (1-4): ") or "1"
    
    if choice == "1":
        create_r_pentomino_animation()
    
    elif choice == "2":
        quick_live_demo()
    
    elif choice == "3":
        create_dual_view_animation()
    
    elif choice == "4":
        print("Custom Parameters:")
        generations = int(input("Generations (default 1000): ") or "1000")
        grid_size = int(input("Grid size (default 100): ") or "100")
        speed = int(input("Animation speed in ms (default 50, lower=faster): ") or "50")
        save = input("Save as GIF? (y/n): ").lower().startswith('y')
        
        create_r_pentomino_animation(generations, grid_size, speed, save)
    
    else:
        print("Running default animation...")
        create_r_pentomino_animation() 