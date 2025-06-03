"""
Quick R-Pentomino Launcher - Complexity Lab

Simple script to quickly run different R-pentomino visualizations.
"""

from game_of_life import GameOfLife
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def run_live_r_pentomino(generations=1000, grid_size=100, speed=50):
    """Run live R-pentomino animation"""
    print(f"R-Pentomino Live Animation - {generations} generations")
    print("Close window to stop. Press any key to continue...")
    
    # Create game
    game = GameOfLife(grid_size, grid_size)
    game.set_initial_state("r_pentomino")
    
    # Setup animation
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(game.grid, cmap='binary', interpolation='nearest')
    
    # Info text
    info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                       verticalalignment='top', fontsize=12,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def animate(frame):
        game.step()
        stats = game.get_statistics()
        im.set_array(game.grid)
        ax.set_title(f'R-Pentomino - Generation {game.generation}', fontsize=16)
        
        info = (f"Gen: {game.generation}\n"
               f"Pop: {stats['population']}\n"
               f"Density: {stats['density']:.4f}")
        info_text.set_text(info)
        
        return [im, info_text]
    
    ani = animation.FuncAnimation(fig, animate, frames=generations, 
                                interval=speed, blit=True, repeat=False)
    plt.show()


def show_r_pentomino_snapshots():
    """Show R-pentomino at different time points"""
    print("R-Pentomino Evolution Snapshots")
    
    game = GameOfLife(80, 80)
    game.set_initial_state("r_pentomino")
    
    # Time points to capture
    snapshots = [0, 10, 50, 100, 200, 500, 1000]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, target_gen in enumerate(snapshots):
        if i >= len(axes):
            break
            
        # Run to target generation
        while game.generation < target_gen:
            game.step()
        
        # Capture snapshot
        stats = game.get_statistics()
        axes[i].imshow(game.grid, cmap='binary', interpolation='nearest')
        axes[i].set_title(f'Gen {game.generation}\nPop: {stats["population"]}')
        axes[i].axis('off')
    
    # Remove empty subplot
    if len(snapshots) < len(axes):
        axes[-1].remove()
    
    plt.suptitle('R-Pentomino Evolution Over Time', fontsize=16)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("=== R-PENTOMINO QUICK LAUNCHER ===")
    print()
    print("Available demonstrations:")
    print("1. Live animation (1000 generations)")
    print("2. Quick live demo (200 generations, faster)")
    print("3. Evolution snapshots")
    print("4. Extended run (2000 generations)")
    print()
    
    try:
        choice = input("Select option (1-4, or Enter for live demo): ").strip()
        
        if choice == "1" or choice == "":
            run_live_r_pentomino(1000, 100, 50)
        elif choice == "2":
            run_live_r_pentomino(200, 60, 30)
        elif choice == "3":
            show_r_pentomino_snapshots()
        elif choice == "4":
            run_live_r_pentomino(2000, 120, 40)
        else:
            print("Running default live demo...")
            run_live_r_pentomino(500, 80, 40)
            
    except KeyboardInterrupt:
        print("\nAnimation stopped by user.")
    except Exception as e:
        print(f"Error: {e}")
        print("Running basic demo...")
        run_live_r_pentomino(100, 50, 100) 