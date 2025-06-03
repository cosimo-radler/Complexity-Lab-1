"""
R-Pentomino Simulation for Complexity Lab

The R-pentomino is famous for its long and complex evolution before stabilizing.
This script runs the R-pentomino for 1000 generations and analyzes its behavior.

The R-pentomino pattern:
 XX
XX
 X

It takes approximately 1103 generations to stabilize completely.
"""

import numpy as np
import matplotlib.pyplot as plt
from game_of_life import GameOfLife


def run_r_pentomino_analysis(generations: int = 1000, grid_size: int = 100):
    """
    Run R-pentomino simulation and analyze its evolution
    
    Args:
        generations: Number of generations to simulate
        grid_size: Size of the grid (grid_size x grid_size)
    """
    print("R-Pentomino Simulation - Complexity Lab")
    print("=" * 50)
    print(f"Running for {generations} generations on {grid_size}x{grid_size} grid")
    print()
    
    # Create game with R-pentomino
    game = GameOfLife(grid_size, grid_size)
    game.set_initial_state("r_pentomino")
    
    # Track statistics
    stats_history = []
    population_history = []
    
    print("Initial R-pentomino configuration:")
    initial_stats = game.get_statistics()
    print(f"Generation {initial_stats['generation']}: Population = {initial_stats['population']}")
    stats_history.append(initial_stats)
    population_history.append(initial_stats['population'])
    
    # Run simulation
    print("\nRunning simulation...")
    checkpoint_intervals = [100, 200, 500, 1000]
    
    for gen in range(1, generations + 1):
        game.step()
        stats = game.get_statistics()
        stats_history.append(stats)
        population_history.append(stats['population'])
        
        # Print checkpoints
        if gen in checkpoint_intervals:
            print(f"Generation {gen}: Population = {stats['population']}")
    
    print(f"\nSimulation complete!")
    final_stats = game.get_statistics()
    print(f"Final state - Generation {final_stats['generation']}: Population = {final_stats['population']}")
    
    # Analyze results
    analyze_r_pentomino_evolution(stats_history, population_history)
    
    # Show final state
    print("\nDisplaying final configuration...")
    game.display()
    
    return game, stats_history


def analyze_r_pentomino_evolution(stats_history, population_history):
    """
    Analyze the evolution patterns of the R-pentomino
    
    Args:
        stats_history: List of statistics dictionaries
        population_history: List of population counts
    """
    print("\n" + "="*50)
    print("ANALYSIS OF R-PENTOMINO EVOLUTION")
    print("="*50)
    
    # Basic statistics
    max_population = max(population_history)
    min_population = min(population_history)
    final_population = population_history[-1]
    
    print(f"Maximum population reached: {max_population}")
    print(f"Minimum population: {min_population}")
    print(f"Final population: {final_population}")
    
    # Find when maximum was reached
    max_gen = population_history.index(max_population)
    print(f"Maximum population reached at generation: {max_gen}")
    
    # Look for stabilization (last 50 generations have same population)
    if len(population_history) >= 50:
        last_50 = population_history[-50:]
        if len(set(last_50)) == 1:
            print(f"Population stabilized at: {last_50[0]}")
        else:
            print("Population has not fully stabilized yet")
            print(f"Population range in last 50 generations: {min(last_50)} - {max(last_50)}")
    
    # Plot evolution
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Full evolution
    plt.subplot(2, 2, 1)
    plt.plot(population_history, linewidth=1)
    plt.title('R-Pentomino Population Evolution (Full)')
    plt.xlabel('Generation')
    plt.ylabel('Population')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: First 200 generations (where most action happens)
    plt.subplot(2, 2, 2)
    plt.plot(population_history[:200], linewidth=1, color='red')
    plt.title('R-Pentomino Evolution (First 200 generations)')
    plt.xlabel('Generation')
    plt.ylabel('Population')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Population differences (rate of change)
    plt.subplot(2, 2, 3)
    differences = [population_history[i+1] - population_history[i] 
                  for i in range(len(population_history)-1)]
    plt.plot(differences, linewidth=1, color='green')
    plt.title('Population Change Rate')
    plt.xlabel('Generation')
    plt.ylabel('Population Change')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Last 200 generations
    plt.subplot(2, 2, 4)
    plt.plot(population_history[-200:], linewidth=1, color='purple')
    plt.title('Final 200 Generations')
    plt.xlabel('Generation (relative to end)')
    plt.ylabel('Population')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Mathematical analysis
    print("\nMATHEMATICAL PROPERTIES:")
    print(f"Growth ratio (max/initial): {max_population / population_history[0]:.2f}")
    print(f"Stabilization ratio (final/max): {final_population / max_population:.2f}")
    
    # Calculate some complexity measures
    total_variation = sum(abs(differences))
    print(f"Total variation: {total_variation}")
    print(f"Average absolute change per generation: {total_variation / len(differences):.2f}")


def interactive_r_pentomino():
    """
    Interactive mode for exploring R-pentomino with different parameters
    """
    print("Interactive R-Pentomino Explorer")
    print("-" * 30)
    
    grid_size = int(input("Enter grid size (50-200, default 100): ") or "100")
    generations = int(input("Enter number of generations (default 1000): ") or "1000")
    
    animate = input("Show animation? (y/n, default n): ").lower().startswith('y')
    
    game = GameOfLife(grid_size, grid_size)
    game.set_initial_state("r_pentomino")
    
    if animate:
        print("Starting animated simulation...")
        game.run_simulation(generations=generations, animate=True)
    else:
        print("Running analysis...")
        run_r_pentomino_analysis(generations, grid_size)


if __name__ == "__main__":
    print("Choose an option:")
    print("1. Run standard R-pentomino analysis (1000 generations)")
    print("2. Interactive mode")
    print("3. Quick demo (100 generations)")
    
    choice = input("Enter choice (1-3): ") or "1"
    
    if choice == "1":
        run_r_pentomino_analysis(1000, 100)
    elif choice == "2":
        interactive_r_pentomino()
    elif choice == "3":
        run_r_pentomino_analysis(100, 60)
    else:
        print("Running default analysis...")
        run_r_pentomino_analysis(1000, 100) 