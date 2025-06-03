"""
R-Pentomino Boundary Effects Analysis - Complexity Lab

Places R-pentominos near grid boundaries to see how edge effects 
and finite grid size affect the evolution and final stable states.

This might finally show different outcomes!
"""

import numpy as np
import matplotlib.pyplot as plt
from game_of_life import GameOfLife
import time


def create_r_pentomino_pattern():
    """Create the basic R-pentomino pattern"""
    return [(0, 1), (0, 2), (1, 0), (1, 1), (2, 1)]


def place_pattern_on_grid(game, pattern, center_x, center_y):
    """Place a pattern on the grid at specified center position"""
    game.grid.fill(0)
    
    for dx, dy in pattern:
        x = center_x + dx - 1
        y = center_y + dy - 1
        
        if 0 <= x < game.height and 0 <= y < game.width:
            game.grid[x, y] = 1


def detect_stability(population_history, stability_threshold=50, min_generations=100):
    """Detect if the population has stabilized"""
    if len(population_history) < min_generations + stability_threshold:
        return False, -1
    
    recent_populations = population_history[-stability_threshold:]
    
    if len(set(recent_populations)) == 1:
        stable_pop = recent_populations[0]
        for i in range(len(population_history) - stability_threshold - 1, -1, -1):
            if population_history[i] != stable_pop:
                return True, i + 1
        return True, 0
    
    return False, -1


def run_boundary_r_pentomino(variation_num, center_x, center_y, description,
                           max_generations=2000, grid_size=80):
    """Run R-pentomino simulation near boundaries"""
    print(f"Running boundary test {variation_num}: {description}")
    print(f"  Position: ({center_x}, {center_y})")
    
    # Initialize game
    game = GameOfLife(grid_size, grid_size)
    
    # Place R-pentomino
    pattern = create_r_pentomino_pattern()
    place_pattern_on_grid(game, pattern, center_x, center_y)
    
    population_history = [game.get_statistics()['population']]
    
    # Run simulation
    for gen in range(1, max_generations + 1):
        game.step()
        stats = game.get_statistics()
        population_history.append(stats['population'])
        
        # Check for stability
        if gen % 10 == 0 and gen > 100:
            is_stable, stability_gen = detect_stability(population_history)
            if is_stable:
                print(f"  → Reached stability at generation {stability_gen}")
                print(f"  → Final population: {stats['population']}")
                return game, population_history, True, stability_gen, gen
        
        # Progress updates
        if gen % 400 == 0:
            print(f"  → Generation {gen}, Population: {stats['population']}")
    
    # Reached max generations
    print(f"  → Cut off at generation {max_generations}")
    final_stats = game.get_statistics()
    print(f"  → Final population: {final_stats['population']}")
    
    is_stable, stability_gen = detect_stability(population_history)
    return game, population_history, is_stable, stability_gen, max_generations


def run_boundary_effects_analysis(max_generations=2000, grid_size=80):
    """
    Run 4 R-pentomino simulations near different boundaries
    """
    print("R-Pentomino Boundary Effects Analysis")
    print("=" * 60)
    print(f"Max generations: {max_generations}")
    print(f"Grid size: {grid_size}x{grid_size}")
    print("Testing how boundary proximity affects evolution...")
    print()
    
    # Define boundary positions (closer to edges)
    boundary_tests = [
        {"center_x": 10, "center_y": 10, "description": "Near top-left corner"},
        {"center_x": grid_size-10, "center_y": 10, "description": "Near top-right corner"},  
        {"center_x": 10, "center_y": grid_size-10, "description": "Near bottom-left corner"},
        {"center_x": grid_size-10, "center_y": grid_size-10, "description": "Near bottom-right corner"}
    ]
    
    results = []
    start_time = time.time()
    
    # Run boundary tests
    for i, test in enumerate(boundary_tests, 1):
        iter_start = time.time()
        game, pop_history, is_stable, stability_gen, final_gen = run_boundary_r_pentomino(
            i, test["center_x"], test["center_y"], test["description"], 
            max_generations, grid_size
        )
        iter_time = time.time() - iter_start
        
        results.append({
            'iteration': i,
            'description': test["description"],
            'position': (test["center_x"], test["center_y"]),
            'game': game,
            'population_history': pop_history,
            'is_stable': is_stable,
            'stability_generation': stability_gen,
            'final_generation': final_gen,
            'final_population': game.get_statistics()['population'],
            'max_population': max(pop_history),
            'runtime': iter_time
        })
        
        print(f"  → Completed in {iter_time:.1f} seconds")
        print()
    
    total_time = time.time() - start_time
    print(f"Total simulation time: {total_time:.1f} seconds")
    print()
    
    # Display results
    display_boundary_results(results)
    plot_boundary_final_states(results)
    plot_boundary_population_evolution(results)
    
    return results


def display_boundary_results(results):
    """Display summary of boundary effect results"""
    print("BOUNDARY EFFECTS RESULTS")
    print("=" * 60)
    
    for result in results:
        print(f"Test {result['iteration']}: {result['description']}")
        print(f"  Position: {result['position']}")
        print(f"  Final Generation: {result['final_generation']}")
        print(f"  Final Population: {result['final_population']}")
        print(f"  Max Population: {result['max_population']}")
        
        if result['is_stable']:
            print(f"  Status: STABLE (reached at generation {result['stability_generation']})")
        else:
            print(f"  Status: CUT OFF (not stable after {result['final_generation']} generations)")
        
        print(f"  Runtime: {result['runtime']:.1f}s")
        print()
    
    # Compare outcomes
    print("BOUNDARY EFFECTS COMPARISON:")
    final_pops = [r['final_population'] for r in results]
    max_pops = [r['max_population'] for r in results]
    stability_gens = [r['stability_generation'] for r in results if r['is_stable']]
    
    print(f"Final populations: {final_pops}")
    print(f"Max populations: {max_pops}")
    if stability_gens:
        print(f"Stability generations: {stability_gens}")
    
    unique_finals = len(set(final_pops))
    unique_maxes = len(set(max_pops))
    
    print(f"\nUnique final populations: {unique_finals}")
    print(f"Unique max populations: {unique_maxes}")
    
    if unique_finals > 1 or unique_maxes > 1:
        print("✓ BOUNDARY EFFECTS DETECTED! Different outcomes observed.")
    else:
        print("→ Even near boundaries, all outcomes were identical.")
        print("  The R-pentomino is remarkably robust!")


def plot_boundary_final_states(results):
    """Plot the final states near boundaries"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    for i, result in enumerate(results):
        ax = axes[i]
        game = result['game']
        
        # Display final state
        im = ax.imshow(game.grid, cmap='binary', interpolation='nearest')
        
        # Create title
        status = "STABLE" if result['is_stable'] else "CUT OFF"
        if result['is_stable']:
            title = (f"Test {result['iteration']}: {result['description']}\n"
                    f"{status} at Gen {result['stability_generation']}, "
                    f"Pop: {result['final_population']}")
        else:
            title = (f"Test {result['iteration']}: {result['description']}\n"
                    f"{status} at Gen {result['final_generation']}, "
                    f"Pop: {result['final_population']}")
        
        ax.set_title(title, fontsize=9)
        ax.axis('off')
        
        # Mark the starting position
        pos_x, pos_y = result['position']
        ax.plot(pos_y, pos_x, 'r+', markersize=10, markeredgewidth=2)
    
    plt.suptitle('R-Pentomino Near Boundaries - Final States', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_boundary_population_evolution(results):
    """Plot population evolution for boundary tests"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    colors = ['blue', 'red', 'green', 'purple']
    
    # Plot 1: Full evolution
    for i, result in enumerate(results):
        generations = list(range(len(result['population_history'])))
        ax1.plot(generations, result['population_history'], 
                color=colors[i], linewidth=1, 
                label=f'Test {result["iteration"]}: {result["description"]}')
        
        # Mark stability point
        if result['is_stable']:
            stable_gen = result['stability_generation']
            stable_pop = result['population_history'][stable_gen]
            ax1.plot(stable_gen, stable_pop, 'o', color=colors[i], markersize=6)
    
    ax1.set_title('Population Evolution - Boundary Effects')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Population')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: First 300 generations
    for i, result in enumerate(results):
        generations = list(range(min(300, len(result['population_history']))))
        pop_subset = result['population_history'][:len(generations)]
        ax2.plot(generations, pop_subset, 
                color=colors[i], linewidth=2, 
                label=f'Test {result["iteration"]}: {result["description"]}')
    
    ax2.set_title('Population Evolution - First 300 Generations (Detail)')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Population')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("=== R-PENTOMINO BOUNDARY EFFECTS ANALYSIS ===")
    print()
    print("This script tests R-pentomino evolution near grid boundaries")
    print("to see if finite grid size and edge effects create different outcomes.")
    print()
    
    # Use smaller grid to emphasize boundary effects
    max_gens = int(input("Maximum generations per simulation (default 2000): ") or "2000")
    grid_size = int(input("Grid size (default 80, smaller to emphasize boundaries): ") or "80")
    
    print()
    print("Starting boundary effects analysis...")
    print()
    
    # Run the analysis
    results = run_boundary_effects_analysis(max_gens, grid_size)
    
    print("\nBoundary effects analysis complete!") 