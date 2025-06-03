"""
R-Pentomino Variations Analysis - Complexity Lab

Runs 4 R-pentomino simulations with different starting positions and orientations
to demonstrate how initial conditions affect the final stable states.

Shows that even small changes can lead to different outcomes.
"""

import numpy as np
import matplotlib.pyplot as plt
from game_of_life import GameOfLife
import time


def create_r_pentomino_pattern():
    """Create the basic R-pentomino pattern (5 cells)"""
    # R-pentomino: 
    #  XX
    # XX
    #  X
    return [(0, 1), (0, 2), (1, 0), (1, 1), (2, 1)]


def rotate_pattern(pattern, rotations=1):
    """Rotate a pattern 90 degrees clockwise, rotations times"""
    for _ in range(rotations % 4):
        # Rotate 90 degrees clockwise: (x, y) -> (y, -x)
        pattern = [(y, -x) for x, y in pattern]
        
        # Normalize to positive coordinates
        min_x = min(x for x, y in pattern)
        min_y = min(y for x, y in pattern)
        pattern = [(x - min_x, y - min_y) for x, y in pattern]
    
    return pattern


def place_pattern_on_grid(game, pattern, center_x, center_y):
    """Place a pattern on the grid at specified center position"""
    game.grid.fill(0)  # Clear grid
    
    for dx, dy in pattern:
        x = center_x + dx - 1  # Adjust for pattern centering
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


def run_r_pentomino_variation(variation_num, center_x, center_y, rotation=0, 
                            max_generations=2000, grid_size=100):
    """
    Run R-pentomino simulation with specified starting position and rotation
    """
    print(f"Running variation {variation_num}...")
    print(f"  Position: ({center_x}, {center_y}), Rotation: {rotation * 90}°")
    
    # Initialize game
    game = GameOfLife(grid_size, grid_size)
    
    # Create and place R-pentomino
    pattern = create_r_pentomino_pattern()
    rotated_pattern = rotate_pattern(pattern, rotation)
    place_pattern_on_grid(game, rotated_pattern, center_x, center_y)
    
    population_history = [game.get_statistics()['population']]
    
    # Run simulation
    for gen in range(1, max_generations + 1):
        game.step()
        stats = game.get_statistics()
        population_history.append(stats['population'])
        
        # Check for stability every 10 generations
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
    print(f"  → Final population: {game.get_statistics()['population']}")
    
    is_stable, stability_gen = detect_stability(population_history)
    return game, population_history, is_stable, stability_gen, max_generations


def run_four_r_pentomino_variations(max_generations=2000, grid_size=100):
    """
    Run 4 R-pentomino variations with different starting conditions
    """
    print("R-Pentomino Variations Analysis - 4 Different Starting Conditions")
    print("=" * 70)
    print(f"Max generations: {max_generations}")
    print(f"Grid size: {grid_size}x{grid_size}")
    print()
    
    # Define 4 different starting conditions
    variations = [
        {"center_x": grid_size//2, "center_y": grid_size//2, "rotation": 0, "name": "Center, 0°"},
        {"center_x": grid_size//3, "center_y": grid_size//3, "rotation": 1, "name": "Upper-left, 90°"},
        {"center_x": 2*grid_size//3, "center_y": grid_size//3, "rotation": 2, "name": "Upper-right, 180°"},
        {"center_x": grid_size//2, "center_y": 2*grid_size//3, "rotation": 3, "name": "Lower-center, 270°"}
    ]
    
    results = []
    start_time = time.time()
    
    # Run variations
    for i, var in enumerate(variations, 1):
        iter_start = time.time()
        game, pop_history, is_stable, stability_gen, final_gen = run_r_pentomino_variation(
            i, var["center_x"], var["center_y"], var["rotation"], max_generations, grid_size
        )
        iter_time = time.time() - iter_start
        
        results.append({
            'iteration': i,
            'name': var["name"],
            'position': (var["center_x"], var["center_y"]),
            'rotation': var["rotation"],
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
    display_variation_results(results)
    plot_variation_final_states(results)
    plot_variation_population_evolution(results)
    
    return results


def display_variation_results(results):
    """Display summary of variation results"""
    print("VARIATION RESULTS SUMMARY")
    print("=" * 70)
    
    for result in results:
        print(f"Variation {result['iteration']} ({result['name']}):")
        print(f"  Position: {result['position']}, Rotation: {result['rotation'] * 90}°")
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
    print("COMPARISON:")
    final_pops = [r['final_population'] for r in results]
    max_pops = [r['max_population'] for r in results]
    
    print(f"Final populations: {final_pops}")
    print(f"Are all final populations the same? {len(set(final_pops)) == 1}")
    print(f"Max populations: {max_pops}")
    print(f"Are all max populations the same? {len(set(max_pops)) == 1}")
    
    if len(set(final_pops)) > 1:
        print("✓ Different starting conditions led to different outcomes!")
    else:
        print("→ All variations converged to the same final state")


def plot_variation_final_states(results):
    """Plot the final states of all 4 variations"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for i, result in enumerate(results):
        ax = axes[i]
        game = result['game']
        
        # Display final state
        im = ax.imshow(game.grid, cmap='binary', interpolation='nearest')
        
        # Create title with variation info
        status = "STABLE" if result['is_stable'] else "CUT OFF"
        if result['is_stable']:
            title = (f"Variation {result['iteration']}: {result['name']}\n"
                    f"{status} at Gen {result['stability_generation']}, "
                    f"Pop: {result['final_population']}")
        else:
            title = (f"Variation {result['iteration']}: {result['name']}\n"
                    f"{status} at Gen {result['final_generation']}, "
                    f"Pop: {result['final_population']}")
        
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    
    plt.suptitle('R-Pentomino Final States - Different Starting Conditions', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_variation_population_evolution(results):
    """Plot population evolution for all variations"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    colors = ['blue', 'red', 'green', 'purple']
    
    # Plot 1: Full evolution
    for i, result in enumerate(results):
        generations = list(range(len(result['population_history'])))
        ax1.plot(generations, result['population_history'], 
                color=colors[i], linewidth=1, 
                label=f'Var {result["iteration"]}: {result["name"]}')
        
        # Mark stability point
        if result['is_stable']:
            stable_gen = result['stability_generation']
            stable_pop = result['population_history'][stable_gen]
            ax1.plot(stable_gen, stable_pop, 'o', color=colors[i], markersize=6)
    
    ax1.set_title('Population Evolution - All 4 Variations')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Population')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: First 300 generations (detail view)
    for i, result in enumerate(results):
        generations = list(range(min(300, len(result['population_history']))))
        pop_subset = result['population_history'][:len(generations)]
        ax2.plot(generations, pop_subset, 
                color=colors[i], linewidth=2, 
                label=f'Var {result["iteration"]}: {result["name"]}')
    
    ax2.set_title('Population Evolution - First 300 Generations')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Population')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("=== R-PENTOMINO VARIATIONS ANALYSIS ===")
    print()
    print("This script runs 4 R-pentomino simulations with different:")
    print("- Starting positions")
    print("- Orientations (rotations)")
    print("to show how initial conditions affect the outcomes.")
    print()
    
    # Configuration
    max_gens = int(input("Maximum generations per simulation (default 2000): ") or "2000")
    grid_size = int(input("Grid size (default 100): ") or "100")
    
    print()
    print("Starting variations analysis...")
    print()
    
    # Run the analysis
    results = run_four_r_pentomino_variations(max_gens, grid_size)
    
    print("\nVariations analysis complete! Check the generated plots.") 