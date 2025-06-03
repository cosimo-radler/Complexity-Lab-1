"""
R-Pentomino Stability Analysis - Complexity Lab

Runs 4 iterations of R-pentomino evolution and analyzes when each reaches stability.
Shows final stable states and reports timing information.

The R-pentomino pattern:
 XX
XX 
 X
"""

import numpy as np
import matplotlib.pyplot as plt
from game_of_life import GameOfLife
import time


def detect_stability(population_history, stability_threshold=50, min_generations=100):
    """
    Detect if the population has stabilized
    
    Args:
        population_history: List of population counts
        stability_threshold: Number of consecutive unchanged generations to consider stable
        min_generations: Minimum generations before checking for stability
    
    Returns:
        (is_stable, stability_generation)
    """
    if len(population_history) < min_generations + stability_threshold:
        return False, -1
    
    # Check last 'stability_threshold' generations
    recent_populations = population_history[-stability_threshold:]
    
    # Check if all recent populations are the same
    if len(set(recent_populations)) == 1:
        # Found stability - find when it started
        stable_pop = recent_populations[0]
        stability_start = len(population_history) - stability_threshold
        
        # Look backwards to find when stability actually began
        for i in range(len(population_history) - stability_threshold - 1, -1, -1):
            if population_history[i] != stable_pop:
                return True, i + 1
        
        return True, 0
    
    return False, -1


def run_single_r_pentomino_to_stability(iteration_num, max_generations=2000, 
                                       grid_size=100, stability_threshold=50):
    """
    Run a single R-pentomino simulation until stability or cutoff
    
    Returns:
        (game, population_history, is_stable, stability_gen, final_gen)
    """
    print(f"Running iteration {iteration_num}...")
    
    # Initialize game
    game = GameOfLife(grid_size, grid_size)
    game.set_initial_state("r_pentomino")
    
    population_history = [game.get_statistics()['population']]
    
    # Run simulation
    for gen in range(1, max_generations + 1):
        game.step()
        stats = game.get_statistics()
        population_history.append(stats['population'])
        
        # Check for stability every 10 generations (after minimum)
        if gen % 10 == 0 and gen > 100:
            is_stable, stability_gen = detect_stability(population_history, stability_threshold)
            if is_stable:
                print(f"  → Reached stability at generation {stability_gen}")
                print(f"  → Final population: {stats['population']}")
                return game, population_history, True, stability_gen, gen
        
        # Progress updates
        if gen % 200 == 0:
            print(f"  → Generation {gen}, Population: {stats['population']}")
    
    # Reached max generations without stability
    print(f"  → Cut off at generation {max_generations}")
    print(f"  → Final population: {game.get_statistics()['population']}")
    
    # Check if it was actually stable at the end
    is_stable, stability_gen = detect_stability(population_history, stability_threshold)
    return game, population_history, is_stable, stability_gen, max_generations


def run_four_r_pentomino_simulations(max_generations=2000, grid_size=100):
    """
    Run 4 R-pentomino simulations and analyze their stability
    """
    print("R-Pentomino Stability Analysis - 4 Iterations")
    print("=" * 60)
    print(f"Max generations: {max_generations}")
    print(f"Grid size: {grid_size}x{grid_size}")
    print(f"Stability threshold: 50 consecutive unchanged generations")
    print()
    
    results = []
    start_time = time.time()
    
    # Run 4 simulations
    for i in range(1, 5):
        iter_start = time.time()
        game, pop_history, is_stable, stability_gen, final_gen = run_single_r_pentomino_to_stability(
            i, max_generations, grid_size
        )
        iter_time = time.time() - iter_start
        
        results.append({
            'iteration': i,
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
    display_results_summary(results)
    plot_final_states(results)
    plot_population_evolution(results)
    
    return results


def display_results_summary(results):
    """Display summary of all simulation results"""
    print("SIMULATION RESULTS SUMMARY")
    print("=" * 60)
    
    for result in results:
        print(f"Iteration {result['iteration']}:")
        print(f"  Final Generation: {result['final_generation']}")
        print(f"  Final Population: {result['final_population']}")
        print(f"  Max Population: {result['max_population']}")
        
        if result['is_stable']:
            print(f"  Status: STABLE (reached at generation {result['stability_generation']})")
        else:
            print(f"  Status: CUT OFF (not stable after {result['final_generation']} generations)")
        
        print(f"  Runtime: {result['runtime']:.1f}s")
        print()


def plot_final_states(results):
    """Plot the final states of all 4 simulations"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    for i, result in enumerate(results):
        ax = axes[i]
        game = result['game']
        
        # Display final state
        im = ax.imshow(game.grid, cmap='binary', interpolation='nearest')
        
        # Create title with status
        status = "STABLE" if result['is_stable'] else "CUT OFF"
        if result['is_stable']:
            title = (f"Iteration {result['iteration']} - {status}\n"
                    f"Stable at Gen {result['stability_generation']}, "
                    f"Pop: {result['final_population']}")
        else:
            title = (f"Iteration {result['iteration']} - {status}\n"
                    f"Final Gen {result['final_generation']}, "
                    f"Pop: {result['final_population']}")
        
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    
    plt.suptitle('R-Pentomino Final States - 4 Iterations', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_population_evolution(results):
    """Plot population evolution for all simulations"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    colors = ['blue', 'red', 'green', 'purple']
    
    # Plot 1: Full evolution
    for i, result in enumerate(results):
        generations = list(range(len(result['population_history'])))
        ax1.plot(generations, result['population_history'], 
                color=colors[i], linewidth=1, label=f'Iteration {result["iteration"]}')
        
        # Mark stability point
        if result['is_stable']:
            stable_gen = result['stability_generation']
            stable_pop = result['population_history'][stable_gen]
            ax1.plot(stable_gen, stable_pop, 'o', color=colors[i], markersize=8)
            ax1.annotate(f'Stable\nGen {stable_gen}', 
                        xy=(stable_gen, stable_pop),
                        xytext=(stable_gen + 100, stable_pop + 10),
                        fontsize=8, ha='center',
                        arrowprops=dict(arrowstyle='->', color=colors[i]))
    
    ax1.set_title('Population Evolution - All 4 Iterations')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Population')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: First 300 generations (where most action happens)
    for i, result in enumerate(results):
        generations = list(range(min(300, len(result['population_history']))))
        pop_subset = result['population_history'][:len(generations)]
        ax2.plot(generations, pop_subset, 
                color=colors[i], linewidth=2, label=f'Iteration {result["iteration"]}')
    
    ax2.set_title('Population Evolution - First 300 Generations (Detail View)')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Population')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def analyze_mathematical_properties(results):
    """Analyze mathematical properties of the simulations"""
    print("MATHEMATICAL ANALYSIS")
    print("=" * 40)
    
    # Overall statistics
    stable_count = sum(1 for r in results if r['is_stable'])
    print(f"Simulations that reached stability: {stable_count}/4")
    
    if stable_count > 0:
        stable_results = [r for r in results if r['is_stable']]
        avg_stability_time = np.mean([r['stability_generation'] for r in stable_results])
        print(f"Average time to stability: {avg_stability_time:.1f} generations")
        
        stability_times = [r['stability_generation'] for r in stable_results]
        print(f"Stability time range: {min(stability_times)} - {max(stability_times)} generations")
    
    # Population statistics
    max_populations = [r['max_population'] for r in results]
    final_populations = [r['final_population'] for r in results]
    
    print(f"\nPopulation Statistics:")
    print(f"Max populations: {max_populations}")
    print(f"Final populations: {final_populations}")
    print(f"Average max population: {np.mean(max_populations):.1f}")
    print(f"Average final population: {np.mean(final_populations):.1f}")


if __name__ == "__main__":
    print("=== R-PENTOMINO STABILITY ANALYSIS ===")
    print()
    print("This script runs 4 R-pentomino simulations to analyze stability patterns.")
    print("Each simulation runs until stability is detected or a cutoff is reached.")
    print()
    
    # Configuration
    max_gens = int(input("Maximum generations per simulation (default 2000): ") or "2000")
    grid_size = int(input("Grid size (default 100): ") or "100")
    
    print()
    print("Starting simulations...")
    print()
    
    # Run the analysis
    results = run_four_r_pentomino_simulations(max_gens, grid_size)
    
    # Additional analysis
    analyze_mathematical_properties(results)
    
    print("\nAnalysis complete! Check the generated plots.") 