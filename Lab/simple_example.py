"""
Simple Game of Life Example - Non-interactive Demo

This script demonstrates the basic functionality without requiring user interaction.
Perfect for understanding the core concepts.
"""

from game_of_life import GameOfLife
import matplotlib.pyplot as plt

def basic_demo():
    """Run a basic demonstration of the Game of Life"""
    
    print("Conway's Game of Life - Basic Demo")
    print("=" * 35)
    
    # Create a small game instance
    game = GameOfLife(width=20, height=20)
    
    # Test different patterns
    patterns = ["glider", "blinker", "block"]
    
    for pattern in patterns:
        print(f"\n--- Testing {pattern.upper()} pattern ---")
        
        # Set the pattern
        game.set_initial_state(pattern)
        
        # Show initial statistics
        stats = game.get_statistics()
        print(f"Initial - Generation: {stats['generation']}, Population: {stats['population']}")
        
        # Run for 10 generations and show population changes
        populations = [stats['population']]
        for i in range(10):
            game.step()
            stats = game.get_statistics()
            populations.append(stats['population'])
            if i < 5:  # Show first 5 generations
                print(f"Gen {stats['generation']}: Population = {stats['population']}")
        
        # Analyze the pattern behavior
        if len(set(populations[-5:])) == 1:
            print(f"Pattern stabilized at population {populations[-1]}")
        elif len(set(populations[-4:])) <= 2:
            print("Pattern appears to be oscillating")
        else:
            print("Pattern is still changing")


def population_analysis():
    """Analyze population dynamics with random initial state"""
    
    print("\n" + "=" * 50)
    print("POPULATION DYNAMICS ANALYSIS")
    print("=" * 50)
    
    # Create game with random initial state
    game = GameOfLife(width=30, height=30)
    game.set_initial_state("random", density=0.35)
    
    # Track population over time
    generations = []
    populations = []
    densities = []
    
    # Run simulation
    for gen in range(50):
        stats = game.get_statistics()
        generations.append(stats['generation'])
        populations.append(stats['population'])
        densities.append(stats['density'])
        
        if gen % 10 == 0:
            print(f"Generation {gen:2d}: Population = {stats['population']:3d}, Density = {stats['density']:.3f}")
        
        game.step()
    
    # Final statistics
    final_stats = game.get_statistics()
    print(f"Final     : Population = {final_stats['population']:3d}, Density = {final_stats['density']:.3f}")
    
    # Simple analysis
    initial_pop = populations[0]
    final_pop = populations[-1]
    max_pop = max(populations)
    min_pop = min(populations)
    
    print(f"\nSummary:")
    print(f"Initial population: {initial_pop}")
    print(f"Final population: {final_pop}")
    print(f"Maximum population: {max_pop}")
    print(f"Minimum population: {min_pop}")
    print(f"Population change: {final_pop - initial_pop:+d}")


def mathematical_verification():
    """Verify that the rules are implemented correctly"""
    
    print("\n" + "=" * 50)
    print("MATHEMATICAL RULE VERIFICATION")
    print("=" * 50)
    
    # Test each rule explicitly
    game = GameOfLife(5, 5)
    
    # Test underpopulation rule
    game.grid.fill(0)
    game.grid[2, 2] = 1  # Single live cell
    game.grid[2, 1] = 1  # One neighbor
    
    print("Testing underpopulation rule:")
    print("Live cell with 1 neighbor should die")
    neighbors = game.count_neighbors(2, 2)
    print(f"Neighbors counted: {neighbors}")
    
    game.step()
    result = game.grid[2, 2]
    print(f"Cell died: {result == 0} (correct: True)")
    
    # Test reproduction rule
    game.grid.fill(0)
    game.grid[1, 2] = 1  # Three neighbors around (2,2)
    game.grid[2, 1] = 1
    game.grid[3, 2] = 1
    
    print("\nTesting reproduction rule:")
    print("Dead cell with 3 neighbors should become alive")
    neighbors = game.count_neighbors(2, 2)
    print(f"Neighbors counted: {neighbors}")
    
    game.step()
    result = game.grid[2, 2]
    print(f"Cell born: {result == 1} (correct: True)")
    
    print("\nRule verification complete!")


if __name__ == "__main__":
    # Run all demonstrations
    basic_demo()
    population_analysis()
    mathematical_verification()
    
    print("\n" + "=" * 50)
    print("Demo complete! You can now:")
    print("1. Run 'python game_of_life.py' for the interactive version")
    print("2. Modify this script to test your own experiments")
    print("3. Study the code to understand the implementation")
    print("=" * 50) 