"""
Conway's Game of Life - Simple Implementation for Complexity Lab

The Game of Life is a cellular automaton with simple rules that can produce complex behavior.

Rules:
1. Any live cell with fewer than two live neighbors dies (underpopulation)
2. Any live cell with two or three live neighbors survives 
3. Any live cell with more than three live neighbors dies (overpopulation)
4. Any dead cell with exactly three live neighbors becomes alive (reproduction)

Mathematical representation:
- Grid: 2D array where 1 = alive, 0 = dead
- Next state depends only on current state and neighbor count
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Tuple, List


class GameOfLife:
    """Simple implementation of Conway's Game of Life"""
    
    def __init__(self, width: int = 50, height: int = 50):
        """
        Initialize the Game of Life grid
        
        Args:
            width: Grid width
            height: Grid height
        """
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=int)
        self.generation = 0
    
    def set_initial_state(self, pattern: str = "random", density: float = 0.3):
        """
        Set the initial state of the grid
        
        Args:
            pattern: "random", "glider", "blinker", "block", or "r_pentomino"
            density: For random pattern, probability of cell being alive
        """
        self.generation = 0
        
        if pattern == "random":
            self.grid = np.random.choice([0, 1], size=(self.height, self.width), 
                                       p=[1-density, density])
        
        elif pattern == "glider":
            # Clear grid and add glider pattern
            self.grid.fill(0)
            glider = [(1, 2), (2, 3), (3, 1), (3, 2), (3, 3)]
            for r, c in glider:
                if 0 <= r < self.height and 0 <= c < self.width:
                    self.grid[r, c] = 1
        
        elif pattern == "blinker":
            # Clear grid and add blinker pattern  
            self.grid.fill(0)
            center_r, center_c = self.height // 2, self.width // 2
            blinker = [(center_r, center_c-1), (center_r, center_c), (center_r, center_c+1)]
            for r, c in blinker:
                if 0 <= r < self.height and 0 <= c < self.width:
                    self.grid[r, c] = 1
        
        elif pattern == "block":
            # Clear grid and add block pattern (still life)
            self.grid.fill(0)
            center_r, center_c = self.height // 2, self.width // 2
            block = [(center_r, center_c), (center_r, center_c+1), 
                    (center_r+1, center_c), (center_r+1, center_c+1)]
            for r, c in block:
                if 0 <= r < self.height and 0 <= c < self.width:
                    self.grid[r, c] = 1
        
        elif pattern == "r_pentomino":
            # Clear grid and add R-pentomino pattern
            # R-pentomino: Famous for complex evolution before stabilizing
            #  XX
            # XX
            #  X
            self.grid.fill(0)
            center_r, center_c = self.height // 2, self.width // 2
            r_pentomino = [
                (center_r-1, center_c), (center_r-1, center_c+1),  # Top row:  XX
                (center_r, center_c-1), (center_r, center_c),      # Mid row: XX 
                (center_r+1, center_c)                             # Bot row:  X
            ]
            for r, c in r_pentomino:
                if 0 <= r < self.height and 0 <= c < self.width:
                    self.grid[r, c] = 1
    
    def count_neighbors(self, row: int, col: int) -> int:
        """
        Count live neighbors for a cell (using periodic boundary conditions)
        
        Args:
            row: Row index
            col: Column index
            
        Returns:
            Number of live neighbors (0-8)
        """
        count = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:  # Skip the cell itself
                    continue
                
                # Periodic boundary conditions (wrap around)
                neighbor_row = (row + dr) % self.height
                neighbor_col = (col + dc) % self.width
                count += self.grid[neighbor_row, neighbor_col]
        
        return count
    
    def step(self):
        """
        Advance the simulation by one generation
        
        Applies Conway's rules to compute the next state
        """
        new_grid = np.zeros_like(self.grid)
        
        for row in range(self.height):
            for col in range(self.width):
                neighbors = self.count_neighbors(row, col)
                current_cell = self.grid[row, col]
                
                # Apply Conway's rules
                if current_cell == 1:  # Live cell
                    if neighbors < 2:       # Underpopulation
                        new_grid[row, col] = 0
                    elif neighbors in [2, 3]:  # Survival
                        new_grid[row, col] = 1
                    else:                   # Overpopulation
                        new_grid[row, col] = 0
                else:  # Dead cell
                    if neighbors == 3:      # Reproduction
                        new_grid[row, col] = 1
        
        self.grid = new_grid
        self.generation += 1
    
    def get_statistics(self) -> dict:
        """
        Get basic statistics about the current state
        
        Returns:
            Dictionary with population count and density
        """
        population = np.sum(self.grid)
        density = population / (self.width * self.height)
        return {
            'generation': self.generation,
            'population': population,
            'density': density
        }
    
    def display(self):
        """Display current state using matplotlib"""
        plt.figure(figsize=(8, 8))
        plt.imshow(self.grid, cmap='binary', interpolation='nearest')
        plt.title(f'Generation {self.generation}')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def run_simulation(self, generations: int = 100, animate: bool = True):
        """
        Run the simulation for a specified number of generations
        
        Args:
            generations: Number of generations to simulate
            animate: Whether to show animation or just final state
        """
        if animate:
            fig, ax = plt.subplots(figsize=(8, 8))
            im = ax.imshow(self.grid, cmap='binary', interpolation='nearest')
            ax.set_title(f'Generation {self.generation}')
            
            def update(frame):
                self.step()
                im.set_array(self.grid)
                ax.set_title(f'Generation {self.generation}')
                return [im]
            
            ani = animation.FuncAnimation(fig, update, frames=generations, 
                                        interval=200, blit=True, repeat=False)
            plt.show()
        else:
            # Just run simulation and show statistics
            stats_history = []
            for _ in range(generations):
                self.step()
                stats_history.append(self.get_statistics())
            
            # Plot population over time
            generations_list = [s['generation'] for s in stats_history]
            populations = [s['population'] for s in stats_history]
            
            plt.figure(figsize=(10, 6))
            plt.plot(generations_list, populations)
            plt.xlabel('Generation')
            plt.ylabel('Population')
            plt.title('Population Over Time')
            plt.grid(True, alpha=0.3)
            plt.show()


# Example usage functions
def demo_patterns():
    """Demonstrate different starting patterns"""
    patterns = ["glider", "blinker", "block", "random"]
    
    for pattern in patterns:
        print(f"\n--- {pattern.upper()} PATTERN ---")
        game = GameOfLife(20, 20)
        game.set_initial_state(pattern)
        
        print("Initial state:")
        stats = game.get_statistics()
        print(f"Population: {stats['population']}, Density: {stats['density']:.3f}")
        
        # Run for a few generations
        for i in range(5):
            game.step()
            stats = game.get_statistics()
            print(f"Gen {stats['generation']}: Population = {stats['population']}")


def analyze_population_dynamics():
    """Analyze how population changes over time"""
    game = GameOfLife(50, 50)
    game.set_initial_state("random", density=0.3)
    
    print("Analyzing population dynamics...")
    game.run_simulation(generations=100, animate=False)


if __name__ == "__main__":
    print("Conway's Game of Life - Complexity Lab")
    print("=" * 40)
    
    # Demo different patterns
    demo_patterns()
    
    # Ask user what they want to do
    print("\nChoose an option:")
    print("1. Watch animated simulation")
    print("2. Analyze population dynamics")
    print("3. Interactive mode")
    
    choice = input("Enter choice (1-3): ")
    
    if choice == "1":
        game = GameOfLife(40, 40)
        game.set_initial_state("random", density=0.3)
        game.run_simulation(generations=200, animate=True)
    
    elif choice == "2":
        analyze_population_dynamics()
    
    elif choice == "3":
        print("Interactive mode - create your own initial configuration!")
        width = int(input("Grid width (20-100): "))
        height = int(input("Grid height (20-100): "))
        pattern = input("Pattern (random/glider/blinker/block): ")
        
        game = GameOfLife(width, height)
        game.set_initial_state(pattern)
        game.run_simulation(generations=150, animate=True) 