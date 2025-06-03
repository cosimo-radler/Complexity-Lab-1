# Conway's Game of Life - Complexity Lab

This is a simple implementation of Conway's Game of Life for educational purposes in the Complexity Lab course.

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the simulation:
   ```bash
   python game_of_life.py
   ```

## Mathematical Foundation

The Game of Life demonstrates how simple rules can create complex behavior:

### Rules
1. **Underpopulation**: Live cell with < 2 neighbors dies
2. **Survival**: Live cell with 2-3 neighbors survives  
3. **Overpopulation**: Live cell with > 3 neighbors dies
4. **Reproduction**: Dead cell with exactly 3 neighbors becomes alive

### Key Concepts
- **Cellular Automaton**: Discrete model with grid of cells
- **Deterministic**: Next state completely determined by current state
- **Local Rules**: Each cell's fate depends only on its immediate neighbors
- **Emergent Behavior**: Complex patterns from simple rules

## Built-in Patterns

- **Glider**: Moving pattern that translates across the grid
- **Blinker**: Oscillating pattern (period-2)
- **Block**: Still life (unchanging pattern)
- **Random**: Random initial configuration

## Key Features

- Mathematically correct implementation of Conway's rules
- Periodic boundary conditions (grid wraps around)
- Population tracking and statistics
- Animation capabilities
- Multiple starting patterns

## Suggestions for Experiments

1. **Pattern Analysis**: Study different initial configurations
2. **Population Dynamics**: Track how population changes over time
3. **Boundary Effects**: Compare periodic vs. fixed boundaries
4. **Density Studies**: How does initial density affect evolution?
5. **Pattern Recognition**: Identify still lifes, oscillators, and spaceships

## Extensions You Can Add

- Different boundary conditions
- Larger grids for complex behavior
- Pattern detection algorithms
- Statistical analysis tools
- Rule variations (different neighbor counts)
- Pattern library loading 