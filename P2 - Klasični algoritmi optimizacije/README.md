# Gradient Search Optimization Demo

A comprehensive Python application demonstrating gradient descent optimization with multiple test functions and constraints.

## Features

### Test Functions
The application includes 6 different test functions:

1. **Sphere (Unimodal)** - Simple convex function: f(x,y) = x² + y²
2. **3rd Order Polynomial (Multimodal)** - Multiple local minima: f(x,y) = x³ - 3x + y³ - 3y
3. **Rastrigin (Highly Multimodal)** - Many local minima with periodic structure
4. **Rosenbrock (Valley-shaped)** - Classic optimization challenge with narrow valley
5. **Ackley (Multimodal)** - Complex landscape with many local minima
6. **Beale (Multimodal)** - Multiple local minima with varying depths

### Constraints

- **Circle Constraint**: Defines a circular feasible region (inside the circle). Center is offset from function minima.
- **Line Constraint**: Linear constraint (x + y ≥ 0) that divides the space into feasible and infeasible regions.
- Both constraints can be activated/deactivated independently
- When both are active, the feasible region is their intersection
- Feasible regions are visually shaded on the plot

### Optimization Algorithm

- **Gradient Descent**: Classic gradient-based optimization
- **Constraint Handling**: Penalty method for handling constraints
- **Configurable Parameters**:
  - Step size (learning rate)
  - Maximum iterations
- **Boundary Handling**: Solutions are clamped to function bounds

### Interactive Features

1. **Click to Set Initial Point**: Click anywhere on the plot to set the starting point
2. **Random Initialization**: Button to generate a random initial point
3. **Step-by-Step Execution**: Run one iteration at a time to observe the algorithm's behavior
4. **Complete Run**: Execute the entire optimization at once
5. **Real-time Visualization**: See the trajectory, current point, best solution, and global minimum
6. **Live Statistics**: Monitor current iteration, objective values, and feasibility

### Visualization

The plot displays:
- **Contour lines**: Function landscape
- **Filled contours**: Colored background showing function values
- **Green star**: Global minimum (theoretical optimum)
- **Blue circle**: Initial point
- **Red circle**: Current point
- **Magenta circle**: Best solution found so far
- **Red line**: Optimization trajectory
- **Constraint boundaries**: Dashed lines for constraints
- **Shaded regions**: Feasible areas (green for circle, blue for line, yellow for intersection)
- **Legend**: Clear identification of all elements

### Statistics Panel

Displays:
- Current iteration number
- Optimization status (Running/Complete)
- Current point coordinates and objective value
- Best solution found (coordinates and value)
- Global minimum (theoretical optimum)
- Feasibility status of current point

## Requirements

```bash
pip install numpy matplotlib tk
```

or

```bash
pip install numpy matplotlib
# tkinter usually comes with Python installation
```

## Usage

Run the application:

```bash
python3 gradient_search_demo.py
```

### Basic Workflow

1. **Select a test function** from the radio buttons (default: Sphere)
2. **Activate constraints** if desired (optional)
3. **Set algorithm parameters**:
   - Step Size: Controls how large each gradient step is (default: 0.05)
   - Max Iterations: Maximum number of optimization steps (default: 100)
4. **Set initial point**:
   - Click on the plot to select a specific location, OR
   - Click "Random Initial Point" button
5. **Run optimization**:
   - Click "Step" to run one iteration at a time, OR
   - Click "Run Complete" to execute all iterations at once
6. **Observe results** in the plot and statistics panel
7. **Reset** to try a different configuration or starting point

### Tips

- **Start with Sphere function** to understand basic gradient descent behavior
- **Try different step sizes**: Too large may cause oscillation, too small may be slow
- **Experiment with constraints**: See how the algorithm handles feasible regions
- **Compare with global minimum**: Check how close the algorithm gets to the optimum
- **Use step-by-step mode**: Great for teaching and understanding the algorithm's behavior
- **Try multimodal functions**: Observe how the algorithm can get stuck in local minima
- **Different initial points**: See how starting position affects the final solution

### Example Scenarios

1. **Unconstrained Optimization**:
   - Select Sphere function
   - Disable both constraints
   - Set initial point far from origin
   - Watch it converge to (0,0)

2. **Constrained Optimization**:
   - Select Sphere function
   - Enable circle constraint
   - Set initial point inside circle
   - Observe how it finds the best point within the feasible region

3. **Local Minima Challenge**:
   - Select Rastrigin function
   - Try different initial points
   - Observe how the algorithm gets trapped in local minima

4. **Complex Constraints**:
   - Enable both circle and line constraints
   - See optimization in the intersection region
   - Try functions with minima outside the feasible region

## Algorithm Details

### Gradient Descent Update

```
x_new = x_old - step_size * ∇f(x_old)
```

### Constraint Handling

The algorithm uses a penalty method:
- Constraints are converted to penalty gradients
- Points outside feasible regions receive penalty forces pushing them back
- Total gradient = objective gradient + penalty gradients

### Numerical Gradient Computation

For functions without analytical gradients:
```
∂f/∂x ≈ [f(x+h, y) - f(x-h, y)] / (2h)
```

## Implementation Notes

- **Framework**: Tkinter for GUI, Matplotlib for visualization
- **Architecture**: Object-oriented design with separate classes for functions, constraints, and optimizer
- **Extensibility**: Easy to add new test functions by subclassing `TestFunction`
- **Numerical stability**: Includes convergence detection and boundary clamping

## Educational Use

This application is ideal for:
- Teaching optimization algorithms
- Demonstrating gradient descent behavior
- Visualizing constraint handling
- Understanding local vs global minima
- Exploring the effect of algorithm parameters
- Interactive demonstrations in lectures

## License

Free to use for educational purposes.
