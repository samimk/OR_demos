"""
Gradient Search Optimization Demo Application
Demonstrates gradient descent optimization with multiple test functions and constraints
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.patches import Circle as CirclePatch
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D


class TestFunction:
    """Base class for test functions"""
    def __init__(self, name, bounds):
        self.name = name
        self.bounds = bounds  # [(xmin, xmax), (ymin, ymax)]

    def evaluate(self, x, y):
        """Evaluate function at point (x, y)"""
        raise NotImplementedError

    def gradient(self, x, y):
        """Compute gradient at point (x, y)"""
        # Numerical gradient approximation
        h = 1e-6
        grad_x = (self.evaluate(x + h, y) - self.evaluate(x - h, y)) / (2 * h)
        grad_y = (self.evaluate(x, y + h) - self.evaluate(x, y - h)) / (2 * h)
        return np.array([grad_x, grad_y])

    def global_minimum(self):
        """Return the global minimum point and value"""
        raise NotImplementedError


class SphereFunction(TestFunction):
    """Sphere function: f(x,y) = x^2 + y^2"""
    def __init__(self):
        super().__init__("Sphere (Unimodal)", [(-5, 5), (-5, 5)])

    def evaluate(self, x, y):
        return x**2 + y**2

    def gradient(self, x, y):
        return np.array([2*x, 2*y])

    def global_minimum(self):
        return (0.0, 0.0, 0.0)


class PolynomialFunction(TestFunction):
    """3rd order polynomial: f(x,y) = x^3 - 3x + y^3 - 3y"""
    def __init__(self):
        super().__init__("3rd Order Polynomial (Multimodal)", [(-2.5, 2.5), (-2.5, 2.5)])

    def evaluate(self, x, y):
        return -(x**3 - 3*x + y**3 - 3*y)

    def gradient(self, x, y):
        return np.array([3*x**2 - 3, 3*y**2 - 3])

    def global_minimum(self):
        # Minimum at (1, 1) with value -4
        return (1.0, 1.0, -4.0)


class RastriginFunction(TestFunction):
    """Rastrigin function: f(x,y) = 20 + x^2 - 10*cos(2πx) + y^2 - 10*cos(2πy)"""
    def __init__(self):
        super().__init__("Rastrigin (Highly Multimodal)", [(-5, 5), (-5, 5)])

    def evaluate(self, x, y):
        return 20 + (x**2 - 10*np.cos(2*np.pi*x)) + (y**2 - 10*np.cos(2*np.pi*y))

    def gradient(self, x, y):
        grad_x = 2*x + 20*np.pi*np.sin(2*np.pi*x)
        grad_y = 2*y + 20*np.pi*np.sin(2*np.pi*y)
        return np.array([grad_x, grad_y])

    def global_minimum(self):
        return (0.0, 0.0, 0.0)


class RosenbrockFunction(TestFunction):
    """Rosenbrock function: f(x,y) = (1-x)^2 + 100*(y-x^2)^2"""
    def __init__(self):
        super().__init__("Rosenbrock (Valley-shaped)", [(-2, 2), (-1, 3)])

    def evaluate(self, x, y):
        return (1 - x)**2 + 100 * (y - x**2)**2

    def gradient(self, x, y):
        grad_x = -2*(1 - x) - 400*x*(y - x**2)
        grad_y = 200*(y - x**2)
        return np.array([grad_x, grad_y])

    def global_minimum(self):
        return (1.0, 1.0, 0.0)


class AckleyFunction(TestFunction):
    """Ackley function"""
    def __init__(self):
        super().__init__("Ackley (Multimodal)", [(-5, 5), (-5, 5)])

    def evaluate(self, x, y):
        a = 20
        b = 0.2
        c = 2 * np.pi
        term1 = -a * np.exp(-b * np.sqrt(0.5 * (x**2 + y**2)))
        term2 = -np.exp(0.5 * (np.cos(c*x) + np.cos(c*y)))
        return term1 + term2 + a + np.e

    def global_minimum(self):
        return (0.0, 0.0, 0.0)


class BealeFunction(TestFunction):
    """Beale function"""
    def __init__(self):
        super().__init__("Beale (Multimodal)", [(-4.5, 4.5), (-4.5, 4.5)])

    def evaluate(self, x, y):
        term1 = (1.5 - x + x*y)**2
        term2 = (2.25 - x + x*y**2)**2
        term3 = (2.625 - x + x*y**3)**2
        return term1 + term2 + term3

    def global_minimum(self):
        return (3.0, 0.5, 0.0)


class CircleConstraint:
    """Circle constraint: (x-cx)^2 + (y-cy)^2 <= r^2"""
    def __init__(self, center_x, center_y, radius):
        self.center_x = center_x
        self.center_y = center_y
        self.radius = radius

    def is_feasible(self, x, y):
        """Check if point is inside the circle (feasible region)"""
        dist_sq = (x - self.center_x)**2 + (y - self.center_y)**2
        return dist_sq <= self.radius**2

    def constraint_value(self, x, y):
        """Return constraint value: g(x) = r^2 - (x-cx)^2 - (y-cy)^2, feasible when g(x) >= 0"""
        dist_sq = (x - self.center_x)**2 + (y - self.center_y)**2
        return self.radius**2 - dist_sq

    def gradient_penalty_exterior(self, x, y):
        """Exterior penalty: penalize infeasible points (outside circle)"""
        g = self.constraint_value(x, y)
        if g >= 0:
            return np.array([0.0, 0.0])  # Inside circle, no penalty
        # Outside circle: penalty = (g(x))^2, gradient pushes towards feasible region
        # ∇penalty = 2*g(x)*∇g(x)
        # ∇g(x) = -2*(x-cx, y-cy)
        grad_g_x = -2 * (x - self.center_x)
        grad_g_y = -2 * (y - self.center_y)
        return 2 * g * np.array([grad_g_x, grad_g_y])

    def gradient_penalty_interior(self, x, y):
        """Interior penalty: penalize points near boundary from inside"""
        g = self.constraint_value(x, y)
        if g <= 0:
            # Outside or on boundary: return large penalty to push back
            return self.gradient_penalty_exterior(x, y) * 100
        # Inside: penalty = -1/g(x), gradient pushes away from boundary
        # ∇penalty = (1/g^2)*∇g(x)
        grad_g_x = -2 * (x - self.center_x)
        grad_g_y = -2 * (y - self.center_y)
        return (1.0 / (g * g)) * np.array([grad_g_x, grad_g_y])


class LineConstraint:
    """Line constraint: ax + by + c >= 0 (one side of line is feasible)"""
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def is_feasible(self, x, y):
        """Check if point is on the feasible side of the line"""
        return self.a * x + self.b * y + self.c >= 0

    def constraint_value(self, x, y):
        """Return constraint value: g(x) = ax + by + c, feasible when g(x) >= 0"""
        return self.a * x + self.b * y + self.c

    def gradient_penalty_exterior(self, x, y):
        """Exterior penalty: penalize infeasible points"""
        g = self.constraint_value(x, y)
        if g >= 0:
            return np.array([0.0, 0.0])  # Feasible, no penalty
        # Infeasible: penalty = (g(x))^2, gradient pushes towards feasible region
        # ∇penalty = 2*g(x)*∇g(x)
        # ∇g(x) = (a, b)
        return 2 * g * np.array([self.a, self.b])

    def gradient_penalty_interior(self, x, y):
        """Interior penalty: penalize points near boundary from inside"""
        g = self.constraint_value(x, y)
        if g <= 0:
            # Outside or on boundary: return large penalty to push back
            return self.gradient_penalty_exterior(x, y) * 100
        # Inside: penalty = -1/g(x), gradient pushes away from boundary
        # ∇penalty = (1/g^2)*∇g(x)
        return (1.0 / (g * g)) * np.array([self.a, self.b])


class GradientSearchOptimizer:
    """Gradient descent optimizer with constraints"""
    def __init__(self, function, step_size=0.1, max_iterations=100,
                 circle_constraint=None, line_constraint=None,
                 use_circle=False, use_line=False, penalty_method='exterior'):
        self.function = function
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.circle_constraint = circle_constraint
        self.line_constraint = line_constraint
        self.use_circle = use_circle
        self.use_line = use_line
        self.penalty_method = penalty_method  # 'exterior' or 'interior'

        # State variables
        self.current_point = None
        self.trajectory = []
        self.values = []
        self.iteration = 0
        self.is_complete = False
        self.penalty_weight = 1000.0 if penalty_method == 'exterior' else 1.0  # Weight for constraint penalties

    def initialize(self, start_point):
        """Initialize the optimization from a starting point"""
        self.current_point = np.array(start_point)
        self.trajectory = [self.current_point.copy()]
        self.values = [self.function.evaluate(*self.current_point)]
        self.iteration = 0
        self.is_complete = False

    def is_feasible(self, x, y):
        """Check if point satisfies all active constraints"""
        feasible = True
        if self.use_circle and self.circle_constraint:
            feasible = feasible and self.circle_constraint.is_feasible(x, y)
        if self.use_line and self.line_constraint:
            feasible = feasible and self.line_constraint.is_feasible(x, y)
        return feasible

    def step(self):
        """Perform one iteration of gradient descent"""
        if self.is_complete or self.iteration >= self.max_iterations:
            self.is_complete = True
            return False

        # Compute objective function gradient
        grad = self.function.gradient(*self.current_point)

        # Add constraint penalties if active
        penalty_grad = np.array([0.0, 0.0])
        if self.use_circle and self.circle_constraint:
            if self.penalty_method == 'exterior':
                penalty_grad += self.penalty_weight * self.circle_constraint.gradient_penalty_exterior(*self.current_point)
            else:  # interior
                penalty_grad += self.penalty_weight * self.circle_constraint.gradient_penalty_interior(*self.current_point)
        if self.use_line and self.line_constraint:
            if self.penalty_method == 'exterior':
                penalty_grad += self.penalty_weight * self.line_constraint.gradient_penalty_exterior(*self.current_point)
            else:  # interior
                penalty_grad += self.penalty_weight * self.line_constraint.gradient_penalty_interior(*self.current_point)

        # Combined gradient
        total_grad = grad + penalty_grad

        # Update position
        new_point = self.current_point - self.step_size * total_grad

        # Clamp to bounds
        bounds = self.function.bounds
        new_point[0] = np.clip(new_point[0], bounds[0][0], bounds[0][1])
        new_point[1] = np.clip(new_point[1], bounds[1][0], bounds[1][1])

        # Update state
        self.current_point = new_point
        self.trajectory.append(self.current_point.copy())
        self.values.append(self.function.evaluate(*self.current_point))
        self.iteration += 1

        # Check for convergence (gradient magnitude)
        if np.linalg.norm(total_grad) < 1e-6:
            self.is_complete = True
            return False

        return True

    def run_complete(self):
        """Run optimization until completion"""
        while self.step():
            pass

    def get_best_solution(self):
        """Return the best solution found so far"""
        if not self.trajectory:
            return None
        best_idx = np.argmin(self.values)
        best_point = self.trajectory[best_idx]
        return best_point, self.values[best_idx]


class GradientSearchApp:
    """Main application window"""
    def __init__(self, root):
        self.root = root
        self.root.title("Gradient Search Optimization Demo")
        self.root.geometry("1400x900")

        # Initialize test functions
        self.functions = [
            SphereFunction(),
            PolynomialFunction(),
            RastriginFunction(),
            RosenbrockFunction(),
            AckleyFunction(),
            BealeFunction()
        ]
        self.current_function = self.functions[0]

        # Initialize constraints (off-center from minimum)
        # Circle: (x+3)^2 + (y+2.5)^2 <= 6.25, center at (-3, -2.5), radius = 2.5
        self.circle_constraint = CircleConstraint(center_x=-3.0, center_y=-2.5, radius=2.5)
        # Line: y + 1.2x <= -5, or -1.2x - y - 5 >= 0 (multiply by -1), or 1.2x + y + 5 <= 0
        # For ax + by + c >= 0 form: -1.2x - y - 5 >= 0
        self.line_constraint = LineConstraint(a=-1.2, b=-1, c=-5)  # -1.2x - y - 5 >= 0

        # Optimizer instance
        self.optimizer = None
        self.initial_point = None

        # Setup GUI
        self.setup_menu()
        self.setup_gui()

        # Initial plot
        self.update_plot()

    def setup_menu(self):
        """Setup the menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="How to Use", command=self.show_usage_dialog)
        help_menu.add_separator()
        help_menu.add_command(label="About", command=self.show_about_dialog)

    def show_usage_dialog(self):
        """Show the How to Use dialog"""
        usage_window = tk.Toplevel(self.root)
        usage_window.title("How to Use")
        usage_window.geometry("700x600")

        # Create a frame with scrollbar
        frame = ttk.Frame(usage_window)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Add scrollbar
        scrollbar = ttk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Create text widget
        text_widget = tk.Text(frame, wrap=tk.WORD, yscrollcommand=scrollbar.set,
                             font=("Arial", 10), padx=10, pady=10)
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=text_widget.yview)

        usage_text = """GRADIENT SEARCH OPTIMIZATION DEMO - HOW TO USE

═══════════════════════════════════════════════════════════════

1. SELECT TEST FUNCTION
   • Choose from 6 different optimization functions
   • Functions include unimodal (Sphere) and multimodal (Rastrigin, etc.)
   • Each function has different characteristics and challenges

2. CHOOSE VISUALIZATION TYPE
   • 2D Contour Plot: Top-down view with contour lines
   • 3D Surface Plot: Interactive 3D surface view
   • Switch between them anytime to see different perspectives

3. CONFIGURE CONSTRAINTS (Optional)
   • Circle Constraint: Feasible region inside the circle
   • Line Constraint: Feasible region on one side of the line
   • Both can be activated independently or together

   Penalty Methods:
   • Exterior Penalty: Penalizes infeasible points (use any start point)
   • Interior Penalty: Maintains feasibility (start from feasible region)

4. SET ALGORITHM PARAMETERS
   • Step Size: Controls gradient descent step length (0.01 - 0.1 typical)
   • Max Iterations: Maximum number of optimization steps (50-200 typical)

5. INITIALIZE STARTING POINT
   Two options:
   • Click on the plot to select a specific point
   • Click "Random Initial Point" button for random initialization

6. RUN OPTIMIZATION
   Two execution modes:
   • Step: Execute one iteration at a time (for teaching/demonstration)
   • Run Complete: Execute all iterations at once

7. OBSERVE RESULTS
   The visualization shows:
   • Green star: Global minimum (theoretical optimum)
   • Blue circle: Initial starting point
   • Red circle: Current point in optimization
   • Magenta circle: Best solution found so far
   • Red line: Optimization trajectory
   • Constraint boundaries (if activated)
   • Shaded feasible regions (if constraints active)

8. MONITOR STATISTICS
   The statistics panel displays:
   • Current iteration number
   • Optimization status (Running/Complete)
   • Penalty method being used
   • Current point coordinates and objective value
   • Best solution found
   • Global minimum (reference)
   • Feasibility status

9. 3D VISUALIZATION CONTROLS (when in 3D mode)
   • Left-click + drag: Rotate the view
   • Right-click + drag: Zoom in/out
   • Middle-click + drag: Pan the view

10. TIPS FOR BEST RESULTS
    • Start with Sphere function to understand basic behavior
    • Use smaller step sizes (0.01-0.05) for smooth convergence
    • Try different initial points to see local minima effects
    • Use Step mode for teaching and understanding the algorithm
    • Compare results with and without constraints
    • Switch to 3D view to understand the function landscape
    • Try Interior penalty from feasible points only
    • Use Exterior penalty when starting from infeasible regions

11. RESET AND TRY AGAIN
    • Click "Reset" button to clear current optimization
    • Change parameters, constraints, or initial point
    • Run again to compare different configurations

═══════════════════════════════════════════════════════════════

For more information, select Help → About"""

        text_widget.insert(1.0, usage_text)
        text_widget.config(state=tk.DISABLED)  # Make read-only

        # Add close button
        btn_frame = ttk.Frame(usage_window)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="Close", command=usage_window.destroy).pack()

    def show_about_dialog(self):
        """Show the About dialog"""
        about_text = """Gradient Search Optimization Demo

Optimizacija resursa

Red. prof. dr Samim Konjicija

2025. godina"""

        messagebox.showinfo("About", about_text)

    def setup_gui(self):
        """Setup the GUI layout"""
        # Main container with two columns
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel: Controls
        left_panel = ttk.Frame(main_frame, width=350)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=5, pady=5)
        left_panel.pack_propagate(False)

        # Right panel: Visualization
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Setup control panel sections
        self.setup_function_selection(left_panel)
        self.setup_visualization_type(left_panel)
        self.setup_constraints(left_panel)
        self.setup_parameters(left_panel)
        self.setup_initialization(left_panel)
        self.setup_controls(left_panel)
        self.setup_statistics(left_panel)

        # Setup visualization
        self.setup_visualization(right_panel)

    def setup_function_selection(self, parent):
        """Setup function selection section"""
        frame = ttk.LabelFrame(parent, text="Test Function", padding=10)
        frame.pack(fill=tk.X, pady=5)

        self.function_var = tk.StringVar(value=self.functions[0].name)
        for func in self.functions:
            rb = ttk.Radiobutton(frame, text=func.name, variable=self.function_var,
                                value=func.name, command=self.on_function_changed)
            rb.pack(anchor=tk.W)

    def setup_visualization_type(self, parent):
        """Setup visualization type selection"""
        frame = ttk.LabelFrame(parent, text="Visualization Type", padding=10)
        frame.pack(fill=tk.X, pady=5)

        self.viz_type_var = tk.StringVar(value="2d")

        rb_2d = ttk.Radiobutton(frame, text="2D Contour Plot",
                               variable=self.viz_type_var,
                               value="2d", command=self.on_viz_type_changed)
        rb_2d.pack(anchor=tk.W)

        rb_3d = ttk.Radiobutton(frame, text="3D Surface Plot",
                               variable=self.viz_type_var,
                               value="3d", command=self.on_viz_type_changed)
        rb_3d.pack(anchor=tk.W)

    def setup_constraints(self, parent):
        """Setup constraints section"""
        frame = ttk.LabelFrame(parent, text="Constraints", padding=10)
        frame.pack(fill=tk.X, pady=5)

        # Circle constraint
        self.circle_var = tk.BooleanVar(value=False)
        cb_circle = ttk.Checkbutton(frame, text="Circle Constraint (feasible inside)",
                                    variable=self.circle_var, command=self.update_plot)
        cb_circle.pack(anchor=tk.W)

        # Line constraint
        self.line_var = tk.BooleanVar(value=False)
        cb_line = ttk.Checkbutton(frame, text="Line Constraint (y + 1.2x ≤ -5)",
                                  variable=self.line_var, command=self.update_plot)
        cb_line.pack(anchor=tk.W)

        # Penalty method selection
        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)
        ttk.Label(frame, text="Penalty Method:").pack(anchor=tk.W)

        self.penalty_method_var = tk.StringVar(value="exterior")
        rb_exterior = ttk.Radiobutton(frame, text="Exterior Penalty",
                                     variable=self.penalty_method_var,
                                     value="exterior")
        rb_exterior.pack(anchor=tk.W, padx=10)

        rb_interior = ttk.Radiobutton(frame, text="Interior Penalty",
                                     variable=self.penalty_method_var,
                                     value="interior")
        rb_interior.pack(anchor=tk.W, padx=10)

    def setup_parameters(self, parent):
        """Setup algorithm parameters section"""
        frame = ttk.LabelFrame(parent, text="Algorithm Parameters", padding=10)
        frame.pack(fill=tk.X, pady=5)

        # Step size
        ttk.Label(frame, text="Step Size:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.step_var = tk.StringVar(value="0.001")
        step_entry = ttk.Entry(frame, textvariable=self.step_var, width=15)
        step_entry.grid(row=0, column=1, pady=2)

        # Max iterations
        ttk.Label(frame, text="Max Iterations:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.iter_var = tk.StringVar(value="10000")
        iter_entry = ttk.Entry(frame, textvariable=self.iter_var, width=15)
        iter_entry.grid(row=1, column=1, pady=2)

    def setup_initialization(self, parent):
        """Setup initialization section"""
        frame = ttk.LabelFrame(parent, text="Initialization", padding=10)
        frame.pack(fill=tk.X, pady=5)

        ttk.Label(frame, text="Click on plot to set initial point").pack(anchor=tk.W)

        btn_random = ttk.Button(frame, text="Random Initial Point", command=self.set_random_initial_point)
        btn_random.pack(fill=tk.X, pady=2)

        # Display current initial point
        self.init_label = ttk.Label(frame, text="Initial Point: Not set")
        self.init_label.pack(anchor=tk.W, pady=2)

    def setup_controls(self, parent):
        """Setup control buttons section"""
        frame = ttk.LabelFrame(parent, text="Controls", padding=10)
        frame.pack(fill=tk.X, pady=5)

        btn_step = ttk.Button(frame, text="Step", command=self.step_optimization)
        btn_step.pack(fill=tk.X, pady=2)

        btn_run = ttk.Button(frame, text="Run Complete", command=self.run_complete)
        btn_run.pack(fill=tk.X, pady=2)

        btn_reset = ttk.Button(frame, text="Reset", command=self.reset_optimization)
        btn_reset.pack(fill=tk.X, pady=2)

    def setup_statistics(self, parent):
        """Setup statistics display section"""
        frame = ttk.LabelFrame(parent, text="Statistics", padding=10)
        frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.stats_text = tk.Text(frame, height=10, wrap=tk.WORD, font=("Courier", 9))
        self.stats_text.pack(fill=tk.BOTH, expand=True)

        self.update_statistics()

    def setup_visualization(self, parent):
        """Setup matplotlib visualization"""
        # Create figure
        self.fig = Figure(figsize=(8, 7), dpi=100)
        self.ax = self.fig.add_subplot(111)

        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Connect click event
        self.canvas.mpl_connect('button_press_event', self.on_plot_click)

    def on_function_changed(self):
        """Handle function selection change"""
        selected_name = self.function_var.get()
        for func in self.functions:
            if func.name == selected_name:
                self.current_function = func
                break
        self.reset_optimization()
        self.update_plot()

    def on_viz_type_changed(self):
        """Handle visualization type change"""
        # Clear and recreate the subplot
        self.fig.clear()
        if self.viz_type_var.get() == "3d":
            self.ax = self.fig.add_subplot(111, projection='3d')
        else:
            self.ax = self.fig.add_subplot(111)
        self.update_plot()

    def set_random_initial_point(self):
        """Set a random initial point within bounds"""
        bounds = self.current_function.bounds
        x = np.random.uniform(bounds[0][0], bounds[0][1])
        y = np.random.uniform(bounds[1][0], bounds[1][1])
        self.initial_point = (x, y)
        self.init_label.config(text=f"Initial Point: ({x:.3f}, {y:.3f})")
        self.update_plot()

    def on_plot_click(self, event):
        """Handle click on plot to set initial point"""
        if event.inaxes == self.ax and event.xdata and event.ydata:
            self.initial_point = (event.xdata, event.ydata)
            self.init_label.config(text=f"Initial Point: ({event.xdata:.3f}, {event.ydata:.3f})")
            self.update_plot()

    def step_optimization(self):
        """Perform one step of optimization"""
        if self.initial_point is None:
            messagebox.showwarning("No Initial Point", "Please set an initial point first.")
            return

        # Initialize optimizer if needed
        if self.optimizer is None:
            try:
                step_size = float(self.step_var.get())
                max_iter = int(self.iter_var.get())
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter valid numbers for parameters.")
                return

            self.optimizer = GradientSearchOptimizer(
                self.current_function,
                step_size=step_size,
                max_iterations=max_iter,
                circle_constraint=self.circle_constraint,
                line_constraint=self.line_constraint,
                use_circle=self.circle_var.get(),
                use_line=self.line_var.get(),
                penalty_method=self.penalty_method_var.get()
            )
            self.optimizer.initialize(self.initial_point)

        # Perform one step
        if not self.optimizer.is_complete:
            self.optimizer.step()
            self.update_plot()
            self.update_statistics()
        else:
            messagebox.showinfo("Complete", "Optimization is complete.")

    def run_complete(self):
        """Run optimization to completion"""
        if self.initial_point is None:
            messagebox.showwarning("No Initial Point", "Please set an initial point first.")
            return

        # Initialize optimizer if needed
        if self.optimizer is None:
            try:
                step_size = float(self.step_var.get())
                max_iter = int(self.iter_var.get())
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter valid numbers for parameters.")
                return

            self.optimizer = GradientSearchOptimizer(
                self.current_function,
                step_size=step_size,
                max_iterations=max_iter,
                circle_constraint=self.circle_constraint,
                line_constraint=self.line_constraint,
                use_circle=self.circle_var.get(),
                use_line=self.line_var.get(),
                penalty_method=self.penalty_method_var.get()
            )
            self.optimizer.initialize(self.initial_point)

        # Run to completion
        self.optimizer.run_complete()
        self.update_plot()
        self.update_statistics()

    def reset_optimization(self):
        """Reset the optimization"""
        self.optimizer = None
        self.initial_point = None
        self.init_label.config(text="Initial Point: Not set")
        self.update_plot()
        self.update_statistics()

    def update_statistics(self):
        """Update the statistics display"""
        self.stats_text.delete(1.0, tk.END)

        if self.optimizer is None or not self.optimizer.trajectory:
            self.stats_text.insert(tk.END, "No optimization running.\n")
            return

        # Current statistics
        stats = f"Iteration: {self.optimizer.iteration}/{self.optimizer.max_iterations}\n"
        stats += f"Status: {'Complete' if self.optimizer.is_complete else 'Running'}\n"
        stats += f"Penalty Method: {self.optimizer.penalty_method.capitalize()}\n\n"

        # Current point
        curr_point = self.optimizer.current_point
        curr_value = self.optimizer.values[-1]
        stats += f"Current Point:\n"
        stats += f"  x = {curr_point[0]:.6f}\n"
        stats += f"  y = {curr_point[1]:.6f}\n"
        stats += f"  f(x,y) = {curr_value:.6f}\n\n"

        # Best solution
        best_point, best_value = self.optimizer.get_best_solution()
        stats += f"Best Solution Found:\n"
        stats += f"  x = {best_point[0]:.6f}\n"
        stats += f"  y = {best_point[1]:.6f}\n"
        stats += f"  f(x,y) = {best_value:.6f}\n\n"

        # Global minimum
        gx, gy, gval = self.current_function.global_minimum()
        stats += f"Global Minimum:\n"
        stats += f"  x = {gx:.6f}\n"
        stats += f"  y = {gy:.6f}\n"
        stats += f"  f(x,y) = {gval:.6f}\n\n"

        # Feasibility
        is_feas = self.optimizer.is_feasible(*curr_point)
        stats += f"Current Feasibility: {'Feasible' if is_feas else 'Infeasible'}\n"

        self.stats_text.insert(tk.END, stats)

    def update_plot(self):
        """Update the visualization"""
        self.ax.clear()

        # Create meshgrid for plotting
        bounds = self.current_function.bounds
        x = np.linspace(bounds[0][0], bounds[0][1], 200)
        y = np.linspace(bounds[1][0], bounds[1][1], 200)
        X, Y = np.meshgrid(x, y)

        # Evaluate function
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = self.current_function.evaluate(X[i, j], Y[i, j])

        # Check visualization type
        if self.viz_type_var.get() == "3d":
            self.update_plot_3d(X, Y, Z, bounds)
        else:
            self.update_plot_2d(X, Y, Z, bounds)

        self.canvas.draw()

    def update_plot_2d(self, X, Y, Z, bounds):
        """Update 2D contour plot"""
        # Plot contours
        levels = 20
        self.ax.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.6)
        self.ax.contourf(X, Y, Z, levels=levels, cmap='viridis', alpha=0.3)

        # Plot constraints and shade feasible regions
        if self.circle_var.get():
            # Create mask for circle constraint
            circle_mask = (X - self.circle_constraint.center_x)**2 + \
                         (Y - self.circle_constraint.center_y)**2 <= self.circle_constraint.radius**2

            # Draw circle boundary
            circle = CirclePatch((self.circle_constraint.center_x, self.circle_constraint.center_y),
                               self.circle_constraint.radius, fill=False, edgecolor='red',
                               linewidth=2, linestyle='--', label='Circle Constraint')
            self.ax.add_patch(circle)

            # Shade feasible region (inside circle)
            self.ax.contourf(X, Y, circle_mask.astype(float), levels=[0.5, 1.5],
                           colors=['green'], alpha=0.15)

        if self.line_var.get():
            # Line constraint: ax + by + c >= 0
            line_mask = self.line_constraint.a * X + self.line_constraint.b * Y + \
                       self.line_constraint.c >= 0

            # Draw line boundary
            x_line = np.array([bounds[0][0], bounds[0][1]])
            y_line = -(self.line_constraint.a * x_line + self.line_constraint.c) / self.line_constraint.b
            self.ax.plot(x_line, y_line, 'b--', linewidth=2, label='Line Constraint')

            # Shade feasible region (one side of line)
            self.ax.contourf(X, Y, line_mask.astype(float), levels=[0.5, 1.5],
                           colors=['blue'], alpha=0.15)

        # If both constraints active, shade intersection
        if self.circle_var.get() and self.line_var.get():
            circle_mask = (X - self.circle_constraint.center_x)**2 + \
                         (Y - self.circle_constraint.center_y)**2 <= self.circle_constraint.radius**2
            line_mask = self.line_constraint.a * X + self.line_constraint.b * Y + \
                       self.line_constraint.c >= 0
            both_mask = circle_mask & line_mask
            self.ax.contourf(X, Y, both_mask.astype(float), levels=[0.5, 1.5],
                           colors=['yellow'], alpha=0.2)

        # Plot global minimum
        gx, gy, _ = self.current_function.global_minimum()
        self.ax.plot(gx, gy, 'g*', markersize=20, label='Global Minimum',
                    markeredgecolor='black', markeredgewidth=1)

        # Plot trajectory if optimizer exists
        if self.optimizer and self.optimizer.trajectory:
            trajectory = np.array(self.optimizer.trajectory)
            self.ax.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2,
                        alpha=0.7, label='Trajectory')

            # Plot initial point
            self.ax.plot(trajectory[0, 0], trajectory[0, 1], 'bo', markersize=10,
                        label='Initial Point', markeredgecolor='black', markeredgewidth=1)

            # Plot current point
            self.ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'ro', markersize=10,
                        label='Current Point', markeredgecolor='black', markeredgewidth=1)

            # Plot best solution
            best_point, _ = self.optimizer.get_best_solution()
            self.ax.plot(best_point[0], best_point[1], 'mo', markersize=12,
                        label='Best Solution', markeredgecolor='black', markeredgewidth=1.5)

        # Plot initial point if set but no optimization yet
        elif self.initial_point:
            self.ax.plot(self.initial_point[0], self.initial_point[1], 'bo', markersize=10,
                        label='Initial Point', markeredgecolor='black', markeredgewidth=1)

        # Labels and legend
        self.ax.set_xlabel('x', fontsize=12)
        self.ax.set_ylabel('y', fontsize=12)
        self.ax.set_title(f'{self.current_function.name}', fontsize=14, fontweight='bold')
        self.ax.legend(loc='upper right', fontsize=9)
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlim(bounds[0])
        self.ax.set_ylim(bounds[1])

    def update_plot_3d(self, X, Y, Z, bounds):
        """Update 3D surface plot"""
        # Plot surface
        surf = self.ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6,
                                     edgecolor='none', antialiased=True)

        # Add contour lines at the bottom
        self.ax.contour(X, Y, Z, levels=10, cmap='viridis',
                       linestyles="solid", offset=np.min(Z), alpha=0.4)

        # Plot constraints
        if self.circle_var.get():
            # Draw circle constraint on the surface
            theta = np.linspace(0, 2*np.pi, 100)
            cx = self.circle_constraint.center_x + self.circle_constraint.radius * np.cos(theta)
            cy = self.circle_constraint.center_y + self.circle_constraint.radius * np.sin(theta)
            # Evaluate Z values on the circle
            cz = np.array([self.current_function.evaluate(cx[i], cy[i]) for i in range(len(cx))])
            self.ax.plot(cx, cy, cz, 'r-', linewidth=3, label='Circle Constraint')

            # Draw circle on the bottom
            cz_bottom = np.full_like(cx, np.min(Z))
            self.ax.plot(cx, cy, cz_bottom, 'r--', linewidth=2, alpha=0.5)

        if self.line_var.get():
            # Draw line constraint
            x_line = np.array([bounds[0][0], bounds[0][1]])
            y_line = -(self.line_constraint.a * x_line + self.line_constraint.c) / self.line_constraint.b
            # Evaluate Z values on the line
            z_line = np.array([self.current_function.evaluate(x_line[i], y_line[i]) for i in range(len(x_line))])
            self.ax.plot(x_line, y_line, z_line, 'b-', linewidth=3, label='Line Constraint')

            # Draw line on the bottom
            z_line_bottom = np.full_like(x_line, np.min(Z))
            self.ax.plot(x_line, y_line, z_line_bottom, 'b--', linewidth=2, alpha=0.5)

        # Plot global minimum
        gx, gy, gval = self.current_function.global_minimum()
        self.ax.scatter([gx], [gy], [gval], c='green', marker='*', s=300,
                       label='Global Minimum', edgecolors='black', linewidths=1.5)

        # Plot trajectory if optimizer exists
        if self.optimizer and self.optimizer.trajectory:
            trajectory = np.array(self.optimizer.trajectory)
            # Evaluate Z values for trajectory
            traj_z = np.array([self.current_function.evaluate(p[0], p[1]) for p in trajectory])

            self.ax.plot(trajectory[:, 0], trajectory[:, 1], traj_z, 'r-',
                        linewidth=2, alpha=0.7, label='Trajectory')

            # Plot initial point
            self.ax.scatter([trajectory[0, 0]], [trajectory[0, 1]], [traj_z[0]],
                           c='blue', marker='o', s=100, label='Initial Point',
                           edgecolors='black', linewidths=1)

            # Plot current point
            self.ax.scatter([trajectory[-1, 0]], [trajectory[-1, 1]], [traj_z[-1]],
                           c='red', marker='o', s=100, label='Current Point',
                           edgecolors='black', linewidths=1)

            # Plot best solution
            best_point, best_value = self.optimizer.get_best_solution()
            self.ax.scatter([best_point[0]], [best_point[1]], [best_value],
                           c='magenta', marker='o', s=120, label='Best Solution',
                           edgecolors='black', linewidths=1.5)

        # Plot initial point if set but no optimization yet
        elif self.initial_point:
            init_z = self.current_function.evaluate(self.initial_point[0], self.initial_point[1])
            self.ax.scatter([self.initial_point[0]], [self.initial_point[1]], [init_z],
                           c='blue', marker='o', s=100, label='Initial Point',
                           edgecolors='black', linewidths=1)

        # Labels and title
        self.ax.set_xlabel('x', fontsize=12)
        self.ax.set_ylabel('y', fontsize=12)
        self.ax.set_zlabel('f(x,y)', fontsize=12)
        self.ax.set_title(f'{self.current_function.name}', fontsize=14, fontweight='bold')
        self.ax.legend(loc='upper left', fontsize=8)
        self.ax.set_xlim(bounds[0])
        self.ax.set_ylim(bounds[1])


def main():
    """Main entry point"""
    root = tk.Tk()
    app = GradientSearchApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
