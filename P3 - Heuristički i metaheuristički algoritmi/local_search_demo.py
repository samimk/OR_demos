"""
Lokalno pretraživanje (Local Search) - Demo aplikacija
Demonstrira osnovno lokalno pretraživanje sa Tkinter GUI
"""

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D

# ============================================================================
# DEFINICIJE TEST FUNKCIJA
# ============================================================================

def quadratic_function(x):
    """Kvadratna funkcija: f(x) = x1^2 + x2^2"""
    return x[0]**2 + x[1]**2

def rastrigin_function(x):
    """Rastrigin funkcija: multimodalna sa mnogo lokalnih minimuma"""
    A = 10
    n = len(x)
    return A * n + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x])

def ackley_function(x):
    """Ackley funkcija: multimodalna sa globalnim minimumom u (0,0)"""
    a = 20
    b = 0.2
    c = 2 * np.pi
    n = len(x)
    sum1 = sum([xi**2 for xi in x])
    sum2 = sum([np.cos(c * xi) for xi in x])
    return -a * np.exp(-b * np.sqrt(sum1/n)) - np.exp(sum2/n) + a + np.e

def griewank_function(x):
    """Griewank funkcija: multimodalna sa globalnim minimumom u (0,0)"""
    sum_part = sum([xi**2 for xi in x]) / 4000
    prod_part = np.prod([np.cos(xi / np.sqrt(i+1)) for i, xi in enumerate(x)])
    return sum_part - prod_part + 1

def levy_function(x):
    """Levy funkcija: multimodalna"""
    w = [1 + (xi - 1) / 4 for xi in x]
    term1 = np.sin(np.pi * w[0])**2
    term2 = sum([(wi - 1)**2 * (1 + 10 * np.sin(np.pi * wi + 1)**2) for wi in w[:-1]])
    term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
    return term1 + term2 + term3

# Mapa funkcija
FUNCTIONS = {
    'Kvadratna': {
        'func': quadratic_function,
        'range': (-5, 5),
        'global_min': (0, 0)
    },
    'Rastrigin': {
        'func': rastrigin_function,
        'range': (-5, 5),
        'global_min': (0, 0)
    },
    'Ackley': {
        'func': ackley_function,
        'range': (-5, 5),
        'global_min': (0, 0)
    },
    'Griewank': {
        'func': griewank_function,
        'range': (-5, 5),
        'global_min': (0, 0)
    },
    'Levy': {
        'func': levy_function,
        'range': (-5, 5),
        'global_min': (1, 1)
    }
}

def generate_neighborhood(x, delta=0.5):
    """
    Generiši okolinu tačke x
    N(x,δ) = {x' ∈ Ω | x' = δ(x)}

    Okolina: 8 tačaka oko trenutne tačke (gore, dole, lijevo, desno + dijagonale)
    """
    neighbors = []

    # Pomeraj u 8 pravaca
    directions = [
        [delta, 0],      # desno
        [-delta, 0],     # lijevo
        [0, delta],      # gore
        [0, -delta],     # dole
        [delta, delta],  # desno-gore
        [-delta, delta], # lijevo-gore
        [delta, -delta], # desno-dole
        [-delta, -delta] # lijevo-dole
    ]

    for d in directions:
        x_new = [x[0] + d[0], x[1] + d[1]]
        neighbors.append(x_new)

    return neighbors

class LocalSearchDemo:
    def __init__(self, root):
        self.root = root
        self.root.title("Lokalno pretraživanje - Demo aplikacija")
        self.root.geometry("1400x900")

        # Parametri
        self.selected_function = 'Kvadratna'
        self.objective_function = FUNCTIONS[self.selected_function]['func']
        self.x_range = FUNCTIONS[self.selected_function]['range']
        self.global_min = FUNCTIONS[self.selected_function]['global_min']
        self.delta = 0.5  # Veličina koraka
        self.max_iterations = 1000  # Maksimalan broj iteracija
        self.view_mode = '2D'  # '2D' ili '3D'

        # Stanje algoritma
        self.current_solution = None
        self.history = []  # Historija rješenja
        self.current_neighbors = []
        self.best_neighbor = None
        self.iteration = 0
        self.finished = False
        self.stop_requested = False  # Flag za zaustavljanje pretraživanja
        self.best_found_solution = None  # Najbolja tačka otkrivena tokom izvršavanja
        self.best_found_value = float('inf')  # Najbolja vrijednost otkrivena tokom izvršavanja

        # Setup GUI
        self.setup_gui()

        # Nacrtaj početni plot
        self.draw_objective_function()
        self.update_info_text()

    def setup_gui(self):
        """Postavi GUI elemente"""

        # Kreiraj meni bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # Help meni
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="O aplikaciji", command=self.on_about)
        help_menu.add_command(label="Uputstvo", command=self.on_help)

        # Glavni kontejner
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Lijeva strana - grafik
        left_frame = ttk.Frame(main_container)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Desna strana - kontrole
        right_frame = ttk.Frame(main_container, width=350)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5, 0))
        right_frame.pack_propagate(False)

        # === GRAFIK ===
        self.fig = Figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.fig, master=left_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Toolbar za navigaciju (omogućava rotiranje 3D prikaza)
        self.toolbar = NavigationToolbar2Tk(self.canvas, left_frame)
        self.toolbar.update()

        # Klik event
        self.canvas.mpl_connect('button_press_event', self.on_click)

        # === KONTROLE ===

        # Naslov
        title_label = ttk.Label(right_frame, text="KONTROLE",
                               font=('Arial', 14, 'bold'))
        title_label.pack(pady=(0, 10))

        # Izbor funkcije
        func_frame = ttk.LabelFrame(right_frame, text="Funkcija", padding=10)
        func_frame.pack(fill=tk.X, pady=5)

        self.func_var = tk.StringVar(value='Kvadratna')
        for func_name in FUNCTIONS.keys():
            ttk.Radiobutton(func_frame, text=func_name,
                          variable=self.func_var, value=func_name,
                          command=self.on_function_changed).pack(anchor=tk.W)

        # Parametri
        params_frame = ttk.LabelFrame(right_frame, text="Parametri", padding=10)
        params_frame.pack(fill=tk.X, pady=5)

        # Delta slider
        ttk.Label(params_frame, text="Delta (veličina koraka):").pack(anchor=tk.W)
        self.delta_var = tk.DoubleVar(value=0.5)
        self.delta_scale = ttk.Scale(params_frame, from_=0.1, to=2.0,
                                    variable=self.delta_var, orient=tk.HORIZONTAL,
                                    command=self.on_delta_changed)
        self.delta_scale.pack(fill=tk.X, pady=(0, 5))
        self.delta_label = ttk.Label(params_frame, text=f"Δ = {self.delta:.1f}")
        self.delta_label.pack(anchor=tk.W)

        # Max iterations slider
        ttk.Label(params_frame, text="Maks. broj iteracija:").pack(anchor=tk.W, pady=(10, 0))
        self.max_iter_var = tk.IntVar(value=1000)
        self.max_iter_scale = ttk.Scale(params_frame, from_=100, to=5000,
                                       variable=self.max_iter_var, orient=tk.HORIZONTAL,
                                       command=self.on_max_iter_changed)
        self.max_iter_scale.pack(fill=tk.X, pady=(0, 5))
        self.max_iter_label = ttk.Label(params_frame, text=f"Max iter = {self.max_iterations}")
        self.max_iter_label.pack(anchor=tk.W)

        # Dugmad
        buttons_frame = ttk.LabelFrame(right_frame, text="Akcije", padding=10)
        buttons_frame.pack(fill=tk.X, pady=5)

        ttk.Button(buttons_frame, text="Klik za start",
                  command=self.show_click_instruction).pack(fill=tk.X, pady=2)
        ttk.Button(buttons_frame, text="Slučajan start",
                  command=self.on_new_start).pack(fill=tk.X, pady=2)
        ttk.Button(buttons_frame, text="Jedan korak",
                  command=self.on_step).pack(fill=tk.X, pady=2)
        ttk.Button(buttons_frame, text="Do kraja",
                  command=self.on_complete).pack(fill=tk.X, pady=2)
        ttk.Button(buttons_frame, text="Zaustavi",
                  command=self.on_stop).pack(fill=tk.X, pady=2)
        ttk.Button(buttons_frame, text="Reset",
                  command=self.on_reset).pack(fill=tk.X, pady=2)

        # Dugmad za prikaz
        view_frame = ttk.LabelFrame(right_frame, text="Prikaz", padding=10)
        view_frame.pack(fill=tk.X, pady=5)

        view_buttons_frame = ttk.Frame(view_frame)
        view_buttons_frame.pack(fill=tk.X)

        ttk.Button(view_buttons_frame, text="2D (Contour)",
                  command=self.switch_to_2d).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        ttk.Button(view_buttons_frame, text="3D (Mesh)",
                  command=self.switch_to_3d).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))

        # Info tekst
        info_frame = ttk.LabelFrame(right_frame, text="Status", padding=10)
        info_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Text widget sa scrollbar-om
        text_frame = ttk.Frame(info_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.info_text = tk.Text(text_frame, wrap=tk.WORD, height=15,
                                font=('Courier', 9),
                                yscrollcommand=scrollbar.set)
        self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.info_text.yview)

    def draw_objective_function(self):
        """Nacrtaj objektivnu funkciju"""
        # Ukloni stari axes i kreiraj novi prema view_mode
        self.fig.clear()

        if self.view_mode == '3D':
            self.ax = self.fig.add_subplot(111, projection='3d')
        else:
            self.ax = self.fig.add_subplot(111)

        # Generiši mrežu tačaka
        x1 = np.linspace(self.x_range[0], self.x_range[1], 100)
        x2 = np.linspace(self.x_range[0], self.x_range[1], 100)
        X1, X2 = np.meshgrid(x1, x2)
        Z = np.zeros_like(X1)

        for i in range(X1.shape[0]):
            for j in range(X1.shape[1]):
                Z[i, j] = self.objective_function([X1[i, j], X2[i, j]])

        if self.view_mode == '3D':
            # 3D mesh prikaz
            self.ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.7,
                                edgecolor='none', antialiased=True)

            # Označi globalni minimum
            z_min = self.objective_function(self.global_min)
            self.ax.scatter([self.global_min[0]], [self.global_min[1]], [z_min],
                           c='green', s=200, marker='*',
                           edgecolors='darkgreen', linewidths=2,
                           label=f'Globalni minimum {self.global_min}', zorder=10)

            self.ax.set_xlabel('x₁', fontsize=11, fontweight='bold')
            self.ax.set_ylabel('x₂', fontsize=11, fontweight='bold')
            self.ax.set_zlabel('f(x)', fontsize=11, fontweight='bold')
            self.ax.set_title(f'Lokalno pretraživanje - {self.selected_function} (3D)',
                             fontsize=13, fontweight='bold')
        else:
            # 2D konturni dijagram
            contour = self.ax.contour(X1, X2, Z, levels=20, cmap='viridis', alpha=0.6)
            self.ax.clabel(contour, inline=True, fontsize=8)

            # Označi globalni minimum
            self.ax.scatter([self.global_min[0]], [self.global_min[1]],
                           c='green', s=200, marker='*',
                           edgecolors='darkgreen', linewidths=2,
                           label=f'Globalni minimum {self.global_min}', zorder=10)

            self.ax.set_xlabel('x₁', fontsize=12, fontweight='bold')
            self.ax.set_ylabel('x₂', fontsize=12, fontweight='bold')
            self.ax.set_title(f'Lokalno pretraživanje - {self.selected_function} (2D)',
                             fontsize=13, fontweight='bold')
            self.ax.grid(True, alpha=0.3)
            self.ax.set_xlim(self.x_range)
            self.ax.set_ylim(self.x_range)

        self.ax.legend(loc='upper right', fontsize=9)

        # Reconnect click event
        self.canvas.mpl_connect('button_press_event', self.on_click)

        self.canvas.draw()

    def update_plot(self):
        """Ažuriraj grafički prikaz"""

        if self.view_mode == '3D':
            # 3D prikaz
            # Nacrtaj historiju (putanju)
            if len(self.history) > 1:
                history_x = [h[0] for h in self.history]
                history_y = [h[1] for h in self.history]
                history_z = [self.objective_function(h) for h in self.history]
                self.ax.plot(history_x, history_y, history_z, 'o-', color='purple',
                            linewidth=2, markersize=6, alpha=0.6, label='Putanja')

            # Nacrtaj trenutno rješenje
            if self.current_solution:
                z_current = self.objective_function(self.current_solution)
                self.ax.scatter([self.current_solution[0]], [self.current_solution[1]], [z_current],
                              c='red', s=200, marker='o',
                              edgecolors='darkred', linewidths=2.5,
                              label='Trenutna tačka', zorder=9)

            # Nacrtaj SVE susjedne tačke
            if self.current_neighbors:
                neighbors_x = [n[0] for n in self.current_neighbors]
                neighbors_y = [n[1] for n in self.current_neighbors]
                neighbors_z = [self.objective_function(n) for n in self.current_neighbors]
                self.ax.scatter(neighbors_x, neighbors_y, neighbors_z,
                              c='orange', s=100, marker='s',
                              edgecolors='darkorange', linewidths=1.5,
                              label='Okolina (8 tačaka)', zorder=7, alpha=0.7)

            # Najbolji susjed
            if self.best_neighbor:
                z_best = self.objective_function(self.best_neighbor)
                self.ax.scatter([self.best_neighbor[0]], [self.best_neighbor[1]], [z_best],
                              c='lime', s=150, marker='D',
                              edgecolors='darkgreen', linewidths=2,
                              label='Najbolji susjed', zorder=8)

            # ★ Najbolja otkrivena tačka
            if self.best_found_solution:
                z_best_found = self.objective_function(self.best_found_solution)
                self.ax.scatter([self.best_found_solution[0]], [self.best_found_solution[1]], [z_best_found],
                              c='gold', s=300, marker='*',
                              edgecolors='darkorange', linewidths=3,
                              label='★ Najbolja otkrivena', zorder=11)
        else:
            # 2D prikaz
            # Nacrtaj historiju (putanju)
            if len(self.history) > 1:
                history_x = [h[0] for h in self.history]
                history_y = [h[1] for h in self.history]
                self.ax.plot(history_x, history_y, 'o-', color='purple',
                            linewidth=2, markersize=6, alpha=0.6, label='Putanja')

            # Nacrtaj trenutno rješenje
            if self.current_solution:
                self.ax.scatter([self.current_solution[0]], [self.current_solution[1]],
                              c='red', s=200, marker='o',
                              edgecolors='darkred', linewidths=2.5,
                              label='Trenutna tačka', zorder=9)

                # Nacrtaj pravougaonik koji predstavlja okolinu
                rect_size = self.delta * 2
                rect = patches.Rectangle((self.current_solution[0] - self.delta,
                                         self.current_solution[1] - self.delta),
                                        rect_size, rect_size,
                                        linewidth=2, edgecolor='orange',
                                        facecolor='orange', alpha=0.1, linestyle='--')
                self.ax.add_patch(rect)

            # Nacrtaj SVE susjedne tačke
            if self.current_neighbors:
                neighbors_x = [n[0] for n in self.current_neighbors]
                neighbors_y = [n[1] for n in self.current_neighbors]
                self.ax.scatter(neighbors_x, neighbors_y,
                              c='orange', s=100, marker='s',
                              edgecolors='darkorange', linewidths=1.5,
                              label='Okolina (8 tačaka)', zorder=7, alpha=0.7)

            # Najbolji susjed
            if self.best_neighbor:
                self.ax.scatter([self.best_neighbor[0]], [self.best_neighbor[1]],
                              c='lime', s=150, marker='D',
                              edgecolors='darkgreen', linewidths=2,
                              label='Najbolji susjed', zorder=8)

            # ★ Najbolja otkrivena tačka
            if self.best_found_solution:
                self.ax.scatter([self.best_found_solution[0]], [self.best_found_solution[1]],
                              c='gold', s=300, marker='*',
                              edgecolors='darkorange', linewidths=3,
                              label='★ Najbolja otkrivena', zorder=11)

        self.ax.legend(loc='upper right', fontsize=9)
        self.canvas.draw()

    def update_info_text(self):
        """Ažuriraj info tekst"""
        self.info_text.delete('1.0', tk.END)

        info = f"═══════════════════════════════════\n"
        info += f"  LOKALNO PRETRAŽIVANJE\n"
        info += f"═══════════════════════════════════\n\n"

        if self.current_solution is None:
            info += "Kliknite na grafik ili odaberite\n'Slučajan start' da počnete.\n"
        else:
            x = self.current_solution
            f_x = self.objective_function(x)

            info += f"Iteracija: {self.iteration}\n\n"
            info += f"Trenutna tačka:\n"
            info += f"  x = [{x[0]:.4f}, {x[1]:.4f}]\n"
            info += f"  f(x) = {f_x:.6f}\n\n"

            # Najbolja otkrivena tačka
            if self.best_found_solution:
                info += f"★ NAJBOLJA OTKRIVENA:\n"
                info += f"  x = [{self.best_found_solution[0]:.4f}, {self.best_found_solution[1]:.4f}]\n"
                info += f"  f(x) = {self.best_found_value:.6f}\n\n"

            # Udaljenost od globalnog minimuma
            dist = np.sqrt((x[0] - self.global_min[0])**2 +
                          (x[1] - self.global_min[1])**2)
            info += f"Udaljenost od globalnog: {dist:.4f}\n\n"

            if self.finished:
                info += "✓ LOKALNI MINIMUM PRONAĐEN!\n"
            else:
                info += "Pretraživanje u toku...\n"

        info += "\n" + "─"*35 + "\n"
        info += "PSEUDOKOD:\n"
        info += "─"*35 + "\n"
        info += "x ← x⁰\n"
        info += "repeat\n"
        info += "  N(x) ← okolina od x\n"
        info += "  x' ← najbolji(N(x))\n"
        info += "  if f(x') < f(x):\n"
        info += "    x ← x'\n"
        info += "until nema poboljšanja\n"

        self.info_text.insert('1.0', info)

    def on_function_changed(self):
        """Promjena funkcije"""
        self.selected_function = self.func_var.get()
        self.objective_function = FUNCTIONS[self.selected_function]['func']
        self.x_range = FUNCTIONS[self.selected_function]['range']
        self.global_min = FUNCTIONS[self.selected_function]['global_min']

        # Resetuj stanje
        self.current_solution = None
        self.history = []
        self.current_neighbors = []
        self.best_neighbor = None
        self.iteration = 0
        self.finished = False
        self.best_found_solution = None
        self.best_found_value = float('inf')

        # Ponovno crtanje
        self.draw_objective_function()
        self.update_info_text()

    def on_delta_changed(self, val):
        """Promjena delta parametra"""
        self.delta = float(val)
        self.delta_label.config(text=f"Δ = {self.delta:.1f}")
        if self.current_solution:
            self.current_neighbors = []
            self.best_neighbor = None
            self.draw_objective_function()
            self.update_plot()

    def on_max_iter_changed(self, val):
        """Promjena maksimalnog broja iteracija"""
        self.max_iterations = int(float(val))
        self.max_iter_label.config(text=f"Max iter = {self.max_iterations}")

    def on_click(self, event):
        """Postavi početnu tačku klikom miša"""
        # Provjeri da li je toolbar u aktivnom modu (zoom, pan, etc.)
        # Ako jeste, ne postavljaj početnu tačku
        if self.toolbar.mode != '':
            return

        if event.inaxes == self.ax:
            x0 = [event.xdata, event.ydata]

            # Ograniči na dozvoljeni prostor
            x0[0] = np.clip(x0[0], self.x_range[0], self.x_range[1])
            x0[1] = np.clip(x0[1], self.x_range[0], self.x_range[1])

            self.current_solution = x0
            self.history = [x0]
            self.iteration = 0
            self.finished = False
            self.current_neighbors = []
            self.best_neighbor = None

            # Postavi početnu tačku kao najbolju
            f_x0 = self.objective_function(x0)
            self.best_found_solution = x0.copy()
            self.best_found_value = f_x0

            print(f"\nPostavljena početna tačku: x = [{x0[0]:.3f}, {x0[1]:.3f}], " +
                  f"f(x) = {f_x0:.3f}")

            self.draw_objective_function()
            self.update_plot()
            self.update_info_text()

    def on_new_start(self):
        """Postavi novu slučajnu početnu tačku"""
        x0 = [np.random.uniform(self.x_range[0], self.x_range[1]),
              np.random.uniform(self.x_range[0], self.x_range[1])]

        self.current_solution = x0
        self.history = [x0]
        self.iteration = 0
        self.finished = False
        self.current_neighbors = []
        self.best_neighbor = None

        # Postavi početnu tačku kao najbolju
        f_x0 = self.objective_function(x0)
        self.best_found_solution = x0.copy()
        self.best_found_value = f_x0

        print(f"\nNova početna tačka: x = [{x0[0]:.3f}, {x0[1]:.3f}], " +
              f"f(x) = {f_x0:.3f}")

        self.draw_objective_function()
        self.update_plot()
        self.update_info_text()

    def on_step(self):
        """Izvrši jedan korak pretraživanja"""
        if self.current_solution is None:
            messagebox.showwarning("Upozorenje", "Prvo odaberite početnu tačku!")
            return

        if self.finished:
            messagebox.showinfo("Info", "Pretraživanje je završeno! Odaberite novu početnu tačku.")
            return

        self.local_search_step()

        self.draw_objective_function()
        self.update_plot()
        self.update_info_text()

    def local_search_step(self):
        """Jedan korak lokalnog pretraživanja"""
        # Generiši okolinu
        neighbors = generate_neighborhood(self.current_solution, self.delta)

        # Ograniči susjedne tačke na dozvoljeni prostor
        neighbors = [[np.clip(n[0], self.x_range[0], self.x_range[1]),
                      np.clip(n[1], self.x_range[0], self.x_range[1])] for n in neighbors]

        self.current_neighbors = neighbors

        # Evaluiraj susjedne tačke
        neighbor_values = [self.objective_function(n) for n in neighbors]
        current_value = self.objective_function(self.current_solution)

        # Nađi najbolju susjednu tačku
        best_neighbor_idx = np.argmin(neighbor_values)
        self.best_neighbor = neighbors[best_neighbor_idx]
        best_value = neighbor_values[best_neighbor_idx]

        # Ažuriraj najbolje otkriveno rješenje
        if current_value < self.best_found_value:
            self.best_found_solution = self.current_solution.copy()
            self.best_found_value = current_value

        print(f"\nIteracija {self.iteration + 1}:")
        print(f"  Trenutno: x = [{self.current_solution[0]:.3f}, {self.current_solution[1]:.3f}], " +
              f"f(x) = {current_value:.3f}")
        print(f"  Najbolji susjed: x = [{self.best_neighbor[0]:.3f}, {self.best_neighbor[1]:.3f}], " +
              f"f(x) = {best_value:.3f}")
        print(f"  ★ NAJBOLJE OTKRIVENO: x = [{self.best_found_solution[0]:.3f}, {self.best_found_solution[1]:.3f}], " +
              f"f(x) = {self.best_found_value:.3f}")

        # Provjeri uslov zaustavljanja
        if best_value >= current_value:
            print("  → LOKALNI MINIMUM PRONAĐEN!")
            self.finished = True
        else:
            # Pomjeri se na bolju tačku
            self.current_solution = self.best_neighbor
            self.history.append(self.best_neighbor)
            self.iteration += 1
            print(f"  → Pomak na bolju tačku")

    def on_complete(self):
        """Izvrši kompletno pretraživanje"""
        if self.current_solution is None:
            messagebox.showwarning("Upozorenje", "Prvo odaberite početnu tačku!")
            return

        if self.finished:
            messagebox.showinfo("Info", "Pretraživanje je već završeno!")
            return

        # Resetuj stop flag
        self.stop_requested = False

        # Izvrši korake dok ne dođemo do završetka ili dok ne bude zatraženo zaustavljanje
        while not self.finished and not self.stop_requested and self.iteration < self.max_iterations:
            self.on_step()
            self.root.update()
            self.root.after(100)  # Pauza od 100ms

        # Prikaži dijalog sa najboljom otkrivenom tačkom
        if self.best_found_solution is not None:
            result_message = (
                f"PRETRAŽIVANJE ZAVRŠENO!\n\n"
                f"★ NAJBOLJA OTKRIVENA TAČKA ★\n\n"
                f"x = [{self.best_found_solution[0]:.6f}, {self.best_found_solution[1]:.6f}]\n\n"
                f"f(x) = {self.best_found_value:.6f}\n\n"
                f"Ukupno iteracija: {self.iteration}"
            )
            messagebox.showinfo("Rezultat pretraživanja", result_message)

        if self.stop_requested:
            print("Pretraživanje zaustavljeno od strane korisnika!")
            self.stop_requested = False

    def on_reset(self):
        """Resetuj aplikaciju"""
        self.current_solution = None
        self.history = []
        self.current_neighbors = []
        self.best_neighbor = None
        self.iteration = 0
        self.finished = False
        self.stop_requested = False
        self.best_found_solution = None
        self.best_found_value = float('inf')

        self.draw_objective_function()
        self.update_info_text()

        print("\nAplikacija resetovana!")

    def on_stop(self):
        """Zaustavi pretraživanje"""
        self.stop_requested = True
        print("\nZaustavljanje pretraživanja zatraženo...")

        # Prikaži dijalog sa najboljom otkrivenom tačkom
        if self.best_found_solution is not None:
            result_message = (
                f"PRETRAŽIVANJE ZAUSTAVLJENO!\n\n"
                f"★ NAJBOLJA OTKRIVENA TAČKA ★\n\n"
                f"x = [{self.best_found_solution[0]:.6f}, {self.best_found_solution[1]:.6f}]\n\n"
                f"f(x) = {self.best_found_value:.6f}\n\n"
                f"Ukupno iteracija: {self.iteration}"
            )
            messagebox.showinfo("Rezultat pretraživanja", result_message)

    def switch_to_2d(self):
        """Prebaci na 2D contour prikaz"""
        if self.view_mode != '2D':
            self.view_mode = '2D'
            self.draw_objective_function()
            if self.current_solution:
                self.update_plot()
            print("\nPrebačeno na 2D (contour) prikaz")

    def switch_to_3d(self):
        """Prebaci na 3D mesh prikaz"""
        if self.view_mode != '3D':
            self.view_mode = '3D'
            self.draw_objective_function()
            if self.current_solution:
                self.update_plot()
            print("\nPrebačeno na 3D (mesh) prikaz")

    def show_click_instruction(self):
        """Prikaži instrukcije za klik"""
        messagebox.showinfo("Instrukcija",
                          "Kliknite bilo gdje na grafiku da postavite početnu tačku!")

    def on_about(self):
        """Prikaži About dialog"""
        messagebox.showinfo("O aplikaciji",
                          "Lokalno pretraživanje - Demo aplikacija\n\n" +
                          "Optimizacija resursa\n\n" +
                          "Red. prof. dr Samim Konjicija\n\n" +
                          "Novembar 2025. godine")

    def on_help(self):
        """Prikaži Help dialog"""
        help_text = """UPUTE ZA KORIŠTENJE - Lokalno pretraživanje

OSNOVNE FUNKCIJE:
• Klik na grafik - postavite početnu tačku
• Slučajan start - generiši slučajnu početnu tačku
• Jedan korak - izvršite jednu iteraciju
• Do kraja - izvršite kompletno pretraživanje
• Zaustavi - zaustavite pretraživanje u toku
• Reset - resetujte aplikaciju

ALGORITAM:
• Lokalno pretraživanje koristi steepest descent strategiju
• U svakoj iteraciji se pomijerite na najbolju susjednu tačku
• Algoritam se zaustavlja kada nema boljeg susjeda (lokalni minimum)

KONTROLE:
• Radio buttons (Funkcija) - izaberite test funkciju
• Slider (Delta) - veličina koraka (0.1 - 2.0)
• Slider (Max iteracija) - maksimalan broj iteracija

LEGENDA:
• Zelena zvijezda - globalni minimum
• Crveni krug - trenutna tačka
• Narančasti kvadrati - susjedne tačke
• Zeleni dijamant - najbolji susjed
• Zlatna zvijezda - najbolja otkrivena tačka
• Ljubičasta linija - putanja pretraživanja"""

        messagebox.showinfo("Pomoć", help_text)

# Pokreni demo
if __name__ == "__main__":
    print("="*80)
    print("Lokalno pretraživanje (Local Search) - Demo Aplikacija")
    print("="*80)
    print("\nFunkcionalnosti:")
    print("  ✓ Lokalno pretraživanje")
    print("  ✓ 5 test funkcija: Kvadratna, Rastrigin, Ackley, Griewank, Levy")
    print("  ✓ Tkinter GUI sa matplotlib canvas-om")
    print("  ✓ Help meni (O aplikaciji i Uputstvo)")
    print("  ✓ Prikaz najbolje otkrivene tačke tokom izvršavanja")
    print("\nKontrole:")
    print("  - Kliknite na grafik da postavite početnu tačku")
    print("  - Odaberite funkciju")
    print("  - Podesite parametre (delta, max iteracije)")
    print("  - Koristite dugmad za izvršavanje koraka")
    print("="*80)

    root = tk.Tk()
    app = LocalSearchDemo(root)
    root.mainloop()
