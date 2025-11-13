"""
Simulirano hlađenje (Simulated Annealing) - Demo aplikacija
Demonstrira lokalno pretraživanje, tabu pretraživanje i simulirano hlađenje sa Tkinter GUI
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

def generate_random_neighbor(x, delta=0.5):
    """
    Generiši slučajnog susjeda iz okoline tačke x
    Koristi se za simulirano hlađenje
    """
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

    # Izaberi slučajan pravac
    d = directions[np.random.randint(0, len(directions))]
    x_new = [x[0] + d[0], x[1] + d[1]]

    return x_new

def points_equal(p1, p2, tolerance=0.01):
    """Provjeri da li su dvije tačke jednake (u okviru tolerancije)"""
    return abs(p1[0] - p2[0]) < tolerance and abs(p1[1] - p2[1]) < tolerance

class SimulatedAnnealingDemo:
    def __init__(self, root):
        self.root = root
        self.root.title("Simulirano hlađenje - Demo aplikacija")
        self.root.geometry("1400x900")

        # Parametri
        self.selected_function = 'Kvadratna'
        self.objective_function = FUNCTIONS[self.selected_function]['func']
        self.x_range = FUNCTIONS[self.selected_function]['range']
        self.global_min = FUNCTIONS[self.selected_function]['global_min']
        self.delta = 0.5  # Veličina koraka
        self.tabu_tenure = 7  # Dužina tabu liste
        self.search_algorithm = 'Lokalno pretraživanje'  # ili 'Tabu pretraživanje' ili 'Simulirano hlađenje'
        self.max_iterations = 5000  # Maksimalan broj iteracija
        self.view_mode = '2D'  # '2D' ili '3D'

        # Parametri za simulirano hlađenje
        self.cooling_schedule = 'Geometrijska'  # 'Linearna', 'Geometrijska', 'Adaptivna'
        self.initial_temp = 100.0  # Početna temperatura T₀
        self.min_temp = 0.01  # Minimalna temperatura
        self.cooling_delta = 1.0  # Delta za linearno hlađenje
        self.cooling_alpha = 0.9  # Alpha za geometrijsko hlađenje (0.85-0.95)
        self.cooling_beta = 0.01  # Beta za adaptivno hlađenje
        self.iterations_per_temp = 10  # Broj iteracija M pri istoj temperaturi

        # Stanje algoritma
        self.current_solution = None
        self.history = []  # Historija rješenja
        self.current_neighbors = []
        self.best_neighbor = None
        self.iteration = 0
        self.finished = False
        self.tabu_list = []  # Tabu lista (za tabu search)
        self.stop_requested = False  # Flag za zaustavljanje pretraživanja
        self.best_found_solution = None  # Najbolja tačka otkrivena tokom izvršavanja
        self.best_found_value = float('inf')  # Najbolja vrijednost otkrivena tokom izvršavanja

        # Stanje za simulirano hlađenje
        self.current_temp = self.initial_temp
        self.temp_iteration = 0  # Brojač iteracija na trenutnoj temperaturi
        self.temp_history = []  # Historija temperatura

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

        # Desna strana - kontrole (sa scrolling-om)
        right_frame = ttk.Frame(main_container, width=350)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5, 0))
        right_frame.pack_propagate(False)

        # Canvas i scrollbar za desni panel
        right_canvas = tk.Canvas(right_frame, width=350, highlightthickness=0)
        right_scrollbar = ttk.Scrollbar(right_frame, orient="vertical", command=right_canvas.yview)
        self.scrollable_right_frame = ttk.Frame(right_canvas)

        self.scrollable_right_frame.bind(
            "<Configure>",
            lambda e: right_canvas.configure(scrollregion=right_canvas.bbox("all"))
        )

        right_canvas.create_window((0, 0), window=self.scrollable_right_frame, anchor="nw")
        right_canvas.configure(yscrollcommand=right_scrollbar.set)

        right_canvas.pack(side="left", fill="both", expand=True)
        right_scrollbar.pack(side="right", fill="y")

        # Omogući scroll sa mišem
        def _on_mousewheel(event):
            right_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        right_canvas.bind_all("<MouseWheel>", _on_mousewheel)

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
        title_label = ttk.Label(self.scrollable_right_frame, text="KONTROLE",
                               font=('Arial', 14, 'bold'))
        title_label.pack(pady=(0, 10))

        # Izbor algoritma
        algo_frame = ttk.LabelFrame(self.scrollable_right_frame, text="Algoritam", padding=10)
        algo_frame.pack(fill=tk.X, pady=5)

        self.algo_var = tk.StringVar(value='Lokalno pretraživanje')
        ttk.Radiobutton(algo_frame, text="Lokalno pretraživanje",
                       variable=self.algo_var, value='Lokalno pretraživanje',
                       command=self.on_algorithm_changed).pack(anchor=tk.W)
        ttk.Radiobutton(algo_frame, text="Tabu pretraživanje",
                       variable=self.algo_var, value='Tabu pretraživanje',
                       command=self.on_algorithm_changed).pack(anchor=tk.W)
        ttk.Radiobutton(algo_frame, text="Simulirano hlađenje",
                       variable=self.algo_var, value='Simulirano hlađenje',
                       command=self.on_algorithm_changed).pack(anchor=tk.W)

        # Izbor funkcije
        func_frame = ttk.LabelFrame(self.scrollable_right_frame, text="Funkcija", padding=10)
        func_frame.pack(fill=tk.X, pady=5)

        self.func_var = tk.StringVar(value='Kvadratna')
        for func_name in FUNCTIONS.keys():
            ttk.Radiobutton(func_frame, text=func_name,
                          variable=self.func_var, value=func_name,
                          command=self.on_function_changed).pack(anchor=tk.W)

        # Parametri - organizovani u tabove
        params_notebook = ttk.Notebook(self.scrollable_right_frame)
        params_notebook.pack(fill=tk.X, pady=5)

        # Tab 1: Opšti parametri (za sve algoritme)
        general_params_frame = ttk.Frame(params_notebook, padding=10)
        params_notebook.add(general_params_frame, text="Opšti")

        # Delta slider
        ttk.Label(general_params_frame, text="Delta (veličina koraka):").pack(anchor=tk.W)
        self.delta_var = tk.DoubleVar(value=0.5)
        self.delta_scale = ttk.Scale(general_params_frame, from_=0.1, to=2.0,
                                    variable=self.delta_var, orient=tk.HORIZONTAL,
                                    command=self.on_delta_changed)
        self.delta_scale.pack(fill=tk.X, pady=(0, 5))
        self.delta_label = ttk.Label(general_params_frame, text=f"Δ = {self.delta:.1f}")
        self.delta_label.pack(anchor=tk.W)

        # Max iterations slider
        ttk.Label(general_params_frame, text="Maks. broj iteracija:").pack(anchor=tk.W, pady=(10, 0))
        self.max_iter_var = tk.IntVar(value=5000)
        self.max_iter_scale = ttk.Scale(general_params_frame, from_=100, to=10000,
                                       variable=self.max_iter_var, orient=tk.HORIZONTAL,
                                       command=self.on_max_iter_changed)
        self.max_iter_scale.pack(fill=tk.X, pady=(0, 5))
        self.max_iter_label = ttk.Label(general_params_frame, text=f"Max iter = {self.max_iterations}")
        self.max_iter_label.pack(anchor=tk.W)

        # Tab 2: Parametri za Tabu pretraživanje
        tabu_params_frame = ttk.Frame(params_notebook, padding=10)
        params_notebook.add(tabu_params_frame, text="Tabu")

        # Tabu tenure slider
        ttk.Label(tabu_params_frame, text="Dužina tabu liste:").pack(anchor=tk.W)
        self.tabu_var = tk.IntVar(value=7)
        self.tabu_scale = ttk.Scale(tabu_params_frame, from_=3, to=50,
                                   variable=self.tabu_var, orient=tk.HORIZONTAL,
                                   command=self.on_tabu_changed)
        self.tabu_scale.pack(fill=tk.X, pady=(0, 5))
        self.tabu_label = ttk.Label(tabu_params_frame, text=f"Tabu lista = {self.tabu_tenure}")
        self.tabu_label.pack(anchor=tk.W)

        # Tab 3: Parametri za simulirano hlađenje
        sa_params_frame = ttk.Frame(params_notebook, padding=10)
        params_notebook.add(sa_params_frame, text="Sim. hlađenje")

        # Izbor funkcije hlađenja
        ttk.Label(sa_params_frame, text="Funkcija hlađenja:").pack(anchor=tk.W)
        self.cooling_var = tk.StringVar(value='Geometrijska')
        cooling_options = ['Linearna', 'Geometrijska', 'Adaptivna']
        self.cooling_combo = ttk.Combobox(sa_params_frame, textvariable=self.cooling_var,
                                         values=cooling_options, state='readonly', width=15)
        self.cooling_combo.pack(anchor=tk.W, pady=(0, 10))
        self.cooling_combo.bind('<<ComboboxSelected>>', self.on_cooling_changed)

        # Početna temperatura T₀
        ttk.Label(sa_params_frame, text="Početna temperatura T₀:").pack(anchor=tk.W)
        self.initial_temp_var = tk.DoubleVar(value=100.0)
        self.initial_temp_scale = ttk.Scale(sa_params_frame, from_=10, to=500,
                                           variable=self.initial_temp_var, orient=tk.HORIZONTAL,
                                           command=self.on_initial_temp_changed)
        self.initial_temp_scale.pack(fill=tk.X, pady=(0, 5))
        self.initial_temp_label = ttk.Label(sa_params_frame, text=f"T₀ = {self.initial_temp:.1f}")
        self.initial_temp_label.pack(anchor=tk.W)

        # Minimalna temperatura
        ttk.Label(sa_params_frame, text="Minimalna temperatura:").pack(anchor=tk.W, pady=(10, 0))
        self.min_temp_var = tk.DoubleVar(value=0.01)
        self.min_temp_scale = ttk.Scale(sa_params_frame, from_=0.001, to=10,
                                       variable=self.min_temp_var, orient=tk.HORIZONTAL,
                                       command=self.on_min_temp_changed)
        self.min_temp_scale.pack(fill=tk.X, pady=(0, 5))
        self.min_temp_label = ttk.Label(sa_params_frame, text=f"T_min = {self.min_temp:.3f}")
        self.min_temp_label.pack(anchor=tk.W)

        # M - broj iteracija po temperaturi
        ttk.Label(sa_params_frame, text="Iteracije po temperaturi (M):").pack(anchor=tk.W, pady=(10, 0))
        self.iter_per_temp_var = tk.IntVar(value=10)
        self.iter_per_temp_scale = ttk.Scale(sa_params_frame, from_=1, to=100,
                                            variable=self.iter_per_temp_var, orient=tk.HORIZONTAL,
                                            command=self.on_iter_per_temp_changed)
        self.iter_per_temp_scale.pack(fill=tk.X, pady=(0, 5))
        self.iter_per_temp_label = ttk.Label(sa_params_frame, text=f"M = {self.iterations_per_temp}")
        self.iter_per_temp_label.pack(anchor=tk.W)

        # Parametri funkcije hlađenja
        self.cooling_params_frame = ttk.Frame(sa_params_frame)
        self.cooling_params_frame.pack(fill=tk.X, pady=(10, 0))
        self.update_cooling_params_ui()

        # Dugmad
        buttons_frame = ttk.LabelFrame(self.scrollable_right_frame, text="Akcije", padding=10)
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
        view_frame = ttk.LabelFrame(self.scrollable_right_frame, text="Prikaz", padding=10)
        view_frame.pack(fill=tk.X, pady=5)

        view_buttons_frame = ttk.Frame(view_frame)
        view_buttons_frame.pack(fill=tk.X)

        ttk.Button(view_buttons_frame, text="2D (Contour)",
                  command=self.switch_to_2d).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        ttk.Button(view_buttons_frame, text="3D (Mesh)",
                  command=self.switch_to_3d).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))

        # Info tekst
        info_frame = ttk.LabelFrame(self.scrollable_right_frame, text="Status", padding=10)
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

    def update_cooling_params_ui(self):
        """Ažuriraj UI za parametre funkcije hlađenja"""
        # Ukloni stare widgete
        for widget in self.cooling_params_frame.winfo_children():
            widget.destroy()

        if self.cooling_schedule == 'Linearna':
            # Delta parameter za linearno hlađenje: T_k+1 = T_k - delta
            ttk.Label(self.cooling_params_frame, text="Delta (δ):").pack(anchor=tk.W)
            self.cooling_delta_var = tk.DoubleVar(value=self.cooling_delta)
            self.cooling_delta_scale = ttk.Scale(self.cooling_params_frame, from_=0.1, to=5.0,
                                                variable=self.cooling_delta_var, orient=tk.HORIZONTAL,
                                                command=self.on_cooling_delta_changed)
            self.cooling_delta_scale.pack(fill=tk.X, pady=(0, 5))
            self.cooling_delta_label = ttk.Label(self.cooling_params_frame,
                                                text=f"δ = {self.cooling_delta:.2f}")
            self.cooling_delta_label.pack(anchor=tk.W)

        elif self.cooling_schedule == 'Geometrijska':
            # Alpha parameter za geometrijsko hlađenje: T_k+1 = alpha * T_k
            ttk.Label(self.cooling_params_frame, text="Alpha (α):").pack(anchor=tk.W)
            self.cooling_alpha_var = tk.DoubleVar(value=self.cooling_alpha)
            self.cooling_alpha_scale = ttk.Scale(self.cooling_params_frame, from_=0.7, to=0.99,
                                                variable=self.cooling_alpha_var, orient=tk.HORIZONTAL,
                                                command=self.on_cooling_alpha_changed)
            self.cooling_alpha_scale.pack(fill=tk.X, pady=(0, 5))
            self.cooling_alpha_label = ttk.Label(self.cooling_params_frame,
                                                text=f"α = {self.cooling_alpha:.3f}")
            self.cooling_alpha_label.pack(anchor=tk.W)

        else:  # Adaptivna
            # Beta parameter za adaptivno hlađenje: T_k+1 = T_k / (1 + beta * T_k)
            ttk.Label(self.cooling_params_frame, text="Beta (β):").pack(anchor=tk.W)
            self.cooling_beta_var = tk.DoubleVar(value=self.cooling_beta)
            self.cooling_beta_scale = ttk.Scale(self.cooling_params_frame, from_=0.001, to=0.1,
                                               variable=self.cooling_beta_var, orient=tk.HORIZONTAL,
                                               command=self.on_cooling_beta_changed)
            self.cooling_beta_scale.pack(fill=tk.X, pady=(0, 5))
            self.cooling_beta_label = ttk.Label(self.cooling_params_frame,
                                               text=f"β = {self.cooling_beta:.4f}")
            self.cooling_beta_label.pack(anchor=tk.W)

    def cool_temperature(self):
        """Umanji temperaturu prema odabranoj funkciji hlađenja"""
        if self.cooling_schedule == 'Linearna':
            # T_k+1 = T_k - delta
            self.current_temp = max(self.min_temp, self.current_temp - self.cooling_delta)
        elif self.cooling_schedule == 'Geometrijska':
            # T_k+1 = alpha * T_k
            self.current_temp = max(self.min_temp, self.cooling_alpha * self.current_temp)
        else:  # Adaptivna
            # T_k+1 = T_k / (1 + beta * T_k)
            self.current_temp = max(self.min_temp,
                                   self.current_temp / (1 + self.cooling_beta * self.current_temp))

        self.temp_history.append(self.current_temp)

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
            self.ax.set_title(f'{self.search_algorithm} - {self.selected_function} (3D)',
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
            self.ax.set_title(f'{self.search_algorithm} - {self.selected_function} (2D)',
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

            # Nacrtaj SVE susjedne tačke (ako nisu SA)
            if self.current_neighbors and self.search_algorithm != 'Simulirano hlađenje':
                neighbors_x = [n[0] for n in self.current_neighbors]
                neighbors_y = [n[1] for n in self.current_neighbors]
                neighbors_z = [self.objective_function(n) for n in self.current_neighbors]
                self.ax.scatter(neighbors_x, neighbors_y, neighbors_z,
                              c='orange', s=100, marker='s',
                              edgecolors='darkorange', linewidths=1.5,
                              label='Okolina (8 tačaka)', zorder=7, alpha=0.7)

            # Označi tabu tačke (ako koristimo tabu search)
            if self.search_algorithm == 'Tabu pretraživanje' and self.tabu_list:
                tabu_x = [t[0] for t in self.tabu_list]
                tabu_y = [t[1] for t in self.tabu_list]
                tabu_z = [self.objective_function(t) for t in self.tabu_list]
                self.ax.scatter(tabu_x, tabu_y, tabu_z,
                              c='red', s=80, marker='x',
                              linewidths=2, label='Tabu lista', zorder=8)

            # Najbolji susjed
            if self.best_neighbor and self.search_algorithm != 'Simulirano hlađenje':
                z_best = self.objective_function(self.best_neighbor)
                self.ax.scatter([self.best_neighbor[0]], [self.best_neighbor[1]], [z_best],
                              c='lime', s=150, marker='D',
                              edgecolors='darkgreen', linewidths=2,
                              label='Najbolji susjed', zorder=8)

            # ★ Najbolja otkrivena tačka (ZLATNA ZVIJEZDA)
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

                # Nacrtaj pravougaonik koji predstavlja okolinu (samo za lokalno i tabu)
                if self.search_algorithm != 'Simulirano hlađenje':
                    rect_size = self.delta * 2
                    rect = patches.Rectangle((self.current_solution[0] - self.delta,
                                             self.current_solution[1] - self.delta),
                                            rect_size, rect_size,
                                            linewidth=2, edgecolor='orange',
                                            facecolor='orange', alpha=0.1, linestyle='--')
                    self.ax.add_patch(rect)

            # Nacrtaj SVE susjedne tačke (ako nisu SA)
            if self.current_neighbors and self.search_algorithm != 'Simulirano hlađenje':
                neighbors_x = [n[0] for n in self.current_neighbors]
                neighbors_y = [n[1] for n in self.current_neighbors]
                self.ax.scatter(neighbors_x, neighbors_y,
                              c='orange', s=100, marker='s',
                              edgecolors='darkorange', linewidths=1.5,
                              label='Okolina (8 tačaka)', zorder=7, alpha=0.7)

            # Označi tabu tačke (ako koristimo tabu search)
            if self.search_algorithm == 'Tabu pretraživanje' and self.tabu_list:
                tabu_x = [t[0] for t in self.tabu_list]
                tabu_y = [t[1] for t in self.tabu_list]
                self.ax.scatter(tabu_x, tabu_y,
                              c='red', s=80, marker='x',
                              linewidths=2, label='Tabu lista', zorder=8)

            # Najbolji susjed
            if self.best_neighbor and self.search_algorithm != 'Simulirano hlađenje':
                self.ax.scatter([self.best_neighbor[0]], [self.best_neighbor[1]],
                              c='lime', s=150, marker='D',
                              edgecolors='darkgreen', linewidths=2,
                              label='Najbolji susjed', zorder=8)

            # ★ Najbolja otkrivena tačka (ZLATNA ZVIJEZDA)
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
        info += f"  ALGORITAM: {self.search_algorithm}\n"
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

            # Info za simulirano hlađenje
            if self.search_algorithm == 'Simulirano hlađenje':
                info += f"Temperatura: T = {self.current_temp:.4f}\n"
                info += f"Funkcija hlađenja: {self.cooling_schedule}\n"
                info += f"Iteracija na T: {self.temp_iteration}/{self.iterations_per_temp}\n\n"

            # Tabu lista info
            if self.search_algorithm == 'Tabu pretraživanje':
                info += f"Tabu lista ({len(self.tabu_list)}/{self.tabu_tenure}):\n"
                if self.tabu_list:
                    # Prikaži sve elemente u tabu listi
                    for i, t in enumerate(self.tabu_list):
                        info += f"  {i+1}. [{t[0]:.3f}, {t[1]:.3f}]\n"
                else:
                    info += "  (prazna)\n"
                info += "\n"

            if self.finished:
                if self.search_algorithm == 'Simulirano hlađenje':
                    info += "✓ MINIMALNA TEMPERATURA DOSTIGNUTA!\n"
                else:
                    info += "✓ LOKALNI MINIMUM PRONAĐEN!\n"
            else:
                info += "Pretraživanje u toku...\n"

        info += "\n" + "─"*35 + "\n"
        info += "PSEUDOKOD:\n"
        info += "─"*35 + "\n"

        if self.search_algorithm == 'Lokalno pretraživanje':
            info += "x ← x⁰\n"
            info += "repeat\n"
            info += "  N(x) ← okolina od x\n"
            info += "  x' ← najbolji(N(x))\n"
            info += "  if f(x') < f(x):\n"
            info += "    x ← x'\n"
            info += "until nema poboljšanja\n"
        elif self.search_algorithm == 'Tabu pretraživanje':
            info += "x ← x⁰, TabuList ← ∅\n"
            info += "repeat\n"
            info += "  N(x) ← okolina od x\n"
            info += "  N'(x) ← N(x) \\ TabuList\n"
            info += "  x' ← najbolji(N'(x))\n"
            info += "  x ← x'\n"
            info += "  add x to TabuList\n"
            info += "  if |TabuList| > tenure:\n"
            info += "    remove oldest from TabuList\n"
            info += "until stop criterion\n"
        else:  # Simulirano hlađenje
            info += "x ← x⁰, T ← T₀\n"
            info += "repeat\n"
            info += "  for i = 1 to M:\n"
            info += "    x' ← random(N(x))\n"
            info += "    Δf ← f(x') - f(x)\n"
            info += "    if Δf < 0:\n"
            info += "      x ← x'\n"
            info += "    else:\n"
            info += "      p ← exp(-Δf/T)\n"
            info += "      if random() < p:\n"
            info += "        x ← x'\n"
            info += "  T ← cool(T)\n"
            info += "until T < T_min\n"

        self.info_text.insert('1.0', info)

    def on_algorithm_changed(self):
        """Promjena algoritma"""
        self.search_algorithm = self.algo_var.get()

        # Resetuj stanje
        self.current_solution = None
        self.history = []
        self.current_neighbors = []
        self.best_neighbor = None
        self.iteration = 0
        self.finished = False
        self.tabu_list = []
        self.best_found_solution = None
        self.best_found_value = float('inf')
        self.current_temp = self.initial_temp
        self.temp_iteration = 0
        self.temp_history = []

        # Ponovno crtanje
        self.draw_objective_function()
        self.update_info_text()

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
        self.tabu_list = []
        self.best_found_solution = None
        self.best_found_value = float('inf')
        self.current_temp = self.initial_temp
        self.temp_iteration = 0
        self.temp_history = []

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

    def on_tabu_changed(self, val):
        """Promjena dužine tabu liste"""
        self.tabu_tenure = int(float(val))
        self.tabu_label.config(text=f"Tabu lista = {self.tabu_tenure}")

        # Skrati tabu listu ako je duža od nove dužine
        if len(self.tabu_list) > self.tabu_tenure:
            self.tabu_list = self.tabu_list[-self.tabu_tenure:]

        self.update_info_text()

    def on_max_iter_changed(self, val):
        """Promjena maksimalnog broja iteracija"""
        self.max_iterations = int(float(val))
        self.max_iter_label.config(text=f"Max iter = {self.max_iterations}")

    def on_initial_temp_changed(self, val):
        """Promjena početne temperature"""
        self.initial_temp = float(val)
        self.initial_temp_label.config(text=f"T₀ = {self.initial_temp:.1f}")

    def on_min_temp_changed(self, val):
        """Promjena minimalne temperature"""
        self.min_temp = float(val)
        self.min_temp_label.config(text=f"T_min = {self.min_temp:.3f}")

    def on_iter_per_temp_changed(self, val):
        """Promjena broja iteracija po temperaturi"""
        self.iterations_per_temp = int(float(val))
        self.iter_per_temp_label.config(text=f"M = {self.iterations_per_temp}")

    def on_cooling_changed(self, event=None):
        """Promjena funkcije hlađenja"""
        self.cooling_schedule = self.cooling_var.get()
        self.update_cooling_params_ui()

    def on_cooling_delta_changed(self, val):
        """Promjena delta parametra za linearno hlađenje"""
        self.cooling_delta = float(val)
        self.cooling_delta_label.config(text=f"δ = {self.cooling_delta:.2f}")

    def on_cooling_alpha_changed(self, val):
        """Promjena alpha parametra za geometrijsko hlađenje"""
        self.cooling_alpha = float(val)
        self.cooling_alpha_label.config(text=f"α = {self.cooling_alpha:.3f}")

    def on_cooling_beta_changed(self, val):
        """Promjena beta parametra za adaptivno hlađenje"""
        self.cooling_beta = float(val)
        self.cooling_beta_label.config(text=f"β = {self.cooling_beta:.4f}")

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
            self.tabu_list = []

            # Resetuj stanje za simulirano hlađenje
            self.current_temp = self.initial_temp
            self.temp_iteration = 0
            self.temp_history = [self.initial_temp]

            # Postavi početnu tačku kao najbolju
            f_x0 = self.objective_function(x0)
            self.best_found_solution = x0.copy()
            self.best_found_value = f_x0

            print(f"\nPostavljena početna tačka: x = [{x0[0]:.3f}, {x0[1]:.3f}], " +
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
        self.tabu_list = []

        # Resetuj stanje za simulirano hlađenje
        self.current_temp = self.initial_temp
        self.temp_iteration = 0
        self.temp_history = [self.initial_temp]

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

        if self.search_algorithm == 'Lokalno pretraživanje':
            self.local_search_step()
        elif self.search_algorithm == 'Tabu pretraživanje':
            self.tabu_search_step()
        else:  # Simulirano hlađenje
            self.simulated_annealing_step()

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

    def tabu_search_step(self):
        """Jedan korak tabu pretraživanja"""
        # Generiši okolinu
        neighbors = generate_neighborhood(self.current_solution, self.delta)

        # Ograniči susjedne tačke na dozvoljeni prostor
        neighbors = [[np.clip(n[0], self.x_range[0], self.x_range[1]),
                      np.clip(n[1], self.x_range[0], self.x_range[1])] for n in neighbors]

        # Filtriraj susjedne tačke koje su u tabu listi
        allowed_neighbors = []
        for n in neighbors:
            is_tabu = False
            for tabu_point in self.tabu_list:
                if points_equal(n, tabu_point):
                    is_tabu = True
                    break
            if not is_tabu:
                allowed_neighbors.append(n)

        # Ako su svi susjedi tabu, dozvoli najbolji (aspiration criterion)
        if not allowed_neighbors:
            print("  NAPOMENA: Svi susjedi su tabu! Koristim aspiration criterion.")
            allowed_neighbors = neighbors

        self.current_neighbors = neighbors

        # Evaluiraj dozvoljene susjedne tačke
        neighbor_values = [self.objective_function(n) for n in allowed_neighbors]
        current_value = self.objective_function(self.current_solution)

        # Nađi najbolju dozvoljenu susjednu tačku
        best_neighbor_idx = np.argmin(neighbor_values)
        self.best_neighbor = allowed_neighbors[best_neighbor_idx]
        best_value = neighbor_values[best_neighbor_idx]

        # Ažuriraj najbolje otkriveno rješenje
        if current_value < self.best_found_value:
            self.best_found_solution = self.current_solution.copy()
            self.best_found_value = current_value

        print(f"\nIteracija {self.iteration + 1} (Tabu Search):")
        print(f"  Trenutno: x = [{self.current_solution[0]:.3f}, {self.current_solution[1]:.3f}], " +
              f"f(x) = {current_value:.3f}")
        print(f"  Tabu lista: {len(self.tabu_list)}/{self.tabu_tenure}")
        print(f"  Dozvoljeni susjedi: {len(allowed_neighbors)}/8")
        print(f"  Najbolji dozvoljeni susjed: x = [{self.best_neighbor[0]:.3f}, {self.best_neighbor[1]:.3f}], " +
              f"f(x) = {best_value:.3f}")
        print(f"  ★ NAJBOLJE OTKRIVENO: x = [{self.best_found_solution[0]:.3f}, {self.best_found_solution[1]:.3f}], " +
              f"f(x) = {self.best_found_value:.3f}")

        # Dodaj trenutnu tačku u tabu listu
        self.tabu_list.append(list(self.current_solution))

        # Ograniči dužinu tabu liste
        if len(self.tabu_list) > self.tabu_tenure:
            removed = self.tabu_list.pop(0)
            print(f"  → Uklonjena iz tabu liste: [{removed[0]:.3f}, {removed[1]:.3f}]")

        # Pomjeri se na najbolju dozvoljenu tačku (čak i ako je gora!)
        self.current_solution = self.best_neighbor
        self.history.append(self.best_neighbor)
        self.iteration += 1

        # Tabu search ne zaustavlja se kod lokalnog minimuma
        # Ali možemo dodati uslov zaustavljanja nakon određenog broja iteracija
        if self.iteration >= self.max_iterations:
            print(f"  → Dostignut maksimalan broj iteracija ({self.max_iterations})!")
            self.finished = True
        else:
            print(f"  → Pomak na tačku (može biti i gora!)")

    def simulated_annealing_step(self):
        """Jedan korak simuliranog hlađenja"""
        current_value = self.objective_function(self.current_solution)

        # Generiši slučajnog susjeda
        neighbor = generate_random_neighbor(self.current_solution, self.delta)

        # Ograniči susjeda na dozvoljeni prostor
        neighbor = [np.clip(neighbor[0], self.x_range[0], self.x_range[1]),
                   np.clip(neighbor[1], self.x_range[0], self.x_range[1])]

        neighbor_value = self.objective_function(neighbor)

        # Računaj promjenu energije
        delta_f = neighbor_value - current_value

        # Odluči da li prihvatiti rješenje
        accepted = False
        if delta_f < 0:
            # Bolje rješenje - uvijek prihvati
            accepted = True
            acceptance_prob = 1.0
        else:
            # Gore rješenje - prihvati sa vjerovatnoćom exp(-Δf/T)
            acceptance_prob = np.exp(-delta_f / self.current_temp) if self.current_temp > 0 else 0
            if np.random.random() < acceptance_prob:
                accepted = True

        # Ažuriraj najbolje otkriveno rješenje
        if current_value < self.best_found_value:
            self.best_found_solution = self.current_solution.copy()
            self.best_found_value = current_value

        print(f"\nIteracija {self.iteration + 1} (Simulated Annealing):")
        print(f"  Temperatura: T = {self.current_temp:.4f}")
        print(f"  Trenutno: x = [{self.current_solution[0]:.3f}, {self.current_solution[1]:.3f}], " +
              f"f(x) = {current_value:.3f}")
        print(f"  Susjed: x = [{neighbor[0]:.3f}, {neighbor[1]:.3f}], " +
              f"f(x) = {neighbor_value:.3f}")
        print(f"  Δf = {delta_f:.3f}, P(prihvat) = {acceptance_prob:.4f}")
        print(f"  ★ NAJBOLJE OTKRIVENO: x = [{self.best_found_solution[0]:.3f}, {self.best_found_solution[1]:.3f}], " +
              f"f(x) = {self.best_found_value:.3f}")

        if accepted:
            self.current_solution = neighbor
            self.history.append(neighbor)
            print(f"  → Prihvaćeno!")
        else:
            print(f"  → Odbijeno!")

        self.iteration += 1
        self.temp_iteration += 1

        # Provjeri da li treba umanjiti temperaturu
        if self.temp_iteration >= self.iterations_per_temp:
            old_temp = self.current_temp
            self.cool_temperature()
            self.temp_iteration = 0
            print(f"  → Hlađenje: T = {old_temp:.4f} → {self.current_temp:.4f}")

            # Provjeri uslov zaustavljanja
            if self.current_temp <= self.min_temp:
                print(f"  → MINIMALNA TEMPERATURA DOSTIGNUTA!")
                self.finished = True

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
        while not self.finished and not self.stop_requested:
            self.on_step()
            self.root.update()
            self.root.after(100)  # Pauza od 100ms

        # Prikaži dijalog sa najboljom otkrivenom tačkom
        if self.best_found_solution is not None:
            # Različita poruka u zavisnosti da li je pretraživanje završeno ili zaustavljeno
            if self.stop_requested:
                title = "PRETRAŽIVANJE ZAUSTAVLJENO!"
            else:
                title = "PRETRAŽIVANJE ZAVRŠENO!"

            result_message = (
                f"{title}\n\n"
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
        self.tabu_list = []
        self.stop_requested = False
        self.best_found_solution = None
        self.best_found_value = float('inf')
        self.current_temp = self.initial_temp
        self.temp_iteration = 0
        self.temp_history = []

        self.draw_objective_function()
        self.update_info_text()

        print("\nAplikacija resetovana!")

    def on_stop(self):
        """Zaustavi pretraživanje"""
        self.stop_requested = True
        print("\nZaustavljanje pretraživanja zatraženo...")

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
                          "Simulirano hlađenje - Demo aplikacija\n\n" +
                          "Optimizacija resursa\n\n" +
                          "Red. prof. dr Samim Konjicija\n\n" +
                          "Novembar 2025. godine")

    def on_help(self):
        """Prikaži Help dialog"""
        help_text = """UPUTE ZA KORIŠTENJE - Simulirano hlađenje

OSNOVNE FUNKCIJE:
• Klik na grafik - postavite početnu tačku
• Slučajan start - generiši slučajnu početnu tačku
• Jedan korak - izvršite jednu iteraciju
• Do kraja - izvršite kompletno pretraživanje
• Reset - resetujte aplikaciju

ALGORITMI:
• Lokalno pretraživanje - zaustavlja se kod lokalnog minimuma
• Tabu pretraživanje - nastavlja pretraživanje izbjegavajući
  nedavno posjećene tačke
• Simulirano hlađenje - probabilistički algoritam inspirisan
  procesom hlađenja metala

KONTROLE:
• Radio buttons (Algoritam) - izaberite algoritam
• Radio buttons (Funkcija) - izaberite test funkciju
• Slider (Delta) - veličina koraka (0.1 - 2.0)
• Slider (Tabu lista) - dužina tabu liste (3 - 50)

SIMULIRANO HLAĐENJE:
• Funkcije hlađenja:
  - Linearna: T_k+1 = T_k - δ
  - Geometrijska: T_k+1 = α × T_k
  - Adaptivna: T_k+1 = T_k / (1 + β × T_k)
• Temperatura kontroliše prihvatanje lošijih rješenja
• Viša temperatura → veća vjerojatnost prihvatanja lošijih rješenja
• Temperatura se postepeno smanjuje prema odabranoj funkciji

PARAMETRI SA:
• T₀ - početna temperatura (10-500)
• T_min - minimalna temperatura (0.001-10)
• M - broj iteracija po temperaturi (1-100)
• δ/α/β - parametri funkcije hlađenja

LEGENDA:
• Zelena zvijezda - globalni minimum
• Crveni krug - trenutna tačka
• Zlatna zvijezda - najbolja otkrivena tačka
• Ljubičasta linija - putanja"""

        messagebox.showinfo("Pomoć", help_text)

# Pokreni demo
if __name__ == "__main__":
    print("="*80)
    print("Simulirano hlađenje (Simulated Annealing) - Demo Aplikacija")
    print("="*80)
    print("\nFunkcionalnosti:")
    print("  ✓ Lokalno pretraživanje")
    print("  ✓ Tabu pretraživanje sa podesivom tabu listom")
    print("  ✓ Simulirano hlađenje sa tri funkcije hlađenja:")
    print("    - Linearna: T_k+1 = T_k - δ")
    print("    - Geometrijska: T_k+1 = α × T_k")
    print("    - Adaptivna: T_k+1 = T_k / (1 + β × T_k)")
    print("  ✓ 5 test funkcija: Kvadratna, Rastrigin, Ackley, Griewank, Levy")
    print("  ✓ Tkinter GUI sa matplotlib canvas-om")
    print("  ✓ Prikaz trenutnog stanja, temperature i najbolje otkrivene tačke")
    print("\nKontrole:")
    print("  - Kliknite na grafik da postavite početnu tačku")
    print("  - Odaberite algoritam i funkciju")
    print("  - Podesite parametre (delta, tabu tenure, SA parametri)")
    print("  - Koristite dugmad za izvršavanje koraka")
    print("="*80)

    root = tk.Tk()
    app = SimulatedAnnealingDemo(root)
    root.mainloop()
