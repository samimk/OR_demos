"""
Genetički algoritam (Genetic Algorithm) - Demo aplikacija
Demonstrira lokalno pretraživanje, tabu pretraživanje, simulirano hlađenje i genetički algoritam sa Tkinter GUI
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

class GeneticAlgorithmDemo:
    def __init__(self, root):
        self.root = root
        self.root.title("Genetički algoritam - Demo aplikacija")
        self.root.geometry("1400x1000")

        # Parametri
        self.selected_function = 'Kvadratna'
        self.objective_function = FUNCTIONS[self.selected_function]['func']
        self.x_range = FUNCTIONS[self.selected_function]['range']
        self.global_min = FUNCTIONS[self.selected_function]['global_min']
        self.delta = 0.5  # Veličina koraka / diskretizacija prostora
        self.tabu_tenure = 7  # Dužina tabu liste
        self.search_algorithm = 'Lokalno pretraživanje'
        self.max_iterations = 5000  # Maksimalan broj iteracija (generacija za GA)
        self.view_mode = '2D'  # '2D' ili '3D'

        # Parametri za simulirano hlađenje
        self.cooling_schedule = 'Geometrijska'
        self.initial_temp = 100.0
        self.min_temp = 0.01
        self.cooling_delta = 1.0
        self.cooling_alpha = 0.9
        self.cooling_beta = 0.01
        self.iterations_per_temp = 10

        # Parametri za GA
        self.population_size = 20
        self.crossover_type = 'Jedna tačka'
        self.crossover_rate = 0.8
        self.mutation_rate = 0.01
        self.selection_method = 'Ruletski točak'
        self.tournament_size = 3
        self.selection_pressure = 1.8  # Za ranking selekciju
        self.elitism = 1
        self.show_fitness_plot = True  # Prikaži grafik fitnessa nakon završetka GA

        # Stanje algoritma
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

        # Stanje za simulirano hlađenje
        self.current_temp = self.initial_temp
        self.temp_iteration = 0
        self.temp_history = []

        # Stanje za GA
        self.population = []  # Lista jedinki [(x1, x2), fitness]
        self.best_fitness_history = []
        self.avg_fitness_history = []

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
        right_frame = ttk.Frame(main_container, width=380)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        right_frame.pack_propagate(False)

        # Canvas i scrollbar za desni panel
        right_canvas = tk.Canvas(right_frame, highlightthickness=0, bg='white')
        right_scrollbar = ttk.Scrollbar(right_frame, orient="vertical", command=right_canvas.yview)

        # Scrollable frame
        self.scrollable_right_frame = ttk.Frame(right_canvas)

        # Bind configure event
        def on_frame_configure(event):
            right_canvas.configure(scrollregion=right_canvas.bbox("all"))

        self.scrollable_right_frame.bind("<Configure>", on_frame_configure)

        # Create window in canvas
        canvas_window = right_canvas.create_window((0, 0), window=self.scrollable_right_frame, anchor="nw")

        # Configure canvas scrolling
        right_canvas.configure(yscrollcommand=right_scrollbar.set)

        # Pack canvas and scrollbar
        right_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Update canvas window width when canvas is resized
        def on_canvas_configure(event):
            right_canvas.itemconfig(canvas_window, width=event.width)

        right_canvas.bind("<Configure>", on_canvas_configure)

        # Mouse wheel scrolling
        def on_mousewheel(event):
            right_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        def bind_mousewheel(event):
            right_canvas.bind_all("<MouseWheel>", on_mousewheel)

        def unbind_mousewheel(event):
            right_canvas.unbind_all("<MouseWheel>")

        right_frame.bind("<Enter>", bind_mousewheel)
        right_frame.bind("<Leave>", unbind_mousewheel)

        # === GRAFIK ===
        self.fig = Figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.fig, master=left_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Toolbar za navigaciju
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
        ttk.Radiobutton(algo_frame, text="Genetički algoritam",
                       variable=self.algo_var, value='Genetički algoritam',
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

        # Tab 1: Opšti parametri
        general_params_frame = ttk.Frame(params_notebook, padding=10)
        params_notebook.add(general_params_frame, text="Opšti")

        # Delta slider
        ttk.Label(general_params_frame, text="Delta (diskretizacija):").pack(anchor=tk.W)
        self.delta_var = tk.DoubleVar(value=0.5)
        self.delta_scale = ttk.Scale(general_params_frame, from_=0.1, to=2.0,
                                    variable=self.delta_var, orient=tk.HORIZONTAL,
                                    command=self.on_delta_changed)
        self.delta_scale.pack(fill=tk.X, pady=(0, 5))
        self.delta_label = ttk.Label(general_params_frame, text=f"Δ = {self.delta:.1f}")
        self.delta_label.pack(anchor=tk.W)

        # Max iterations slider
        ttk.Label(general_params_frame, text="Maks. broj iteracija (generacija):").pack(anchor=tk.W, pady=(10, 0))
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

        ttk.Label(sa_params_frame, text="Funkcija hlađenja:").pack(anchor=tk.W)
        self.cooling_var = tk.StringVar(value='Geometrijska')
        cooling_options = ['Linearna', 'Geometrijska', 'Adaptivna']
        self.cooling_combo = ttk.Combobox(sa_params_frame, textvariable=self.cooling_var,
                                         values=cooling_options, state='readonly')
        self.cooling_combo.pack(fill=tk.X, pady=(0, 10))
        self.cooling_combo.bind('<<ComboboxSelected>>', self.on_cooling_changed)

        ttk.Label(sa_params_frame, text="Početna temperatura T₀:").pack(anchor=tk.W)
        self.initial_temp_var = tk.DoubleVar(value=100.0)
        self.initial_temp_scale = ttk.Scale(sa_params_frame, from_=10, to=500,
                                           variable=self.initial_temp_var, orient=tk.HORIZONTAL,
                                           command=self.on_initial_temp_changed)
        self.initial_temp_scale.pack(fill=tk.X, pady=(0, 5))
        self.initial_temp_label = ttk.Label(sa_params_frame, text=f"T₀ = {self.initial_temp:.1f}")
        self.initial_temp_label.pack(anchor=tk.W)

        ttk.Label(sa_params_frame, text="Minimalna temperatura:").pack(anchor=tk.W, pady=(10, 0))
        self.min_temp_var = tk.DoubleVar(value=0.01)
        self.min_temp_scale = ttk.Scale(sa_params_frame, from_=0.001, to=10,
                                       variable=self.min_temp_var, orient=tk.HORIZONTAL,
                                       command=self.on_min_temp_changed)
        self.min_temp_scale.pack(fill=tk.X, pady=(0, 5))
        self.min_temp_label = ttk.Label(sa_params_frame, text=f"T_min = {self.min_temp:.3f}")
        self.min_temp_label.pack(anchor=tk.W)

        ttk.Label(sa_params_frame, text="Iteracije po temperaturi (M):").pack(anchor=tk.W, pady=(10, 0))
        self.iter_per_temp_var = tk.IntVar(value=10)
        self.iter_per_temp_scale = ttk.Scale(sa_params_frame, from_=1, to=100,
                                            variable=self.iter_per_temp_var, orient=tk.HORIZONTAL,
                                            command=self.on_iter_per_temp_changed)
        self.iter_per_temp_scale.pack(fill=tk.X, pady=(0, 5))
        self.iter_per_temp_label = ttk.Label(sa_params_frame, text=f"M = {self.iterations_per_temp}")
        self.iter_per_temp_label.pack(anchor=tk.W)

        self.cooling_params_frame = ttk.Frame(sa_params_frame)
        self.cooling_params_frame.pack(fill=tk.X, pady=(10, 0))
        self.update_cooling_params_ui()

        # Tab 4: Genetički algoritam
        ga_params_frame = ttk.Frame(params_notebook, padding=10)
        params_notebook.add(ga_params_frame, text="GA")

        # Veličina populacije
        ttk.Label(ga_params_frame, text="Veličina populacije:").pack(anchor=tk.W)
        self.pop_size_var = tk.IntVar(value=20)
        self.pop_size_scale = ttk.Scale(ga_params_frame, from_=10, to=100,
                                       variable=self.pop_size_var, orient=tk.HORIZONTAL,
                                       command=self.on_pop_size_changed)
        self.pop_size_scale.pack(fill=tk.X, pady=(0, 5))
        self.pop_size_label = ttk.Label(ga_params_frame, text=f"Pop = {self.population_size}")
        self.pop_size_label.pack(anchor=tk.W)

        # Operator ukrštanja
        ttk.Label(ga_params_frame, text="Operator ukrštanja:").pack(anchor=tk.W, pady=(10, 0))
        self.crossover_type_var = tk.StringVar(value='Jedna tačka')
        crossover_options = ['Jedna tačka', 'Dvije tačke', 'Uniformno']
        self.crossover_combo = ttk.Combobox(ga_params_frame, textvariable=self.crossover_type_var,
                                           values=crossover_options, state='readonly')
        self.crossover_combo.pack(fill=tk.X, pady=(0, 5))
        self.crossover_combo.bind('<<ComboboxSelected>>', self.on_crossover_type_changed)

        # Vjerovatnoća ukrštanja
        ttk.Label(ga_params_frame, text="Vjerovatnoća ukrštanja (pₓ):").pack(anchor=tk.W, pady=(10, 0))
        self.crossover_rate_var = tk.DoubleVar(value=0.8)
        self.crossover_rate_scale = ttk.Scale(ga_params_frame, from_=0.0, to=1.0,
                                             variable=self.crossover_rate_var, orient=tk.HORIZONTAL,
                                             command=self.on_crossover_rate_changed)
        self.crossover_rate_scale.pack(fill=tk.X, pady=(0, 5))
        self.crossover_rate_label = ttk.Label(ga_params_frame, text=f"pₓ = {self.crossover_rate:.2f}")
        self.crossover_rate_label.pack(anchor=tk.W)

        # Vjerovatnoća mutacije
        ttk.Label(ga_params_frame, text="Vjerovatnoća mutacije (pₘ):").pack(anchor=tk.W, pady=(10, 0))
        self.mutation_rate_var = tk.DoubleVar(value=0.01)
        self.mutation_rate_scale = ttk.Scale(ga_params_frame, from_=0.0, to=0.1,
                                            variable=self.mutation_rate_var, orient=tk.HORIZONTAL,
                                            command=self.on_mutation_rate_changed)
        self.mutation_rate_scale.pack(fill=tk.X, pady=(0, 5))
        self.mutation_rate_label = ttk.Label(ga_params_frame, text=f"pₘ = {self.mutation_rate:.3f}")
        self.mutation_rate_label.pack(anchor=tk.W)

        # Metoda selekcije
        ttk.Label(ga_params_frame, text="Metoda selekcije:").pack(anchor=tk.W, pady=(10, 0))
        self.selection_method_var = tk.StringVar(value='Ruletski točak')
        selection_options = ['Ruletski točak', 'Rangiranje', 'Turnirska']
        self.selection_combo = ttk.Combobox(ga_params_frame, textvariable=self.selection_method_var,
                                           values=selection_options, state='readonly')
        self.selection_combo.pack(fill=tk.X, pady=(0, 5))
        self.selection_combo.bind('<<ComboboxSelected>>', self.on_selection_method_changed)

        # Frame za dodatne parametre selekcije
        self.selection_params_frame = ttk.Frame(ga_params_frame)
        self.selection_params_frame.pack(fill=tk.X, pady=(10, 0))
        self.update_selection_params_ui()

        # Elitizam
        ttk.Label(ga_params_frame, text="Elitizam (broj jedinki):").pack(anchor=tk.W, pady=(10, 0))
        self.elitism_var = tk.IntVar(value=1)
        self.elitism_scale = ttk.Scale(ga_params_frame, from_=0, to=10,
                                      variable=self.elitism_var, orient=tk.HORIZONTAL,
                                      command=self.on_elitism_changed)
        self.elitism_scale.pack(fill=tk.X, pady=(0, 5))
        self.elitism_label = ttk.Label(ga_params_frame, text=f"Elite = {self.elitism}")
        self.elitism_label.pack(anchor=tk.W)

        # Checkbox za prikaz grafa fitnessa
        ttk.Label(ga_params_frame, text="").pack(pady=(10, 0))  # Spacer
        self.show_fitness_plot_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ga_params_frame, text="Prikaži tok izvršavanja GA nakon završetka",
                       variable=self.show_fitness_plot_var,
                       command=self.on_show_fitness_plot_changed).pack(anchor=tk.W)

        # Tab 5: Prikaz/Vizualizacija tab
        view_tab_frame = ttk.Frame(params_notebook, padding=10)
        params_notebook.add(view_tab_frame, text="Prikaz")

        ttk.Label(view_tab_frame, text="Režim vizualizacije:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(0, 10))

        view_buttons_frame_tab = ttk.Frame(view_tab_frame)
        view_buttons_frame_tab.pack(fill=tk.X, pady=5)

        ttk.Button(view_buttons_frame_tab, text="2D (Contour)",
                  command=self.switch_to_2d).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(view_buttons_frame_tab, text="3D (Mesh)",
                  command=self.switch_to_3d).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))

        info_label = ttk.Label(view_tab_frame,
                              text="2D: Konturni dijagram sa oznakama\n3D: Površinski prikaz funkcije",
                              justify=tk.LEFT,
                              foreground='gray')
        info_label.pack(anchor=tk.W, pady=(10, 0))

        # Tab 6: Status/Info tab
        status_frame = ttk.Frame(params_notebook, padding=10)
        params_notebook.add(status_frame, text="Status")

        text_frame = ttk.Frame(status_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar_status = ttk.Scrollbar(text_frame)
        scrollbar_status.pack(side=tk.RIGHT, fill=tk.Y)

        self.info_text = tk.Text(text_frame, wrap=tk.WORD, height=25,
                                font=('Courier', 9),
                                yscrollcommand=scrollbar_status.set)
        self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_status.config(command=self.info_text.yview)

        # Dugmad
        buttons_frame = ttk.LabelFrame(self.scrollable_right_frame, text="Akcije", padding=10)
        buttons_frame.pack(fill=tk.X, pady=5)

        self.btn_click = ttk.Button(buttons_frame, text="Klik za start",
                  command=self.show_click_instruction)
        self.btn_click.pack(fill=tk.X, pady=2)

        self.btn_random = ttk.Button(buttons_frame, text="Slučajan start",
                  command=self.on_new_start)
        self.btn_random.pack(fill=tk.X, pady=2)

        ttk.Button(buttons_frame, text="Jedan korak",
                  command=self.on_step).pack(fill=tk.X, pady=2)
        ttk.Button(buttons_frame, text="Do kraja",
                  command=self.on_complete).pack(fill=tk.X, pady=2)
        ttk.Button(buttons_frame, text="Zaustavi",
                  command=self.on_stop).pack(fill=tk.X, pady=2)
        ttk.Button(buttons_frame, text="Reset",
                  command=self.on_reset).pack(fill=tk.X, pady=2)

        # Inicijalno stanje dugmadi
        self.update_button_states()

    def update_button_states(self):
        """Ažuriraj stanje dugmadi na osnovu algoritma"""
        if self.search_algorithm == 'Genetički algoritam':
            self.btn_click.config(state='disabled')
            self.btn_random.config(text="Inicijaliziraj populaciju")
        else:
            self.btn_click.config(state='normal')
            self.btn_random.config(text="Slučajan start")

    def update_selection_params_ui(self):
        """Ažuriraj UI za parametre selekcije"""
        for widget in self.selection_params_frame.winfo_children():
            widget.destroy()

        if self.selection_method == 'Turnirska':
            ttk.Label(self.selection_params_frame, text="Veličina turnira:").pack(anchor=tk.W)
            self.tournament_size_var = tk.IntVar(value=self.tournament_size)
            self.tournament_size_scale = ttk.Scale(self.selection_params_frame, from_=2, to=10,
                                                  variable=self.tournament_size_var, orient=tk.HORIZONTAL,
                                                  command=self.on_tournament_size_changed)
            self.tournament_size_scale.pack(fill=tk.X, pady=(0, 5))
            self.tournament_size_label = ttk.Label(self.selection_params_frame,
                                                  text=f"Turnir = {self.tournament_size}")
            self.tournament_size_label.pack(anchor=tk.W)
        elif self.selection_method == 'Rangiranje':
            ttk.Label(self.selection_params_frame, text="Pritisak selekcije (SP):").pack(anchor=tk.W)
            self.selection_pressure_var = tk.DoubleVar(value=self.selection_pressure)
            self.selection_pressure_scale = ttk.Scale(self.selection_params_frame, from_=1.0, to=2.0,
                                                     variable=self.selection_pressure_var, orient=tk.HORIZONTAL,
                                                     command=self.on_selection_pressure_changed)
            self.selection_pressure_scale.pack(fill=tk.X, pady=(0, 5))
            self.selection_pressure_label = ttk.Label(self.selection_params_frame,
                                                     text=f"SP = {self.selection_pressure:.2f}")
            self.selection_pressure_label.pack(anchor=tk.W)

    def update_cooling_params_ui(self):
        """Ažuriraj UI za parametre funkcije hlađenja"""
        for widget in self.cooling_params_frame.winfo_children():
            widget.destroy()

        if self.cooling_schedule == 'Linearna':
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
            self.current_temp = max(self.min_temp, self.current_temp - self.cooling_delta)
        elif self.cooling_schedule == 'Geometrijska':
            self.current_temp = max(self.min_temp, self.cooling_alpha * self.current_temp)
        else:  # Adaptivna
            self.current_temp = max(self.min_temp,
                                   self.current_temp / (1 + self.cooling_beta * self.current_temp))

        self.temp_history.append(self.current_temp)

    # GA Helper funkcije
    def discretize_space(self):
        """Generiši diskretne tačke u prostoru sa korakom delta"""
        x_values = np.arange(self.x_range[0], self.x_range[1] + self.delta, self.delta)
        y_values = np.arange(self.x_range[0], self.x_range[1] + self.delta, self.delta)
        return x_values, y_values

    def initialize_population_ga(self):
        """Inicijalizuj populaciju za GA"""
        self.population = []
        x_values, y_values = self.discretize_space()

        for _ in range(self.population_size):
            x = np.random.choice(x_values)
            y = np.random.choice(y_values)
            individual = [x, y]
            fitness = self.objective_function(individual)
            self.population.append({'genes': individual, 'fitness': fitness})

        # Sortiraj po fitnessu (minimizacija)
        self.population.sort(key=lambda ind: ind['fitness'])

        # Postavi najbolju jedinku
        self.best_found_solution = self.population[0]['genes'].copy()
        self.best_found_value = self.population[0]['fitness']

        self.best_fitness_history = [self.best_found_value]
        self.avg_fitness_history = [np.mean([ind['fitness'] for ind in self.population])]

    def selection_ga(self):
        """Izvrši selekciju i vrati jednu jedinku"""
        if self.selection_method == 'Ruletski točak':
            # Inverzija fitnessa za minimizaciju
            max_fitness = max([ind['fitness'] for ind in self.population])
            adjusted_fitnesses = [max_fitness - ind['fitness'] + 1 for ind in self.population]
            total_fitness = sum(adjusted_fitnesses)

            if total_fitness == 0:
                return self.population[np.random.randint(len(self.population))]

            r = np.random.uniform(0, total_fitness)
            cumulative = 0
            for i, fitness in enumerate(adjusted_fitnesses):
                cumulative += fitness
                if cumulative >= r:
                    return self.population[i]
            return self.population[-1]

        elif self.selection_method == 'Rangiranje':
            n = len(self.population)
            # Population je već sortirana
            probabilities = [(2 - self.selection_pressure) + 2 * (self.selection_pressure - 1) * (n - i - 1) / (n - 1)
                           for i in range(n)]
            total_prob = sum(probabilities)
            r = np.random.uniform(0, total_prob)
            cumulative = 0
            for i, prob in enumerate(probabilities):
                cumulative += prob
                if cumulative >= r:
                    return self.population[i]
            return self.population[0]

        else:  # Turnirska
            tournament = [self.population[np.random.randint(len(self.population))]
                         for _ in range(self.tournament_size)]
            return min(tournament, key=lambda ind: ind['fitness'])

    def crossover_ga(self, parent1, parent2):
        """Ukrštanje dva roditelja"""
        if np.random.random() > self.crossover_rate:
            return [parent1['genes'].copy(), parent2['genes'].copy()]

        x_values, y_values = self.discretize_space()

        if self.crossover_type == 'Jedna tačka':
            # Ukrštanje na prvoj koordinati
            child1 = [parent1['genes'][0], parent2['genes'][1]]
            child2 = [parent2['genes'][0], parent1['genes'][1]]
        elif self.crossover_type == 'Dvije tačke':
            # Uniformno ukrštanje
            child1 = [parent1['genes'][0] if np.random.random() < 0.5 else parent2['genes'][0],
                     parent1['genes'][1] if np.random.random() < 0.5 else parent2['genes'][1]]
            child2 = [parent2['genes'][0] if np.random.random() < 0.5 else parent1['genes'][0],
                     parent2['genes'][1] if np.random.random() < 0.5 else parent1['genes'][1]]
        else:  # Uniformno
            child1 = [parent1['genes'][0] if np.random.random() < 0.5 else parent2['genes'][0],
                     parent1['genes'][1] if np.random.random() < 0.5 else parent2['genes'][1]]
            child2 = [parent2['genes'][0] if np.random.random() < 0.5 else parent1['genes'][0],
                     parent2['genes'][1] if np.random.random() < 0.5 else parent1['genes'][1]]

        return [child1, child2]

    def mutate_ga(self, individual):
        """Mutiraj jedinku"""
        x_values, y_values = self.discretize_space()

        # Mutiraj x koordinatu
        if np.random.random() < self.mutation_rate:
            individual[0] = np.random.choice(x_values)

        # Mutiraj y koordinatu
        if np.random.random() < self.mutation_rate:
            individual[1] = np.random.choice(y_values)

        return individual

    def ga_step(self):
        """Jedan korak GA"""
        new_population = []

        # Elitizam
        for i in range(min(self.elitism, len(self.population))):
            new_population.append({
                'genes': self.population[i]['genes'].copy(),
                'fitness': self.population[i]['fitness']
            })

        # Generiši ostatak populacije
        while len(new_population) < self.population_size:
            parent1 = self.selection_ga()
            parent2 = self.selection_ga()

            children = self.crossover_ga(parent1, parent2)

            for child_genes in children:
                if len(new_population) >= self.population_size:
                    break

                child_genes = self.mutate_ga(child_genes)

                # Ograniči na prostor
                child_genes[0] = np.clip(child_genes[0], self.x_range[0], self.x_range[1])
                child_genes[1] = np.clip(child_genes[1], self.x_range[0], self.x_range[1])

                fitness = self.objective_function(child_genes)
                new_population.append({'genes': child_genes, 'fitness': fitness})

        self.population = new_population
        self.population.sort(key=lambda ind: ind['fitness'])

        # Ažuriraj najbolje rješenje
        if self.population[0]['fitness'] < self.best_found_value:
            self.best_found_solution = self.population[0]['genes'].copy()
            self.best_found_value = self.population[0]['fitness']

        # Zapamti historiju
        self.best_fitness_history.append(self.population[0]['fitness'])
        avg_fitness = np.mean([ind['fitness'] for ind in self.population])
        self.avg_fitness_history.append(avg_fitness)

        self.iteration += 1

        print(f"\nGeneracija {self.iteration}:")
        print(f"  Najbolji: x = {self.population[0]['genes']}, f(x) = {self.population[0]['fitness']:.6f}")
        print(f"  Prosječan fitness: {avg_fitness:.6f}")
        print(f"  ★ NAJBOLJE OTKRIVENO: x = {self.best_found_solution}, f(x) = {self.best_found_value:.6f}")

        # Provjeri uslov zaustavljanja
        if self.iteration >= self.max_iterations:
            print(f"  → Dostignut maksimalan broj generacija!")
            self.finished = True

    def draw_objective_function(self):
        """Nacrtaj objektivnu funkciju"""
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
            self.ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.7,
                                edgecolor='none', antialiased=True)

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
            contour = self.ax.contour(X1, X2, Z, levels=20, cmap='viridis', alpha=0.6)
            self.ax.clabel(contour, inline=True, fontsize=8)

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

        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.draw()

    def update_plot(self):
        """Ažuriraj grafički prikaz"""

        if self.search_algorithm == 'Genetički algoritam':
            # Prikaz za GA - prikaži populaciju
            if self.view_mode == '3D':
                if self.population:
                    pop_x = [ind['genes'][0] for ind in self.population]
                    pop_y = [ind['genes'][1] for ind in self.population]
                    pop_z = [ind['fitness'] for ind in self.population]
                    self.ax.scatter(pop_x, pop_y, pop_z,
                                  c='blue', s=50, marker='o',
                                  alpha=0.6, label='Populacija', zorder=7)

                    # Najbolja jedinka
                    best = self.population[0]
                    self.ax.scatter([best['genes'][0]], [best['genes'][1]], [best['fitness']],
                                  c='red', s=200, marker='o',
                                  edgecolors='darkred', linewidths=2.5,
                                  label='Najbolja jedinka', zorder=9)

                if self.best_found_solution:
                    z_best_found = self.objective_function(self.best_found_solution)
                    self.ax.scatter([self.best_found_solution[0]], [self.best_found_solution[1]], [z_best_found],
                                  c='gold', s=300, marker='*',
                                  edgecolors='darkorange', linewidths=3,
                                  label='★ Najbolja otkrivena', zorder=11)
            else:
                # 2D prikaz
                if self.population:
                    pop_x = [ind['genes'][0] for ind in self.population]
                    pop_y = [ind['genes'][1] for ind in self.population]
                    self.ax.scatter(pop_x, pop_y,
                                  c='blue', s=50, marker='o',
                                  alpha=0.6, label='Populacija', zorder=7)

                    # Najbolja jedinka
                    best = self.population[0]
                    self.ax.scatter([best['genes'][0]], [best['genes'][1]],
                                  c='red', s=200, marker='o',
                                  edgecolors='darkred', linewidths=2.5,
                                  label='Najbolja jedinka', zorder=9)

                if self.best_found_solution:
                    self.ax.scatter([self.best_found_solution[0]], [self.best_found_solution[1]],
                                  c='gold', s=300, marker='*',
                                  edgecolors='darkorange', linewidths=3,
                                  label='★ Najbolja otkrivena', zorder=11)
        else:
            # Ostali algoritmi - originalni kod
            if self.view_mode == '3D':
                if len(self.history) > 1:
                    history_x = [h[0] for h in self.history]
                    history_y = [h[1] for h in self.history]
                    history_z = [self.objective_function(h) for h in self.history]
                    self.ax.plot(history_x, history_y, history_z, 'o-', color='purple',
                                linewidth=2, markersize=6, alpha=0.6, label='Putanja')

                if self.current_solution:
                    z_current = self.objective_function(self.current_solution)
                    self.ax.scatter([self.current_solution[0]], [self.current_solution[1]], [z_current],
                                  c='red', s=200, marker='o',
                                  edgecolors='darkred', linewidths=2.5,
                                  label='Trenutna tačka', zorder=9)

                if self.current_neighbors and self.search_algorithm != 'Simulirano hlađenje':
                    neighbors_x = [n[0] for n in self.current_neighbors]
                    neighbors_y = [n[1] for n in self.current_neighbors]
                    neighbors_z = [self.objective_function(n) for n in self.current_neighbors]
                    self.ax.scatter(neighbors_x, neighbors_y, neighbors_z,
                                  c='orange', s=100, marker='s',
                                  edgecolors='darkorange', linewidths=1.5,
                                  label='Okolina (8 tačaka)', zorder=7, alpha=0.7)

                if self.search_algorithm == 'Tabu pretraživanje' and self.tabu_list:
                    tabu_x = [t[0] for t in self.tabu_list]
                    tabu_y = [t[1] for t in self.tabu_list]
                    tabu_z = [self.objective_function(t) for t in self.tabu_list]
                    self.ax.scatter(tabu_x, tabu_y, tabu_z,
                                  c='red', s=80, marker='x',
                                  linewidths=2, label='Tabu lista', zorder=8)

                if self.best_neighbor and self.search_algorithm != 'Simulirano hlađenje':
                    z_best = self.objective_function(self.best_neighbor)
                    self.ax.scatter([self.best_neighbor[0]], [self.best_neighbor[1]], [z_best],
                                  c='lime', s=150, marker='D',
                                  edgecolors='darkgreen', linewidths=2,
                                  label='Najbolji susjed', zorder=8)

                if self.best_found_solution:
                    z_best_found = self.objective_function(self.best_found_solution)
                    self.ax.scatter([self.best_found_solution[0]], [self.best_found_solution[1]], [z_best_found],
                                  c='gold', s=300, marker='*',
                                  edgecolors='darkorange', linewidths=3,
                                  label='★ Najbolja otkrivena', zorder=11)
            else:
                # 2D prikaz
                if len(self.history) > 1:
                    history_x = [h[0] for h in self.history]
                    history_y = [h[1] for h in self.history]
                    self.ax.plot(history_x, history_y, 'o-', color='purple',
                                linewidth=2, markersize=6, alpha=0.6, label='Putanja')

                if self.current_solution:
                    self.ax.scatter([self.current_solution[0]], [self.current_solution[1]],
                                  c='red', s=200, marker='o',
                                  edgecolors='darkred', linewidths=2.5,
                                  label='Trenutna tačka', zorder=9)

                    if self.search_algorithm != 'Simulirano hlađenje':
                        rect_size = self.delta * 2
                        rect = patches.Rectangle((self.current_solution[0] - self.delta,
                                                 self.current_solution[1] - self.delta),
                                                rect_size, rect_size,
                                                linewidth=2, edgecolor='orange',
                                                facecolor='orange', alpha=0.1, linestyle='--')
                        self.ax.add_patch(rect)

                if self.current_neighbors and self.search_algorithm != 'Simulirano hlađenje':
                    neighbors_x = [n[0] for n in self.current_neighbors]
                    neighbors_y = [n[1] for n in self.current_neighbors]
                    self.ax.scatter(neighbors_x, neighbors_y,
                                  c='orange', s=100, marker='s',
                                  edgecolors='darkorange', linewidths=1.5,
                                  label='Okolina (8 tačaka)', zorder=7, alpha=0.7)

                if self.search_algorithm == 'Tabu pretraživanje' and self.tabu_list:
                    tabu_x = [t[0] for t in self.tabu_list]
                    tabu_y = [t[1] for t in self.tabu_list]
                    self.ax.scatter(tabu_x, tabu_y,
                                  c='red', s=80, marker='x',
                                  linewidths=2, label='Tabu lista', zorder=8)

                if self.best_neighbor and self.search_algorithm != 'Simulirano hlađenje':
                    self.ax.scatter([self.best_neighbor[0]], [self.best_neighbor[1]],
                                  c='lime', s=150, marker='D',
                                  edgecolors='darkgreen', linewidths=2,
                                  label='Najbolji susjed', zorder=8)

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

        if self.search_algorithm == 'Genetički algoritam':
            if not self.population:
                info += "Kliknite 'Inicijaliziraj populaciju'\nza početak.\n"
            else:
                info += f"Generacija: {self.iteration}\n\n"
                info += f"Veličina populacije: {len(self.population)}\n"
                info += f"Najbolji fitness: {self.population[0]['fitness']:.6f}\n"
                avg_fitness = np.mean([ind['fitness'] for ind in self.population])
                info += f"Prosječan fitness: {avg_fitness:.6f}\n\n"

                if self.best_found_solution:
                    info += f"★ NAJBOLJA OTKRIVENA:\n"
                    info += f"  x = {self.best_found_solution}\n"
                    info += f"  f(x) = {self.best_found_value:.6f}\n\n"

                if self.finished:
                    info += "✓ DOSTIGNUT MAKSIMALAN\n  BROJ GENERACIJA!\n"
                else:
                    info += "Evolucija u toku...\n"
        elif self.current_solution is None:
            info += "Kliknite na grafik ili odaberite\n'Slučajan start' da počnete.\n"
        else:
            x = self.current_solution
            f_x = self.objective_function(x)

            info += f"Iteracija: {self.iteration}\n\n"
            info += f"Trenutna tačka:\n"
            info += f"  x = [{x[0]:.4f}, {x[1]:.4f}]\n"
            info += f"  f(x) = {f_x:.6f}\n\n"

            if self.best_found_solution:
                info += f"★ NAJBOLJA OTKRIVENA:\n"
                info += f"  x = [{self.best_found_solution[0]:.4f}, {self.best_found_solution[1]:.4f}]\n"
                info += f"  f(x) = {self.best_found_value:.6f}\n\n"

            dist = np.sqrt((x[0] - self.global_min[0])**2 +
                          (x[1] - self.global_min[1])**2)
            info += f"Udaljenost od globalnog: {dist:.4f}\n\n"

            if self.search_algorithm == 'Simulirano hlađenje':
                info += f"Temperatura: T = {self.current_temp:.4f}\n"
                info += f"Funkcija hlađenja: {self.cooling_schedule}\n"
                info += f"Iteracija na T: {self.temp_iteration}/{self.iterations_per_temp}\n\n"

            if self.search_algorithm == 'Tabu pretraživanje':
                info += f"Tabu lista ({len(self.tabu_list)}/{self.tabu_tenure}):\n"
                if self.tabu_list:
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

        self.info_text.insert('1.0', info)

    # Event handleri za GA parametre
    def on_pop_size_changed(self, val):
        self.population_size = int(float(val))
        self.pop_size_label.config(text=f"Pop = {self.population_size}")

    def on_crossover_type_changed(self, event=None):
        self.crossover_type = self.crossover_type_var.get()

    def on_crossover_rate_changed(self, val):
        self.crossover_rate = float(val)
        self.crossover_rate_label.config(text=f"pₓ = {self.crossover_rate:.2f}")

    def on_mutation_rate_changed(self, val):
        self.mutation_rate = float(val)
        self.mutation_rate_label.config(text=f"pₘ = {self.mutation_rate:.3f}")

    def on_selection_method_changed(self, event=None):
        self.selection_method = self.selection_method_var.get()
        self.update_selection_params_ui()

    def on_tournament_size_changed(self, val):
        self.tournament_size = int(float(val))
        self.tournament_size_label.config(text=f"Turnir = {self.tournament_size}")

    def on_selection_pressure_changed(self, val):
        self.selection_pressure = float(val)
        self.selection_pressure_label.config(text=f"SP = {self.selection_pressure:.2f}")

    def on_elitism_changed(self, val):
        self.elitism = int(float(val))
        self.elitism_label.config(text=f"Elite = {self.elitism}")

    def on_show_fitness_plot_changed(self):
        self.show_fitness_plot = self.show_fitness_plot_var.get()

    # Ostali event handleri
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
        self.population = []
        self.best_fitness_history = []
        self.avg_fitness_history = []

        # Ažuriraj stanje dugmadi
        self.update_button_states()

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
        self.population = []
        self.best_fitness_history = []
        self.avg_fitness_history = []

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
        if self.search_algorithm == 'Genetički algoritam':
            return  # Onemogući klik za GA

        if self.toolbar.mode != '':
            return

        if event.inaxes == self.ax:
            x0 = [event.xdata, event.ydata]

            x0[0] = np.clip(x0[0], self.x_range[0], self.x_range[1])
            x0[1] = np.clip(x0[1], self.x_range[0], self.x_range[1])

            self.current_solution = x0
            self.history = [x0]
            self.iteration = 0
            self.finished = False
            self.current_neighbors = []
            self.best_neighbor = None
            self.tabu_list = []

            self.current_temp = self.initial_temp
            self.temp_iteration = 0
            self.temp_history = [self.initial_temp]

            f_x0 = self.objective_function(x0)
            self.best_found_solution = x0.copy()
            self.best_found_value = f_x0

            print(f"\nPostavljena početna tačka: x = [{x0[0]:.3f}, {x0[1]:.3f}], " +
                  f"f(x) = {f_x0:.3f}")

            self.draw_objective_function()
            self.update_plot()
            self.update_info_text()

    def on_new_start(self):
        """Postavi novu slučajnu početnu tačku ili inicijalizuj populaciju"""
        if self.search_algorithm == 'Genetički algoritam':
            # Inicijalizuj populaciju
            self.initialize_population_ga()
            self.iteration = 0
            self.finished = False

            print(f"\nInicijalizirana populacija od {self.population_size} jedinki")
            print(f"Najbolja jedinka: x = {self.population[0]['genes']}, f(x) = {self.population[0]['fitness']:.6f}")

            self.draw_objective_function()
            self.update_plot()
            self.update_info_text()
        else:
            # Slučajan start za ostale algoritme
            x0 = [np.random.uniform(self.x_range[0], self.x_range[1]),
                  np.random.uniform(self.x_range[0], self.x_range[1])]

            self.current_solution = x0
            self.history = [x0]
            self.iteration = 0
            self.finished = False
            self.current_neighbors = []
            self.best_neighbor = None
            self.tabu_list = []

            self.current_temp = self.initial_temp
            self.temp_iteration = 0
            self.temp_history = [self.initial_temp]

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
        if self.search_algorithm == 'Genetički algoritam':
            if not self.population:
                messagebox.showwarning("Upozorenje", "Prvo inicijalizirajte populaciju!")
                return

            if self.finished:
                messagebox.showinfo("Info", "GA je završen! Inicijalizirajte novu populaciju.")
                return

            self.ga_step()
        else:
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
        neighbors = generate_neighborhood(self.current_solution, self.delta)

        neighbors = [[np.clip(n[0], self.x_range[0], self.x_range[1]),
                      np.clip(n[1], self.x_range[0], self.x_range[1])] for n in neighbors]

        self.current_neighbors = neighbors

        neighbor_values = [self.objective_function(n) for n in neighbors]
        current_value = self.objective_function(self.current_solution)

        best_neighbor_idx = np.argmin(neighbor_values)
        self.best_neighbor = neighbors[best_neighbor_idx]
        best_value = neighbor_values[best_neighbor_idx]

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

        if best_value >= current_value:
            print("  → LOKALNI MINIMUM PRONAĐEN!")
            self.finished = True
        else:
            self.current_solution = self.best_neighbor
            self.history.append(self.best_neighbor)
            self.iteration += 1
            print(f"  → Pomak na bolju tačku")

    def tabu_search_step(self):
        """Jedan korak tabu pretraživanja"""
        neighbors = generate_neighborhood(self.current_solution, self.delta)

        neighbors = [[np.clip(n[0], self.x_range[0], self.x_range[1]),
                      np.clip(n[1], self.x_range[0], self.x_range[1])] for n in neighbors]

        allowed_neighbors = []
        for n in neighbors:
            is_tabu = False
            for tabu_point in self.tabu_list:
                if points_equal(n, tabu_point):
                    is_tabu = True
                    break
            if not is_tabu:
                allowed_neighbors.append(n)

        if not allowed_neighbors:
            print("  NAPOMENA: Svi susjedi su tabu! Koristim aspiration criterion.")
            allowed_neighbors = neighbors

        self.current_neighbors = neighbors

        neighbor_values = [self.objective_function(n) for n in allowed_neighbors]
        current_value = self.objective_function(self.current_solution)

        best_neighbor_idx = np.argmin(neighbor_values)
        self.best_neighbor = allowed_neighbors[best_neighbor_idx]
        best_value = neighbor_values[best_neighbor_idx]

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

        self.tabu_list.append(list(self.current_solution))

        if len(self.tabu_list) > self.tabu_tenure:
            removed = self.tabu_list.pop(0)
            print(f"  → Uklonjena iz tabu liste: [{removed[0]:.3f}, {removed[1]:.3f}]")

        self.current_solution = self.best_neighbor
        self.history.append(self.best_neighbor)
        self.iteration += 1

        if self.iteration >= self.max_iterations:
            print(f"  → Dostignut maksimalan broj iteracija ({self.max_iterations})!")
            self.finished = True
        else:
            print(f"  → Pomak na tačku (može biti i gora!)")

    def simulated_annealing_step(self):
        """Jedan korak simuliranog hlađenja"""
        current_value = self.objective_function(self.current_solution)

        neighbor = generate_random_neighbor(self.current_solution, self.delta)

        neighbor = [np.clip(neighbor[0], self.x_range[0], self.x_range[1]),
                   np.clip(neighbor[1], self.x_range[0], self.x_range[1])]

        neighbor_value = self.objective_function(neighbor)

        delta_f = neighbor_value - current_value

        accepted = False
        if delta_f < 0:
            accepted = True
            acceptance_prob = 1.0
        else:
            acceptance_prob = np.exp(-delta_f / self.current_temp) if self.current_temp > 0 else 0
            if np.random.random() < acceptance_prob:
                accepted = True

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

        if self.temp_iteration >= self.iterations_per_temp:
            old_temp = self.current_temp
            self.cool_temperature()
            self.temp_iteration = 0
            print(f"  → Hlađenje: T = {old_temp:.4f} → {self.current_temp:.4f}")

            if self.current_temp <= self.min_temp:
                print(f"  → MINIMALNA TEMPERATURA DOSTIGNUTA!")
                self.finished = True

    def show_fitness_evolution_plot(self):
        """Prikaži grafik evolucije fitnessa kroz generacije"""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))

        generations = list(range(len(self.best_fitness_history)))

        ax.plot(generations, self.best_fitness_history, 'b-', linewidth=2, label='Najbolji fitness')
        ax.plot(generations, self.avg_fitness_history, 'r--', linewidth=2, label='Srednji fitness')

        ax.set_xlabel('Generacija', fontsize=12, fontweight='bold')
        ax.set_ylabel('Fitness', fontsize=12, fontweight='bold')
        ax.set_title('Tok izvršavanja genetičkog algoritma', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=11)

        # Dodaj informacije o finalnom stanju
        final_best = self.best_fitness_history[-1]
        final_avg = self.avg_fitness_history[-1]
        info_text = f'Finalni najbolji: {final_best:.6f}\nFinalni srednji: {final_avg:.6f}'
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.show()

    def on_complete(self):
        """Izvrši kompletno pretraživanje"""
        if self.search_algorithm == 'Genetički algoritam':
            if not self.population:
                messagebox.showwarning("Upozorenje", "Prvo inicijalizirajte populaciju!")
                return
        else:
            if self.current_solution is None:
                messagebox.showwarning("Upozorenje", "Prvo odaberite početnu tačku!")
                return

        if self.finished:
            messagebox.showinfo("Info", "Pretraživanje je već završeno!")
            return

        self.stop_requested = False

        while not self.finished and not self.stop_requested:
            self.on_step()
            self.root.update()
            self.root.after(100)

        if self.best_found_solution is not None:
            if self.stop_requested:
                title = "PRETRAŽIVANJE ZAUSTAVLJENO!"
            else:
                title = "PRETRAŽIVANJE ZAVRŠENO!"

            result_message = (
                f"{title}\n\n"
                f"★ NAJBOLJA OTKRIVENA TAČKA ★\n\n"
                f"x = {self.best_found_solution}\n\n"
                f"f(x) = {self.best_found_value:.6f}\n\n"
                f"Ukupno iteracija: {self.iteration}"
            )
            messagebox.showinfo("Rezultat pretraživanja", result_message)

            # Prikaži grafik evolucije fitnessa ako je GA i ako je opcija uključena
            if (self.search_algorithm == 'Genetički algoritam' and
                self.show_fitness_plot and
                len(self.best_fitness_history) > 1 and
                not self.stop_requested):
                self.show_fitness_evolution_plot()

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
        self.population = []
        self.best_fitness_history = []
        self.avg_fitness_history = []

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
            if self.current_solution or self.population:
                self.update_plot()
            print("\nPrebačeno na 2D (contour) prikaz")

    def switch_to_3d(self):
        """Prebaci na 3D mesh prikaz"""
        if self.view_mode != '3D':
            self.view_mode = '3D'
            self.draw_objective_function()
            if self.current_solution or self.population:
                self.update_plot()
            print("\nPrebačeno na 3D (mesh) prikaz")

    def show_click_instruction(self):
        """Prikaži instrukcije za klik"""
        if self.search_algorithm == 'Genetički algoritam':
            messagebox.showinfo("Info",
                              "Za Genetički algoritam koristite\n'Inicijaliziraj populaciju'!")
        else:
            messagebox.showinfo("Instrukcija",
                              "Kliknite bilo gdje na grafiku da postavite početnu tačku!")

    def on_about(self):
        """Prikaži About dialog"""
        messagebox.showinfo("O aplikaciji",
                          "Genetički algoritam - Demo aplikacija\n\n" +
                          "Optimizacija resursa\n\n" +
                          "Red. prof. dr Samim Konjicija\n\n" +
                          "Novembar 2025. godine")

    def on_help(self):
        """Prikaži Help dialog"""
        help_text = """UPUTE ZA KORIŠTENJE

ALGORITMI:
• Lokalno pretraživanje
• Tabu pretraživanje
• Simulirano hlađenje
• Genetički algoritam (GA)

GA PARAMETRI:
• Veličina populacije - broj jedinki
• Operator ukrštanja - način kombinovanja
• Vjerovatnoća ukrštanja (pₓ)
• Vjerovatnoća mutacije (pₘ)
• Metoda selekcije:
  - Ruletski točak
  - Rangiranje (sa pritiskom selekcije)
  - Turnirska (sa veličinom turnira)
• Elitizam - broj najboljih jedinki

KONTROLE ZA GA:
• Inicijaliziraj populaciju - stvori
  početnu populaciju
• Jedan korak - izvrši jednu generaciju
• Do kraja - izvrši sve generacije

LEGENDA:
• Zelena zvijezda - globalni minimum
• Plavi krugovi - populacija (GA)
• Crveni krug - najbolja jedinka/tačka
• Zlatna zvijezda - najbolja otkrivena"""

        messagebox.showinfo("Pomoć", help_text)

# Pokreni demo
if __name__ == "__main__":
    print("="*80)
    print("Genetički algoritam (Genetic Algorithm) - Demo Aplikacija")
    print("="*80)
    print("\nFunkcionalnosti:")
    print("  ✓ Lokalno pretraživanje")
    print("  ✓ Tabu pretraživanje")
    print("  ✓ Simulirano hlađenje")
    print("  ✓ Genetički algoritam (GA)")
    print("    - Različiti operatori ukrštanja")
    print("    - Različite metode selekcije")
    print("    - Elitizam")
    print("  ✓ 5 test funkcija")
    print("  ✓ Tkinter GUI sa matplotlib canvas-om")
    print("="*80)

    root = tk.Tk()
    app = GeneticAlgorithmDemo(root)
    root.mainloop()
