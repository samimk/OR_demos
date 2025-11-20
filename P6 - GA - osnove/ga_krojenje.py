"""
Genetički algoritam za problem krojenja drvenih ploča
Optimizacija resursa - Red. prof. dr Samim Konjicija, 2025.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import random
import math
import copy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

# Definicija Tetris oblika (svaki oblik je lista koordinata relativnih od reference tačke)
SHAPES = {
    'I': [(0, 0), (0, 1), (0, 2), (0, 3)],  # Vertikalna linija
    'O': [(0, 0), (0, 1), (1, 0), (1, 1)],  # Kvadrat
    'T': [(0, 0), (0, 1), (0, 2), (1, 1)],  # T oblik
    'L': [(0, 0), (0, 1), (0, 2), (1, 0)],  # L oblik
    'J': [(0, 0), (0, 1), (0, 2), (1, 2)],  # Obrnuti L
    'S': [(0, 1), (0, 2), (1, 0), (1, 1)],  # S oblik
    'Z': [(0, 0), (0, 1), (1, 1), (1, 2)]   # Z oblik
}

SHAPE_COLORS = {
    'I': '#00f0f0',
    'O': '#f0f000',
    'T': '#a000f0',
    'L': '#f0a000',
    'J': '#0000f0',
    'S': '#00f000',
    'Z': '#f00000'
}

class Shape:
    """Klasa koja predstavlja jedan oblik za smještanje na ploču"""
    def __init__(self, shape_type, x=0, y=0, rotation=0):
        self.type = shape_type
        self.x = x
        self.y = y
        self.rotation = rotation  # 0, 90, 180, 270
        self.coords = SHAPES[shape_type]

    def get_rotated_coords(self):
        """Vraća koordinate oblika nakon rotacije"""
        coords = []
        for dx, dy in self.coords:
            # Rotacija koordinata
            if self.rotation == 0:
                coords.append((dx, dy))
            elif self.rotation == 90:
                coords.append((dy, -dx))
            elif self.rotation == 180:
                coords.append((-dx, -dy))
            elif self.rotation == 270:
                coords.append((-dy, dx))
        return coords

    def get_absolute_coords(self):
        """Vraća apsolutne koordinate oblika na ploči"""
        rotated = self.get_rotated_coords()
        return [(self.x + dx, self.y + dy) for dx, dy in rotated]

    def get_area(self):
        """Vraća površinu oblika (broj jediničnih kvadrata)"""
        return len(self.coords)

    def get_bounds(self):
        """Vraća granice oblika (min_x, max_x, min_y, max_y)"""
        abs_coords = self.get_absolute_coords()
        xs = [x for x, y in abs_coords]
        ys = [y for x, y in abs_coords]
        return min(xs), max(xs), min(ys), max(ys)


class BoardSolution:
    """Klasa koja predstavlja jedno rješenje (raspored oblika na ploči)"""
    def __init__(self, board_width, board_height, shapes_list, placement_mode='free'):
        self.board_width = board_width
        self.board_height = board_height
        self.shapes = shapes_list  # Lista Shape objekata
        self.fitness = 0
        self.placement_mode = placement_mode  # 'free' ili 'gravity'

    def is_valid_placement(self, shape):
        """Provjerava da li je smještaj oblika validan"""
        abs_coords = shape.get_absolute_coords()

        # Provjera granica ploče
        for x, y in abs_coords:
            if x < 0 or x >= self.board_width or y < 0 or y >= self.board_height:
                return False

        # Provjera preklapanja sa drugim oblicima
        occupied = set()
        for s in self.shapes:
            if s is shape:
                continue
            for coord in s.get_absolute_coords():
                occupied.add(coord)

        for coord in abs_coords:
            if coord in occupied:
                return False

        # Za gravity mod, provjeri da li je oblik na dnu ili naslonjen na drugi oblik
        if self.placement_mode == 'gravity':
            if not self.is_supported(shape, occupied):
                return False

        return True

    def is_supported(self, shape, occupied):
        """Provjerava da li je oblik podržan (na dnu ili na drugom obliku)"""
        abs_coords = shape.get_absolute_coords()

        for x, y in abs_coords:
            # Ako je na dnu ploče, podržan je
            if y == 0:
                return True

            # Provjeri da li je direktno iznad nekog drugog oblika
            if (x, y - 1) in occupied:
                return True

        # Ako nijedan dio oblika nije na dnu niti na drugom obliku
        return False

    def calculate_fitness(self):
        """Računa fitness na osnovu načina popunjavanja"""
        if self.placement_mode == 'free':
            return self.calculate_fitness_free()
        else:  # gravity
            return self.calculate_fitness_gravity()

    def calculate_fitness_free(self):
        """Računa fitness kao ukupnu površinu validno smještenih oblika (slobodno popunjavanje)"""
        total_area = 0
        for shape in self.shapes:
            if self.is_valid_placement(shape):
                total_area += shape.get_area()
        self.fitness = total_area
        return self.fitness

    def calculate_fitness_gravity(self):
        """Računa fitness sa kaznom za praznine u popunjenim redovima (gravitacijski mod)"""
        # Prvo odredi koji su oblici validno smješteni
        valid_shapes = []
        occupied = set()

        for shape in self.shapes:
            if self.is_valid_placement(shape):
                valid_shapes.append(shape)
                for coord in shape.get_absolute_coords():
                    occupied.add(coord)

        # Površina validno smještenih oblika
        total_area = sum(shape.get_area() for shape in valid_shapes)

        # Kazna za praznine u popunjenim redovima
        penalty = 0

        # Pronađi najviši red koji ima barem jedno popunjeno polje
        max_filled_row = -1
        for x, y in occupied:
            if y > max_filled_row:
                max_filled_row = y

        # Za svaki red od dna do najvišeg popunjenog reda
        for row in range(max_filled_row + 1):
            filled_in_row = 0
            for col in range(self.board_width):
                if (col, row) in occupied:
                    filled_in_row += 1

            # Broj praznina u ovom redu
            gaps = self.board_width - filled_in_row
            penalty += gaps

        # Fitness = površina - kazna za praznine
        self.fitness = total_area - penalty * 0.5  # Faktor 0.5 da kazna ne bude prevelika
        return self.fitness

    def get_valid_shapes(self):
        """Vraća listu validno smještenih oblika"""
        valid = []
        for shape in self.shapes:
            if self.is_valid_placement(shape):
                valid.append(shape)
        return valid


class GACuttingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GA - Krojenje drvenih ploča")
        self.root.geometry("1400x900")

        # Parametri ploče
        self.board_width = 10
        self.board_height = 10
        self.placement_mode = 'free'  # 'free' ili 'gravity'

        # Parametri oblika
        self.shape_counts = {shape: 0 for shape in SHAPES.keys()}

        # GA parametri
        self.population_size = 50
        self.max_generations = 100
        self.crossover_rate = 0.8
        self.mutation_rate = 0.1
        self.selection_method = 'Ruletski točak'
        self.crossover_type = 'Jedna tačka'
        self.elitism = 2
        self.tournament_size = 3

        # Stanje
        self.population = []
        self.generation = 0
        self.best_solution = None
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.running = False
        self.paused = False

        self.setup_ui()

    def setup_ui(self):
        """Postavljanje korisničkog interfejsa"""
        # Meni bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # Help meni
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Uputstvo", command=self.show_instructions)
        help_menu.add_separator()
        help_menu.add_command(label="About", command=self.show_about)

        # Glavni kontejner
        main_container = ttk.Frame(self.root, padding=10)
        main_container.pack(fill=tk.BOTH, expand=True)

        # Lijeva strana - Kontrole
        left_frame = ttk.Frame(main_container)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # Desna strana - Vizualizacija
        right_frame = ttk.Frame(main_container)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # === LIJEVA STRANA ===

        # Notebook za tabove
        notebook = ttk.Notebook(left_frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Postavke ploče
        board_tab = ttk.Frame(notebook, padding=10)
        notebook.add(board_tab, text="Ploča")

        ttk.Label(board_tab, text="Dimenzije ploče:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(0, 5))

        ttk.Label(board_tab, text="Širina:").pack(anchor=tk.W)
        self.width_var = tk.IntVar(value=10)
        tk.Spinbox(board_tab, from_=5, to=20, textvariable=self.width_var, width=10).pack(anchor=tk.W, pady=(0, 10))

        ttk.Label(board_tab, text="Visina:").pack(anchor=tk.W)
        self.height_var = tk.IntVar(value=10)
        tk.Spinbox(board_tab, from_=5, to=20, textvariable=self.height_var, width=10).pack(anchor=tk.W, pady=(0, 10))

        ttk.Separator(board_tab, orient='horizontal').pack(fill=tk.X, pady=10)

        ttk.Label(board_tab, text="Način popunjavanja:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(0, 5))

        self.placement_var = tk.StringVar(value='free')

        placement_frame = ttk.Frame(board_tab)
        placement_frame.pack(anchor=tk.W, pady=5)

        ttk.Radiobutton(placement_frame, text="Slobodno", variable=self.placement_var,
                       value='free').pack(anchor=tk.W)
        ttk.Radiobutton(placement_frame, text="Gravitacijski (od dna)", variable=self.placement_var,
                       value='gravity').pack(anchor=tk.W)

        ttk.Label(board_tab, text="", font=('Arial', 8)).pack(anchor=tk.W, pady=(5, 0))
        ttk.Label(board_tab, text="Slobodno: oblici se mogu postaviti bilo gdje",
                 font=('Arial', 8), foreground='gray').pack(anchor=tk.W)
        ttk.Label(board_tab, text="Gravitacijski: oblici moraju biti na dnu",
                 font=('Arial', 8), foreground='gray').pack(anchor=tk.W)
        ttk.Label(board_tab, text="ili naslonjeni na druge oblike + kazna",
                 font=('Arial', 8), foreground='gray').pack(anchor=tk.W)
        ttk.Label(board_tab, text="za praznine u popunjenim redovima",
                 font=('Arial', 8), foreground='gray').pack(anchor=tk.W)

        # Tab 2: Oblici
        shapes_tab = ttk.Frame(notebook, padding=10)
        notebook.add(shapes_tab, text="Oblici")

        ttk.Label(shapes_tab, text="Broj komada po obliku:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(0, 10))

        self.shape_vars = {}
        for shape_name in SHAPES.keys():
            frame = ttk.Frame(shapes_tab)
            frame.pack(fill=tk.X, pady=5)

            # Boja indikatora
            color_label = tk.Label(frame, bg=SHAPE_COLORS[shape_name], width=2)
            color_label.pack(side=tk.LEFT, padx=(0, 5))

            ttk.Label(frame, text=f"{shape_name}:", width=3).pack(side=tk.LEFT)

            var = tk.IntVar(value=2)
            self.shape_vars[shape_name] = var
            tk.Spinbox(frame, from_=0, to=20, textvariable=var, width=8).pack(side=tk.LEFT)

        # Tab 3: GA parametri
        ga_tab = ttk.Frame(notebook, padding=10)
        notebook.add(ga_tab, text="GA parametri")

        ttk.Label(ga_tab, text="Veličina populacije:").pack(anchor=tk.W)
        self.pop_size_var = tk.IntVar(value=50)
        tk.Spinbox(ga_tab, from_=10, to=200, textvariable=self.pop_size_var, width=15).pack(anchor=tk.W, pady=(0, 10))

        ttk.Label(ga_tab, text="Maks. broj generacija:").pack(anchor=tk.W)
        self.max_gen_var = tk.IntVar(value=100)
        tk.Spinbox(ga_tab, from_=10, to=500, textvariable=self.max_gen_var, width=15).pack(anchor=tk.W, pady=(0, 10))

        ttk.Label(ga_tab, text="Metoda selekcije:").pack(anchor=tk.W)
        self.selection_var = tk.StringVar(value='Ruletski točak')
        ttk.Combobox(ga_tab, textvariable=self.selection_var,
                    values=['Ruletski točak', 'Turnirska', 'Ranking'],
                    state='readonly', width=15).pack(anchor=tk.W, pady=(0, 10))

        ttk.Label(ga_tab, text="Tip ukrštanja:").pack(anchor=tk.W)
        self.crossover_var = tk.StringVar(value='Jedna tačka')
        ttk.Combobox(ga_tab, textvariable=self.crossover_var,
                    values=['Jedna tačka', 'Dvije tačke', 'Uniformno'],
                    state='readonly', width=15).pack(anchor=tk.W, pady=(0, 10))

        ttk.Label(ga_tab, text="Vjerovatnoća ukrštanja:").pack(anchor=tk.W)
        self.crossover_rate_var = tk.DoubleVar(value=0.8)
        ttk.Scale(ga_tab, from_=0.0, to=1.0, variable=self.crossover_rate_var,
                 orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(0, 5))
        self.crossover_label = ttk.Label(ga_tab, text="0.80")
        self.crossover_label.pack(anchor=tk.W, pady=(0, 10))
        self.crossover_rate_var.trace('w', self.update_crossover_label)

        ttk.Label(ga_tab, text="Vjerovatnoća mutacije:").pack(anchor=tk.W)
        self.mutation_rate_var = tk.DoubleVar(value=0.1)
        ttk.Scale(ga_tab, from_=0.0, to=0.5, variable=self.mutation_rate_var,
                 orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(0, 5))
        self.mutation_label = ttk.Label(ga_tab, text="0.10")
        self.mutation_label.pack(anchor=tk.W, pady=(0, 10))
        self.mutation_rate_var.trace('w', self.update_mutation_label)

        ttk.Label(ga_tab, text="Elitizam:").pack(anchor=tk.W)
        self.elitism_var = tk.IntVar(value=2)
        tk.Spinbox(ga_tab, from_=0, to=10, textvariable=self.elitism_var, width=15).pack(anchor=tk.W, pady=(0, 10))

        ttk.Label(ga_tab, text="Veličina turnira:").pack(anchor=tk.W)
        self.tournament_var = tk.IntVar(value=3)
        tk.Spinbox(ga_tab, from_=2, to=10, textvariable=self.tournament_var, width=15).pack(anchor=tk.W)

        # Kontrolna dugmad
        button_frame = ttk.Frame(left_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))

        self.btn_start = ttk.Button(button_frame, text=f"{chr(0x25B6)} Pokreni GA", command=self.start_ga)
        self.btn_start.pack(fill=tk.X, pady=2)

        self.btn_pause = ttk.Button(button_frame, text="Pauziraj", command=self.pause_ga, state='disabled')
        self.btn_pause.pack(fill=tk.X, pady=2)

        self.btn_step = ttk.Button(button_frame, text="Jedan korak", command=self.step_ga, state='disabled')
        self.btn_step.pack(fill=tk.X, pady=2)

        self.btn_reset = ttk.Button(button_frame, text=f"{chr(0x1F504)} Reset", command=self.reset_ga)
        self.btn_reset.pack(fill=tk.X, pady=2)

        # === DESNA STRANA ===

        # Gornji dio - Prikaz rješenja
        top_frame = ttk.LabelFrame(right_frame, text="Prikaz rješenja", padding=10)
        top_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Lijevi canvas - Najbolje rješenje
        left_canvas_frame = ttk.Frame(top_frame)
        left_canvas_frame.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(left_canvas_frame, text="Najbolje rješenje (globalno)", font=('Arial', 10, 'bold')).pack()
        self.board_canvas = tk.Canvas(left_canvas_frame, bg='white', width=400, height=400)
        self.board_canvas.pack()

        # Središnji info panel
        info_frame = ttk.Frame(top_frame)
        info_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        ttk.Label(info_frame, text="Statistika:", font=('Arial', 11, 'bold')).pack(anchor=tk.W, pady=(0, 10))

        ttk.Label(info_frame, text="Generacija:").pack(anchor=tk.W)
        self.gen_label = ttk.Label(info_frame, text="0", font=('Arial', 12, 'bold'))
        self.gen_label.pack(anchor=tk.W, pady=(0, 10))

        ttk.Label(info_frame, text="Najbolji fitness:").pack(anchor=tk.W)
        self.best_fitness_label = ttk.Label(info_frame, text="0", font=('Arial', 12, 'bold'))
        self.best_fitness_label.pack(anchor=tk.W, pady=(0, 10))

        ttk.Label(info_frame, text="Srednji fitness:").pack(anchor=tk.W)
        self.avg_fitness_label = ttk.Label(info_frame, text="0.00", font=('Arial', 12, 'bold'))
        self.avg_fitness_label.pack(anchor=tk.W, pady=(0, 10))

        ttk.Label(info_frame, text="Validnih oblika:").pack(anchor=tk.W)
        self.valid_shapes_label = ttk.Label(info_frame, text="0", font=('Arial', 12, 'bold'))
        self.valid_shapes_label.pack(anchor=tk.W, pady=(0, 10))

        ttk.Label(info_frame, text="Iskorištenost:").pack(anchor=tk.W)
        self.utilization_label = ttk.Label(info_frame, text="0.0%", font=('Arial', 12, 'bold'))
        self.utilization_label.pack(anchor=tk.W)

        # Desni canvas - Odabrana jedinka iz populacije
        right_canvas_frame = ttk.Frame(top_frame)
        right_canvas_frame.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(right_canvas_frame, text="Odabrana jedinka", font=('Arial', 10, 'bold')).pack()

        # Combobox za izbor jedinke
        selector_frame = ttk.Frame(right_canvas_frame)
        selector_frame.pack(pady=5)

        ttk.Label(selector_frame, text="Jedinka #:").pack(side=tk.LEFT, padx=(0, 5))
        self.individual_var = tk.StringVar(value="1")
        self.individual_combo = ttk.Combobox(selector_frame, textvariable=self.individual_var,
                                            width=8, state='normal')
        self.individual_combo.pack(side=tk.LEFT)
        self.individual_combo.bind('<<ComboboxSelected>>', self.on_individual_selected)
        self.individual_combo.bind('<Return>', self.on_individual_selected)

        self.individual_canvas = tk.Canvas(right_canvas_frame, bg='white', width=400, height=400)
        self.individual_canvas.pack(pady=(5, 0))

        # Label za fitness odabrane jedinke
        self.individual_fitness_label = ttk.Label(right_canvas_frame, text="Fitness: -",
                                                 font=('Arial', 10))
        self.individual_fitness_label.pack(pady=5)

        # Donji dio - Grafik fitnessa
        bottom_frame = ttk.LabelFrame(right_frame, text="Evolucija fitnessa", padding=10)
        bottom_frame.pack(fill=tk.BOTH, expand=True)

        # Matplotlib grafik
        self.fig = Figure(figsize=(8, 3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel('Generacija')
        self.ax.set_ylabel('Fitness')
        self.ax.set_title('Najbolji i srednji fitness kroz generacije')
        self.ax.grid(True, alpha=0.3)

        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=bottom_frame)
        self.canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_crossover_label(self, *args):
        self.crossover_label.config(text=f"{self.crossover_rate_var.get():.2f}")

    def update_mutation_label(self, *args):
        self.mutation_label.config(text=f"{self.mutation_rate_var.get():.2f}")

    def create_random_solution(self):
        """Kreira nasumično rješenje"""
        shapes_list = []

        for shape_type, count in self.shape_counts.items():
            for _ in range(count):
                x = random.randint(0, self.board_width - 1)
                y = random.randint(0, self.board_height - 1)
                rotation = random.choice([0, 90, 180, 270])
                shapes_list.append(Shape(shape_type, x, y, rotation))

        solution = BoardSolution(self.board_width, self.board_height, shapes_list, self.placement_mode)
        solution.calculate_fitness()
        return solution

    def initialize_population(self):
        """Inicijalizuje populaciju"""
        self.population = []
        for _ in range(self.population_size):
            solution = self.create_random_solution()
            self.population.append(solution)

        # Sortiraj po fitnessu
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        self.best_solution = self.population[0]

    def selection(self):
        """Vrši selekciju roditelja"""
        if self.selection_method == 'Ruletski točak':
            return self.roulette_selection()
        elif self.selection_method == 'Turnirska':
            return self.tournament_selection()
        else:  # Ranking
            return self.ranking_selection()

    def roulette_selection(self):
        """Ruletska selekcija"""
        # Dodaj 1 da izbjegneš nula fitness
        fitnesses = [ind.fitness + 1 for ind in self.population]
        total_fitness = sum(fitnesses)

        if total_fitness == 0:
            return random.choice(self.population)

        pick = random.uniform(0, total_fitness)
        current = 0

        for ind in self.population:
            current += ind.fitness + 1
            if current >= pick:
                return ind

        return self.population[-1]

    def tournament_selection(self):
        """Turnirska selekcija"""
        tournament = random.sample(self.population, min(self.tournament_size, len(self.population)))
        return max(tournament, key=lambda x: x.fitness)

    def ranking_selection(self):
        """Ranking selekcija"""
        n = len(self.population)
        # Population je već sortirana, najbolji ima najviši rank
        ranks = list(range(1, n + 1))
        total_rank = sum(ranks)

        pick = random.uniform(0, total_rank)
        current = 0

        for i, ind in enumerate(self.population):
            current += ranks[i]
            if current >= pick:
                return ind

        return self.population[0]

    def crossover(self, parent1, parent2):
        """Ukrštanje dva roditelja"""
        if random.random() > self.crossover_rate:
            # Vrati duboke kopije roditelja
            return copy.deepcopy(parent1), copy.deepcopy(parent2)

        child1_shapes = []
        child2_shapes = []

        if self.crossover_type == 'Jedna tačka':
            point = random.randint(1, len(parent1.shapes) - 1)
            # Duboke kopije oblika
            child1_shapes = copy.deepcopy(parent1.shapes[:point]) + copy.deepcopy(parent2.shapes[point:])
            child2_shapes = copy.deepcopy(parent2.shapes[:point]) + copy.deepcopy(parent1.shapes[point:])

        elif self.crossover_type == 'Dvije tačke':
            points = sorted(random.sample(range(1, len(parent1.shapes)), 2))
            child1_shapes = (copy.deepcopy(parent1.shapes[:points[0]]) +
                           copy.deepcopy(parent2.shapes[points[0]:points[1]]) +
                           copy.deepcopy(parent1.shapes[points[1]:]))
            child2_shapes = (copy.deepcopy(parent2.shapes[:points[0]]) +
                           copy.deepcopy(parent1.shapes[points[0]:points[1]]) +
                           copy.deepcopy(parent2.shapes[points[1]:]))

        else:  # Uniformno
            for i in range(len(parent1.shapes)):
                if random.random() < 0.5:
                    child1_shapes.append(copy.deepcopy(parent1.shapes[i]))
                    child2_shapes.append(copy.deepcopy(parent2.shapes[i]))
                else:
                    child1_shapes.append(copy.deepcopy(parent2.shapes[i]))
                    child2_shapes.append(copy.deepcopy(parent1.shapes[i]))

        child1 = BoardSolution(self.board_width, self.board_height, child1_shapes, self.placement_mode)
        child2 = BoardSolution(self.board_width, self.board_height, child2_shapes, self.placement_mode)

        child1.calculate_fitness()
        child2.calculate_fitness()

        return child1, child2

    def mutate(self, solution):
        """Mutira rješenje"""
        if random.random() > self.mutation_rate:
            return solution

        # Izaberi nasumični oblik za mutaciju
        if len(solution.shapes) > 0:
            shape = random.choice(solution.shapes)

            # Nasumično mijenjaj poziciju ili rotaciju
            mutation_type = random.choice(['position', 'rotation', 'both'])

            if mutation_type in ['position', 'both']:
                shape.x = random.randint(0, self.board_width - 1)
                shape.y = random.randint(0, self.board_height - 1)

            if mutation_type in ['rotation', 'both']:
                shape.rotation = random.choice([0, 90, 180, 270])

        solution.calculate_fitness()
        return solution

    def evolve(self):
        """Vrši jednu generaciju evolucije"""
        new_population = []

        # Elitizam - zadrži najbolje (duboka kopija!)
        for i in range(self.elitism):
            if i < len(self.population):
                elite_copy = copy.deepcopy(self.population[i])
                new_population.append(elite_copy)

        # Kreiraj novu populaciju
        while len(new_population) < self.population_size:
            parent1 = self.selection()
            parent2 = self.selection()

            child1, child2 = self.crossover(parent1, parent2)

            child1 = self.mutate(child1)
            child2 = self.mutate(child2)

            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)

        self.population = new_population
        self.population.sort(key=lambda x: x.fitness, reverse=True)

        # Ažuriraj najbolje rješenje kroz sve generacije
        if self.population[0].fitness > self.best_solution.fitness:
            self.best_solution = copy.deepcopy(self.population[0])

        # Statistika - koristi globalno najbolje rješenje, ne trenutnu populaciju
        best_fitness = self.best_solution.fitness
        avg_fitness = sum(ind.fitness for ind in self.population) / len(self.population)

        self.best_fitness_history.append(best_fitness)
        self.avg_fitness_history.append(avg_fitness)

        self.generation += 1

    def draw_board(self):
        """Crta ploču sa oblicima"""
        self.board_canvas.delete('all')

        if self.best_solution is None:
            return

        # Dimenzije canvas-a
        canvas_width = self.board_canvas.winfo_width()
        canvas_height = self.board_canvas.winfo_height()

        if canvas_width <= 1:
            canvas_width = 500
        if canvas_height <= 1:
            canvas_height = 500

        # Veličina ćelije
        cell_size = min(canvas_width // self.board_width, canvas_height // self.board_height)

        # Crtaj grid
        for i in range(self.board_width + 1):
            x = i * cell_size
            self.board_canvas.create_line(x, 0, x, self.board_height * cell_size, fill='#ddd')

        for i in range(self.board_height + 1):
            y = i * cell_size
            self.board_canvas.create_line(0, y, self.board_width * cell_size, y, fill='#ddd')

        # Crtaj validne oblike
        valid_shapes = self.best_solution.get_valid_shapes()

        for shape in valid_shapes:
            color = SHAPE_COLORS[shape.type]
            abs_coords = shape.get_absolute_coords()

            for x, y in abs_coords:
                x1 = x * cell_size
                y1 = y * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size

                self.board_canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline='black', width=2)

        # Ažuriraj statistiku
        self.gen_label.config(text=str(self.generation))
        self.best_fitness_label.config(text=str(self.best_solution.fitness))

        if len(self.avg_fitness_history) > 0:
            self.avg_fitness_label.config(text=f"{self.avg_fitness_history[-1]:.2f}")

        self.valid_shapes_label.config(text=str(len(valid_shapes)))

        max_area = self.board_width * self.board_height
        utilization = (self.best_solution.fitness / max_area) * 100 if max_area > 0 else 0
        self.utilization_label.config(text=f"{utilization:.1f}%")

    def draw_individual(self, solution, canvas):
        """Crta odabranu jedinku na zadati canvas"""
        canvas.delete('all')

        if solution is None:
            return

        # Dimenzije canvas-a
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()

        if canvas_width <= 1:
            canvas_width = 400
        if canvas_height <= 1:
            canvas_height = 400

        # Veličina ćelije
        cell_size = min(canvas_width // self.board_width, canvas_height // self.board_height)

        # Crtaj grid
        for i in range(self.board_width + 1):
            x = i * cell_size
            canvas.create_line(x, 0, x, self.board_height * cell_size, fill='#ddd')

        for i in range(self.board_height + 1):
            y = i * cell_size
            canvas.create_line(0, y, self.board_width * cell_size, y, fill='#ddd')

        # Crtaj validne oblike
        valid_shapes = solution.get_valid_shapes()

        for shape in valid_shapes:
            color = SHAPE_COLORS[shape.type]
            abs_coords = shape.get_absolute_coords()

            for x, y in abs_coords:
                x1 = x * cell_size
                y1 = y * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size

                canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline='black', width=2)

    def on_individual_selected(self, event=None):
        """Event handler kada se odabere jedinka iz comboboxa"""
        if not self.population:
            messagebox.showwarning("Upozorenje", "Populacija nije inicijalizirana!")
            return

        try:
            index = int(self.individual_var.get()) - 1  # Convert to 0-based index
            if 0 <= index < len(self.population):
                solution = self.population[index]
                self.draw_individual(solution, self.individual_canvas)
                self.individual_fitness_label.config(text=f"Fitness: {solution.fitness:.2f}")
            else:
                messagebox.showwarning("Upozorenje",
                                      f"Jedinka #{index+1} ne postoji u populaciji!\n"
                                      f"Dostupne jedinke: 1-{len(self.population)}")
        except ValueError:
            messagebox.showwarning("Upozorenje", "Unesite validan broj jedinke!")

    def update_individual_combo(self):
        """Ažurira listu jedinki u comboboxu"""
        if self.population:
            values = [str(i+1) for i in range(len(self.population))]
            self.individual_combo['values'] = values

            # Ako je trenutna vrijednost validna, osvježi prikaz
            try:
                current = int(self.individual_var.get())
                if 1 <= current <= len(self.population):
                    self.on_individual_selected()
            except ValueError:
                pass

    def update_plot(self):
        """Ažurira grafik fitnessa"""
        self.ax.clear()

        if len(self.best_fitness_history) > 0:
            generations = list(range(len(self.best_fitness_history)))
            self.ax.plot(generations, self.best_fitness_history, 'b-', linewidth=2, label='Najbolji')
            self.ax.plot(generations, self.avg_fitness_history, 'r--', linewidth=1, label='Srednji')

            self.ax.set_xlabel('Generacija')
            self.ax.set_ylabel('Fitness')
            self.ax.set_title('Najbolji i srednji fitness kroz generacije')
            self.ax.grid(True, alpha=0.3)
            self.ax.legend()

        self.canvas_plot.draw()

    def start_ga(self):
        """Pokreće GA"""
        if not self.running:
            # Učitaj parametre
            self.board_width = self.width_var.get()
            self.board_height = self.height_var.get()
            self.placement_mode = self.placement_var.get()
            self.population_size = self.pop_size_var.get()
            self.max_generations = self.max_gen_var.get()
            self.crossover_rate = self.crossover_rate_var.get()
            self.mutation_rate = self.mutation_rate_var.get()
            self.selection_method = self.selection_var.get()
            self.crossover_type = self.crossover_var.get()
            self.elitism = self.elitism_var.get()
            self.tournament_size = self.tournament_var.get()

            # Učitaj oblike
            self.shape_counts = {shape: var.get() for shape, var in self.shape_vars.items()}

            if sum(self.shape_counts.values()) == 0:
                messagebox.showwarning("Upozorenje", "Morate dodati barem jedan oblik!")
                return

            # Inicijalizuj
            self.generation = 0
            self.best_fitness_history = []
            self.avg_fitness_history = []
            self.initialize_population()

            self.running = True
            self.paused = False

            # Ažuriraj dugmad
            self.btn_start.config(state='disabled')
            self.btn_pause.config(state='normal')
            self.btn_step.config(state='disabled')

            self.draw_board()
            self.update_plot()
            self.update_individual_combo()
            self.run_generation()

    def run_generation(self):
        """Izvršava generacije u petlji"""
        if not self.running or self.paused:
            return

        if self.generation < self.max_generations:
            self.evolve()
            self.draw_board()
            self.update_plot()
            self.update_individual_combo()
            self.root.after(50, self.run_generation)
        else:
            self.running = False
            self.btn_start.config(state='normal')
            self.btn_pause.config(state='disabled')
            self.btn_step.config(state='disabled')
            messagebox.showinfo("Završeno", f"GA je završio nakon {self.generation} generacija.\n"
                                           f"Najbolji fitness: {self.best_solution.fitness}")

    def pause_ga(self):
        """Pauzira/nastavlja GA"""
        if self.running:
            if self.paused:
                self.paused = False
                self.btn_pause.config(text="Pauziraj")
                self.btn_step.config(state='disabled')
                self.run_generation()
            else:
                self.paused = True
                self.btn_pause.config(text="Nastavi")
                self.btn_step.config(state='normal')

    def step_ga(self):
        """Izvršava jedan korak GA"""
        if self.running and self.paused and self.generation < self.max_generations:
            self.evolve()
            self.draw_board()
            self.update_plot()
            self.update_individual_combo()

    def reset_ga(self):
        """Resetuje GA"""
        self.running = False
        self.paused = False
        self.generation = 0
        self.population = []
        self.best_solution = None
        self.best_fitness_history = []
        self.avg_fitness_history = []

        self.btn_start.config(state='normal')
        self.btn_pause.config(state='disabled', text="Pauziraj")
        self.btn_step.config(state='disabled')

        self.board_canvas.delete('all')
        self.individual_canvas.delete('all')
        self.individual_combo['values'] = []
        self.individual_var.set("1")
        self.individual_fitness_label.config(text="Fitness: -")

        self.gen_label.config(text="0")
        self.best_fitness_label.config(text="0")
        self.avg_fitness_label.config(text="0.00")
        self.valid_shapes_label.config(text="0")
        self.utilization_label.config(text="0.0%")

        self.ax.clear()
        self.ax.set_xlabel('Generacija')
        self.ax.set_ylabel('Fitness')
        self.ax.set_title('Najbolji i srednji fitness kroz generacije')
        self.ax.grid(True, alpha=0.3)
        self.canvas_plot.draw()

    def show_about(self):
        """Prikazuje About dijalog"""
        messagebox.showinfo(
            "O programu",
            "GA - Krojenje drvenih ploča\n\n"
            "Demonstracija primjene genetičkog algoritma\n"
            "za optimizaciju krojenja drvenih ploča\n"
            "sa Tetris oblicima.\n\n"
            "Optimizacija resursa\n"
            "Red. prof. dr Samim Konjicija\n"
            "Novembar 2025."
        )

    def show_instructions(self):
        """Prikazuje uputstvo za korištenje"""
        instructions = """UPUTSTVO ZA KORIŠTENJE

=== POSTAVKE PLOČE ===
• Dimenzije ploče: Odredite širinu i visinu ploče (5-20)
• Način popunjavanja:
  - Slobodno: Oblici se mogu postaviti bilo gdje
  - Gravitacijski: Oblici moraju biti na dnu ili naslonjeni na druge oblike

=== OBLICI ===
• Odaberite broj komada za svaki Tetris oblik (I, O, T, L, J, S, Z)
• Boje indikatora pokazuju boju svakog oblika na ploči

=== GA PARAMETRI ===
• Veličina populacije: Broj rješenja u populaciji (10-200)
• Maks. broj generacija: Broj generacija evolucije (10-500)
• Metoda selekcije: Ruletski točak, Turnirska ili Ranking
• Tip ukrštanja: Jedna tačka, Dvije tačke ili Uniformno
• Vjerovatnoća ukrštanja: Šansa da se izvrši ukrštanje (0.0-1.0)
• Vjerovatnoća mutacije: Šansa da se izvrši mutacija (0.0-0.5)
• Elitizam: Broj najboljih jedinki koji se čuvaju (0-10)
• Veličina turnira: Broj učesnika u turnirskoj selekciji (2-10)

=== KONTROLE ===
▶ Pokreni GA: Pokreće izvršavanje algoritma
⏸ Pauziraj: Pauzira/nastavlja izvršavanje
⏭ Jedan korak: Izvršava jednu generaciju (samo u pauzi)
🔄 Reset: Resetuje sve na početne vrijednosti

=== PRIKAZ RJEŠENJA ===
• Lijevi canvas: Prikazuje najbolje pronađeno rješenje (globalno)
• Desni canvas: Prikazuje odabranu jedinku iz trenutne populacije
• Combo box: Omogućava izbor jedinke iz populacije za prikaz
• Statistika: Prikazuje trenutne vrijednosti fitnessa i iskorištenosti
• Graf: Prati evoluciju najboljeg i srednjeg fitnessa kroz generacije

=== FITNESS ===
• Slobodno popunjavanje: Površina validno smještenih oblika
• Gravitacijski mod: Površina - kazna za praznine u popunjenim redovima

=== SAVJETI ===
• Počnite sa manjim brojem oblika za brže rezultate
• Gravitacijski mod daje kompaktnije rješenja ali je zahtjevniji
• Veći elitizam čuva najbolja rješenja ali smanjuje raznolikost
• Veća populacija daje bolje rezultate ali sporije izvršavanje
"""
        # Kreiraj prozor sa scroll barom
        instruction_window = tk.Toplevel(self.root)
        instruction_window.title("Uputstvo")
        instruction_window.geometry("700x600")

        # Text widget sa scrollbarom
        frame = ttk.Frame(instruction_window, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        text_widget = tk.Text(frame, wrap=tk.WORD, font=('Courier', 10))
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)

        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        text_widget.insert('1.0', instructions)
        text_widget.configure(state='disabled')

        # Dugme za zatvaranje
        btn_close = ttk.Button(instruction_window, text="Zatvori", command=instruction_window.destroy)
        btn_close.pack(pady=10)


def main():
    root = tk.Tk()
    app = GACuttingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
