"""
Lokalno pretraživanje (Local Search) - Demo aplikacija
Demonstrira osnovno lokalno pretraživanje iz prezentacije (slajdovi 18-24)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, RadioButtons
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches

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
    def __init__(self):
        # Parametri
        self.selected_function = 'Kvadratna'
        self.objective_function = FUNCTIONS[self.selected_function]['func']
        self.x_range = FUNCTIONS[self.selected_function]['range']
        self.global_min = FUNCTIONS[self.selected_function]['global_min']
        self.delta = 0.5  # Veličina koraka
        self.plot_type = 'contour'  # 'contour' ili '3d'

        # Stanje algoritma
        self.current_solution = None
        self.history = []  # Historija rješenja
        self.current_neighbors = []
        self.best_neighbor = None
        self.iteration = 0
        self.finished = False

        # Početne postavke za reset
        self.initial_state = {
            'function': self.selected_function,
            'delta': self.delta,
            'plot_type': self.plot_type
        }

        # Setup figura
        self.setup_figure()

    def setup_figure(self):
        """Postavi figuru"""
        self.fig = plt.figure(figsize=(16, 10))

        # Glavni plot
        if self.plot_type == 'contour':
            self.ax_main = plt.subplot2grid((4, 3), (0, 0), colspan=2, rowspan=3)
        else:
            self.ax_main = self.fig.add_subplot(2, 3, (1, 5), projection='3d')

        # Info panel
        self.ax_info = plt.subplot2grid((4, 3), (3, 0))
        self.ax_info.axis('off')

        # Kontrole
        self.ax_controls = plt.subplot2grid((4, 3), (3, 1))
        self.ax_controls.axis('off')

        # Radio buttons za izbor funkcije
        self.ax_radio_func = plt.subplot2grid((4, 3), (0, 2))

        # Radio buttons za tip prikaza
        self.ax_radio_plot = plt.subplot2grid((4, 3), (1, 2))

        # Slider kontrola
        self.ax_slider = plt.subplot2grid((4, 3), (2, 2))
        self.ax_slider.axis('off')

        # Dugmad
        self.ax_buttons = plt.subplot2grid((4, 3), (3, 2))
        self.ax_buttons.axis('off')

        # Nacrtaj početni plot
        self.draw_objective_function()

        # Elementi za crtanje
        self.scatter_current = None
        self.scatter_neighbors = None
        self.scatter_best_neighbor = None
        self.scatter_history = None
        self.neighborhood_patch = None
        self.surface = None

        # Dodaj kontrole
        self.add_controls()

        # Inicijalni info tekst
        self.update_info_text()

        # Event za klik mišem
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        plt.tight_layout()

    def draw_objective_function(self):
        """Nacrtaj objektive funkciju"""
        self.ax_main.clear()

        # Generiši mrežu tačaka
        x1 = np.linspace(self.x_range[0], self.x_range[1], 200)
        x2 = np.linspace(self.x_range[0], self.x_range[1], 200)
        X1, X2 = np.meshgrid(x1, x2)
        Z = np.zeros_like(X1)

        for i in range(X1.shape[0]):
            for j in range(X1.shape[1]):
                Z[i, j] = self.objective_function([X1[i, j], X2[i, j]])

        if self.plot_type == 'contour':
            # Konturni dijagram
            contour = self.ax_main.contour(X1, X2, Z, levels=20, cmap='viridis', alpha=0.6)
            self.ax_main.clabel(contour, inline=True, fontsize=8)

            # Označi globalni minimum
            self.ax_main.scatter([self.global_min[0]], [self.global_min[1]],
                                c='green', s=200, marker='*',
                                edgecolors='darkgreen', linewidths=2,
                                label=f'Globalni minimum {self.global_min}', zorder=10)

            self.ax_main.set_xlabel('x₁', fontsize=13, fontweight='bold')
            self.ax_main.set_ylabel('x₂', fontsize=13, fontweight='bold')
            self.ax_main.set_title(f'Lokalno pretraživanje - {self.selected_function}',
                                   fontsize=14, fontweight='bold')
            self.ax_main.grid(True, alpha=0.3)
            self.ax_main.legend(loc='upper right')
            self.ax_main.set_xlim(self.x_range)
            self.ax_main.set_ylim(self.x_range)
        else:
            # 3D mesh plot
            self.surface = self.ax_main.plot_surface(X1, X2, Z, cmap='viridis',
                                                     alpha=0.6, edgecolor='none')

            # Označi globalni minimum
            z_min = self.objective_function(list(self.global_min))
            self.ax_main.scatter([self.global_min[0]], [self.global_min[1]], [z_min],
                                c='green', s=200, marker='*',
                                edgecolors='darkgreen', linewidths=2, zorder=10)

            self.ax_main.set_xlabel('x₁', fontsize=11, fontweight='bold')
            self.ax_main.set_ylabel('x₂', fontsize=11, fontweight='bold')
            self.ax_main.set_zlabel('f(x)', fontsize=11, fontweight='bold')
            self.ax_main.set_title(f'Lokalno pretraživanje - {self.selected_function}',
                                   fontsize=14, fontweight='bold')

    def add_controls(self):
        """Dodaj kontrolne dugmad i widget-e"""

        # Radio buttons - izbor funkcije
        self.radio_func = RadioButtons(self.ax_radio_func,
                                       list(FUNCTIONS.keys()),
                                       active=list(FUNCTIONS.keys()).index(self.selected_function))
        self.radio_func.on_clicked(self.on_function_changed)
        self.ax_radio_func.set_title('Funkcija', fontweight='bold', fontsize=12)
        # Povećaj veličinu teksta i krugova za radio buttons
        for label in self.radio_func.labels:
            label.set_fontsize(11)
        # Kompatibilnost sa različitim verzijama matplotlib-a
        try:
            if hasattr(self.radio_func, 'circles'):
                for circle in self.radio_func.circles:
                    circle.set_radius(0.08)
            elif hasattr(self.radio_func, '_buttons'):
                # U novijim verzijama _buttons je PathCollection
                if hasattr(self.radio_func._buttons, 'set_sizes'):
                    # set_sizes prima površinu (pi * r^2), pa koristimo 200 za veći krug
                    num_buttons = len(self.radio_func.labels)
                    self.radio_func._buttons.set_sizes([200] * num_buttons)
                else:
                    # Starije verzije gde _buttons je lista
                    for circle in self.radio_func._buttons:
                        if hasattr(circle, 'set_radius'):
                            circle.set_radius(0.08)
        except (TypeError, AttributeError):
            # Ako ne uspe, nastavi bez promene veličine krugova
            pass

        # Radio buttons - tip prikaza
        self.radio_plot = RadioButtons(self.ax_radio_plot,
                                       ['contour', '3D mesh'],
                                       active=0 if self.plot_type == 'contour' else 1)
        self.radio_plot.on_clicked(self.on_plot_type_changed)
        self.ax_radio_plot.set_title('Tip prikaza', fontweight='bold', fontsize=12)
        # Povećaj veličinu teksta i krugova za radio buttons
        for label in self.radio_plot.labels:
            label.set_fontsize(11)
        # Kompatibilnost sa različitim verzijama matplotlib-a
        try:
            if hasattr(self.radio_plot, 'circles'):
                for circle in self.radio_plot.circles:
                    circle.set_radius(0.08)
            elif hasattr(self.radio_plot, '_buttons'):
                # U novijim verzijama _buttons je PathCollection
                if hasattr(self.radio_plot._buttons, 'set_sizes'):
                    # set_sizes prima površinu (pi * r^2), pa koristimo 200 za veći krug
                    num_buttons = len(self.radio_plot.labels)
                    self.radio_plot._buttons.set_sizes([200] * num_buttons)
                else:
                    # Starije verzije gde _buttons je lista
                    for circle in self.radio_plot._buttons:
                        if hasattr(circle, 'set_radius'):
                            circle.set_radius(0.08)
        except (TypeError, AttributeError):
            # Ako ne uspe, nastavi bez promene veličine krugova
            pass

        # Slider: Delta (veličina koraka)
        ax_slider_delta = plt.axes([0.74, 0.37, 0.15, 0.02])
        self.slider_delta = Slider(ax_slider_delta, 'Delta (Δx):',
                                   0.1, 2.0, valinit=0.5, valstep=0.1)
        self.slider_delta.on_changed(self.on_delta_changed)

        # Dugmad
        button_height = 0.04
        button_width = 0.12
        button_spacing = 0.05

        # Dugme: Klik za početnu tačku
        ax_button_click = plt.axes([0.68, 0.28, button_width, button_height])
        self.button_click = Button(ax_button_click, 'Klik za start',
                                   color='lightcoral', hovercolor='coral')
        self.button_click.on_clicked(lambda x: print("Kliknite na grafik da postavite početnu tačku"))

        # Dugme: Slučajna početna tačka
        ax_button_new = plt.axes([0.68, 0.23, button_width, button_height])
        self.button_new = Button(ax_button_new, 'Slučajan start',
                                color='lightgreen', hovercolor='green')
        self.button_new.on_clicked(self.on_new_start)

        # Dugme: Jedan korak
        ax_button_step = plt.axes([0.68, 0.18, button_width, button_height])
        self.button_step = Button(ax_button_step, 'Jedan korak',
                                  color='lightblue', hovercolor='blue')
        self.button_step.on_clicked(self.on_step)

        # Dugme: Kompletno pretraživanje
        ax_button_complete = plt.axes([0.68, 0.13, button_width, button_height])
        self.button_complete = Button(ax_button_complete, 'Do kraja',
                                      color='lightyellow', hovercolor='yellow')
        self.button_complete.on_clicked(self.on_complete)

        # Dugme: Reset
        ax_button_reset = plt.axes([0.68, 0.08, button_width, button_height])
        self.button_reset = Button(ax_button_reset, 'Reset',
                                   color='lightgray', hovercolor='gray')
        self.button_reset.on_clicked(self.on_reset)

        # Dugme: About
        ax_button_about = plt.axes([0.81, 0.28, button_width, button_height])
        self.button_about = Button(ax_button_about, 'About',
                                   color='lightcyan', hovercolor='cyan')
        self.button_about.on_clicked(self.on_about)

        # Dugme: Help
        ax_button_help = plt.axes([0.81, 0.23, button_width, button_height])
        self.button_help = Button(ax_button_help, 'Help',
                                  color='lightsteelblue', hovercolor='steelblue')
        self.button_help.on_clicked(self.on_help)

    def on_function_changed(self, label):
        """Promjena funkcije"""
        self.selected_function = label
        self.objective_function = FUNCTIONS[label]['func']
        self.x_range = FUNCTIONS[label]['range']
        self.global_min = FUNCTIONS[label]['global_min']

        # Resetuj stanje
        self.current_solution = None
        self.history = []
        self.current_neighbors = []
        self.best_neighbor = None
        self.iteration = 0
        self.finished = False

        # Ponovno crtanje
        self.redraw_plot()

    def on_plot_type_changed(self, label):
        """Promjena tipa prikaza"""
        self.plot_type = 'contour' if label == 'contour' else '3d'
        self.redraw_plot()

    def redraw_plot(self):
        """Ponovno crtaj cijelu figuru"""
        # Disconnect event handler prije zatvaranja figure
        if hasattr(self, 'cid') and self.cid is not None:
            self.fig.canvas.mpl_disconnect(self.cid)

        plt.close(self.fig)
        self.setup_figure()
        plt.show(block=False)

    def on_click(self, event):
        """Postavi početnu tačku klikom miša"""
        if event.inaxes == self.ax_main and self.plot_type == 'contour':
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

            print(f"\nPostavljena početna tačka klikom: x = [{x0[0]:.3f}, {x0[1]:.3f}], f(x) = {self.objective_function(x0):.3f}")

            self.update_plot()
            self.update_info_text()

    def on_new_start(self, event):
        """Postavi novu slučajnu početnu tačku"""
        # Slučajna početna tačka
        x0 = [np.random.uniform(self.x_range[0], self.x_range[1]),
              np.random.uniform(self.x_range[0], self.x_range[1])]

        self.current_solution = x0
        self.history = [x0]
        self.iteration = 0
        self.finished = False
        self.current_neighbors = []
        self.best_neighbor = None

        print(f"\nNova početna tačka: x = [{x0[0]:.3f}, {x0[1]:.3f}], f(x) = {self.objective_function(x0):.3f}")

        self.update_plot()
        self.update_info_text()

    def on_step(self, event):
        """Izvrši jedan korak lokalnog pretraživanja"""
        if self.current_solution is None:
            print("Prvo odaberite početnu tačku!")
            return

        if self.finished:
            print("Lokalno pretraživanje je završeno! Odaberite novu početnu tačku.")
            return

        # Generiši okolinu
        neighbors = generate_neighborhood(self.current_solution, self.delta)

        # Ograniči susjedne tačke na dozvoljeni prostor
        neighbors = [[np.clip(n[0], self.x_range[0], self.x_range[1]),
                      np.clip(n[1], self.x_range[0], self.x_range[1])] for n in neighbors]

        self.current_neighbors = neighbors

        # Evaluiraj susjedne tačke
        neighbor_values = [self.objective_function(n) for n in neighbors]
        current_value = self.objective_function(self.current_solution)

        # Nađi najbolju susjednu tačku (Steepest Descent)
        best_neighbor_idx = np.argmin(neighbor_values)
        self.best_neighbor = neighbors[best_neighbor_idx]
        best_value = neighbor_values[best_neighbor_idx]

        print(f"\nIteracija {self.iteration + 1}:")
        print(f"  Trenutno: x = [{self.current_solution[0]:.3f}, {self.current_solution[1]:.3f}], f(x) = {current_value:.3f}")
        print(f"  Najbolji susjed: x = [{self.best_neighbor[0]:.3f}, {self.best_neighbor[1]:.3f}], f(x) = {best_value:.3f}")

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

        self.update_plot()
        self.update_info_text()

    def on_complete(self, event):
        """Izvrši kompletno lokalno pretraživanje"""
        if self.current_solution is None:
            print("Prvo odaberite početnu tačku!")
            return

        if self.finished:
            print("Lokalno pretraživanje je već završeno!")
            return

        # Izvrši korake dok ne dođemo do lokalnog minimuma
        max_iterations = 100
        for _ in range(max_iterations):
            if self.finished:
                break
            self.on_step(None)
            plt.pause(0.3)

        if not self.finished:
            print("Dostignut maksimalan broj iteracija!")

    def on_reset(self, event):
        """Resetuj aplikaciju na početne postavke"""
        print("\nResetovanje aplikacije...")

        # Vrati početne postavke
        self.selected_function = self.initial_state['function']
        self.objective_function = FUNCTIONS[self.selected_function]['func']
        self.x_range = FUNCTIONS[self.selected_function]['range']
        self.global_min = FUNCTIONS[self.selected_function]['global_min']
        self.delta = self.initial_state['delta']
        self.plot_type = self.initial_state['plot_type']

        # Resetuj stanje algoritma
        self.current_solution = None
        self.history = []
        self.current_neighbors = []
        self.best_neighbor = None
        self.iteration = 0
        self.finished = False

        # Ponovno crtanje
        self.redraw_plot()

        print("Aplikacija resetovana!")

    def update_plot(self):
        """Ažuriraj grafički prikaz"""

        if self.plot_type == 'contour':
            self.update_contour_plot()
        else:
            self.update_3d_plot()

    def update_contour_plot(self):
        """Ažuriraj 2D contour plot"""

        # Obriši prethodne elemente
        if self.scatter_current:
            self.scatter_current.remove()
        if self.scatter_neighbors:
            self.scatter_neighbors.remove()
        if self.scatter_best_neighbor:
            self.scatter_best_neighbor.remove()
        if self.scatter_history:
            self.scatter_history.remove()
        if self.neighborhood_patch:
            self.neighborhood_patch.remove()

        # Nacrtaj historiju (putanju)
        if len(self.history) > 1:
            history_x = [h[0] for h in self.history]
            history_y = [h[1] for h in self.history]
            self.scatter_history = self.ax_main.plot(history_x, history_y,
                                                     'o-', color='purple',
                                                     linewidth=2, markersize=8,
                                                     alpha=0.6, label='Putanja')[0]

        # Nacrtaj trenutno rješenje
        if self.current_solution:
            self.scatter_current = self.ax_main.scatter([self.current_solution[0]],
                                                        [self.current_solution[1]],
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
            self.neighborhood_patch = self.ax_main.add_patch(rect)

        # Nacrtaj SVE susjedne tačke (svih 8)
        if self.current_neighbors:
            neighbors_x = [n[0] for n in self.current_neighbors]
            neighbors_y = [n[1] for n in self.current_neighbors]
            self.scatter_neighbors = self.ax_main.scatter(neighbors_x, neighbors_y,
                                                         c='orange', s=100, marker='s',
                                                         edgecolors='darkorange',
                                                         linewidths=1.5,
                                                         label='Okolina (8 tačaka)',
                                                         zorder=7, alpha=0.7)

        # Označi najbolju susjednu tačku posebno
        if self.best_neighbor:
            self.scatter_best_neighbor = self.ax_main.scatter([self.best_neighbor[0]],
                                                              [self.best_neighbor[1]],
                                                              c='lime', s=150, marker='D',
                                                              edgecolors='darkgreen',
                                                              linewidths=2,
                                                              label='Najbolji susjed',
                                                              zorder=8)

        self.ax_main.legend(loc='upper right', fontsize=10)
        self.fig.canvas.draw_idle()

    def update_3d_plot(self):
        """Ažuriraj 3D plot"""

        # Za 3D plot, crtamo samo tačke
        if self.current_solution:
            z_current = self.objective_function(self.current_solution)
            self.ax_main.scatter([self.current_solution[0]],
                                [self.current_solution[1]],
                                [z_current],
                                c='red', s=200, marker='o',
                                edgecolors='darkred', linewidths=2.5,
                                label='Trenutna tačka', zorder=10)

        # Nacrtaj putanju
        if len(self.history) > 1:
            history_x = [h[0] for h in self.history]
            history_y = [h[1] for h in self.history]
            history_z = [self.objective_function(h) for h in self.history]
            self.ax_main.plot(history_x, history_y, history_z,
                             'o-', color='purple', linewidth=2, markersize=8,
                             alpha=0.8, label='Putanja', zorder=10)

        # Nacrtaj susjedne tačke
        if self.current_neighbors:
            neighbors_x = [n[0] for n in self.current_neighbors]
            neighbors_y = [n[1] for n in self.current_neighbors]
            neighbors_z = [self.objective_function(n) for n in self.current_neighbors]
            self.ax_main.scatter(neighbors_x, neighbors_y, neighbors_z,
                                c='orange', s=100, marker='s',
                                edgecolors='darkorange', linewidths=1.5,
                                label='Okolina (8 tačaka)', zorder=10, alpha=0.7)

        # Najbolji susjed
        if self.best_neighbor:
            z_best = self.objective_function(self.best_neighbor)
            self.ax_main.scatter([self.best_neighbor[0]],
                                [self.best_neighbor[1]],
                                [z_best],
                                c='lime', s=150, marker='D',
                                edgecolors='darkgreen', linewidths=2,
                                label='Najbolji susjed', zorder=10)

        self.fig.canvas.draw_idle()

    def update_info_text(self):
        """Ažuriraj info tekst"""
        self.ax_info.clear()
        self.ax_info.axis('off')

        if self.current_solution is None:
            info_text = "Kliknite na grafik ili 'Slučajan start' da počnete"
        else:
            x = self.current_solution
            f_x = self.objective_function(x)

            info_text = f"Iteracija: {self.iteration}\n"
            info_text += f"Trenutna tačka: x = [{x[0]:.4f}, {x[1]:.4f}]\n"
            info_text += f"Vrijednost: f(x) = {f_x:.4f}\n"

            # Udaljenost od globalnog minimuma
            dist = np.sqrt((x[0] - self.global_min[0])**2 + (x[1] - self.global_min[1])**2)
            info_text += f"Udaljenost od globalnog: {dist:.4f}\n"

            if self.finished:
                info_text += "\n✓ LOKALNI MINIMUM PRONAĐEN!"
            else:
                info_text += "\nPretraživanje u toku..."

        self.ax_info.text(0.05, 0.95, info_text, transform=self.ax_info.transAxes,
                         fontsize=10, verticalalignment='top', family='monospace',
                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        # Pseudokod u info panelu
        pseudocode = "\nPseudokod:\n"
        pseudocode += "x ← x⁰\n"
        pseudocode += "repeat\n"
        pseudocode += "  N(x) ← okolina od x\n"
        pseudocode += "  x' ← najbolji(N(x))\n"
        pseudocode += "  if f(x') < f(x):\n"
        pseudocode += "    x ← x'\n"
        pseudocode += "until nema poboljšanja"

        self.ax_controls.clear()
        self.ax_controls.axis('off')
        self.ax_controls.text(0.05, 0.95, pseudocode, transform=self.ax_controls.transAxes,
                             fontsize=9, verticalalignment='top', family='monospace',
                             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

        self.fig.canvas.draw_idle()

    def on_delta_changed(self, val):
        """Promjena delta parametra"""
        self.delta = val
        if self.current_solution:
            # Ne resetujemo neighbours, samo ažuriramo prikaz
            self.current_neighbors = []
            self.best_neighbor = None
            self.update_plot()

    def on_about(self, event):
        """Prikaži About dialog"""
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()  # Sakrij glavni prozor
        messagebox.showinfo("O aplikaciji",
                           "Optimizacija resursa\n\n"
                           "Red. prof. dr Samim Konjicija\n\n"
                           "Novembar 2025. godine")
        root.destroy()

    def on_help(self, event):
        """Prikaži Help dialog"""
        import tkinter as tk
        from tkinter import messagebox

        help_text = """UPUTE ZA KORIŠTENJE - Lokalno pretraživanje

OSNOVNE FUNKCIJE:
• Klik na grafik - postavite početnu tačku za pretraživanje
• Slučajan start - generiši slučajnu početnu tačku
• Jedan korak - izvršite jednu iteraciju algoritma
• Do kraja - izvršite kompletno pretraživanje do lokalnog minimuma
• Reset - vratite aplikaciju na početne postavke

KONTROLE:
• Radio buttons (Funkcija) - izaberite test funkciju za optimizaciju
  (Kvadratna, Rastrigin, Ackley, Griewank, Levy)
• Radio buttons (Tip prikaza) - izaberite contour ili 3D mesh prikaz
• Slider (Delta) - podesite veličinu koraka pretrage (0.1 - 2.0)

LEGENDA:
• Zelena zvijezda - globalni minimum funkcije
• Crveni krug - trenutna tačka
• Narančasti kvadrati - susjedne tačke (okolina 8 tačaka)
• Zeleni dijamant - najbolji susjed
• Ljubičasta linija - putanja pretraživanja

NAPOMENA:
Algoritam koristi steepest descent strategiju - u svakoj iteraciji
se pomijerite na najbolju susjednu tačku dok god ima poboljšanja."""

        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("Pomoć", help_text)
        root.destroy()

# Pokreni demo
if __name__ == "__main__":
    print("="*80)
    print("Lokalno pretraživanje (Local Search) - Proširena Demo Aplikacija")
    print("="*80)
    print("\nFunkcionalnosti:")
    print("  ✓ Izbor funkcije: Kvadratna, Rastrigin, Ackley, Griewank, Levy")
    print("  ✓ Tip prikaza: Contour ili 3D mesh")
    print("  ✓ Prikaz svih 8 tačaka okoline + najbolja označena posebno")
    print("  ✓ Postavljanje početne tačke klikom miša")
    print("  ✓ Reset dugme")
    print("\nKontrole:")
    print("  - Kliknite na grafik da postavite početnu tačku")
    print("  - 'Slučajan start': Generiši slučajnu početnu tačku")
    print("  - 'Jedan korak': Izvrši jednu iteraciju")
    print("  - 'Do kraja': Izvrši pretraživanje do lokalnog minimuma")
    print("  - 'Reset': Vrati aplikaciju na početne postavke")
    print("  - Radio buttons: Izaberite funkciju i tip prikaza")
    print("  - Slider: Promijeni veličinu koraka (delta)")
    print("="*80)

    demo = LocalSearchDemo()
    plt.show()
