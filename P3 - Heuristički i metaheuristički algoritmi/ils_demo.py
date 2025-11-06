import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk

class ILSVisualizer:
    def __init__(self, master):
        self.master = master
        self.master.title("ILS - Iterated Local Search na multimodalnim funkcijama")

        # Parametri
        self.A = 10
        self.x_opt = 2  # Pomak minimuma na x=2
        self.num_cycles = 5
        self.current_cycle = 0
        self.animation_running = False
        self.search_step = 0.01
        self.function_type = "rastrigin"  # Tip funkcije: "rastrigin" ili "levy"

        # Istorija
        self.starting_points = []
        self.local_minima = []
        self.quadratic_functions = []
        self.search_paths = []  # Putanje lokalnog pretraživanja

        # GUI Setup
        self.setup_gui()

        # Bind resize event
        self.master.bind('<Configure>', self.on_resize)

        # Inicijalni prikaz
        self.reset_search()

    def rastrigin_1d(self, x):
        """Rastrigin funkcija sa jednom varijablom, pomaknuta tako da je minimum u x=2"""
        x_shifted = x - self.x_opt
        return self.A + x_shifted**2 - self.A * np.cos(2 * np.pi * x_shifted)

    def rastrigin_derivative(self, x):
        """Izvod Rastrigin funkcije"""
        x_shifted = x - self.x_opt
        return 2 * x_shifted + 2 * np.pi * self.A * np.sin(2 * np.pi * x_shifted)

    def levy_1d(self, x):
        """
        Levy funkcija (1D verzija) - asimetrična multimodalna funkcija.
        Globalni minimum je u x=1, ali ima mnogo lokalnih minimuma.
        Funkcija je asimetrična i veoma valovita.
        """
        w = 1 + (x - 1) / 4
        term1 = np.sin(np.pi * w)**2
        term2 = (w - 1)**2 * (1 + 10 * np.sin(np.pi * w + 1)**2)
        return term1 + term2

    def levy_derivative(self, x):
        """Izvod Levy funkcije - numerički izvod za stabilnost"""
        h = 1e-7
        return (self.levy_1d(x + h) - self.levy_1d(x - h)) / (2 * h)

    def griewank_1d(self, x):
        """
        Griewank funkcija (1D verzija) - asimetrična multimodalna funkcija.
        Kombinacija kvadratne komponente i kosinusne komponente.
        Globalni minimum je u x=0, ali ima mnogo lokalnih minimuma.
        Asimetrija dolazi od skaliranja i cosinus člana.
        """
        # Pomjeramo funkciju tako da minimum bude oko x=1.5
        x_shifted = x - 1.5
        term1 = (x_shifted**2) / 200
        term2 = np.cos(x_shifted / np.sqrt(1.5))
        return term1 - term2 + 1 + 0.1 * x_shifted  # Dodajemo linearni član za asimetriju

    def griewank_derivative(self, x):
        """Izvod Griewank funkcije - numerički izvod"""
        h = 1e-7
        return (self.griewank_1d(x + h) - self.griewank_1d(x - h)) / (2 * h)

    def ackley_1d(self, x):
        """
        Ackley funkcija (1D verzija) - asimetrična multimodalna funkcija.
        Karakteristična po gotovo ravnoj spoljašnjoj oblasti sa centralnim pikom.
        Globalni minimum je u x=0 (pomjeren na x=2).
        Ima veliki broj lokalnih minimuma.
        """
        # Pomjeramo funkciju tako da minimum bude oko x=2
        x_shifted = x - 2
        a = 20
        b = 0.2
        c = 2 * np.pi

        term1 = -a * np.exp(-b * np.abs(x_shifted))
        term2 = -np.exp(np.cos(c * x_shifted))

        # Dodajemo blagi asimetrični član
        asym_term = 0.05 * x_shifted

        return term1 + term2 + a + np.e + asym_term

    def ackley_derivative(self, x):
        """Izvod Ackley funkcije - numerički izvod"""
        h = 1e-7
        return (self.ackley_1d(x + h) - self.ackley_1d(x - h)) / (2 * h)

    def objective_function(self, x):
        """Wrapper za odabranu objektivnu funkciju"""
        if self.function_type == "levy":
            return self.levy_1d(x)
        elif self.function_type == "griewank":
            return self.griewank_1d(x)
        elif self.function_type == "ackley":
            return self.ackley_1d(x)
        else:  # rastrigin
            return self.rastrigin_1d(x)

    def objective_derivative(self, x):
        """Wrapper za izvod odabrane funkcije"""
        if self.function_type == "levy":
            return self.levy_derivative(x)
        elif self.function_type == "griewank":
            return self.griewank_derivative(x)
        elif self.function_type == "ackley":
            return self.ackley_derivative(x)
        else:  # rastrigin
            return self.rastrigin_derivative(x)

    def fit_quadratic(self, points):
        """Fituje kvadratnu funkciju kroz date tačke (x, y)"""
        x_vals = np.array([p[0] for p in points])
        y_vals = np.array([p[1] for p in points])

        # Fitujemo polinom drugog stepena
        coeffs = np.polyfit(x_vals, y_vals, 2)
        return coeffs  # vraća [a, b, c] za ax^2 + bx + c

    def find_quadratic_minimum(self, coeffs):
        """Nalazi minimum kvadratne funkcije"""
        a, b, c = coeffs
        if a <= 0:
            # Ako je parabola okrenuta nadole, nema smislenog minimuma
            return None
        x_min = -b / (2 * a)
        return x_min

    def local_search(self, x0):
        """
        Jednostavno lokalno pretraživanje koristeći gradijentni spust
        sa adaptivnim korakom i provjerom okolnih tačaka.
        Čuva putanju svih koraka za vizualizaciju.
        """
        x = x0
        step = self.search_step * 10  # Početni korak
        max_iterations = 10000
        tolerance = 1e-6
        no_improvement_count = 0
        best_value = self.objective_function(x)

        # Čuvanje putanje
        path = [(x, self.objective_function(x))]

        for iteration in range(max_iterations):
            current_value = self.objective_function(x)

            # Izračunaj gradijent
            grad = self.objective_derivative(x)

            # Probaj korak u smjeru negativnog gradijenta
            x_new = x - step * np.sign(grad) if abs(grad) > 1e-10 else x

            # Ograniči na domen
            x_new = np.clip(x_new, -3.5, 7.5)

            new_value = self.objective_function(x_new)

            # Ako smo poboljšali rješenje
            if new_value < current_value - tolerance:
                x = x_new
                best_value = new_value
                no_improvement_count = 0
                step = min(step * 1.1, 0.5)  # Povećaj korak (ali ne previše)

                # Dodaj u putanju samo značajne korake
                if len(path) < 100 or iteration % max(1, len(path) // 50) == 0:
                    path.append((x, self.objective_function(x)))
            else:
                # Ako nismo poboljšali, smanji korak
                step *= 0.5
                no_improvement_count += 1

                # Ako je korak postao vrlo mali, pokušaj još malo pa prekini
                if step < tolerance:
                    # Finija lokalna pretraga oko trenutne pozicije
                    search_radius = 0.1
                    test_points = np.linspace(x - search_radius, x + search_radius, 20)
                    test_points = np.clip(test_points, -3.5, 7.5)
                    test_values = [self.objective_function(xi) for xi in test_points]
                    best_idx = np.argmin(test_values)
                    x = test_points[best_idx]
                    path.append((x, self.objective_function(x)))
                    break

                # Ako dugo nema poboljšanja, prekini
                if no_improvement_count > 50:
                    break

        # Dodaj finalnu tačku ako nije već dodana
        if path[-1][0] != x:
            path.append((x, self.objective_function(x)))

        return x, path

    def setup_gui(self):
        # Kontrolni panel
        control_frame = ttk.Frame(self.master, padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Izbor funkcije
        ttk.Label(control_frame, text="Funkcija:").grid(row=0, column=0, sticky=tk.W)
        self.function_var = tk.StringVar(value="rastrigin")
        function_combo = ttk.Combobox(control_frame, textvariable=self.function_var,
                                      values=["rastrigin", "levy", "griewank", "ackley"],
                                      width=12, state="readonly")
        function_combo.grid(row=0, column=1, padx=5)
        function_combo.bind("<<ComboboxSelected>>", self.on_function_change)

        # Broj ciklusa
        ttk.Label(control_frame, text="Broj ciklusa:").grid(row=0, column=2, sticky=tk.W, padx=(20, 0))
        self.cycles_var = tk.IntVar(value=5)
        cycles_spinbox = tk.Spinbox(control_frame, from_=3, to=20,
                                     textvariable=self.cycles_var, width=10)
        cycles_spinbox.grid(row=0, column=3, padx=5)

        # Korak lokalnog pretraživanja
        ttk.Label(control_frame, text="Korak pretrage:").grid(row=0, column=4, sticky=tk.W, padx=(20, 0))
        self.step_var = tk.DoubleVar(value=0.01)
        step_spinbox = tk.Spinbox(control_frame, from_=0.001, to=0.1, increment=0.005,
                                   textvariable=self.step_var, width=10)
        step_spinbox.grid(row=0, column=5, padx=5)

        # Dugmad
        self.start_button = ttk.Button(control_frame, text="Pokreni ILS",
                                       command=self.start_ils)
        self.start_button.grid(row=0, column=6, padx=5)

        self.reset_button = ttk.Button(control_frame, text="Resetuj",
                                       command=self.reset_search)
        self.reset_button.grid(row=0, column=7, padx=5)

        self.step_button = ttk.Button(control_frame, text="Sljedeći korak",
                                      command=self.next_step)
        self.step_button.grid(row=0, column=8, padx=5)

        self.status_label = ttk.Label(control_frame, text="Spremno za pokretanje",
                                      wraplength=1100)
        self.status_label.grid(row=1, column=0, columnspan=9, pady=5)

        # Frame za matplotlib (koristi samo grid)
        self.plot_frame = ttk.Frame(self.master)
        self.plot_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)

        # Konfiguriši grid težine za resizing
        self.master.columnconfigure(0, weight=1)
        self.master.rowconfigure(1, weight=1)
        self.plot_frame.columnconfigure(0, weight=1)
        self.plot_frame.rowconfigure(0, weight=1)

        # Matplotlib figura
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)

        # Koristi grid umjesto pack
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    def on_function_change(self, event=None):
        """Handler za promenu funkcije"""
        self.function_type = self.function_var.get()
        self.reset_search()

    def on_resize(self, event):
        """Handler za resize event"""
        # Ažuriraj veličinu figure pri promjeni veličine prozora
        if hasattr(self, 'canvas') and hasattr(self, 'plot_frame'):
            # Sačekaj malo prije redraw-a da izbjegnemo previše refresh-eva
            if hasattr(self, '_resize_job'):
                self.master.after_cancel(self._resize_job)
            self._resize_job = self.master.after(100, self.update_plot_size)

    def update_plot_size(self):
        """Ažuriraj veličinu plota na osnovu veličine frame-a"""
        try:
            frame_width = self.plot_frame.winfo_width()
            frame_height = self.plot_frame.winfo_height()

            if frame_width > 100 and frame_height > 100:  # Validna veličina
                # Konvertuj pixele u inče (pretpostavljamo 100 DPI)
                fig_width = frame_width / 100
                fig_height = frame_height / 100

                self.fig.set_size_inches(fig_width, fig_height)
                self.canvas.draw()
        except:
            pass

    def plot_current_state(self):
        """Iscrtava trenutno stanje pretrage"""
        self.ax.clear()

        # Iscrtaj objektivnu funkciju
        x = np.linspace(-3.5, 7.5, 1000)
        y = self.objective_function(x)
        func_names = {
            "rastrigin": "Rastrigin funkcija",
            "levy": "Levy funkcija",
            "griewank": "Griewank funkcija",
            "ackley": "Ackley funkcija"
        }
        func_name = func_names.get(self.function_type, "Rastrigin funkcija")
        self.ax.plot(x, y, 'b-', linewidth=2, label=func_name)

        # Iscrtaj putanje lokalnog pretraživanja
        path_colors = ['purple', 'orange', 'brown', 'pink', 'cyan', 'magenta', 'lime', 'navy']
        for i, path in enumerate(self.search_paths):
            if len(path) > 1:
                path_x = [p[0] for p in path]
                path_y = [p[1] for p in path]
                color = path_colors[i % len(path_colors)]

                # Iscrtaj putanju sa tačkicama
                self.ax.plot(path_x, path_y, 'o-', color=color, markersize=4,
                           linewidth=1.5, alpha=0.6, label=f'Putanja {i+1}' if i < 3 else '')

                # Označi početnu i krajnju tačku
                self.ax.plot(path_x[0], path_y[0], 'o', color=color, markersize=10,
                           markeredgecolor='darkgreen', markeredgewidth=2, zorder=5)
                self.ax.plot(path_x[-1], path_y[-1], 's', color=color, markersize=10,
                           markeredgecolor='darkred', markeredgewidth=2, zorder=5)

        # Iscrtaj pronađene minimume
        for i, (start, minimum) in enumerate(zip(self.starting_points, self.local_minima)):
            self.ax.plot(start, self.objective_function(start), 'go', markersize=12,
                        label='Početne tačke' if i == 0 else '', zorder=6,
                        markeredgecolor='darkgreen', markeredgewidth=2)
            self.ax.plot(minimum, self.objective_function(minimum), 'ro', markersize=12,
                        label='Lokalni minimumi' if i == 0 else '', zorder=6,
                        markeredgecolor='darkred', markeredgewidth=2)

            # Oznake tačaka
            self.ax.text(start, self.objective_function(start) + 2, f'{i+1}',
                        ha='center', fontsize=9, fontweight='bold', color='darkgreen')

        # Iscrtaj kvadratne funkcije (tek od 3. ciklusa)
        quad_colors = ['darkviolet', 'darkorange', 'saddlebrown', 'deeppink', 'darkcyan']
        for i, (coeffs, points) in enumerate(self.quadratic_functions):
            x_quad = np.linspace(-3.5, 7.5, 200)
            y_quad = coeffs[0] * x_quad**2 + coeffs[1] * x_quad + coeffs[2]

            # Ograniči y vrijednosti za bolji prikaz
            y_max = 60 if self.function_type == "rastrigin" else 100
            y_quad_clipped = np.clip(y_quad, -5, y_max)

            color = quad_colors[i % len(quad_colors)]
            self.ax.plot(x_quad, y_quad_clipped, '--', linewidth=2, alpha=0.5,
                        color=color, label=f'Kvadratna aproks. (ciklus {i+3})')

            # Označi minimum kvadratne funkcije
            x_min = self.find_quadratic_minimum(coeffs)
            if x_min is not None and -3.5 <= x_min <= 7.5:  # Ako je minimum u domenu
                y_min = coeffs[0] * x_min**2 + coeffs[1] * x_min + coeffs[2]
                if -5 <= y_min <= y_max:
                    self.ax.plot(x_min, y_min, 's', color=color, markersize=12,
                                label='Min. kvadratne f.' if i == len(self.quadratic_functions)-1 else '',
                                zorder=6, markeredgecolor='black', markeredgewidth=1)
                    self.ax.axvline(x=x_min, color=color, linestyle=':', alpha=0.3)

        # Označi globalni minimum
        global_mins = {
            "rastrigin": self.x_opt,
            "levy": 1.0,
            "griewank": 1.5,
            "ackley": 2.0
        }
        global_min_x = global_mins.get(self.function_type, self.x_opt)
        self.ax.plot(global_min_x, self.objective_function(global_min_x), 'y*', markersize=20,
                    label=f'Globalni minimum (x={global_min_x})', zorder=10,
                    markeredgecolor='black', markeredgewidth=1.5)

        self.ax.set_xlabel('x', fontsize=12, fontweight='bold')
        self.ax.set_ylabel('f(x)', fontsize=12, fontweight='bold')
        title = f'ILS Pretraga ({func_name}) - Ciklus {self.current_cycle}/{self.num_cycles}'
        self.ax.set_title(title, fontsize=14, fontweight='bold')
        self.ax.legend(loc='upper right', fontsize=8, ncol=2)
        self.ax.grid(True, alpha=0.3)

        # Prilagodi y-lim u zavisnosti od funkcije
        y_limits = {
            "rastrigin": (-2, 50),
            "levy": (-1, 80),
            "griewank": (-1, 25),
            "ackley": (-1, 25)
        }
        y_lim = y_limits.get(self.function_type, (-2, 50))
        self.ax.set_ylim(y_lim[0], y_lim[1])
        self.ax.set_xlim(-3.5, 7.5)

        self.fig.tight_layout()
        self.canvas.draw()

    def reset_search(self):
        """Resetuje pretragu"""
        self.current_cycle = 0
        self.starting_points = []
        self.local_minima = []
        self.quadratic_functions = []
        self.search_paths = []
        self.num_cycles = self.cycles_var.get()
        self.search_step = self.step_var.get()
        self.animation_running = False
        func_names = {
            "rastrigin": "Rastrigin",
            "levy": "Levy",
            "griewank": "Griewank",
            "ackley": "Ackley"
        }
        func_name = func_names.get(self.function_type, "Rastrigin")
        self.status_label.config(text=f"Spremno za pokretanje ({func_name} funkcija)")
        self.plot_current_state()

    def next_step(self):
        """Izvršava sljedeći korak ILS algoritma"""
        if self.current_cycle >= self.num_cycles:
            self.status_label.config(text="Pretraga završena!")
            return

        self.current_cycle += 1
        self.search_step = self.step_var.get()

        if self.current_cycle == 1:
            # Prvi ciklus - slučajna početna tačka lijevo od optimuma
            x0 = np.random.uniform(-3, 0)
            self.status_label.config(text=f"Ciklus 1: Slučajna početna tačka x={x0:.3f}")
        elif self.current_cycle == 2:
            # Drugi ciklus - slučajna tačka desno od optimuma
            x0 = np.random.uniform(4, 7)
            self.status_label.config(text=f"Ciklus 2: Slučajna početna tačka x={x0:.3f}")
        else:
            # Od trećeg ciklusa - minimum kvadratne funkcije kroz sve pronađene minimume
            points = [(x, self.objective_function(x)) for x in self.local_minima]
            coeffs = self.fit_quadratic(points)
            x0 = self.find_quadratic_minimum(coeffs)

            # Provjeri da li je parabola okrenuta prema gore
            if x0 is None or coeffs[0] <= 0:
                # Ako je okrenuta nadole, uzmi slučajnu tačku
                x0 = np.random.uniform(-3, 7)
                self.status_label.config(text=f"Ciklus {self.current_cycle}: "
                                        f"Kvadratna f. nema minimum (a={coeffs[0]:.3f}), "
                                        f"nova slučajna tačka x={x0:.3f}")
            else:
                x0 = np.clip(x0, -3.5, 7.5)  # Ograniči na domen
                self.quadratic_functions.append((coeffs, self.local_minima.copy()))
                self.status_label.config(text=f"Ciklus {self.current_cycle}: "
                                        f"Početna tačka iz min. kvadratne f. x={x0:.3f} "
                                        f"(a={coeffs[0]:.3f}, b={coeffs[1]:.3f}, c={coeffs[2]:.3f})")

        # Lokalno pretraživanje - sada vraća i putanju
        self.starting_points.append(x0)
        x_min, path = self.local_search(x0)
        self.local_minima.append(x_min)
        self.search_paths.append(path)

        self.status_label.config(text=self.status_label.cget("text") +
                                f" → Lokalni min: x={x_min:.3f}, f(x)={self.objective_function(x_min):.3f} "
                                f"(koraka: {len(path)})")

        self.plot_current_state()

        if self.current_cycle >= self.num_cycles:
            best_idx = np.argmin([self.objective_function(x) for x in self.local_minima])
            best_x = self.local_minima[best_idx]
            best_y = self.objective_function(best_x)
            global_mins = {
                "rastrigin": self.x_opt,
                "levy": 1.0,
                "griewank": 1.5,
                "ackley": 2.0
            }
            global_min_x = global_mins.get(self.function_type, self.x_opt)
            self.status_label.config(text=f"Pretraga završena! "
                                    f"Najbolje rješenje: x={best_x:.4f}, f(x)={best_y:.4f} "
                                    f"(Greška od globalnog: {abs(best_x - global_min_x):.4f})")

    def start_ils(self):
        """Pokreće kompletnu ILS pretragu"""
        self.reset_search()
        for i in range(self.num_cycles):
            self.master.after(1500 * (i + 1), self.next_step)

# Pokretanje aplikacije
if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1400x750")  # Početna veličina prozora
    app = ILSVisualizer(root)
    root.mainloop()
