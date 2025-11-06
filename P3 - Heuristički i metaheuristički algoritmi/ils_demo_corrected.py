"""
ILS (Iterated Local Search) Demo - ISPRAVLJENA verzija
Kvadratna interpolacija prolazi kroz PRONAĐENE LOKALNE MINIMUME
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from scipy.optimize import minimize_scalar
from scipy.interpolate import lagrange

def rastrigin(x, A=10):
    """Rastrigin funkcija: f(x) = A + x^2 - A*cos(2*pi*x)"""
    return A + x**2 - A * np.cos(2 * np.pi * x)

class ILSDemo:
    def __init__(self):
        # Parametri
        self.x_range = (-5, 5)
        self.n_cycles = 5
        self.A = 10
        
        # Podaci za algoritam
        self.local_minima = []  # Lista (x_min, f(x_min)) - PRONAĐENI MINIMUMI
        self.starting_points = []  # Lista početnih tačaka za svaki ciklus
        self.current_cycle = 0
        
        # Setup figura
        self.setup_figure()
        
    def setup_figure(self):
        """Postavi figuru i sve elemente"""
        self.fig = plt.figure(figsize=(14, 10))
        
        # Glavni plot
        self.ax_main = plt.subplot2grid((3, 2), (0, 0), colspan=2, rowspan=2)
        
        # Info panel
        self.ax_info = plt.subplot2grid((3, 2), (2, 0))
        self.ax_info.axis('off')
        
        # Kontrole
        self.ax_controls = plt.subplot2grid((3, 2), (2, 1))
        self.ax_controls.axis('off')
        
        # Nacrtaj Rastrigin funkciju
        self.x_plot = np.linspace(self.x_range[0], self.x_range[1], 1000)
        self.y_plot = rastrigin(self.x_plot, self.A)
        
        self.ax_main.plot(self.x_plot, self.y_plot, 'b-', linewidth=2.5, 
                         label='Rastrigin funkcija', alpha=0.8)
        self.ax_main.axvline(x=0, color='green', linestyle='--', linewidth=2, 
                            alpha=0.5, label='Globalni minimum (x=0)')
        self.ax_main.set_xlabel('x', fontsize=13, fontweight='bold')
        self.ax_main.set_ylabel('f(x)', fontsize=13, fontweight='bold')
        self.ax_main.set_title('ILS - Kvadratna interpolacija kroz pronađene minimume', 
                               fontsize=14, fontweight='bold')
        self.ax_main.grid(True, alpha=0.3)
        self.ax_main.legend(loc='upper right')
        
        # Elementi za crtanje
        self.scatter_minima = None
        self.line_quadratic = None
        self.scatter_starts = None
        self.scatter_pred_start = None
        
        # Dodaj kontrole
        self.add_controls()
        
        # Inicijalni info tekst
        self.update_info_text()
        
        plt.tight_layout()
        
    def add_controls(self):
        """Dodaj kontrolne dugmad i slidere"""
        
        # Dugme: Start/Reset
        ax_button_start = plt.axes([0.15, 0.02, 0.15, 0.04])
        self.button_start = Button(ax_button_start, 'Pokreni ILS', 
                                   color='lightgreen', hovercolor='green')
        self.button_start.on_clicked(self.on_start_clicked)
        
        # Dugme: Jedan korak
        ax_button_step = plt.axes([0.32, 0.02, 0.15, 0.04])
        self.button_step = Button(ax_button_step, 'Jedan korak', 
                                  color='lightblue', hovercolor='blue')
        self.button_step.on_clicked(self.on_step_clicked)
        
        # Dugme: Reset
        ax_button_reset = plt.axes([0.49, 0.02, 0.15, 0.04])
        self.button_reset = Button(ax_button_reset, 'Reset', 
                                   color='lightcoral', hovercolor='red')
        self.button_reset.on_clicked(self.on_reset_clicked)
        
        # Slider: Broj ciklusa
        ax_slider = plt.axes([0.7, 0.03, 0.2, 0.02])
        self.slider_cycles = Slider(ax_slider, 'Broj ciklusa:', 
                                    1, 20, valinit=5, valstep=1)
        self.slider_cycles.on_changed(self.on_slider_changed)
        
    def find_local_minimum(self, x_start):
        """
        Pronađi lokalni minimum počevši od x_start
        Ograniči pretragu na lokalni region oko početne tačke
        """
        local_range = 2.0  # Radius lokalnog pretraživanja
        x_min_bound = max(self.x_range[0], x_start - local_range)
        x_max_bound = min(self.x_range[1], x_start + local_range)
        
        result = minimize_scalar(
            lambda x: rastrigin(x, self.A),
            bounds=(x_min_bound, x_max_bound),
            method='bounded',
            options={'xatol': 1e-5}
        )
        
        return result.x, result.fun
        
    def quadratic_interpolation(self):
        """
        KLJUČNA FUNKCIJA:
        Napravi kvadratnu interpolaciju kroz PRONAĐENE LOKALNE MINIMUME
        (ne kroz početne tačke!)
        
        Returns:
            x_min: x koordinata minimuma kvadratne funkcije
            poly: Lagrange polinom
        """
        if len(self.local_minima) < 2:
            return None, None
            
        # Uzmi zadnje 2-3 PRONAĐENA MINIMUMA
        points = self.local_minima[-3:] if len(self.local_minima) >= 3 else self.local_minima[-2:]
        
        # Ekstrakcija x i f(x) koordinata MINIMUMA
        x_points = np.array([p[0] for p in points])
        y_points = np.array([p[1] for p in points])
        
        # Lagrange interpolacija - kvadratna funkcija kroz pronađene minimume
        poly = lagrange(x_points, y_points)
        
        # Nađi minimum te kvadratne funkcije
        x_dense = np.linspace(self.x_range[0], self.x_range[1], 1000)
        y_dense = poly(x_dense)
        
        # Filtriranje nevažećih vrijednosti
        valid_mask = np.isfinite(y_dense)
        if np.any(valid_mask):
            y_valid = y_dense[valid_mask]
            x_valid = x_dense[valid_mask]
            min_idx = np.argmin(y_valid)
            x_min = x_valid[min_idx]
        else:
            # Fallback: srednja vrijednost
            x_min = np.mean(x_points)
        
        # Ograniči na dozvoljeni range
        x_min = np.clip(x_min, self.x_range[0], self.x_range[1])
        
        return x_min, poly
        
    def perform_one_cycle(self):
        """Izvrši jedan ILS ciklus"""
        
        if self.current_cycle == 0:
            # Ciklus 1: Slučajna početna tačka
            x_start = np.random.uniform(self.x_range[0], self.x_range[1])
            print(f"\nCiklus 1: Slučajna početna tačka x_start = {x_start:.4f}")
            
        elif self.current_cycle == 1:
            # Ciklus 2: Još jedna slučajna početna tačka
            x_start = np.random.uniform(self.x_range[0], self.x_range[1])
            print(f"\nCiklus 2: Slučajna početna tačka x_start = {x_start:.4f}")
            
        else:
            # Ciklus 3+: Kvadratna interpolacija kroz PRONAĐENE MINIMUME
            x_start, poly = self.quadratic_interpolation()
            self.current_poly = poly
            print(f"\nCiklus {self.current_cycle + 1}: Interpolirana početna tačka x_start = {x_start:.4f}")
            print(f"  Interpolacija kroz minimume: {[f'({p[0]:.2f}, {p[1]:.2f})' for p in self.local_minima[-3:]]}")
        
        # Sačuvaj početnu tačku
        self.starting_points.append(x_start)
        
        # Pronađi lokalni minimum od te početne tačke
        x_min, f_min = self.find_local_minimum(x_start)
        
        print(f"  Pronađen lokalni minimum: x_min = {x_min:.4f}, f(x_min) = {f_min:.4f}")
        
        # Sačuvaj pronađeni minimum
        self.local_minima.append((x_min, f_min))
        
        self.current_cycle += 1
        
        # Ažuriraj prikaz
        self.update_plot()
        self.update_info_text()
        
    def update_plot(self):
        """Ažuriraj grafički prikaz"""
        
        # Obriši prethodne elemente
        if self.scatter_minima:
            self.scatter_minima.remove()
        if self.line_quadratic:
            for line in self.line_quadratic:
                line.remove()
        if self.scatter_starts:
            self.scatter_starts.remove()
        if self.scatter_pred_start:
            self.scatter_pred_start.remove()
            
        # Nacrtaj pronađene lokalne MINIMUME (crveni krugovi)
        if self.local_minima:
            x_mins = [p[0] for p in self.local_minima]
            y_mins = [p[1] for p in self.local_minima]
            self.scatter_minima = self.ax_main.scatter(x_mins, y_mins, 
                                                       c='red', s=150, marker='o',
                                                       edgecolors='darkred', linewidths=2.5,
                                                       label='Pronađeni lokalni minimumi', 
                                                       zorder=5)
            
            # Označi svaki minimum brojem
            for i, (x, y) in enumerate(self.local_minima):
                self.ax_main.annotate(f'{i+1}', (x, y), 
                                     textcoords="offset points", xytext=(0,12),
                                     ha='center', fontsize=11, fontweight='bold',
                                     color='darkred',
                                     bbox=dict(boxstyle='round,pad=0.3', 
                                             facecolor='white', 
                                             edgecolor='darkred', linewidth=1.5))
        
        # Nacrtaj početne tačke (narandžasti kvadrati)
        if self.starting_points:
            x_starts = self.starting_points
            y_starts = [rastrigin(x, self.A) for x in x_starts]
            self.scatter_starts = self.ax_main.scatter(x_starts, y_starts, 
                                                       c='orange', s=120, marker='s',
                                                       edgecolors='darkorange', linewidths=2,
                                                       label='Početne tačke', zorder=4, 
                                                       alpha=0.7)
        
        # Nacrtaj kvadratnu interpolaciju (zelena isprekidana linija)
        if self.current_cycle > 2 and hasattr(self, 'current_poly'):
            x_interp = np.linspace(self.x_range[0], self.x_range[1], 500)
            y_interp = self.current_poly(x_interp)
            
            # Ograniči y da ne ide previsoko
            y_interp = np.clip(y_interp, self.ax_main.get_ylim()[0], 
                              self.ax_main.get_ylim()[1])
            
            self.line_quadratic = self.ax_main.plot(x_interp, y_interp, 
                                                   'g--', linewidth=2.5, alpha=0.7,
                                                   label='Kvadratna interpolacija')
            
            # Označi predviđenu početnu tačku (lime zvijezda)
            if self.starting_points:
                x_pred = self.starting_points[-1]
                y_pred = rastrigin(x_pred, self.A)
                self.scatter_pred_start = self.ax_main.scatter([x_pred], [y_pred],
                                                               c='lime', s=300, 
                                                               marker='*',
                                                               edgecolors='darkgreen',
                                                               linewidths=2.5,
                                                               label='Predviđena početna tačka',
                                                               zorder=6)
        
        # Nacrtaj strelice od početne tačke do pronađenog minimuma
        for i in range(len(self.local_minima)):
            x_start = self.starting_points[i]
            x_end = self.local_minima[i][0]
            y_start = rastrigin(x_start, self.A)
            y_end = self.local_minima[i][1]
            
            self.ax_main.annotate('', xy=(x_end, y_end), xytext=(x_start, y_start),
                                 arrowprops=dict(arrowstyle='->', lw=2.5, 
                                               color='purple', alpha=0.6))
        
        self.ax_main.legend(loc='upper right', fontsize=9)
        self.fig.canvas.draw_idle()
        
    def update_info_text(self):
        """Ažuriraj info tekst"""
        self.ax_info.clear()
        self.ax_info.axis('off')
        
        info_text = f"Ciklus: {self.current_cycle} / {self.n_cycles}\n\n"
        
        if self.local_minima:
            info_text += "Pronađeni lokalni minimumi:\n"
            for i, (x, f_x) in enumerate(self.local_minima):
                x_start = self.starting_points[i]
                info_text += f"  {i+1}. Start: x={x_start:.3f} → Min: x={x:.3f}, f(x)={f_x:.3f}\n"
            
            # Najbolji minimum
            best_idx = np.argmin([f for _, f in self.local_minima])
            best_x, best_f = self.local_minima[best_idx]
            info_text += f"\n🎯 Najbolji: x={best_x:.4f}, f(x)={best_f:.4f}\n"
            info_text += f"Odstupanje od globalnog: {abs(best_x):.4f}"
        else:
            info_text += "Pritisnite 'Jedan korak' ili 'Pokreni ILS'"
        
        self.ax_info.text(0.05, 0.95, info_text, transform=self.ax_info.transAxes,
                         fontsize=9, verticalalignment='top', family='monospace',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        self.fig.canvas.draw_idle()
        
    def on_start_clicked(self, event):
        """Pokreni kompletnu ILS sekvencu"""
        self.on_reset_clicked(None)
        
        for _ in range(self.n_cycles):
            self.perform_one_cycle()
            plt.pause(0.8)
            
    def on_step_clicked(self, event):
        """Izvrši jedan korak"""
        if self.current_cycle < self.n_cycles:
            self.perform_one_cycle()
        else:
            print("Dostignut maksimalan broj ciklusa!")
            
    def on_reset_clicked(self, event):
        """Reset algoritma"""
        self.local_minima = []
        self.starting_points = []
        self.current_cycle = 0
        
        if hasattr(self, 'current_poly'):
            delattr(self, 'current_poly')
        
        # Obriši elemente
        if self.scatter_minima:
            self.scatter_minima.remove()
            self.scatter_minima = None
        if self.line_quadratic:
            for line in self.line_quadratic:
                line.remove()
            self.line_quadratic = None
        if self.scatter_starts:
            self.scatter_starts.remove()
            self.scatter_starts = None
        if self.scatter_pred_start:
            self.scatter_pred_start.remove()
            self.scatter_pred_start = None
            
        # Ponovo nacrtaj osnovnu funkciju
        self.ax_main.clear()
        self.ax_main.plot(self.x_plot, self.y_plot, 'b-', linewidth=2.5, 
                         label='Rastrigin funkcija', alpha=0.8)
        self.ax_main.axvline(x=0, color='green', linestyle='--', linewidth=2, 
                            alpha=0.5, label='Globalni minimum (x=0)')
        self.ax_main.set_xlabel('x', fontsize=13, fontweight='bold')
        self.ax_main.set_ylabel('f(x)', fontsize=13, fontweight='bold')
        self.ax_main.set_title('ILS - Kvadratna interpolacija kroz pronađene minimume', 
                               fontsize=14, fontweight='bold')
        self.ax_main.grid(True, alpha=0.3)
        self.ax_main.legend(loc='upper right')
        
        self.update_info_text()
        self.fig.canvas.draw_idle()
        
    def on_slider_changed(self, val):
        """Promjena broja ciklusa"""
        self.n_cycles = int(val)
        self.update_info_text()

# Pokreni demo
if __name__ == "__main__":
    print("="*80)
    print("ILS (Iterated Local Search) Demo - ISPRAVLJENA verzija")
    print("="*80)
    print("\nKLJUČNA RAZLIKA:")
    print("  - Kvadratna interpolacija prolazi kroz PRONAĐENE LOKALNE MINIMUME")
    print("  - Minimum te kvadratne funkcije postaje nova početna tačka")
    print("\nKontrole:")
    print("  - 'Pokreni ILS': Automatski izvrši sve cikluse")
    print("  - 'Jedan korak': Izvrši jedan ciklus manuelno")
    print("  - 'Reset': Ponovo pokreni algoritam")
    print("  - Slider: Promijeni broj ciklusa (1-20)")
    print("\nAlgoritam:")
    print("  1. Ciklus 1: x_start = slučajno → nađi minimum → zapamti minimum")
    print("  2. Ciklus 2: x_start = slučajno → nađi minimum → zapamti minimum")
    print("  3. Ciklus 3: Kvadratna f-ja kroz minimume 1 i 2 → min te f-je = x_start")
    print("  4. Ciklus 4+: Kvadratna f-ja kroz zadnja 3 minimuma → min = x_start")
    print("="*80)
    
    demo = ILSDemo()
    plt.show()
