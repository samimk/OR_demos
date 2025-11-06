"""
Lokalno pretraživanje (Local Search) - Demo aplikacija
Demonstrira osnovno lokalno pretraživanje iz prezentacije (slajdovi 18-24)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import matplotlib.patches as patches

def objective_function(x):
    """
    Multimodalna funkcija za demonstraciju:
    f(x) = x1^2 + x2^2 (kvadratna funkcija sa jednostavnom okolinom)
    """
    return x[0]**2 + x[1]**2

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
        # Ograniči na dozvoljeni prostor
        x_new[0] = np.clip(x_new[0], -5, 5)
        x_new[1] = np.clip(x_new[1], -5, 5)
        neighbors.append(x_new)
    
    return neighbors

class LocalSearchDemo:
    def __init__(self):
        # Parametri
        self.x_range = (-5, 5)
        self.delta = 0.5  # Veličina koraka
        
        # Stanje algoritma
        self.current_solution = None
        self.history = []  # Historija rješenja
        self.current_neighbors = []
        self.iteration = 0
        self.finished = False
        
        # Setup figura
        self.setup_figure()
        
    def setup_figure(self):
        """Postavi figuru"""
        self.fig = plt.figure(figsize=(14, 10))
        
        # Glavni plot - konturni dijagram
        self.ax_main = plt.subplot2grid((3, 2), (0, 0), colspan=2, rowspan=2)
        
        # Info panel
        self.ax_info = plt.subplot2grid((3, 2), (2, 0))
        self.ax_info.axis('off')
        
        # Kontrole
        self.ax_controls = plt.subplot2grid((3, 2), (2, 1))
        self.ax_controls.axis('off')
        
        # Nacrtaj konturni dijagram objektive funkcije
        x1 = np.linspace(self.x_range[0], self.x_range[1], 200)
        x2 = np.linspace(self.x_range[0], self.x_range[1], 200)
        X1, X2 = np.meshgrid(x1, x2)
        Z = X1**2 + X2**2
        
        contour = self.ax_main.contour(X1, X2, Z, levels=20, cmap='viridis', alpha=0.6)
        self.ax_main.clabel(contour, inline=True, fontsize=8)
        
        # Označi globalni minimum
        self.ax_main.scatter([0], [0], c='green', s=200, marker='*', 
                            edgecolors='darkgreen', linewidths=2,
                            label='Globalni minimum (0,0)', zorder=10)
        
        self.ax_main.set_xlabel('x₁', fontsize=13, fontweight='bold')
        self.ax_main.set_ylabel('x₂', fontsize=13, fontweight='bold')
        self.ax_main.set_title('Lokalno pretraživanje - Steepest Descent', 
                               fontsize=14, fontweight='bold')
        self.ax_main.grid(True, alpha=0.3)
        self.ax_main.legend(loc='upper right')
        self.ax_main.set_xlim(self.x_range)
        self.ax_main.set_ylim(self.x_range)
        
        # Elementi za crtanje
        self.scatter_current = None
        self.scatter_neighbors = None
        self.scatter_history = None
        self.neighborhood_patch = None
        
        # Dodaj kontrole
        self.add_controls()
        
        # Inicijalni info tekst
        self.update_info_text()
        
        plt.tight_layout()
        
    def add_controls(self):
        """Dodaj kontrolne dugmad"""
        
        # Dugme: Nova početna tačka
        ax_button_new = plt.axes([0.15, 0.02, 0.15, 0.04])
        self.button_new = Button(ax_button_new, 'Nova početna tačka', 
                                color='lightgreen', hovercolor='green')
        self.button_new.on_clicked(self.on_new_start)
        
        # Dugme: Jedan korak
        ax_button_step = plt.axes([0.32, 0.02, 0.15, 0.04])
        self.button_step = Button(ax_button_step, 'Jedan korak', 
                                  color='lightblue', hovercolor='blue')
        self.button_step.on_clicked(self.on_step)
        
        # Dugme: Kompletno pretraživanje
        ax_button_complete = plt.axes([0.49, 0.02, 0.15, 0.04])
        self.button_complete = Button(ax_button_complete, 'Do kraja', 
                                      color='lightyellow', hovercolor='yellow')
        self.button_complete.on_clicked(self.on_complete)
        
        # Slider: Delta (veličina koraka)
        ax_slider = plt.axes([0.7, 0.03, 0.2, 0.02])
        self.slider_delta = Slider(ax_slider, 'Delta (Δx):', 
                                   0.1, 2.0, valinit=0.5, valstep=0.1)
        self.slider_delta.on_changed(self.on_delta_changed)
        
    def on_new_start(self, event):
        """Postavi novu slučajnu početnu tačku"""
        # Slučajna početna tačka
        x0 = [np.random.uniform(self.x_range[0], self.x_range[1]),
              np.random.uniform(self.x_range[0], self.x_range[1])]
        
        self.current_solution = x0
        self.history = [x0]
        self.iteration = 0
        self.finished = False
        
        print(f"\nNova početna tačka: x = [{x0[0]:.3f}, {x0[1]:.3f}], f(x) = {objective_function(x0):.3f}")
        
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
        self.current_neighbors = neighbors
        
        # Evaluiraj susjedne tačke
        neighbor_values = [objective_function(n) for n in neighbors]
        current_value = objective_function(self.current_solution)
        
        # Nađi najbolju susjednu tačku (Steepest Descent)
        best_neighbor_idx = np.argmin(neighbor_values)
        best_neighbor = neighbors[best_neighbor_idx]
        best_value = neighbor_values[best_neighbor_idx]
        
        print(f"\nIteracija {self.iteration + 1}:")
        print(f"  Trenutno: x = [{self.current_solution[0]:.3f}, {self.current_solution[1]:.3f}], f(x) = {current_value:.3f}")
        print(f"  Najbolji susjed: x = [{best_neighbor[0]:.3f}, {best_neighbor[1]:.3f}], f(x) = {best_value:.3f}")
        
        # Provjeri uslov zaustavljanja
        if best_value >= current_value:
            print("  → LOKALNI MINIMUM PRONAĐEN!")
            self.finished = True
        else:
            # Pomjeri se na bolju tačku
            self.current_solution = best_neighbor
            self.history.append(best_neighbor)
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
            
    def update_plot(self):
        """Ažuriraj grafički prikaz"""
        
        # Obriši prethodne elemente
        if self.scatter_current:
            self.scatter_current.remove()
        if self.scatter_neighbors:
            self.scatter_neighbors.remove()
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
                                                        label='Trenutna tačka', zorder=8)
            
            # Nacrtaj pravougaonik koji predstavlja okolinu
            rect_size = self.delta * 2
            rect = patches.Rectangle((self.current_solution[0] - self.delta, 
                                     self.current_solution[1] - self.delta),
                                    rect_size, rect_size,
                                    linewidth=2, edgecolor='orange', 
                                    facecolor='orange', alpha=0.1, linestyle='--')
            self.neighborhood_patch = self.ax_main.add_patch(rect)
        
        # Nacrtaj susjedne tačke
        if self.current_neighbors:
            neighbors_x = [n[0] for n in self.current_neighbors]
            neighbors_y = [n[1] for n in self.current_neighbors]
            self.scatter_neighbors = self.ax_main.scatter(neighbors_x, neighbors_y, 
                                                         c='orange', s=80, marker='s',
                                                         edgecolors='darkorange', 
                                                         linewidths=1.5,
                                                         label='Susjedne tačke (okolina)', 
                                                         zorder=7, alpha=0.7)
        
        self.ax_main.legend(loc='upper right', fontsize=10)
        self.fig.canvas.draw_idle()
        
    def update_info_text(self):
        """Ažuriraj info tekst"""
        self.ax_info.clear()
        self.ax_info.axis('off')
        
        if self.current_solution is None:
            info_text = "Pritisnite 'Nova početna tačka' da počnete"
        else:
            x = self.current_solution
            f_x = objective_function(x)
            
            info_text = f"Iteracija: {self.iteration}\n"
            info_text += f"Trenutna tačka: x = [{x[0]:.4f}, {x[1]:.4f}]\n"
            info_text += f"Vrijednost: f(x) = {f_x:.4f}\n"
            info_text += f"Udaljenost od globalnog: {np.sqrt(x[0]**2 + x[1]**2):.4f}\n"
            
            if self.finished:
                info_text += "\n✓ LOKALNI MINIMUM PRONAĐEN!"
            else:
                info_text += "\nPretraživanje u toku..."
                
            # Pseudokod
            info_text += "\n\nPseudokod:\n"
            info_text += "x ← x⁰\n"
            info_text += "repeat\n"
            info_text += "  Ω' ← ∅\n"
            info_text += "  repeat\n"
            info_text += "    izabrati x' ∈ N(x,δ)\n"
            info_text += "    if f(x') < f(x):\n"
            info_text += "      uvrstiti x' u Ω'\n"
            info_text += "  x ← najbolje(Ω')\n"
            info_text += "until f(x) nije poboljšano"
        
        self.ax_info.text(0.05, 0.95, info_text, transform=self.ax_info.transAxes,
                         fontsize=9, verticalalignment='top', family='monospace',
                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        self.fig.canvas.draw_idle()
        
    def on_delta_changed(self, val):
        """Promjena delta parametra"""
        self.delta = val
        if self.current_solution:
            self.update_plot()

# Pokreni demo
if __name__ == "__main__":
    print("="*80)
    print("Lokalno pretraživanje (Local Search) - Demo")
    print("="*80)
    print("\nOsnovno lokalno pretraživanje (Steepest Descent)")
    print("\nKontrole:")
    print("  - 'Nova početna tačka': Odaberi slučajnu početnu tačku")
    print("  - 'Jedan korak': Izvrši jednu iteraciju")
    print("  - 'Do kraja': Izvrši pretraživanje dok ne nađe lokalni minimum")
    print("  - Slider: Promijeni veličinu koraka (delta)")
    print("\nAlgoritam:")
    print("  1. Započni od početne tačke x⁰")
    print("  2. Generiši okolinu N(x,δ)")
    print("  3. Evaluiraj sve susjedne tačke")
    print("  4. Pomjeri se na najbolju susjednu tačku (ako postoji poboljšanje)")
    print("  5. Ponovi dok ne nađeš lokalni minimum")
    print("="*80)
    
    demo = LocalSearchDemo()
    plt.show()
