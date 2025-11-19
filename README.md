# OR_demos

Demo aplikacije za predmet **Optimizacija resursa**, Red. prof. dr Samim Konjicija, 2025. godina

## Pregled demo aplikacija

### P1 - Uvod

#### laptop_optimizer_demo.html
Interaktivna web aplikacija za demonstraciju problema optimizacije izbora laptopa. Prikazuje osnovne koncepte optimizacije kroz praktičan primjer gdje se biraju laptopi sa ciljem maksimiziranja performansi uz ograničen budžet.

---

### P2 - Klasični algoritmi optimizacije

#### dfs_bfs_demo.html
Web demo koji vizualizira algoritme pretraživanja grafa:
- **DFS (Depth-First Search)** - pretraživanje u dubinu
- **BFS (Breadth-First Search)** - pretraživanje u širinu

Omogućava interaktivno iscrtavanje grafa i praćenje koraka algoritama.

#### dp_demo.html
Demonstracija **dinamičkog programiranja** kroz klasične probleme:
- Problem ranca (Knapsack problem)
- Fibonacci brojevi
- Najduža zajednička podsekvenca (LCS)

Prikazuje tablice memorisanja i optimalne podstrukture.

#### gradient_search_demo.html / gradient_search_demo.py
Vizualizacija algoritama **gradijentnog pretraživanja**:
- Gradient Descent
- Prikaz konvergencije prema lokalnim/globalnim optimumima
- Dostupno kao web i Python/Tkinter aplikacija

---

### P3 - Heuristički i metaheuristički algoritmi

#### local_search_demo.py
Python/Tkinter aplikacija koja demonstrira **lokalnu pretragu** (Hill Climbing):
- Iterativno poboljšanje trenutnog rješenja
- Vizualizacija pretraživanja u 2D prostoru
- Prikaz lokalnih optimuma

#### ils_demo.py / ils_demo_corrected.py
**Iterated Local Search (ILS)** demo aplikacije:
- Kombinacija lokalne pretrage i perturbacije
- Vizualizacija eskejpa iz lokalnih optimuma
- Praćenje najbolje pronađene tačke

---

### P4 - Tabu pretraživanje

#### tabu_search_demo.html / tabu_search_demo.py
Demonstracija **Tabu Search** algoritma:
- Tabu lista za sprječavanje cikličnog vraćanja
- Short-term i long-term memorija
- Vizualizacija pretraživanja uz zabranjene poteze
- Dostupno kao web i Python/Tkinter aplikacija

#### tabu_knapsack_demo.html
Primjena Tabu Search algoritma na **problem ranca**:
- Demonstracija rada sa diskretnim problemima
- Aspiracioni kriterijum
- Praćenje istorije rješenja

---

### P5 - Simulirano hlađenje

#### simulated_annealing_demo.html / simulated_annealing_demo.py
**Simulated Annealing** demo sa poređenjem algoritama:
- Lokalna pretraga
- Tabu pretraživanje
- Simulirano hlađenje

Omogućava konfigurisanje:
- Temperature parametara (T₀, Tₘᵢₙ)
- Sheme hlađenja (Geometrijsko, Logaritamsko, Linearno)
- Vizualizacija funkcija i historije pretraživanja

---

### P6 - GA - osnove

#### ga_visualization.html
Kompletna web aplikacija za demonstraciju **genetičkog algoritma**:
- Populacijska optimizacija
- Fitness funkcije (kvadratna, Rastrigin, Ackley, Griewank, Levy)
- Operatori selekcije (Ruletski točak, Ranking, Turnirska)
- Operatori ukrštanja (Jedna tačka, Dvije tačke, Uniformno)
- Mutacija sa konfigurisanom vjerovatnoćom
- Elitizam
- Cilj optimizacije (maksimizacija/minimizacija)
- Graf evolucije fitnessa kroz generacije
- Prikaz populacije hromozoma

#### ga_operatori.html
Interaktivna vizualizacija **genetičkih operatora**:
- Operator ukrštanja (one-point, two-point, uniform)
- Operator mutacije
- 2D vizualizacija u ravni
- Binarna reprezentacija sa prikaz izmijenjenih bita
- Konfigurisanje vjerovatnoća (pₑ, pₘ)
- Prikaz više pokušaja operacija sa statistikom

#### genetic_algorithm_demo.py
Unificirana Python/Tkinter aplikacija koja integriše:
- Lokalnu pretragu
- Tabu pretraživanje
- Simulirano hlađenje
- **Genetički algoritam** sa svim parametrima

GA specifične opcije:
- Veličina populacije
- Operatori ukrštanja i selekcije
- Parametri mutacije
- Elitizam
- Grafički prikaz evolucije fitnessa nakon završetka

---

## Pokretanje demo aplikacija

### Web aplikacije (HTML)
Otvorite HTML fajlove direktno u web browseru (preporučuje se Chrome, Firefox ili Edge).

### Python aplikacije
Potreban Python 3.x sa bibliotekama:
- `tkinter` (uobičajeno dolazi sa Python instalacijom)
- `numpy`
- `matplotlib`

Pokretanje:
```bash
python3 <naziv_aplikacije>.py
```

---

## Licenca i kontakt

Materijali kreirani za potrebe predmeta Optimizacija resursa.
© 2025. Red. prof. dr Samim Konjicija
