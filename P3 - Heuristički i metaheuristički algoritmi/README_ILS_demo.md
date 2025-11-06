# ILS (Iterated Local Search) Demo - Dokumentacija

## 📊 Pregled

Interaktivna Python aplikacija koja demonstrira **Iterated Local Search (ILS)** algoritam na Rastrigin funkciji sa inteligentnom perturbacijom kroz kvadratnu interpolaciju.

---

## 🎯 Koncept

### Rastrigin funkcija
```
f(x) = A + x² - A·cos(2πx)
```
- **Globalni minimum**: x = 0, f(0) = 0
- **Karakteristika**: Multimodalna - ima mnogo lokalnih minimuma
- **Težina**: Lako zaglibiti u lokalnom minimumu

### ILS sa kvadratnom interpolacijom

**Klasični ILS**:
1. Nađi lokalni minimum
2. Perturbuj rješenje (slučajno)
3. Ponovi

**Naša pametna varijanta**:
1. **Ciklus 1**: Slučajna početna tačka → pronađi lokalni minimum
2. **Ciklus 2**: Nova slučajna početna tačka → pronađi lokalni minimum
3. **Ciklus 3+**: 
   - Napravi kvadratnu interpolaciju kroz pronađene minimume
   - Minimum interpolacije = nova početna tačka
   - Pronađi lokalni minimum od te tačke

**Zašto je ovo pametno?**
- Koristi informacije iz prethodnih pretraživanja
- Predviđa gdje bi mogao biti globalni minimum
- Brže konvergira nego slučajna perturbacija

---

## 🖥️ Interaktivna aplikacija

### Pokretanje:
```bash
python ils_rastrigin_demo.py
```

### Kontrole:

| Dugme | Funkcija |
|-------|----------|
| **Pokreni ILS** | Automatski izvrši sve cikluse sa pauzama |
| **Jedan korak** | Izvrši jedan ciklus manuelno (korak-po-korak) |
| **Reset** | Ponovo pokreni algoritam |
| **Slider** | Promijeni broj ciklusa (1-20) |

### Vizuelni elementi:

| Element | Boja | Značenje |
|---------|------|----------|
| Plava linija | Plava | Rastrigin funkcija |
| Zelena linija | Zelena isprekidana | Globalni minimum (x=0) |
| Crveni krug | Crveni | Pronađeni lokalni minimumi |
| Narandžasti kvadrat | Narandžasti | Početna tačka pretraživanja |
| Ljubičasta strelica | Ljubičasta | Putanja od početne do minimuma |
| Zelena isprekidana | Zelena | Kvadratna interpolacija |
| Zelena zvijezda | Lime/zelena | Predviđena početna tačka |

---

## 📸 Automatska demonstracija

### Pokretanje:
```bash
python ils_auto_demo.py
```

### Generisane slike:

1. **ils_demo_cycle_1.png** - Prvi ciklus (slučajna početna tačka)
2. **ils_demo_cycle_2.png** - Drugi ciklus (još jedna slučajna)
3. **ils_demo_cycle_3.png** - Treći ciklus (početak interpolacije)
4. **ils_demo_cycle_4.png** - Četvrti ciklus
5. **ils_demo_cycle_5.png** - Peti ciklus
6. **ils_demo_cycle_6.png** - Šesti ciklus
7. **ils_demo_all_cycles.png** - Uporedni prikaz svih 6 ciklusa

---

## 🔬 Rezultati demo izvršavanja

### Pronađeni lokalni minimumi:

| Ciklus | Početna tačka | Lokalni minimum | f(x) | Komentar |
|--------|---------------|-----------------|------|----------|
| 1 | -1.2546 | **0.0000** | **0.0000** | ✅ Pronađen globalni! |
| 2 | 4.5071 | 3.9798 | 15.9192 | Desni lokalni |
| 3 | -5.0000 (interp) | -3.9798 | 15.9192 | Lijevi lokalni |
| 4 | -0.0050 (interp) | 0.9950 | 0.9950 | Blizu globalnog |
| 5 | -0.0050 (interp) | 0.9950 | 0.9950 | Ponovo isti |
| 6 | -5.0000 (interp) | -3.9798 | 15.9192 | Ponovo lijevi |

### Finalni rezultat:
- 🎯 **Najbolji minimum**: x = 0.0000, f(x) = 0.0000
- ✅ **Tačnost**: 100.00%
- 🏆 **Globalni minimum pronađen u prvom ciklusu!**

---

## 💡 Ključni koncepti

### 1. Lokalno pretraživanje
```python
def find_local_minimum(x_start, x_range):
    # Ograniči pretragu na region oko x_start
    local_range = 2.0
    x_min = x_start - local_range
    x_max = x_start + local_range
    
    # Koristi scipy.optimize
    result = minimize_scalar(f, bounds=(x_min, x_max), method='bounded')
    return result.x, result.fun
```

### 2. Kvadratna interpolacija
```python
def quadratic_interpolation(local_minima):
    # Uzmi posljednje 2-3 tačke
    points = local_minima[-3:]
    
    x_points = [p[0] for p in points]
    y_points = [p[1] for p in points]
    
    # Lagrange interpolacija
    poly = lagrange(x_points, y_points)
    
    # Nađi minimum polinoma
    x_min = argmin(poly(x_dense))
    return x_min
```

### 3. ILS glavna petlja
```python
for cycle in range(n_cycles):
    if cycle < 2:
        x_start = random()  # Slučajno
    else:
        x_start = quadratic_interpolation(minima)  # Pametno
    
    x_min = find_local_minimum(x_start)
    minima.append(x_min)
```

---

## 🎓 Primjena u prezentaciji

### Scenario 1: Demonstracija ILS koncepta
1. Pokreni interaktivnu aplikaciju
2. Klikni "Jedan korak" nekoliko puta
3. Objasni:
   - "Evo kako ILS radi..."
   - "Prvi put - slučajno"
   - "Drugi put - slučajno"
   - "Treći put - pametna interpolacija!"

### Scenario 2: Poređenje sa običnim lokalnim pretraživanjem
1. Pokaži kako obično LS zapada u prvi lokalni minimum
2. Pokaži kako ILS izbegava taj problem
3. "ILS pokušava više puta iz različitih tačaka"

### Scenario 3: Pokazati vrednost interpolacije
1. Pokreni demo 2-3 puta
2. Uporedi rezultate sa/bez interpolacije
3. "Interpolacija predviđa gdje bi mogao biti globalni minimum"

---

## ⚙️ Tehnički detalji

### Zavisnosti:
```bash
pip install numpy matplotlib scipy
```

### Parametri:
- **x_range**: (-5, 5) - Opseg pretrage
- **A**: 10 - Parametar Rastrigin funkcije
- **n_cycles**: 1-20 (podešivo) - Broj ILS ciklusa
- **local_range**: 2.0 - Opseg lokalnog pretraživanja

### Struktura koda:
```
ILSDemo/
├── __init__()              # Inicijalizacija
├── setup_figure()          # Setup GUI
├── find_local_minimum()    # Lokalno pretraživanje
├── quadratic_interpolation() # Interpolacija
├── perform_one_cycle()     # Jedan ILS ciklus
├── update_plot()           # Ažuriraj prikaz
└── event handlers          # Kontrole
```

---

## 📚 Dodatne napomene

### Prednosti ove varijante ILS:
✅ Koristi znanje iz prethodnih iteracija  
✅ Brža konvergencija od slučajne perturbacije  
✅ Vizuelno intuitivna (vidi se interpolacija)  
✅ Pogodna za edukaciju  

### Ograničenja:
⚠️ Kvadratna interpolacija može loše predvidjeti za 3+ lokalnih minimuma  
⚠️ Zavisi od kvaliteta prvih nekoliko slučajnih tačaka  
⚠️ Ne garantuje pronalaženje globalnog minimuma  

### Moguća poboljšanja:
1. Dodati "diversification" - ako se dugo ne poboljšava, probaj vrlo udaljenu tačku
2. Koristiti veći stepen polinoma (kubna interpolacija)
3. Dodati "tabu" mehanizam - ne vraćaj se na već istražene regije

---

## 🔗 Veza sa drugim materijalima

- **Slajd 18-24**: Lokalno pretraživanje - teorija
- **Slajd 26-27**: ILS algoritam - teorija
- **Ova demo**: ILS algoritam - praksa

---

## 📁 Lista fajlova

```
/home/claude/
├── ils_rastrigin_demo.py      # Interaktivna aplikacija
└── ils_auto_demo.py           # Automatska demonstracija

/mnt/user-data/outputs/
├── ils_demo_cycle_1.png       # Ciklus 1
├── ils_demo_cycle_2.png       # Ciklus 2
├── ils_demo_cycle_3.png       # Ciklus 3
├── ils_demo_cycle_4.png       # Ciklus 4
├── ils_demo_cycle_5.png       # Ciklus 5
├── ils_demo_cycle_6.png       # Ciklus 6
├── ils_demo_all_cycles.png    # Uporedno
└── README_ILS_demo.md         # Ova dokumentacija
```

---

## ✅ Checklist za prezentaciju

- [ ] Testirano na projektoru (čitljivost)
- [ ] Pripremljeni primjeri sa različitim seed-ovima
- [ ] Backup slike (ako aplikacija ne radi)
- [ ] Objašnjenje interpolacije spremno
- [ ] Poređenje sa običnim LS spremno

---

**Autor**: Dr Samim Konjicija  
**Kurs**: Optimizacija resursa  
**Datum**: Novembar 2025

**Napomena**: Ova demo aplikacija je edukacijski alat. Za proizvodne primjene, koristiti specijalizovane biblioteke kao što su scikit-optimize ili scipy.optimize.
