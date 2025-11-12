# P4 - Tabu pretraživanje

Demo aplikacija za demonstraciju **Tabu pretraživanja** (Tabu Search) i poređenje sa **Lokalnim pretraživanjem**.

## Sadržaj

- `tabu_search_demo.py` - Interaktivna Tkinter aplikacija za demonstraciju algoritama

## Funkcionalnosti

### Algoritmi
1. **Lokalno pretraživanje** - Klasičan algoritam lokalnog pretraživanja koji se zaustavlja kod prvog lokalnog minimuma
2. **Tabu pretraživanje** - Metaheuristički algoritam koji koristi tabu listu za izbjegavanje nedavno posjećenih tačaka

### Test funkcije
- **Kvadratna** funkcija (f(x) = x₁² + x₂²)
- **Rastrigin** funkcija (multimodalna)
- **Ackley** funkcija (multimodalna)
- **Griewank** funkcija (multimodalna)
- **Levy** funkcija (multimodalna)

### Kontrole
- Izbor algoritma (Lokalno / Tabu pretraživanje)
- Izbor test funkcije
- Podešavanje veličine koraka (Delta: 0.1 - 2.0)
- Podešavanje dužine tabu liste (3 - 20)
- Izvršavanje po koracima ili do kraja
- Postavljanje početne tačke klikom ili slučajno

## Tabu pretraživanje

Tabu pretraživanje je metaheuristički algoritam koji proširuje lokalno pretraživanje dodavanjem **tabu liste** - memorije koja pamti nedavno posjećene tačke.

### Ključne karakteristike:
- **Tabu lista**: Pamti posljednjih N posjećenih tačaka (N je dužina tabu liste)
- **Izbjegavanje ciklusa**: Sprječava vraćanje na nedavno posjećene tačke
- **Istraživanje**: Omogućava prelazak na gore rješenje ako to vodi ka neistraženim oblastima
- **Aspiration criterion**: Ako su svi susjedi tabu, dozvoljava najbolji susjed

### Razlike u odnosu na lokalno pretraživanje:
1. Lokalno pretraživanje se **zaustavlja** kod prvog lokalnog minimuma
2. Tabu pretraživanje **nastavlja pretraživanje** i može pobjeći iz lokalnih minimuma
3. Tabu pretraživanje može napraviti **korak prema gorem rješenju** ako to vodi ka neistraženim oblastima

## Pokretanje

```bash
python3 tabu_search_demo.py
```

## Upute za korištenje

1. **Odaberite algoritam** - Lokalno ili Tabu pretraživanje
2. **Odaberite test funkciju** - Jedna od 5 dostupnih funkcija
3. **Podesite parametre**:
   - Delta - veličina koraka u okolini
   - Dužina tabu liste - koliko tačaka pamtiti (samo za tabu search)
4. **Postavite početnu tačku**:
   - Kliknite na grafik, ili
   - Koristite "Slučajan start"
5. **Izvršite pretraživanje**:
   - "Jedan korak" - izvršite jednu iteraciju
   - "Do kraja" - izvršite kompletno pretraživanje

## Vizualizacija

- 🟢 **Zelena zvijezda** - Globalni minimum funkcije
- 🔴 **Crveni krug** - Trenutna tačka
- 🟠 **Narančasti kvadrati** - Susjedne tačke (okolina)
- ❌ **Crveni X** - Tačke u tabu listi (zabranjena područja)
- 💎 **Zeleni dijamant** - Najbolji dozvoljeni susjed
- 💜 **Ljubičasta linija** - Putanja pretraživanja

## Autor

Red. prof. dr Samim Konjicija
Optimizacija resursa
Novembar 2025. godine
