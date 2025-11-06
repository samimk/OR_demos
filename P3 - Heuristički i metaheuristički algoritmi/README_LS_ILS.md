# Lokalno pretraživanje i ILS - Demo aplikacije

## 📋 Pregled

Dvije interaktivne Python aplikacije koje demonstriraju:
1. **Osnovno lokalno pretraživanje** (Local Search) - Slajdovi 18-24
2. **Ponavljano lokalno pretraživanje** (ILS) sa kvadratnom interpolacijom - Slajdovi 26-27

---

## 🎯 Aplikacija 1: Osnovno lokalno pretraživanje

### Fajl: `local_search_demo.py`

### Opis
Demonstrira osnovno lokalno pretraživanje (Steepest Descent) na kvadratnoj funkciji f(x) = x₁² + x₂².

### Algoritam

```
x ← x⁰
v ← f(x⁰)
repeat
  Ω' ← ∅
  repeat
    izabrati x' ∈ N(x, δ)
    v' ← f(x')
    if v' < v then
      uvrstiti x' u Ω'
    endif
  until ZavrsenoPretrazivanjeOkoline(x)
  x ← IzborNovogRjesenja(Ω')
until UslovZaustavljanja()
```

### Kontrole

| Dugme | Funkcija |
|-------|----------|
| **Nova početna tačka** | Odaberi slučajnu početnu tačku x⁰ |
| **Jedan korak** | Izvrši jednu iteraciju algoritma |
| **Do kraja** | Automatski izvrši dok ne nađe lokalni minimum |
| **Slider (Δx)** | Promijeni veličinu koraka (0.1-2.0) |

### Vizuelni elementi

| Element | Boja | Značenje |
|---------|------|----------|
| Zelena zvijezda | Zelena | Globalni minimum (0,0) |
| Crveni krug | Crveni | Trenutna tačka |
| Narandžasti kvadrati | Narandžasti | Susjedne tačke (okolina) |
| Isprekidani pravougaonik | Narandžasti | Granica okoline N(x,δ) |
| Ljubičasta linija | Ljubičasta | Putanja algoritma |

### Definicija okoline

**N(x,δ)**: 8 tačaka oko trenutne tačke x
- ±δ po x₁ osi
- ±δ po x₂ osi  
- ±δ po dijagonalama

Ukupno 8 susjednih tačaka u svakoj iteraciji.

### Primjer izvršavanja

```
Iteracija 1:
  Trenutno: x = [3.456, -2.123], f(x) = 16.456
  Najbolji susjed: x = [2.956, -1.623], f(x) = 11.378
  → Pomak na bolju tačku

Iteracija 2:
  Trenutno: x = [2.956, -1.623], f(x) = 11.378
  Najbolji susjed: x = [2.456, -1.123], f(x) = 7.294
  → Pomak na bolju tačku

...

Iteracija N:
  Trenutno: x = [0.123, 0.087], f(x) = 0.023
  Najbolji susjed: x = [0.123, 0.087], f(x) = 0.023
  → LOKALNI MINIMUM PRONAĐEN!
```

### Ključne karakteristike

✅ **Prednosti:**
- Jednostavan za razumijevanje
- Garantovano nalazi lokalni minimum
- Brza konvergencija za konveksne funkcije

⚠️ **Nedostaci:**
- Zapada u lokalne minimume
- Ne pronalazi globalni minimum na multimodalnim funkcijama
- Zavisi od izbora početne tačke

---

## 🎯 Aplikacija 2: ILS sa kvadratnom interpolacijom

### Fajl: `ils_demo_corrected.py`

### Opis
Demonstrira ILS algoritam na Rastrigin funkciji sa **inteligentnom perturbacijom** kroz kvadratnu interpolaciju.

### Rastrigin funkcija
```
f(x) = A + x² - A·cos(2πx)
A = 10
Globalni minimum: x = 0, f(0) = 0
```

### KLJUČNA IDEJA

**Kvadratna interpolacija prolazi kroz PRONAĐENE LOKALNE MINIMUME!**

```
Ciklus 1: x_start = slučajno (npr. x=-5)
          → Lokalni minimum pronađen: x₁ ≈ -4, f(x₁) ≈ 16

Ciklus 2: x_start = slučajno (npr. x=3.5)  
          → Lokalni minimum pronađen: x₂ ≈ 4, f(x₂) ≈ 16

Ciklus 3: Kvadratna f-ja kroz (x₁, f(x₁)) i (x₂, f(x₂))
          → Minimum kvadratne f-je: x_start ≈ 1.5
          → Lokalni minimum pronađen: x₃ ≈ 1, f(x₃) ≈ 1

Ciklus 4: Kvadratna f-ja kroz (x₁, f(x₁)), (x₂, f(x₂)), (x₃, f(x₃))
          → Minimum kvadratne f-je: x_start ≈ 0.2
          → Lokalni minimum pronađen: x₄ ≈ 0, f(x₄) ≈ 0 ✓
```

### Algoritam

```
x* ← LS(x⁰)
Ωmem ← ∅
repeat
  if ciklus < 2:
    x' ← slučajna tačka
  else:
    poly ← kvadratna_interpolacija(local_minima)
    x' ← argmin(poly)
  endif
  
  x'* ← LS(x')
  x* ← UslovPrihvatanja(x*, x'*, Ωmem)
  Ωmem ← AzuriranjeMemorije(x*, x'*, Ωmem)
until UslovZaustavljanja()
```

### Kontrole

| Dugme | Funkcija |
|-------|----------|
| **Pokreni ILS** | Automatski izvrši sve cikluse (sa pauzama) |
| **Jedan korak** | Izvrši jedan ILS ciklus manuelno |
| **Reset** | Resetuj algoritam |
| **Slider** | Promijeni broj ciklusa (1-20) |

### Vizuelni elementi

| Element | Boja | Značenje |
|---------|------|----------|
| Plava linija | Plava | Rastrigin funkcija |
| Zelena isprekidana | Zelena | Globalni minimum (x=0) |
| Crveni krugovi | Crveni | Pronađeni lokalni minimumi |
| Narandžasti kvadrati | Narandžasti | Početne tačke pretraživanja |
| Ljubičaste strelice | Ljubičaste | Putanje: start → minimum |
| Zelena isprekidana kriva | Zelena | Kvadratna interpolacija |
| Lime zvijezda | Lime/zelena | Predviđena početna tačka |

### Matematički zapis

**Kvadratna interpolacija** (Lagrange):

Za N pronađenih minimuma: {(x₁, f₁), (x₂, f₂), ..., (xₙ, fₙ)}

```
P(x) = Σᵢ fᵢ · Lᵢ(x)

gdje je: Lᵢ(x) = Πⱼ≠ᵢ (x - xⱼ)/(xᵢ - xⱼ)
```

Nova početna tačka:
```
x_start = argmin P(x)
          x∈[-5,5]
```

### Primjer izvršavanja

```
================================================================================
Ciklus 1: Slučajna početna tačka x_start = -1.2546
  Pronađen lokalni minimum: x_min = 0.0000, f(x_min) = 0.0000

Ciklus 2: Slučajna početna tačka x_start = 4.5071
  Pronađen lokalni minimum: x_min = 3.9798, f(x_min) = 15.9192

Ciklus 3: Interpolirana početna tačka x_start = -5.0000
  Interpolacija kroz minimume: ['(-0.00, 0.00)', '(3.98, 15.92)']
  Pronađen lokalni minimum: x_min = -3.9798, f(x_min) = 15.9192

Ciklus 4: Interpolirana početna tačka x_start = -0.0050
  Interpolacija kroz minimume: ['(-0.00, 0.00)', '(3.98, 15.92)', '(-3.98, 15.92)']
  Pronađen lokalni minimum: x_min = 0.9950, f(x_min) = 0.9950

...

🎯 Najbolji minimum: x = 0.0000, f(x) = 0.0000
✅ Tačnost: 100.00%
================================================================================
```

### Ključne karakteristike

✅ **Prednosti nad običnim LS:**
- Koristi informacije iz prethodnih iteracija
- Inteligentno predviđa gdje bi mogao biti globalni minimum
- Brža konvergencija od slučajne perturbacije
- Vizuelno intuitivna (vidi se kvadratna funkcija)

✅ **Prednosti nad običnim ILS sa slučajnom perturbacijom:**
- Umjesto potpuno slučajne nove početne tačke
- Kvadratna interpolacija "uči" iz pronađenih minimuma
- Minimum interpolacije = obrazovana pretpostavka o globalnom minimumu

⚠️ **Ograničenja:**
- Kvadratna interpolacija može loše predvidjeti za 3+ minimuma
- Ne garantuje pronalaženje globalnog minimuma
- Zavisi od kvaliteta prva 2 slučajne početne tačke

---

## 🔬 Poređenje algoritama

| Karakteristika | Lokalno pretraživanje | ILS |
|----------------|----------------------|-----|
| **Broj pokušaja** | 1 | Više (konfigurisano) |
| **Početna tačka** | Fiksirana x⁰ | Više različitih tačaka |
| **Perturbacija** | Nema | Kvadratna interpolacija |
| **Memorija** | Nema | Pamti pronađene minimume |
| **Pronalazi globalni** | Samo ako je x⁰ blizu | Većaвероватноћа |
| **Brzina** | Brzo | Sporije (više LS izvršavanja) |

---

## 🎓 Za prezentaciju

### Scenario 1: Demonstracija osnovnog LS
1. Pokreni `local_search_demo.py`
2. Klikni "Nova početna tačka"
3. Objasni: "Evo početne tačke... algoritam gleda oko sebe..."
4. Klikni "Jedan korak" nekoliko puta
5. "Vidite kako se pomjera prema minimumu... uvijek bira najbolju susjednu tačku"
6. Klikni "Do kraja"
7. "I zapao je u lokalni minimum! Ne zna da postoji bolji."

### Scenario 2: Problem sa LS
1. "Problem: Lokalno pretraživanje uvijek zapada u prvi lokalni minimum"
2. Promijeni početnu tačku par puta
3. "Svaki put različit rezultat... Kako to riješiti?"

### Scenario 3: Uvod u ILS
1. Pokreni `ils_demo_corrected.py`
2. "ILS: Pokušaj više puta, ali pameti!"
3. Klikni "Jedan korak" 2 puta
4. "Prva dva puta: slučajno, kao obično LS"

### Scenario 4: Kvadratna interpolacija
1. Nastavi sa "Jedan korak"
2. "Treći put: INTERPOLACIJA! Vidi zelenu liniju?"
3. "Prolazi kroz dva pronađena minimuma"
4. "Minimum te zelene linije = nova početna tačka"
5. "Algoritam PREDVIĐA gdje bi mogao biti globalni minimum!"

### Scenario 5: Pokazati vrednost
1. "Umjesto slučajnog... koristimo znanje iz prošlosti"
2. Klikni "Reset" pa "Pokreni ILS"
3. "Gledajte kako konvergira prema x=0!"

---

## 📊 Dijagram toka

### Lokalno pretraživanje
```
┌─────────────┐
│    Start    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  x ← x⁰     │
└──────┬──────┘
       │
   ┌───▼────┐
   │Generiši│
   │okolinu │
   │ N(x,δ) │
   └───┬────┘
       │
       ▼
   ┌────────────┐
   │ Evaluiraj  │
   │susjedne    │◄────┐
   │tačke       │     │
   └───┬────────┘     │
       │              │
       ▼              │
   ┌───────────┐      │
   │Postoji    │  Ne  │
   │bolja?     ├──────┘
   └───┬───────┘
       │Da
       ▼
   ┌────────────┐
   │x ← najbolja│
   │tačka       │
   └───┬────────┘
       │
       ▼
┌──────────────┐
│Lokalni       │
│minimum       │
│pronađen!     │
└──────────────┘
```

### ILS
```
┌─────────────┐
│    Start    │
└──────┬──────┘
       │
  ┌────▼─────┐
  │ Ciklus 1:│
  │ slučajno │
  └────┬─────┘
       │
       ▼
  ┌──────────┐
  │    LS    │──► Minimum 1
  └────┬─────┘
       │
  ┌────▼─────┐
  │ Ciklus 2:│
  │ slučajno │
  └────┬─────┘
       │
       ▼
  ┌──────────┐
  │    LS    │──► Minimum 2
  └────┬─────┘
       │
  ┌────▼─────────────┐
  │ Ciklus 3+:       │
  │ Interpolacija    │
  │ kroz minimume    │
  └────┬─────────────┘
       │
       ▼
  ┌──────────────────┐
  │ Minimum          │
  │ interpolacije =  │
  │ x_start          │
  └────┬─────────────┘
       │
       ▼
  ┌──────────┐
  │    LS    │──► Novi minimum
  └────┬─────┘
       │
       ▼
  ┌───────────┐
  │Ponovi dok │
  │ne dođeš do│
  │N ciklusa  │
  └───────────┘
```

---

## 💾 Fajlovi

```
/mnt/user-data/outputs/
├── local_search_demo.py          # Demo osnovnog LS
├── ils_demo_corrected.py         # ILS sa kvadratnom interpolacijom
└── README_LS_ILS.md             # Ova dokumentacija
```

---

## 🚀 Pokretanje

```bash
# Osnovno lokalno pretraživanje
python local_search_demo.py

# ILS sa kvadratnom interpolacijom
python ils_demo_corrected.py
```

---

## 📚 Veza sa predavanjem

- **Slajd 18-24**: Osnovno lokalno pretraživanje → `local_search_demo.py`
- **Slajd 26-27**: ILS algoritam → `ils_demo_corrected.py`

---

## ✅ Checklist za prezentaciju

- [ ] Testirano na projektoru
- [ ] Pripremljeni primjeri sa različitim seed-ovima
- [ ] Backup slike (ako GUI ne radi)
- [ ] Objašnjenje interpolacije spremno
- [ ] Razlika LS vs ILS jasna

---

**Napomena**: Ove demo aplikacije su edukacijski alat za razumijevanje lokalnog pretraživanja i ILS algoritma. Za proizvodne primjene, koristiti specijalizovane biblioteke.
