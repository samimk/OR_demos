# Poređenje Test Funkcija za ILS Demo

## Brza referenca

| Funkcija | Tip | Globalni min (x) | Težina | Asimetrija | Preporučeni ciklusi |
|----------|-----|------------------|--------|------------|-------------------|
| **Rastrigin** | Periodična | 2.0 | ⭐⭐ | Niska | 5-7 |
| **Levy** | Kompleksna | 1.0 | ⭐⭐⭐⭐ | Visoka | 7-10 |
| **Griewank** | Hibridna | 1.5 | ⭐⭐⭐ | Srednja | 5-8 |
| **Ackley** | Eksponencijalna | 2.0 | ⭐⭐⭐⭐ | Visoka | 8-12 |

## Vizuelne karakteristike

### Rastrigin funkcija
```
Oblik: ~\/~\/~\/~  (regularni talasi)
│     ╱╲    ╱╲    ╱╲    ╱╲
│    ╱  ╲  ╱  ╲  ╱  ╲  ╱  ╲
│___╱____╲╱____╲╱____╲╱____╲___
```
- Periodična struktura
- Ravnomerno raspoređeni lokalni minimumi
- Predvidiva, ali izazovna

### Levy funkcija
```
Oblik: ~\~/\/~\~  (neregularni talasi)
│   ╱╲ ╱╲╱╲  ╱╲
│  ╱  V  │  ╱  ╲
│_╱_____╱╲_╱____╲___
```
- Neregularna struktura
- Različite dubine lokalnih minimuma
- Asimetrična raspodela

### Griewank funkcija
```
Oblik: ~\/\/\/\~  (glatka sa talasima)
│    _/‾‾‾‾‾‾\_
│   / /\/\/\/\ \
│__/___________\___
```
- Kvadratna osnova sa kosinusnim oscilacijama
- Asimetrična zbog linearnog člana
- Srednje izazovna

### Ackley funkcija
```
Oblik: ‾‾‾\___/‾‾‾  (ravan-strm-ravan)
│‾‾‾‾‾‾\ /\  /‾‾‾‾‾‾
│       V  \/
│__________________
```
- Ravna spoljna oblast
- Strm centralni deo sa oscilacijama
- Izazovna za gradijentne metode

## Matematičke definicije

### 1. Rastrigin
```
f(x) = A + (x-2)² - A·cos(2π(x-2))
gde je A = 10
```

### 2. Levy
```
w = 1 + (x-1)/4
f(x) = sin²(πw) + (w-1)²[1 + 10sin²(πw+1)]
```

### 3. Griewank
```
x' = x - 1.5
f(x) = x'²/200 - cos(x'/√1.5) + 1 + 0.1·x'
```

### 4. Ackley
```
x' = x - 2
f(x) = -20·exp(-0.2|x'|) - exp(cos(2πx')) + 20 + e + 0.05·x'
```

## Kada koristiti svaku funkciju?

### Rastrigin - Za početnu demonstraciju
- ✅ Učenje osnovnih koncepata ILS
- ✅ Razumevanje lokalnih vs globalnih minimuma
- ✅ Demonstracija kvadratne aproksimacije

### Levy - Za napredniju analizu
- ✅ Testiranje robusnosti algoritma
- ✅ Demonstracija asimetričnih problema
- ✅ Analiza teških slučajeva optimizacije

### Griewank - Za poređenje sa Rastrigin
- ✅ Demonstracija uticaja skaliranja
- ✅ Analiza uticaja asimetrije
- ✅ Srednja težina između Rastrigin i Levy

### Ackley - Za ekstremne slučajeve
- ✅ Testiranje u ravnim oblastima
- ✅ Demonstracija potrebe za perturbacijom
- ✅ Analiza konvergencije u teškim uslovima

## Tipični rezultati

### Broj koraka do lokalnog minimuma (prosek):
- **Rastrigin**: 50-150 koraka
- **Levy**: 100-300 koraka
- **Griewank**: 60-180 koraka
- **Ackley**: 150-400 koraka

### Uspešnost pronalaženja globalnog minimuma (5 ciklusa):
- **Rastrigin**: ~70%
- **Levy**: ~40%
- **Griewank**: ~60%
- **Ackley**: ~35%

## Preporuke za prezentaciju

1. **Počnite sa Rastrigin** - učenje osnovnih koncepata
2. **Pređite na Griewank** - demonstracija asimetrije
3. **Uporedite sa Levy** - pokazivanje izazova
4. **Završite sa Ackley** - demonstracija ekstremnih slučajeva

## Ključne lekcije po funkciji

### Rastrigin
> "Čak i simetrične funkcije mogu imati mnogo lokalnih minimuma"

### Levy
> "Asimetrija čini problem značajno težim za optimizaciju"

### Griewank
> "Skaliranje i mali detalji mogu mnogo promeniti težinu problema"

### Ackley
> "Ravne oblasti mogu biti podjednako problematične kao i strme"

## Napredne tehnike po funkciji

### Za Rastrigin:
- Koristi regularnu mrežu početnih tačaka
- Manji koraci pretrage (0.005-0.01)

### Za Levy:
- Veći broj ciklusa (10+)
- Agresivnija perturbacija

### Za Griewank:
- Prilagodi veličinu koraka tokom pretrage
- Fokusiraj se na centralnu oblast

### Za Ackley:
- Veliki broj kratkih pokušaja
- Kombinuj sa random restart strategijom
