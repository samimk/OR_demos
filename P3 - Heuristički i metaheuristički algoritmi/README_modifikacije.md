# Modifikacije ILS Demo Aplikacije

## Pregled izmena

Ova Python aplikacija je modifikovana tako da sada uključuje:

### 1. **Tri dodatne asimetrične multimodalne funkcije**

Pored Rastrigin funkcije, sada su dostupne još **tri asimetrične test funkcije**:

#### **Levy funkcija**
- **Asimetrična**: Za razliku od Rastrigin funkcije koja je relativno simetrična, Levy funkcija ima asimetričan oblik
- **Multimodalna**: Ima veliki broj lokalnih minimuma
- **Globalni minimum**: x = 1.0, f(1.0) = 0
- **Matematička definicija**:
  ```
  w = 1 + (x - 1) / 4
  f(x) = sin²(πw) + (w-1)²[1 + 10sin²(πw+1)]
  ```

#### **Griewank funkcija**
- **Asimetrična**: Kombinacija kvadratne i kosinusne komponente sa linearnim članom za asimetriju
- **Multimodalna**: Veliki broj lokalnih minimuma, slična Rastrigin funkciji po strukturi
- **Globalni minimum**: x = 1.5, f(1.5) ≈ 0
- **Matematička definicija**:
  ```
  x' = x - 1.5
  f(x) = x'²/200 - cos(x'/√1.5) + 1 + 0.1·x'
  ```
- **Karakteristike**: Lakša za optimizaciju od Levy, ali teža od Rastrigin funkcije

#### **Ackley funkcija**
- **Asimetrična**: Kombinacija eksponencijalnih i trigonometrijskih funkcija
- **Multimodalna**: Karakteristična po gotovo ravnoj spoljašnjoj oblasti sa mnogo lokalnih minimuma u centralnoj regiji
- **Globalni minimum**: x = 2.0, f(2.0) ≈ 0
- **Matematička definicija**:
  ```
  x' = x - 2
  a = 20, b = 0.2, c = 2π
  f(x) = -a·exp(-b·|x'|) - exp(cos(c·x')) + a + e + 0.05·x'
  ```
- **Karakteristike**: Vrlo izazovna za lokalno pretraživanje zbog ravne spoljašnje oblasti

### 2. **Vizualizacija putanja lokalnog pretraživanja**

Aplikacija sada prikazuje **svaki korak** lokalnog pretraživanja od početne tačke do lokalnog minimuma:

- **Obojene putanje**: Svaka putanja lokalnog pretraživanja ima svoju boju
- **Tačkice duž putanje**: Prikazuju pojedinačne korake algoritma
- **Početne tačke**: Označene zelenim kružićima sa zelenim ivicama
- **Lokalni minimumi**: Označeni crvenim kvadratićima sa crvenim ivicama
- **Broj koraka**: U status baru se prikazuje broj koraka potrebnih za dostizanje lokalnog minimuma

### 3. **Novi kontrolni elementi**

- **Izbor funkcije**: Dropdown meni za odabir između 4 funkcije: "rastrigin", "levy", "griewank", i "ackley"
- **Automatsko resetovanje**: Promena funkcije automatski resetuje pretragu
- **Prilagođeni prikaz**: Y-osa i globalni minimum se automatski prilagođavaju odabranoj funkciji

## Upotreba

1. **Pokretanje aplikacije**:
   ```bash
   python3 ils_demo.py
   ```

2. **Odabir funkcije**:
   - Kliknite na dropdown "Funkcija" u gornjem levom uglu
   - Izaberite između: "rastrigin", "levy", "griewank", ili "ackley"

3. **Podešavanje parametara**:
   - **Broj ciklusa**: Broj iteracija ILS algoritma (3-20)
   - **Korak pretrage**: Veličina koraka za lokalno pretraživanje (0.001-0.1)

4. **Pokretanje pretrage**:
   - **"Pokreni ILS"**: Automatski pokreće sve cikluse sa pauzama
   - **"Sljedeći korak"**: Izvršava jedan po jedan ciklus ručno
   - **"Resetuj"**: Briše sve rezultate i vraća na početno stanje

## Karakteristike vizualizacije

### Legenda elemenata:
- 🟢 **Zeleni krugovi** - Početne tačke pretrage
- 🔴 **Crveni kvadrati** - Pronađeni lokalni minimumi
- 🟣🟠🟤 **Obojene linije sa tačkicama** - Putanje lokalnog pretraživanja
- ⭐ **Žuta zvezda** - Globalni minimum funkcije
- 📈 **Isprekidane linije** - Kvadratne aproksimacije (od 3. ciklusa)

### Putanje lokalnog pretraživanja:
Svaka putanja pokazuje:
- Kako algoritam gradijentnog spusta nalazi lokalni minimum
- Broj koraka potrebnih za konvergenciju
- Efektivnost lokalnog pretraživanja za različite početne tačke

## Razlike između funkcija

### Rastrigin funkcija:
- Relativno simetrična
- Periodični lokalni minimumi
- Globalni minimum na x = 2.0
- **Težina**: ⭐⭐ (lakša - referentna funkcija)
- Idealna za demonstraciju osnovnih koncepata ILS

### Levy funkcija:
- **Asimetrična struktura**
- Kompleksnija multimodalna površina
- Globalni minimum na x = 1.0
- **Težina**: ⭐⭐⭐⭐ (teža)
- Izazovna za optimizaciju zbog asimetrije i visoke modalnosti

### Griewank funkcija:
- **Asimetrična** (linearni član)
- Slična Rastrigin strukturi ali sa drugačijim skaliranjem
- Globalni minimum na x = 1.5
- **Težina**: ⭐⭐⭐ (srednja)
- Dobra za demonstraciju uticaja različitih početnih tačaka

### Ackley funkcija:
- **Visoko asimetrična**
- Ravna spoljna oblast sa strmim centralnim delom
- Globalni minimum na x = 2.0
- **Težina**: ⭐⭐⭐⭐ (teža)
- Izazovna zbog kombinacije eksponencijalnih i trigonometrijskih komponenti
- Odlična za testiranje robusnosti algoritama

## Tehnički detalji

### Algoritam lokalnog pretraživanja:
- Gradijentni spust sa adaptivnim korakom
- Numerički izvod za Levy, Griewank i Ackley funkcije (stabilnost)
- Analitički izvod za Rastrigin funkciju (preciznost)
- Praćenje putanje sa ograničenjem broja tačaka (performanse)

### Optimalni parametri po funkciji:
| Funkcija | Preporučeni broj ciklusa | Preporučeni korak |
|----------|-------------------------|-------------------|
| Rastrigin | 5-7 | 0.01 |
| Levy | 7-10 | 0.005 |
| Griewank | 5-8 | 0.01 |
| Ackley | 8-12 | 0.005 |

### Poboljšanja u kodu:
- `objective_function()` - Wrapper za trenutno odabranu funkciju
- `objective_derivative()` - Wrapper za izvod trenutne funkcije
- `local_search()` - Sada vraća tuple (minimum, putanja)
- `search_paths[]` - Lista svih putanja za vizualizaciju

## Dodatne napomene

- Aplikacija se automatski prilagođava veličini prozora
- Status bar prikazuje detaljne informacije o svakom koraku
- Kvadratna aproksimacija se koristi od 3. ciklusa za pametnije biranje novih početnih tačaka
- Greška od globalnog minimuma se prikazuje na kraju pretrage

## Preporuke za eksperimentisanje

1. **Poredite funkcije**: Pokrenite ILS na svim funkcijama sa istim parametrima i uporedite rezultate
2. **Testirajte osetljivost**: Promenite korak pretrage i broj ciklusa da vidite kako utiču na performanse
3. **Posmatrajte putanje**: Obratite pažnju kako različite funkcije vode do različitih obrazaca lokalnog pretraživanja
4. **Analizirajte konvergenciju**: Broj koraka do lokalnog minimuma može mnogo varirati između funkcija

## Edukativna vrednost

Ova aplikacija demonstrira:
- Kako asimetrične funkcije mogu biti teže za optimizaciju
- Važnost izbora dobrih početnih tačaka
- Ulogu kvadratne aproksimacije u metaheurističkim algoritmima
- Razliku između lokalnog i globalnog pretraživanja
- Kako ILS algoritam poboljšava jednostavno lokalno pretraživanje

## Autor modifikacija

Modifikacije napravljene: 06.11.2025
- Dodavanje tri asimetrične funkcije: Levy, Griewank i Ackley
- Implementacija vizualizacije putanja lokalnog pretraživanja
- Poboljšanja u GUI i kontrolnim elementima
- Automatsko prilagođavanje prikaza za različite funkcije
