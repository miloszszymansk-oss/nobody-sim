# SPEC — Symulator grawitacyjny N ciał (nbody-sim)

Wersja 0.1 (Dzień 1). Ten dokument to kontrakt: kod jest pisany PRZECIW niemu, testy weryfikują jego twierdzenia. Zmiany w SPEC tylko świadome, z wpisem w changelogu na dole.

**Cel projektu:** symulator N ciał z (1) integratorami symplektycznymi i dowodem empirycznym ich przewagi, (2) algorytmem Barnes-Hut O(N log N), (3) walidacją naukową na prawie Keplera i wielkościach zachowanych, (4) interaktywną wizualizacją web.

**Definition of Done (tydzień):** testy zielone · wykres dryfu energii leapfrog vs RK4 · wykres skalowania brute vs Barnes-Hut · demo web · repo publiczne z instrukcją reprodukcji.

---

## 1. Model fizyczny

### 1.1 Równania ruchu

N ciał punktowych o masach m_i, pozycjach **r**_i, prędkościach **v**_i. Grawitacja Newtona — przyspieszenie ciała i:

```
a_i = G · Σ_{j≠i}  m_j · (r_j − r_i) / |r_j − r_i|³
```

To układ równań różniczkowych 2. rzędu: znasz to z liceum jako F = ma, tylko F zależy od pozycji wszystkich ciał naraz — dlatego nie ma rozwiązania analitycznego dla N ≥ 3 i potrzebujemy numeryki.

### 1.2 Softening (zmiękczenie)

Gdy dwa ciała się zbliżają, |r_j − r_i| → 0 i przyspieszenie wybucha do nieskończoności — krok czasowy tego nie udźwignie. Standardowy zabieg: parametr ε (epsilon) w mianowniku:

```
a_i = G · Σ_{j≠i}  m_j · (r_j − r_i) / (|r_j − r_i|² + ε²)^{3/2}
```

**Intuicja:** ε "rozmywa" masy punktowe na kulki o promieniu ~ε. Cena: siła przestaje być dokładnie newtonowska na dystansach ≲ ε. Dla testów Keplera (2 ciała, brak bliskich spotkań) ustawiamy ε = 0, dla gromad ε > 0.

**Pułapka nr 1 (częsty bug):** energia potencjalna musi używać TEGO SAMEGO ε co siła:

```
V = −G · Σ_{i<j}  m_i m_j / √(|r_i − r_j|² + ε²)
```

Inaczej test zachowania energii kłamie. Siła to minus gradient potencjału — sprawdź na papierze, że pochodna −dV/dr daje wzór na siłę z ε.

### 1.3 Jednostki

Astronomiczne: długość w AU, czas w latach, masa w masach Słońca (M_☉). Wtedy:

```
G = 4π² ≈ 39.478  [AU³ · M_☉⁻¹ · yr⁻²]
```

Sprawdzenie (III prawo Keplera): T² = 4π²a³/(G·M) → dla a = 1 AU, M = 1 M_☉: T = 1 rok. Ziemia na orbicie kołowej: v = √(GM/r) = 2π AU/yr. Te okrągłe liczby to gotowe asercje testowe.

### 1.4 Wielkości zachowane (nasze "sędziowie poprawności")

| Wielkość | Wzór | Zachowana bo… |
|---|---|---|
| Energia E = T + V | Σ ½m_i v_i² + V | brak sił zależnych od czasu |
| Pęd całkowity **P** | Σ m_i **v**_i | III zasada dynamiki (siły parami przeciwne) |
| Moment pędu **L** | Σ m_i **r**_i × **v**_i | siły centralne (wzdłuż linii łączącej ciała) |

Symulator NIE wymusza tych praw — one mają wyjść same z poprawnej numeryki. To jest właśnie test.

---

## 2. Integratory

### 2.1 Leapfrog (velocity-Verlet, wariant kick-drift-kick)

```
v_{½}  = v_n + a(r_n) · dt/2        # kick  (pół kroku prędkości)
r_{n+1} = r_n + v_{½} · dt           # drift (pełny krok pozycji)
v_{n+1} = v_{½} + a(r_{n+1}) · dt/2  # kick  (domknięcie prędkości)
```

Koszt: 1 wywołanie a(r) na krok (kick końcowy = kick początkowy następnego kroku; w implementacji v0 liczymy a raz na krok i cache'ujemy).

Własności: **rząd 2** (błąd globalny ~ dt²), **symplektyczny**, **odwracalny w czasie** (puszczony z −dt wraca po własnych śladach).

**Skąd się bierze (szkic wyprowadzenia):** energia układu to H = T(v) + V(r), część kinetyczna + potencjalna. Rozbijamy ewolucję na dwa proste ruchy, z których każdy umiemy rozwiązać DOKŁADNIE:
- **kick** — działa samo V: pozycje zamrożone, prędkości dostają impuls a·dt/2 (dokładne, bo a nie zmienia się, gdy r stoi);
- **drift** — działa samo T: prędkości zamrożone, pozycje płyną v·dt (dokładne, bo ruch jednostajny).

Sekwencja kick(dt/2) → drift(dt) → kick(dt/2) to tzw. złożenie Stranga. Symetria złożenia kasuje błąd rzędu 1 → zostaje rząd 2.

**Intuicja symplektyczności (kluczowa rzecz w projekcie):** każdy z podkroków jest dokładnym rozwiązaniem "połowy fizyki", więc zachowuje objętość w przestrzeni fazowej (pozycje×pędy) — a złożenie takich map też. Konsekwencja (tzw. backward error analysis): leapfrog rozwiązuje DOKŁADNIE pewien minimalnie zaburzony układ ("cieniowy hamiltonian" H̃ = H + O(dt²)), który ma własną, ściśle zachowaną energię. Dlatego błąd energii leapfroga **oscyluje w ograniczonym paśmie ~dt² i nie rośnie z czasem**. To jak zegarek, który tyka nierówno, ale nigdy się nie spóźnia narastająco.

### 2.2 RK4 (klasyczny Runge-Kutta) — kontrola negatywna

Stan y = (r, v), pochodna f(y) = (v, a(r)). Standardowe k₁…k₄:

```
k₁ = f(y_n)
k₂ = f(y_n + dt/2 · k₁)
k₃ = f(y_n + dt/2 · k₂)
k₄ = f(y_n + dt · k₃)
y_{n+1} = y_n + dt/6 · (k₁ + 2k₂ + 2k₃ + k₄)
```

Rząd 4 — na krótkim odcinku DOKŁADNIEJSZY od leapfroga. Ale nie jest symplektyczny: co krok gubi/dokłada odrobinę energii w sposób systematyczny, więc błąd energii **dryfuje ~liniowo z czasem** — jak procent składany. Koszt: 4 wywołania a(r) na krok (4× drożej!).

### 2.3 Eksperyment rozstrzygający (wynik nr 1 do write-upu)

Orbita eliptyczna e = 0.9 (błędy kumulują się przy peryhelium), 10³–10⁴ okresów, dt = T/300. Wykres |ΔE/E₀|(t) w skali log: leapfrog — płaskie pasmo; RK4 — narastający dryf. Do kompletu: rząd zbieżności na krótkim odcinku (log-log błąd vs dt: nachylenie 2 vs 4) — pokazuje, że rozumiemy różnicę "dokładność lokalna" vs "stabilność strukturalna".

Poza zakresem D1 (świadomie): adaptacyjny krok czasowy — naiwna adaptacja psuje symplektyczność; notatka w write-upie jako rozszerzenie.

---

## 3. Barnes-Hut (implementacja: Dzień 2, kontrakt: dziś)

Problem: brute force liczy N² par. Barnes-Hut: **odległe grupy ciał ≈ jedno ciało w ich środku masy**.

Algorytm: (1) buduj oktree — sześcian dzielony rekurencyjnie na 8; każdy węzeł przechowuje masę całkowitą i środek masy poddrzewa. (2) Siła na ciało i: schodź od korzenia; jeśli `s/d < θ` (s — bok komórki, d — odległość ciała od środka masy węzła), potraktuj węzeł jak punkt; inaczej wejdź w dzieci. θ ≈ 0.5–0.8; θ = 0 odtwarza brute force.

**Intuicja błędu:** zaniedbujemy to, że grupa nie jest idealnie punktowa — błąd względny siły ~ (s/d)², czyli kontrolowany przez θ. Złożoność: O(N log N).

Przypadki brzegowe (testy D2): ciało wewnątrz komórki → zawsze otwieraj; pomijanie samooddziaływania; identyczne pozycje → limit głębokości + ε. Kontrakt: `accel_barnes_hut` ma IDENTYCZNĄ sygnaturę jak `accel_brute` — wymienny backend. Test zgodności: θ = 0 → wynik równy brute co do błędu maszynowego; θ = 0.5 → błąd względny < 1% na gromadzie 10³ ciał.

---

## 4. Architektura kodu

### 4.1 Zasady

1. **Tablice, nie obiekty.** Stan to trzy tablice NumPy: `pos (N,3)`, `vel (N,3)`, `mass (N,)`. Klasyczny błąd początkujących: klasa `Body` i pętla po liściе obiektów — 100× wolniej. Wektoryzacja to nasza jedyna pętla.
2. **Funkcje czyste.** Krok integratora bierze stan, zwraca nowy stan. Zero globali, zero mutacji ukrytych — testowalność przede wszystkim. (Optymalizacja in-place dopiero po profilingu, D3.)
3. **Integrator nie zna fizyki.** Dostaje funkcję `accel_fn(pos) -> (N,3)` jako argument. Dzięki temu: (a) testujemy integratory na oscylatorze harmonicznym, gdzie znamy wynik dokładny, (b) Barnes-Hut wpina się bez zmiany integratora.
4. **SPEC → test → kod.** Nic nie jest "zrobione" bez testu odwołującego się do sekcji SPEC.

### 4.2 Moduły i sygnatury (kontrakt interfejsów)

```
src/nbody/
├── bodies.py        # stan i generatory układów
├── forces.py        # backendy sił: brute (D1) + fabryka make_accel
├── barnes_hut.py    # oktree + trawersal frontierowy, w pełni wektorowe (D2)
├── integrators.py   # leapfrog, euler_cromer, rk4 — agnostyczne wobec fizyki
├── diagnostics.py   # energia, pęd, L, elementy orbitalne
└── sim.py           # pętla symulacji + zapis historii + eksport JSON
```

```python
# bodies.py
@dataclass
class System:
    pos: np.ndarray   # (N,3) [AU]
    vel: np.ndarray   # (N,3) [AU/yr]
    mass: np.ndarray  # (N,)  [M_sun]

def sun_earth() -> System                                   # test kanoniczny
def two_body(m1, m2, a, e) -> System    # start w peryhelium, układ środka masy
def uniform_cluster(n, radius, total_mass, seed) -> System  # zimna kula
def to_barycentric(s: System) -> System # zeruje pęd całkowity (dryf środka masy)

# forces.py
def accel_brute(pos, mass, G=G_ASTRO, eps=0.0) -> np.ndarray   # (N,3)
def make_accel(mass, G, eps, backend="brute") -> Callable      # fabryka accel_fn

# integrators.py — wspólna sygnatura, wymienność
def leapfrog_step(pos, vel, dt, accel_fn) -> tuple[pos, vel]
def rk4_step(pos, vel, dt, accel_fn) -> tuple[pos, vel]

# diagnostics.py
def total_energy(s, G=G_ASTRO, eps=0.0) -> float   # MUSI używać tego samego eps co siły (§1.2)
def momentum(s) -> np.ndarray                       # (3,)
def angular_momentum(s) -> np.ndarray               # (3,)
def orbital_elements(rel_pos, rel_vel, gm) -> tuple[a, e]  # z wektora Laplace'a-Rungego-Lenza

# sim.py
@dataclass
class Config:
    dt: float; n_steps: int; G: float = G_ASTRO; eps: float = 0.0
    integrator: str = "leapfrog"; force: str = "brute"; record_every: int = 1

def run(system: System, cfg: Config) -> History   # History: time, pos, energia, L (próbkowane)
```

Wzory do `two_body` (start w peryhelium, μ = G(m₁+m₂)):
```
r_p = a(1−e)                     # odległość w peryhelium
v_p = √( μ(1+e) / (a(1−e)) )     # prędkość styczna (z vis-viva)
```
oraz do `orbital_elements`: 1/a = 2/|r| − v²/μ (vis-viva), e z wektora ekscentryczności **e** = (**v**×**l**)/μ − **r̂**, gdzie **l** = **r**×**v**. Sprawdź vis-vivę na papierze z zasady zachowania energii — licealne przekształcenie.

### 4.2b Format eksportu historii: `nbody-history/1` (D3, dla odtwarzacza D4)

JSON z PŁASKIMI tablicami liczb (zero zagnieżdżonych obiektów per krok) — w JS parsuje się raz przez `Float64Array.from(j.pos)`:

| pole | typ/kształt | uwaga |
|---|---|---|
| schema | "nbody-history/1" | wersjonowanie kontraktu |
| n, k | int | liczba ciał, liczba próbek |
| meta | obiekt | parametry eksperymentu (G, eps, theta, dt, ...) |
| time | [k] | czasy próbek |
| mass | [n] | masy |
| pos | [k·n·3] płaska | pos[(k·n + i)·3 + c]: krok-major, ciało-minor, xyz |
| energy | [k] | energia całkowita (z eps jak w siłach, §1.2) |
| angular_momentum | [k·3] płaska | wektor L |

`decimals` (domyślnie 4) kontroluje rozmiar pliku (~6 B/liczbę).

### 4.3 Wydajność (świadome decyzje)

Brute force przez broadcasting tworzy tablicę (N,N,3): dla N = 2000 to ~100 MB — OK; dla N = 5000 ~600 MB — za dużo. Kontrakt: brute wspiera N ≤ 3000, powyżej — komunikat kierujący na Barnes-Hut. Chunking i numba: dopiero po profilingu (D3), z benchmarkiem przed/po. Zasada: żadnej optymalizacji bez pomiaru.

---

## 5. Plan testów (pytest; test = egzekwowalna klauzula SPEC)

| # | Test | Własność | Tolerancja (kalibracja: 1. uruchomienie, potem zamrożona) |
|---|---|---|---|
| T1 | `test_forces::two_body_analytic` | siła 2 ciał = wzór ręczny | 1e-14 (błąd maszynowy) |
| T2 | `test_forces::newton_third_law` | Σ m_i·a_i = 0 | 1e-12 |
| T3 | `test_integrators::sho_exact` | oscylator harmoniczny vs cos(ωt) | rząd zbieżności: leapfrog 2.0±0.1, RK4 4.0±0.2 (log-log) |
| T4 | `test_conservation::circular_orbit` | r = 1 AU stałe, T = 1 yr (100 orbit) | |Δr|/r < 1e-4 przy dt = T/1000 |
| T5 | `test_conservation::kepler_ellipse` | a, e stałe (e = 0.5, 100 orbit) | Δa/a, Δe < 1e-3 |
| T6 | `test_conservation::energy_leapfrog` | pasmo |ΔE/E| ograniczone, bez trendu | < 1e-3 przy dt = T/300; brak korelacji z t |
| T7 | `test_conservation::energy_rk4_drifts` | RK4 dryfuje (kontrola negatywna) | |ΔE(t_end)| > |ΔE(t_end/10)| systematycznie |
| T8 | `test_conservation::momentum_L` | P, L stałe | 1e-10 względnie |
| T9 (D2) | `test_barnes_hut::matches_brute` | θ=0 ≡ brute; θ=0.5 błąd < 1% | j.w. |

Zasada tolerancji: pierwszy przebieg pokazuje realny błąd → ustawiamy próg z zapasem 3–5× → od tej pory każda regresja = czerwony test.

---

## 6. Eksperymenty do write-upu (skrypty w `experiments/`, każdy zapisuje PNG + CSV)

1. `exp_energy_drift.py` — §2.3: leapfrog vs RK4, e = 0.9, długi horyzont.
2. `exp_convergence.py` — błąd vs dt (log-log) dla obu integratorów + koszt (liczba wywołań accel) vs błąd: "RK4 bywa tańszy na krótko, leapfrog wygrywa na długo".
3. `exp_scaling.py` (D3) — czas kroku vs N: brute ~N², BH ~N log N; punkt przecięcia.
4. `exp_cluster.py` (D3) — ZREALIZOWANE jako zderzenie dwóch sfer Plummera (ambitniejsze niż pierwotny kolaps zimnej kuli; kolaps został testem jednostkowym energii BH).
   Matematyka próbkowania Plummera (sekcja referencyjna do write-upu):
   - gęstość ρ(r) = (3M/4πa³)(1+r²/a²)^{-5/2}; masa skumulowana M(<r) = M·r³/(r²+a²)^{3/2};
   - promień: odwrócenie CDF → r = a/√(u^{-2/3}−1), u~U(0,1), ogon r>10a repróbkowany;
   - prędkość: ergodyczna f(E) ∝ (−E)^{7/2} ⇒ dla q=v/v_esc: g(q) ∝ q²(1−q²)^{7/2}, akceptacja-odrzucenie pod g_max = g(√(2/9)); v_esc = √(2GM)·(r²+a²)^{-1/4};
   - testy samplera: 2T/|W| ≈ 1 (W = −(3π/32)GM²/a), r_połowy masy = a/√(2^{2/3}−1) ≈ 1.305a, v ≤ v_esc.
   Scenariusz: M=2×0.5, a=0.5, separacja d=3, parametr zderzenia b=0.4, v_rel=3 (E_orb=−2.16 < 0 ⇒ związany), N=800, dt=1.25e-3, BH θ=0.5, ε wg §8.1.

---

## 7. Repo i praktyki (dla portfolio)

```
nbody-sim/
├── SPEC.md  README.md  pyproject.toml  .gitignore
├── src/nbody/          # pakiet (src-layout, pip install -e ".[dev]")
├── tests/              # pytest
├── experiments/        # skrypty → figures/
└── benchmarks/         # D3
```

- Python ≥ 3.10, zależności: numpy, matplotlib; dev: pytest, ruff. Type hints wszędzie, docstringi NumPy-style.
- Git od dziś: konto GitHub → repo `nbody-sim` (public) → commity małe, wiadomości w trybie rozkazującym ("Add leapfrog integrator with SHO test"). Pierwszy commit: SPEC + szkielet. Push robisz ze swojej maszyny (git albo GitHub Desktop).
- D5: README z sekcjami Motivation / Physics / Results (wykresy!) / Reproduce (`pip install -e ".[dev]" && pytest && python experiments/...`) + CI (GitHub Actions: pytest na push).

---

## 8. Decyzje (zamknięte w 0.2)

1. Softening gromad: ε = 0.05·R/N^(1/3) — PRZYJĘTE.
2. Wizualizacja: wariant (a) — pełna symulacja w Pythonie (single source of truth), eksport historii do kompaktowego JSON (`History.to_json`), frontend HTML/JS jako interaktywny odtwarzacz trajektorii. PRZYJĘTE.
3. Nazwa repo: `nbody-sim`. PRZYJĘTE.

## Changelog
- 0.4 (D3: profiling, Plummer, eksport JSON):
  - PROFILING (benchmarks/profile_bh.py, raport w benchmarks/out/): metodologia = cProfile/pstats + A/B na identycznych wejściach. DWIE hipotezy sfalsyfikowane pomiarem:
    (1) hipoteza briefu "sortowanie dominuje budowę": grouping przyspieszony 2.52× (jeden argsort zamiast unique+argsort, `_group_keys` reużywa `order` w CSR), ALE build to 2% pełnego accel, sortowania 1.0% całości — zysk realny, wpływ marginalny;
    (2) hipoteza "większe bucket-leaves przyspieszą trawersal": sweep leaf_size∈{1..64} @ N=3000: 144.6→313.7 ms (wolniej!), za to p99 błędu 1.51e-2→2.76e-3. Wniosek: leaf_size zostaje 1 (szybkość), dokumentowany jako pokrętło dokładność/koszt.
    Prawdziwy hotspot: own-time trawersalu 75.1% (fancy indexing + alokacje masek per poziom); kandydat na numba/persistent buffers PO maratonie — wpis do roadmapy, nie grzebiemy dalej bez potrzeby.
  - bodies.plummer_sphere: sampler Plummera (matematyka w §6.4), 4 testy statystyczne zielone (wiriał 2T/|W|∈(0.85,1.15) obserwowany ~1.0).
  - History.to_json → schema `nbody-history/1` (§4.2b), round-trip test.
  - exp_cluster (§6.4): merger związany, 1600 kroków, 33 s wall; energia |ΔE/E| do 4.3e-2 — budżet błędu zdominowany przez bliskie spotkania w gęstym rdzeniu (skok w pierwszych krokach relaksacji + wzrost przy pierwszym przejściu); remedia poza zakresem D3: mniejszy dt / większe ε / krok adaptacyjny (§2.3-notatka). figures/cluster_merger.png + JSON dla odtwarzacza D4.
  - Suita: 30/30.
- 0.3 (D2, zmiana nocna):
  - §3 ZAIMPLEMENTOWANE, architektura inna niż podręcznikowa rekurencja (celowo): budowa drzewa POZIOMAMI na kluczach oktantowych (np.unique + bincount na komórkach zajętych), trawersal jako FRONTIER par (ciało, węzeł) przetwarzany poziom po poziomie czystymi operacjami tablicowymi. Powód: rekurencja per-ciało w Pythonie ma stałe ~10³× gorsze od numpy; wersja frontierowa utrzymuje O(N log N) przy stałych na poziomie C.
  - Przypadki brzegowe wg kontraktu: koincydentne pozycje → bucket-leaf na max_depth=20 (limit int64 dla kluczy); bezpieczeństwo self-force: θ < 1/√3 gwarantuje, że zaakceptowany węzeł nie zawiera celu (max odległość COM-narożnik = s√3); θ=0 ≡ brute co do kolejności sumowania (test: 1e-12).
  - UWAGA fizyczna: siła BH nie jest parami antysymetryczna → pęd zachowany tylko do O(błędu siły); T2 (III zasada) dotyczy wyłącznie brute.
  - T9 skalibrowane z pomiaru (gromada 10³, θ=0.5): mediana błędu 5.3·10⁻³, p99 2.6·10⁻² → progi: mediana < 1e-2, p99 < 5e-2. Metryka max ODRZUCONA świadomie (eksploduje przy kasowaniu się sił: 15% na ciele o niemal zerowej sile wypadkowej). Dodany test WŁASNOŚCI: skalowanie błędu z θ — obserwowane ~θ³ (5.8e-4 / 5.3e-3 / 2.1e-2 dla θ=0.25/0.5/0.8), lepsze niż podręcznikowe (s/d)², bo d liczone do COM zeruje dipol. Do write-upu.
  - Test energii BH: pierwsza wersja całkowała zimną kulę PRZEZ kolaps (t_ff≈0.18 przy G=4π²) i mierzyła sztywność całkowania zamiast błędu drzewa. Poprawka: horyzont pre-kolaps t=0.1; kontrola: BH 7.9e-4 vs brute 1.2e-3 — monopol nie psuje zachowania energii.
  - §2: dodany euler_cromer (symplektyczny Euler, rząd 1) — trzeci punkt danych: symplektyczność ⊥ rząd dokładności (pasmo O(dt) bez trendu; testy: nachylenie 1.00, pasmo bez dryfu). STEPPERS: leapfrog | euler_cromer | rk4.
  - §6.3 WYNIK (figures/scaling.png, benchmarks/out/): przecięcie brute/BH przy N≈1.07·10³; N=3000: BH 2.0× szybszy (81 ms vs 163 ms); N=3·10⁴: BH 1.32 s/eval (brute: niewykonalne pamięciowo). Skalowanie BH lekko ponad N log N (koszt sortowań przy budowie) — kandydat do profilingu D3.
  - Config: +theta (0.5); make_accel: +theta, +leaf_size. Suita: 23/23.
- 0.2 (D1, po implementacji):
  - §8: decyzje zamknięte (ε-heurystyka, wizualizacja = eksport + odtwarzacz, nazwa repo).
  - §4.2: `sim.run` ma szybką ścieżkę leapfrog (cache przyspieszenia: 1 wywołanie accel/krok zamiast 2; kick zamykający kroku n = kick otwierający n+1). Matematycznie identyczna z referencyjnym `leapfrog_step` — wymuszone testem równości bitowej.
  - §5/T3 — LEKCJA: pomiar błędu globalnego w punkcie startowym po pełnym okresie (start v=0) jest zdegenerowany: amplituda jest (niemal) dokładna, a czysty błąd fazy w ekstremum cosinusa wchodzi kwadratowo → leapfrog udaje rząd 4, RK4 ~5 (superzbieżność punktu symetrycznego). Poprawka: pomiar w fazie generycznej (0.85 T) na normie stanu √(Δx² + (Δv/ω)²). Rządy 2.00 i 4.00 potwierdzone.
  - §2.3 — WYNIK D1 (figures/energy_drift.png, dwa panele):
    (A) e=0.9, dt=T/3000, 300 orbit: pasmo leapfroga |ΔE/E| ≈ 1.1·10⁻² (płaskie, bez trendu), dryf RK4 liniowy w t, 4.4·10⁻⁴ po 300 orbitach → ekstrapolowane przecięcie ≈ 7.4·10³ orbit (poza budżetem obliczeniowym sesji: ~13 min CPU).
    (B) walidacja ekstrapolacji przy tańszych parametrach: e=0.6, dt=T/200, 2500 orbit — przecięcie PRZEWIDZIANE z pilota 200 orbit: ~1100; ZAOBSERWOWANE: ≈1106. Pasmo leapfroga 7.35·10⁻³ identyczne po 200 i 2500 orbitach; RK4 kończy na 1.67·10⁻² (2.3× ponad pasmem).
    Lekcja metodologiczna: RK4 rzędu 4 przy drobnym dt jest lokalnie tak dokładny, że sekularność ujawnia się dopiero na długim horyzoncie — projekt eksperymentu musi dobierać (e, dt, horyzont) pod tezę i budżet, a ekstrapolację walidować niezależnym przebiegiem.
- 0.1 (D1): pierwsza wersja.
