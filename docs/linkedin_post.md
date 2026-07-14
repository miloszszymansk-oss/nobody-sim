# LinkedIn post (EN) — pairs with docs/ARTICLE.md

I spent 7 days building an N-body gravity simulator in pure NumPy — directing an AI
pair-engineer through a spec-first workflow where nothing ships without a test or a
figure behind it.

The part worth sharing isn't the simulator. It's that we were measurably wrong three times:

→ Our convergence test reported order 4 for a 2nd-order integrator. The integrator was
fine — the measurement point was degenerate (phase error hides quadratically at a
cosine extremum).

→ I assumed tree-build sorting was the bottleneck. We made it 2.5× faster… and the
profiler showed the build is 2% of runtime. The walk owns 75%.

→ "Bigger octree buckets = faster walk"? Slower by 2.2× — but 5× more accurate. It
shipped as an accuracy dial instead.

One prediction landed beautifully: extrapolated the leapfrog-vs-RK4 energy crossover
from a cheap 200-orbit pilot at ~1,100 orbits; the full run observed it at 1,106.

Everything is public: architecture contract (SPEC), 30 physics tests, the failure
changelog, an interactive trajectory player, Docker + CI.

Repo + deep dive: [LINK]
Live demo: [LINK]

#computationalphysics #python #numpy #softwareengineering #buildinpublic

---

# Wariant PL (krótszy, pod polskie sieci)

7 dni, symulator N ciał w czystym NumPy, workflow spec-first z AI jako pair-engineerem.
Najcenniejsze nie jest to, co zadziałało, tylko to, co pomiary obaliły: test zbieżności
kłamał (zdegenerowany punkt pomiaru), "oczywista" optymalizacja sortowania dotknęła 2%
runtime'u, a większe buckety w drzewie okazały się wolniejsze i dokładniejsze naraz.
Jedna prognoza trafiła w punkt: przecięcie błędu energii przewidziane na ~1100 orbit,
zaobserwowane na 1106.

Repo + artykuł: [LINK]
