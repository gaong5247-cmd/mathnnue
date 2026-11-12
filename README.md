# MathNNUE 2.0 — Self-Evolving Mathematical Problem Lab

A native **Rust + egui** experimental laboratory with two independently trained neural problem-generators/solver policies. Starts from elementary arithmetic and progressively unlocks equations, calculus, combinatorics, algebra, university material and **IMO-themed certified templates**. The long-range ambition is to discover difficult, original problems; this version **does not** claim general mathematical creativity, IMO performance, mathematical research breakthroughs, or the ability to prove arbitrary conjectures.

## Windows installation

1. In GitHub Actions → **Build and test** → select the latest successful run.
2. Download **MathNNUE-windows-x64** from Artifacts, unzip and run `mathnnue.exe` (Windows 10/11 x64).
3. Click **▶ 학습 / 재개**. Neural weights start randomly initialized; the models must train. Progress is persisted in `%LOCALAPPDATA%\MathNNUE\checkpoints\latest.json`. An older `./checkpoints/latest.json` checkpoint can also be migrated.
4. You can independently preview advanced-stage examiner questions from the creative preview panel at any time. This does **not** skip or fabricate the student curriculum.
5. To try your own expressions use `1+1`, `3x+7=22`, `x^2-5x+6=0`, `diff 2x^3-3x^2+4x+1 at x=2`, `integral 0 2 3x^2+4x`.

Source build (Rust stable):

```sh
cargo test
cargo run --release -- --smoke
cargo run --release
```

The Windows `.exe` is created automatically by GitHub Actions on each push. This repository does not contain a pre-trained high-performance neural checkpoint.

## Structure

- `src/solver.rs` / `src/neural.rs`: elementary arithmetic solved by a neural **digit-transition** policy, including carry/borrow and multiplication/division. The exact-answer arithmetic oracle is not consulted during student inference.
- `src/examiner.rs`: independent elementary examiner NNUE selects operations, difficulty, digit widths and operand digits; REINFORCE updates with student mistakes.
- `src/advanced.rs`: certified math challenge grammar, 30 task families and 19 additional stages. A symbolic teacher can check and score the constructed challenges; constrained integer/rational/symbolic formats are deliberate. Some templates are much easier than their stage labels might suggest. **No unsolved problems are claimed.**
- `src/creative.rs`: **separate** advanced examiner NNUE, action-head sampling, multi-proposal novelty search, structure-history deduplication and reward from both novelty and student errors. Advanced student NNUE learns to select a mathematical solution **program** from prompt features. The explicit program then applies symbolic algorithms: **the higher-stage solver is hybrid, not a pure neural theorem prover**.
- `src/input.rs`: limited input parser for linear/multivariate? No: only integer linear/monic small-integer-root polynomial equations, polynomial derivative evaluation and zero-based definite integrals of quadratic+linear terms.
- `src/trainer.rs`: student–examiner self-play worker, GUI updates, checkpoint migration, 20,000 teaching questions then **100,000 separate balanced questions**. Advancement requires answer accuracy, successful step rate, and completion all at least 90% on a full exam. With 100,000 questions, structural uniqueness is reported separately; early stages necessarily repeat some structures.
- `src/main.rs`: Windows graphical application, live NNUE choices, learning events, manually submitted math, creative preview, exam performance.

## Study curriculum

**Elementary 10 stages:** one-digit addition, subtraction and regrouping, multiplication, division, signed integers and mixed arithmetic.

**Higher 19 stages:** fractions and algebra → systems → quadratic/cubic/quartic polynomials → sequences/logarithms/trigonometry → limits/derivatives/integrals → multivariable calculus and linear algebra → probability/combinatorics/number theory → IMO-themed algebra/number theory/combinatorics → verified cross-topic research-style *exercises*.

A generated question must pass structural restrictions. Examinations use independently seeded randomization, balance eligible topic families and difficulty, do not train weights, and report both exact unique prompts and structural signatures. Training policies encourage novelty; they do not certify mathematical originality. Passing a registered template exam is **not** equivalent to solving an unseen IMO problem or reaching undergraduate/graduate mathematical maturity.

## Practical constraints / next research steps

- The current composer has a *finite* vocabulary of verified problem families. Sampling new parameter values and recombining families is a useful start, **not** unrestricted invention of novel concepts. Need grammar expansion, formal semantics, proof search and formal proof checking (e.g. Lean, Isabelle) for genuine originality and research-level verification.
- Student arithmetic is neural; advanced math uses **neural program selection** plus deterministic algorithms. Building an end-to-end proof-producing neural solver is a separate research project.
- The 90% gate could be reached by learning the templates and is not a reliable estimate of out-of-distribution generalization. Future exams should include adversarial holdouts, truly new families and external independent collections.
- 100k evaluations and long training can use substantial CPU resources. There is no claim that arbitrary hard problems will eventually become solvable just by training longer.

## Security

No network service, no auto-updates, and no external API key. Everything runs locally; GitHub Actions downloads are build artifacts, not signatures. For reproducibility build from source yourself.
