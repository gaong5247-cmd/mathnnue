# MathNNUE 1.0 · Student × Examiner

**Two independently trained neural networks** in a native Rust/egui desktop laboratory. A *Student NNUE* solves arithmetic, while an *Examiner NNUE* learns to set harder questions and automatically examines the student. The system starts at grade-one single-digit addition and raises the curriculum only after a balanced 100,000-question examination.

> **Important limitation:** this is a research/learning prototype with a complete training loop and desktop application, **not** a pretrained mathematical genius. On first launch the student weights are random; even `1+1` may be wrong. A grade transition is **not guaranteed**. If it does not reach 90% on each required metric, it stays at the same stage. Large numbers, fractions, equations with variables, multi-digit divisors, algebra, and arbitrary mathematical functions are **not supported** yet. No misleading fallback solver substitutes oracle answers for neural predictions.

## Features

- **Student NNUE**: a sparse feature accumulator (operation + digit A + digit B + carry/borrow/remainder) feeding a 96-wide ReLU layer and separate digit/state heads. All elementary digit arithmetic is predicted by the network. For bigger expressions a deterministic column layout composes neural transitions; **the numbers themselves are never calculated by SymPy or a hidden calculator at inference**.
- **Examiner NNUE**: a separate sparse 56-wide policy network that picks operation, digit width, difficulty and whether to compose two operations. Its policy is reinforced on student failures; it retains explicit exploration rather than only memorizing a fixed worksheet. A seeded random instance generator materializes valid integer problems based on the NNUE's choices.
- **Teacher and validator**: exact Rust integer arithmetic is confined to question creation, teaching labels and **independent grading**. The teacher is *not* called during `Student::solve`.
- **Automatic curriculum**: 20,000 training questions per attempt followed by an exactly 100,000-question exam. The exam balances operation type, digit width and difficulty so the examiner cannot "win" by testing just easy examples. **Answer accuracy ≥90%, intermediate-step accuracy ≥90% and completion rate ≥90%** are all required to advance. No auto-advancement from shortened tests.
- **Real-time egui console**: see the current expression, student guess, trusted answer, NNUE digit-level predictions, neural examiner policy, test progress, score history and per-operation breakdown. Type `1+1`, `(12+7)*3`, etc. to test the student yourself, or ask the examiner to produce a worksheet with optionally hidden answers.
- **Resume**: checkpoints store both models, optimizer moments, stage, exam history and counters in `checkpoints/latest.json` (automatically after each exam, on exit, or on request).
- **Offline**: no cloud APIs, no accounts and no external mathematics libraries at inference; runs on CPU.

## Build / run

Requires **Rust stable**, installed with [rustup.rs](https://rustup.rs). On Windows 11 install the MSVC C++ Build Tools if rustup requires them.

```powershell
git clone https://github.com/gaong5247-cmd/mathnnue.git
cd mathnnue
cargo run --release
```

Use **▶ 학습 / 재개** to start. The app remains interactive while the worker thread is training or examining. Closing the window asks the worker to save and shut down. While an exam is underway, **Ⅱ 일시정지** suspends exam processing at the next 256-question boundary. Training itself is not run on the GPU; small sparse CPU nets were chosen for predictable memory and latency.

For server/console training without a window:

```powershell
cargo run --release -- --headless --rounds 1
```

`--rounds 1` runs until the next completed 100,000-question examination. Increase `--rounds` to repeat. Your checkpoints are the same as the GUI's. **It is normal for the same stage to repeat many times.**

Tests:

```powershell
cargo test
```

## Planned curriculum

| Stage | Subject | Key scope |
|---|---|---|
| 1 | Grade 1 | 0–9 addition |
| 2 | Grade 1 | Addition/subtraction within 20 |
| 3 | Grade 2 | Two-digit addition/subtraction |
| 4 | Grade 3 | Up to three digits, basic times tables |
| 5 | Grade 3 | Exact division by one digit |
| 6 | Grade 4 | Four-digit operations |
| 7 | Grade 4 | Multiplication by up to two digits |
| 8 | Grade 5 | Five digits and two-step expressions |
| 9 | Grade 6 | Signed numbers and two-step expressions |
| 10 | Extension | Up to six-digit operands, up to three-digit multipliers |

**Scope notes:** Exact division is currently limited to a *one-digit divisor*. Only integer answers are supported. "Extension" does **not** imply all school math is mastered: no decimals, fractions, algebra, geometry or proofs. The architecture can be extended by adding new neural primitives and exam families.

## How passing works

Each 100,000-question examination is *created* by the Examiner NNUE, but fairness constraints force even representation of operations, operand widths and carry-difficulty categories. The policy still controls allowed composition choices and its learned distribution drives training questions and user-requested quizzes. Exact grading is out-of-band.

- **정답률 / answer accuracy**: complete expressions with the exact correct final integer answer, divided by all exam questions.
- **풀이 단계 정확도 / intermediate step accuracy**: correct neural results at all operator nodes, divided by all intermediate operator nodes. Wrong intermediate results cannot be hidden by a coincidentally correct final answer.
- **완주율 / completion**: expressions for which the neural solver returns a complete, structurally valid answer without falling back to a calculation engine.

All three must exceed or equal **90.00%**. Every attempt logs the number of **unique** expressions; early grade-one tests necessarily repeat some questions because the problem space is smaller than 100,000. Higher difficulty is never claimed solely from duplicate-heavy grades.

## Architecture / source

```text
src/
  neural.rs    sparse accumulator, forward pass, differentiable Adam training
  solver.rs    student prediction, digit transitions, neural multi-digit composition, teaching examples
  examiner.rs  separate trainable policy NNUE with masked sampling and REINFORCE
  math.rs      syllabus, parser, instance generation, isolated exact-value oracle
  trainer.rs   continuous self-play, balanced 100k exams, grade gate, checkpoint & worker
  main.rs      native egui lab, graphs, direct prompts and worksheet generation
```

### Why this is "NNUE-like" and not a conventional algebra solver

The first layer adds **only active sparse feature-weight rows** instead of multiplying a dense feature vector. This is the useful NNUE idea behind the name, although the current implementation recomputes the small accumulator for each digit rather than incrementally updating a persistent board-like state. Digit arithmetic is learned from examples and gradients. A minimal procedural scaffold provides position order and decimal carrying; it is **not** claimed to be an unrestricted, end-to-end sequence model.

The system uses **supervised digit targets** for Student learning plus **failure-driven reinforcement** for Examiner learning. The two models do not magically invent universally valid new mathematics from scratch; adding more question types requires new appropriate representations, teachers and honest held-out evaluations.

## Windows CI

GitHub Actions builds on Windows and Linux, runs unit tests and uploads a Windows executable as a downloadable workflow artifact. The workflow outcome is the authoritative build check (not a benchmark of 90% accuracy).

MIT license.
