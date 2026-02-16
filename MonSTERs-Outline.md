Absolutely. Below is a clean **sort/organization** of what you’ve drafted into the **four paper buckets**, with (a) what each paper is *about*, (b) the *core claim*, (c) the *sections* it should contain, and (d) **where each chunk of your draft content belongs** (mapped to your numbered timeline + the later “Triumvirate / Radford Doctrine / Sutton” pieces).

I’ll keep your voice where it helps (cowboy scientist / Triumvirate), but I’ll also show you where to park it so it doesn’t muddy the technical contribution.

---

# Shared framing that can be reused across all 4 papers

These are “common intro primitives” you can reuse, then tailor per paper:

## The Triumvirate (the three fixed assumptions)

This is your strongest unifying frame:

1. **Fixed vocabulary / fixed output head** (softmax over a predetermined set)
2. **Fixed context window / fixed attention pattern** (compute grows with token count)
3. **Fixed positional contract** (indices + PE as the only address scheme)

You also have a 4th theme (“continual learning not possible”)—that fits best as a **consequence** of the Triumvirate, not a fourth member, unless you really want a “Quadrumvirate.” If you keep it as the Triumvirate, you stay sharper.

## The translation tax

This is the “why this matters” glue:

> In exchange for a battle‑tested baseline, we pay a translation tax: each new domain must be contorted to fit a fixed token alphabet, fixed context/attention policy, and fixed positional interface.

## The “single-task illusion”

This is the key insight you discovered:

> In single-task regimes, transformers can often learn around arbitrary positional/structural priors (row-major, patching, etc.), so PE debates look inconclusive. The brittleness shows up under **multi-task interference**, where a shared interface must serve incompatible geometries.

## The Radford Doctrine slogan

This belongs best in Paper 4 and optionally as an “epigraph” in Papers 1–3:

> Pick a task that eats the rest.

(And in your framing: pick an *interface/objective* that collapses tasks rather than enumerates them.)

## Your “Sutton correction”

This is great, but keep it short and place it in Paper 3 or Paper 4:

> The problem isn’t “human knowledge.” The problem is freezing arbitrary architectural commitments (one-hot symbols, fixed attention routing, fixed positional contracts). Those aren’t neutral defaults—they’re anti-knowledge constraints.

---

# Paper 1

## **Spatial Reasoning MonSTERs**

**Subtitle idea:** *Minkowski SpaceTime Encoding Rotors as a 1D–4D Drop‑in Replacement for RoPE*

### What this paper is (core claim)

MonSTERs is a **positional/structural encoding contract**: a principled extension of RoPE from 1D to **4D (t,x,y,z)** that preserves the “absolute → relative” property while supporting absolute+relative fusion.

This paper is not about memory policies, vocab, RL, or DreamCoder. Keep it brutally about **positional encoding as a unified contract across dimensionalities**.

### Sections (recommended)

1. **Motivation: why higher‑D PE is fragmented**

   * “RoPE converged in 1D; no consensus in 2D/3D/4D.”
   * Axial RoPE limitations (diagonals / separability)
   * “Single-task illusion” explanation (why many methods look “fine” in narrow domains)

2. **Design goals**

   * Unified 1D–4D contract
   * Relative-position identity like RoPE
   * Compatible with linear attention / efficient attention
   * Supports absolute+relative fusion (not “relative only”)

3. **MonSTERs formulation**

   * 4D coordinate scheme
   * Lorentz rotors / Minkowski metric rationale
   * Drop‑in Q/K transformation view

4. **Ablations**

   * 1D RoPE vs axial 2D RoPE vs mixed RoPE vs MonSTERs
   * Where it helps (OOD generalization; multi-task interference)

5. **Evaluation suite (the point of the paper)**

   * Not just ARC score
   * Include controlled “geometry stress tests”: diagonals, rotations, resizing, variable grid sizes, multi-task mixtures (1D + 2D + 3D/4D)

6. **Why single-task improvements look small**

   * You already have this insight; make it explicit and measurable

### Where your existing draft content goes

**Use these timeline items here:**

* (4) anisotropy of row-major indexing → motivation
* (5) axial/mixed RoPE issues (diagonals) → motivation + related work
* (6) need 4D for layers + time + x + y → motivation
* (7) MonSTERs definition → method
* (9) HRM results only +3–7% → “single-task illusion” section
* (10) why PE debates persist in vision → thesis support
* (11) need multi-domain training to prove advantage → evaluation design

**Keep OUT of Paper 1 (park elsewhere):**

* quaternion tokens / softmax replacement (Paper 3)
* dynamic attention threads / RL policy / working memory (Paper 2)
* dreamcoder / continual vocab growth (Paper 3)
* big “MonSTER models do everything” claim (Paper 4)

---

# Paper 2

## **Dynamic Attention Threads for Transformer Working Memory**

**Subtitle idea:** *Reducing Attention-Score Computation via Learned Query/Key Routing over a Token Bank*

### What this paper is (core claim)

This is not primarily “context pollution.” Your sharper version is:

> Standard attention computes far too many scores (O(L²)) because the interface forces the model to treat *everything* as mutually relevant by default. Dynamic Attention Threads replace fixed attention patterns with a **learned compute-routing policy** that selects which comparisons to pay for.

This paper lives and dies on **compute scaling** and **meta-learning over sets of examples**.

### Sections (recommended)

1. **Problem: attention compute is the bottleneck**

   * Fixed window + full attention (or fixed sparse patterns) is an architectural commitment
   * Multi-example ARC-style episodes explode token comparisons

2. **Failure mode in existing ARC-style transformers**

   * You already have the key anecdote: more than one grid hurts (meta-learning failure)
   * Emphasize: it’s not just “too many tokens,” it’s “too many pairwise comparisons”

3. **Dynamic Attention Threads**

   * Token bank as environment/state
   * Policy selects:

     * a small set of **query tokens**
     * for each query, a small set of **key tokens** (or key-groups)
   * Threads = recurrent loops of “read → update → write”

4. **Working memory / notepad**

   * Per-grid notes
   * Cross-example notes
   * Hypothesis testing loop (try transformation, verify on examples, revise)

5. **Complexity analysis (this is the spine)**

   * Baseline attention: O(L²)
   * Threads: O(Q·K) per step, or O(T·Q·K) over T steps
   * Show regimes where compute drops dramatically

6. **Training**

   * RL policy vs differentiable routing (you can mention alternatives briefly, but keep your method central)
   * Rewards: task success + compute penalty + stability (optional)

7. **Experiments**

   * Accuracy vs compute curves
   * Performance vs number of demonstrations (should not degrade)
   * Generalization with fixed compute budget

### Where your existing draft content goes

**Use these timeline items here:**

* (12) token explosion / context difficulty → motivation
* (15) HRM misses meta-learning; performance degrades with more grids → motivation
* (15) full dynamic memory / policy selects queries + keys / focus + zoom → method (this is the heart)
* (17) “recursive loop module” / reasoning via observations + scratchpad → method & training

**Optionally reference (but don’t deep dive):**

* (8) HRM paper drops / recurrent transformer baseline → related work + baseline

**Keep OUT of Paper 2:**

* Minkowski/Lorentz/MonSTERs math (Paper 1)
* quaternion token synthesis and softmax replacement (Paper 3)
* dreamcoder and hierarchical compression tokens (Paper 3)
* “MonSTER models do everything” framing (Paper 4)

---

# Paper 3

## **Relay Learning**

**Subtitle idea:** *Typed Value Tokens, Synthesized Outputs, and Continual Abstraction Growth*

### What this paper is (core claim)

This is your “interface layer” paper: it attacks the **fixed vocabulary / fixed symbol interface** problem and ties it to continual learning.

Your cleanest technical nucleus here is:

1. **Typed value tokens** (value-parameterized, not ID-only)
2. **Synthesized outputs** (avoid giant softmax over huge discrete alphabets)
3. **Abstraction growth + pruning** (DreamCoder-style library formation + consolidation)

This paper is where your Sutton correction belongs too (because this is about “what is knowledge / what is arbitrary constraint”).

### Sections (recommended)

1. **Problem: fixed vocab is an arbitrary choke point**

   * RGB as canonical example (16M possibilities)
   * Softmax scaling + rare-token update pathology
   * More generally: token ID vocabularies are “anti-knowledge” (structureless)

2. **Typed value tokens**

   * Type channel: COLOR, INT, GRIDCELL, SYMBOL, ACTION, etc.
   * Value channel: continuous structured payload (quaternion, vector, tuple, etc.)
   * Latent tokens: learned abstractions created by the system

3. **Quaternion color representation + O(1) invertibility (your idea)**

   * RGB → quaternion
   * Type-specific projection
   * Output distribution over value space instead of ID softmax

4. **Continual abstraction growth (DreamCoder link)**

   * New “named” abstractions can be created, reused, pruned
   * “Utility tracking” maps to promotion/pruning rules
   * This is the bridge from implicit latents → reusable concepts

5. **Hierarchical token compression objective**

   * sentence token, paragraph token, chapter token analogy
   * reconstruction loss + multi-scale prediction
   * emphasize: this is a *learned vocabulary synthesizer*

6. **Relay framing (philosophy, keep it short)**

   * Not “no human knowledge”
   * Remove arbitrary constraints; let structure be learnable
   * Relay between system discoveries and human refinement

7. **Experiments**

   * RGB/value synthesis vs discrete vocab baselines
   * Continual growth & pruning: does performance improve on new domains without forgetting?

### Where your existing draft content goes

**Use these timeline items here:**

* (13) vocab explosion / softmax cost / rare token updates → motivation
* (14) quaternion paper inspiration; synthesize tokens; replace softmax → method
* (16) dreamcoder continual learning/refinement; grow reusable programs/options → method
* (17) hierarchical compression tokens + reconstruction + multi-scale rewards → method/training
* Your Sutton correction paragraphs → framing section

**Keep OUT of Paper 3:**

* Lorentz/MonSTERs math (Paper 1)
* Dynamic attention threads compute-routing (Paper 2)
* “We unified everything into a single model and did all tasks” (Paper 4)

---

# Paper 4

## **MonSTER Models are Domain-Agnostic Multitask Learners**

**Subtitle idea:** *Replacing the Transformer’s Static Interface with Dynamic Vocabulary, Dynamic Routing, and a Unified 4D Coordinate Contract*

### What this paper is (core claim)

This is the “GPT‑2 homage” systems paper:

> If you unfreeze the transformer interface (vocab, compute routing, positional contract), you don’t need “one model per domain.” You can train a single learner across 1D, 2D+t, and 3D+t tasks without constantly swapping architectures.

This is where you use the Triumvirate + translation tax + Radford doctrine + single-task illusion as the narrative engine.

### Sections (recommended)

1. **Intro: the Triumvirate and the translation tax**

   * This is where your “cowboy scientist” paragraph can live (short, 1–2 paragraphs max)
   * Then immediately snap into technical framing

2. **Why ARC-AGI is the stress test**

   * Few-shot episode reasoning
   * Multi-example in context (meta-learning)
   * Grid/space/time structure

3. **Three interface upgrades (the unified model)**

   * MonSTERs: unified 1D–4D positional contract
   * Dynamic Attention Threads: learned compute routing to avoid O(L²)
   * Typed value tokens + synthesis + abstraction growth: escape fixed vocab, enable continual learning

4. **Training stack**

   * Multi-domain curriculum: 1D (language/math), 2D+t (ARC, Sudoku), 3D+t (mazes, cube-like, folding)
   * Your p5.js code–image corpus fits well here as an example of a *domain that forces structured representation + synthesis*

5. **Results: multitask without modality-specific architectures**

   * This paper needs the “big table”: tasks vs performance
   * Key emphasis: transfer when structure overlaps, isolation when it doesn’t

6. **Ablations**

   * Remove MonSTERs → what breaks?
   * Remove threads → compute blows up / meta-learning degrades
   * Remove typed synthesis → vocab bottleneck / continual learning stalls

7. **Discussion**

   * The “single-task illusion” reframed as why the field hasn’t converged on 2D/3D PE
   * Why your interface approach changes the question

### Where your existing draft content goes

**Use these timeline items here:**

* (1)–(3) ARC origin story + Karpathy + p5.js plan + dataset → intro + training data
* (8) HRM appears and changes expectations → related work + motivation
* (10) single-task illusion → discussion framing
* (11) multi-domain proof requirement → training/eval framing
* (12)–(15) compute/tokens/meta-learning failure → motivation for threads
* (16)–(17) dreamcoder + synthesized vocab + compression objective → motivation for relay

**Use your “Triumvirate / cowboy scientist” writing here**
That whole block about:

* “These three assumptions… undisturbed existence… cowboy scientist…”
  belongs at the very top of Paper 4 as a tone-setting hook.

But: keep it short so it doesn’t become the paper. Think: 10–15 lines, then cut to the actual claim.

---

# Where the “ARC highlights continual learning weakness” argument belongs

You have a thread in the draft about:

> “What evidence shows ARC-AGI highlights lack of continual learning?”

That belongs in **Paper 4 discussion**, and possibly as a motivation paragraph in Paper 3.

The clean way to say it (without overclaiming) is:

* ARC is episodic/meta-learning: each puzzle is a mini-task.
* Most strong approaches adapt per-task (TTT/fine-tune) and reset, because compounding updates across tasks risks interference.
* That gap between “episodic adaptation” and “accumulate reusable abstractions across tasks” is exactly what you mean by continual learning.

You don’t need to litigate the whole CL literature in Paper 4; just position ARC as a stressor that exposes the interface rigidity.

---

# Cut list: what to trim so each paper stays publishable

Right now Paper 3 and Paper 4 are at risk of becoming “everything everywhere all at once.” Here’s the safe trimming rule:

* **Paper 1:** only MonSTERs math + PE experiments.
* **Paper 2:** only routing/threads + compute scaling experiments.
* **Paper 3:** only typed tokens + synthesis + abstraction growth objective.
* **Paper 4:** only the integrated system + multitask results.

If you keep that separation, you can still cross-reference the other papers as “components,” but each one will stand on its own.

---

# Optional: a clean “table of contents” view

If you want a single glance:

1. **MonSTERs** = *unified coordinate contract (1D–4D)*
2. **Threads** = *learned compute routing / working memory (reduce attention-score compute)*
3. **Relay** = *escape fixed vocab via typed value synthesis + abstraction growth*
4. **MonSTER Models** = *the integrated system that learns across domains without modality-specific architectures*

---

If you want, I can take your exact text and do a **literal cut-and-paste restructure** into four separate documents (with headings and minimal rewriting), but the mapping above is the “editor’s blueprint” that will keep the papers clean and non-overlapping.
