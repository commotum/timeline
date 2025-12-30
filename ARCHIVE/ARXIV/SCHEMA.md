# Data Pipeline Schema & Operation Specification

This document defines a **strict, auditable, automation-friendly data pipeline model**.

The system is designed around **explicit state transitions**, **row conservation**, and **machine-verifiable invariants**.

---

## Core Principles

1. **Every operation produces new file(s)**  
   No in-place mutation. All dataset states are immutable once written.

2. **All row movement is explicit**  
   No row may disappear unless explicitly modeled.

3. **Time is linear and numbered**  
   Each step `T` consumes one or more source states and produces new destination states.

4. **Semantics live in metadata, not filenames**  
   Filenames encode sequence only. Meaning is carried in the log.

---

## Log Entry Schema

Each pipeline step is represented as a log entry.

### Required Fields

```yaml
T: integer                       # Monotonic step index
action: ACTION_TYPE              # See Action Types below
source: path | list | pattern    # One or many input files
destination: path | map          # One or many output files
````

### Optional Fields

```yaml
predicate: string | null         # Row selection condition
group_key: list[string]          # For DEDUPE
aggregation: map                 # For DEDUPE
sort_key: list[string]           # For SORT
transform: string                # For NORMALIZE
cut: boolean                     # For EXTRACT (always true by definition)
remainder_status: pending|complete
invariants: list[string]         # Machine-checkable assertions
```

---

## Action Types

### 1. SPLIT

**Purpose**
Shard a single dataset into multiple files **without filtering**.

**Definition**

* 1 → many
* No predicates
* No row loss

**Schema**

```yaml
action: SPLIT
predicate: null
```

**Invariant**

```text
total_rows(source) == sum_rows(destination/*)
```

---

### 2. MERGE

**Purpose**
Combine multiple datasets into one **without filtering**.

**Definition**

* many → 1
* No predicates
* No row loss

**Schema**

```yaml
action: MERGE
predicate: null
```

**Invariant**

```text
rows(destination) == sum_rows(source/*)
```

---

### 3. EXTRACT (Primitive, Mandatory Remainder)

**Purpose**
Move a subset of rows to a new file **as a CUT**, emitting the remainder as a new file.

**Definition**

* 1 → 2
* Predicate-based
* Always produces:

  * `extracted`
  * `remainder`

**Schema**

```yaml
action: EXTRACT
predicate: <row condition>
cut: true
destination:
  extracted: <path>
  remainder: <path>
remainder_status: pending | complete
```

**Invariants**

```text
rows(source) == rows(extracted) + rows(remainder)
```

**Notes**

* FILTER does not exist as a concept.
* PARTITION is unnecessary when EXTRACT always emits remainder.

---

### 4. NORMALIZE

**Purpose**
Create a canonicalized version of a dataset while preserving all rows.

**Definition**

* 1 → 1
* Full copy, then transformation
* Predicate gates *where* transform applies, not selection

**Schema**

```yaml
action: NORMALIZE
predicate: <optional>
transform: <description>
```

**Invariants**

```text
rows(destination) == rows(source)
```

---

### 5. DEDUPE

**Purpose**
Collapse groups of rows into one row per group while preserving multiplicity via counts.

**Definition**

* many → one per group
* Identity-destroying but information-preserving

**Schema**

```yaml
action: DEDUPE
group_key: [col1, col2, ...]
count_column: duplicate_count
aggregation:
  duplicate_count: count | sum
```

**Invariants**

```text
sum(duplicate_count in destination) == rows(source)
destination unique by group_key
```

**Notes**

* Internal grouping is not modeled as a file.
* DEDUPE is atomic at the pipeline level.

---

### 6. SORT

**Purpose**
Reorder rows for presentation or deterministic output.

**Definition**

* 1 → 1
* No data change

**Schema**

```yaml
action: SORT
sort_key: [col1, col2, ...]
```

**Invariant**

```text
multiset(rows(source)) == multiset(rows(destination))
```

---

## Remainder Semantics

For `EXTRACT`, the `remainder_status` field indicates workflow intent:

* `pending`
  Remainder may be processed by later steps.
* `complete`
  Remainder will not be re-entered into this pipeline.

This replaces implicit “double back later” logic with an explicit marker.

---

## Conservation Rules (Global)

Across the entire pipeline:

1. **Row conservation**

   ```text
   sum(rows(all terminal outputs)) == rows(original input)
   ```

2. **Count conservation**

   ```text
   sum(duplicate_count across leaves) == rows(before first DEDUPE)
   ```

3. **No silent loss**
   Every row must appear in exactly one descendant state unless explicitly discarded by a modeled action.

---

## Minimal Complete Action Set

With the above rules, the system is complete with:

* SPLIT
* MERGE
* EXTRACT (with mandatory remainder)
* NORMALIZE
* DEDUPE
* SORT

No additional primitives are required.

---

## Design Outcome

This model guarantees:

* Deterministic replay
* Auditable lineage
* Machine-verifiable correctness
* No semantic leakage into filenames
* No hidden or implicit state transitions

---

End of specification.