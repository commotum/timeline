[PROJECT-BACKGROUND]
You are consolidating timeline-style Markdown files from /home/jake/Developer/timeline/DRAFTS into a single consistent format. Each source file represents a research paper classification where every entry should have a clear Driver (the motivation or gap) and Outcome (the paper's primary response). The goal is to normalize formatting only, not to deduplicate or fact-check.

[ORIGINAL-GENERIC-TEMPLATE]
# <Research Paper Classification Name>

**Criteria:**

1. Works within <general model / paradigm / domain>.
2. The paper is motivated by a clearly identifiable **Driver** -- a limitation, pressure, gap, or unmet need in prior work or practice.
3. The paper's **primary contribution** is an **Outcome** -- a concrete or conceptual response directly addressing that Driver.

**Examples:**

- *<Paper Title>* (<Year>)
  **Driver:** <what forces the contribution to exist>
  **Outcome:** <what the paper produces or changes>

[GENERAL-ENTRY-FORM]
- **Title** (Year)
  **Driver:** <short statement>
  **Outcome:** <short statement> ([Source Label][Ref ID])

[LIST-OF-ERRORS-MISTAKE-TYPES]
- Title lines include authors, venues, or year ranges instead of just title + year.
- Some entries use "Improves On / Adaptation" or "Critique / Improvement" instead of Driver / Outcome.
- Some entries are missing URLs, reuse a single URL for many entries, or have mismatched references.
- Reference IDs are inconsistent or reused across files.
- URLs sometimes point to landing pages or include tracking parameters (utm_*).
- Non-entry prose appears (intro paragraphs, "if you want..." notes, placeholders).
- Nested bullets contain valid papers that must be extracted as entries.
- Titles are truncated with ellipses or include extra qualifiers like "borderline."

[INSTRUCTIONS-FOR-REFORMAT-AND-OUTPUT]
1. Read the source draft file at [SOURCE_MD_PATH]. Treat it as the only input.
2. Extract every paper entry in order. Include nested bullet items that are actual papers. Skip non-paper prose, notes, or placeholders.
3. For each entry:
   - Title: keep only the paper title. Remove authors/venue names. Keep acronyms in parentheses.
   - Year: use a 4-digit year. If missing on the line, infer from the nearest section heading (e.g., "## 2024 ..."). If still unknown, use "Year unknown".
   - Driver/Outcome: map "Improves On" -> Driver, "Adaptation" -> Outcome. Map "Critique" -> Driver, "Improvement" -> Outcome. If neither exists, set to "Not stated in source".
   - Notes like "borderline" should be appended to Outcome as "Note: ...".
4. References:
   - If a URL is present, include it as the Source Label + Ref ID pair.
   - If multiple URLs exist for the entry, choose the most direct public URL available in the source (prefer a PDF when clearly provided).
   - If no URL is present, use "Source Missing" as the Source Label and create a Ref ID. Set its URL to "MISSING".
   - Ref IDs must be unique in the output. If you reuse an existing Ref ID, make it unique by prefixing with the source file stem and a counter.
5. Output format must follow [GENERAL-ENTRY-FORM] exactly.
6. Append the new entries to the output file at [OUTPUT-PATH]. Do not overwrite or delete existing content.
7. After the batch of entries you add, include the corresponding [Ref ID]: URL definitions. Do not add any other commentary.

[OUTPUT-PATH] /home/jake/Developer/timeline/CONSOLIDATED.md
