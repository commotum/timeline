# Data Quality Issues Taxonomy (PDF-Only Target)

Goal: output a CSV with `year`, `title`, and `url`, where `url` points to a publicly available or downloadable PDF.

## Missing or Placeholder URL
- Detection: url is empty or matches placeholders like `MISSING`, `TBD`, `N/A`.
- Plan: search for an official PDF (publisher/venue), then arXiv PDF, then open-access mirror; if none found, flag for manual research or drop from output.

## URL Contains Extra Text or Annotations
- Detection: url field has spaces, quotes, or labels appended after a URL.
- Plan: extract the first valid URL; discard annotations; if multiple URLs exist, follow "Same Work Listed Multiple Times".

## Landing/Abstract/DOI Page (Non-PDF)
- Detection: url lacks `.pdf` and matches `/abs/`, `/doi/`, `/article/`, or venue landing pages.
- Plan: convert to a PDF URL when a stable pattern exists; otherwise locate a publicly accessible PDF and replace; if none, flag.

## HTML Full-Text Render (Non-PDF)
- Detection: url contains `/html/` or `/full/`.
- Plan: replace with the canonical PDF on the same host (e.g., arXiv `/pdf/ID.pdf`); if unavailable, treat as non-PDF.

## Non-paper Context URL
- Detection: url points to a competition, wiki, blog, or project page rather than a paper.
- Plan: find the actual paper PDF and replace; if no paper exists, drop from output.

## Non-canonical Mirror Source
- Detection: domains like archive.org, personal homepages, lab sites, institutional repositories, or paths with `~`/`dspace`.
- Plan: replace with the official publisher/venue PDF or canonical arXiv PDF; keep mirror only when it is the only publicly available PDF.

## Version-specific arXiv URL
- Detection: arXiv URL ends with `v#` (e.g., `v2`).
- Plan: normalize to the versionless canonical PDF `https://arxiv.org/pdf/ID.pdf`.

## Inconsistent arXiv URL Forms
- Detection: mix of `/abs/`, `/pdf/`, `/html/` arXiv URLs.
- Plan: standardize all arXiv entries to `https://arxiv.org/pdf/ID.pdf` to meet the PDF requirement.

## Year-URL Metadata Mismatch
- Detection: year inferred from URL/DOI conflicts with the `year` column.
- Plan: verify year from authoritative metadata; correct the year or update the URL; if ambiguous, flag for review.

## Exact Duplicate Row
- Detection: identical `year`, `title`, `url` rows repeat.
- Plan: deduplicate by keeping a single row per unique key.

## Same Work Listed Multiple Times (Different URLs)
- Detection: same normalized title/year appears with different URLs.
- Plan: choose one canonical PDF URL using the priority order below, then merge rows.

## Title Variants or Aliases
- Detection: case changes, parentheticals, or minor wording differences.
- Plan: normalize titles for matching (casefold, remove parentheticals/aliases); keep one canonical title in output and merge duplicates.

## Canonical PDF Selection Priority
1) Official publisher/venue PDF that is publicly accessible.
2) arXiv PDF.
3) Open-access repository mirror (only if no official PDF is available).
