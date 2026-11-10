# Final visual review

The final paper has 12 pages, with 11 pages of body text and the AI acknowledgment and references on page 12. The title, both corresponding authors, ORCIDs, mechanism figure, algorithms, result tables and quantitative figures were visually inspected. All 38 PDF fonts are embedded. No overfull boxes, unresolved references or manuscript placeholders remain. The last reference page is balanced using the standard IEEE bibliography trigger.

All three regenerated quantitative PDFs pass collision checks with zero FAIL and zero WARN findings. The minimum rendered glyph is 5.04 pt in each. The main scatter and sensitivity figure pass the 1.5 pt multi-panel alignment gate; decomposition is a single panel. Every panel and its legend was inspected. The retained mechanism and single-decision plots have their existing source-to-figure mappings.

The source preflight has two reviewed warnings. TIFF is unnecessary for the submission because the included figures are vector PDFs; 600 dpi PNGs are previews. All displayed log-scale values are positive. The geometric-mean helper explicitly tests positivity, and the zero-call K=5 Top-5 result is not in the displayed K=10,20,30 panels. Main-scatter values exceed its lower plotting bound. There is no hidden removal of main queries.

The scatter points are per-query means over 30 sampling runs. The decomposition aggregates those same queries; sensitivity points use 10 runs per query and the stated geometric aggregation. Their purpose is cost comparison, with run-to-run variation reported in the paper and retained in the supplied summaries and execution records. The figures do not imply confidence intervals.
