# Corrected quantitative figure contract

The established Python backend and existing visual layouts are reused. Quantitative marks are regenerated from actual-prefix replay summaries. No observations, methods or queries are removed from the plotted comparison.

- Main scatter. Each point represents one complete query, with its cost averaged over 30 sampling executions. Three matched panels compare VILLE with the three certified baseline families using the same queries. The panels share scale, dimensions and labels. Repeated sampling uncertainty is summarized separately in the manuscript's cost-CV statement, avoiding unreadable overlapping intervals on the 75-point scatter.
- Decomposition. Dataset-level geometric means compare the supplied confidence-sequence and boundary/paired implementations with and without proxies. The combined betting and paired-construction effect is the stated estimand. The display shows implementation-family comparisons across the same queries, with equal dataset membership.
- Sensitivity. Ten sampling executions per query underpin every point. Proxy and margin panels aggregate the year and product groupings using the stated geometric mean. The group-count panel compares ORDER BY and Top-5 for K=10,20,30. Every curve retains the same aggregation convention and query membership as the original study.

All exported PDF glyphs must reach 5 points at final physical size. The plots retain editable PDF/SVG text. PNG previews are exported at 600 dpi. Multi-panel alignment is measured at rendering, and the final PDFs receive text and collision audits. Figures are inspected individually and at manuscript placement. The mechanism diagram and single-decision trajectory use their existing source data and retain their own scientific mapping.
