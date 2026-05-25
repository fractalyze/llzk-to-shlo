# Documentation style

## Markdown footnotes

The pre-commit `mdformat` hook runs with `mdformat-gfm` and
`mdformat-frontmatter` only — **no** `mdformat-footnote`. Raw GFM footnote
definitions (`[^id]: text`) are escaped to `\[^id\]: text` on autoformat. Use
**inline numeric markers** (`¹` `²` …) and a "Notes:" sub-list immediately under
the table. To switch styles, add `mdformat-footnote` to
`.pre-commit-config.yaml`'s `mdformat` `additional_dependencies` first.
