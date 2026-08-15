# Manuscript source

The canonical LaTeX lives one level up from the repo `paper/` directory:

- [`../../paper/main.tex`](../../paper/main.tex)
- [`../../paper/references.bib`](../../paper/references.bib)

Build from repo root:

```bash
make -C paper pdf
```

Paper B does not maintain a duplicate TeX tree. Edit files under `paper/` directly.
