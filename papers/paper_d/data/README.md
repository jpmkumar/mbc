# Paper D — local BreakHis archive (not on GitHub)

Place the official UFPR tarball here:

```text
papers/paper_d/data/BreaKHis_v1.tar.gz
```

This directory is gitignored. Do not commit the archive.

## Copy from Downloads (run in macOS Terminal)

Cursor cannot read `~/Downloads` under macOS privacy. After the browser
download finishes, run:

```bash
cd /Users/muthu/ResTest/paper1/mbc
./papers/paper_d/scripts/install_breakhis_archive.sh
```

Or point at an explicit path:

```bash
./papers/paper_d/scripts/install_breakhis_archive.sh ~/Downloads/BreaKHis_v1.tar.gz
```

The script waits until the file size is stable, copies into this folder, prints
SHA-256 (for `paper_d_lock.json`), and refuses to overwrite without `--force`.
