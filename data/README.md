# `data/`

User-owned raw inputs. Exploratory data analysis reads from here
(or another path you choose) and never modifies the files. Reports
and notebooks live in `data_analysis/`. Large local files can be
gitignored by pattern (`raw/`, `*.parquet`); do not ignore all of
`data/` if you want small samples committed.
