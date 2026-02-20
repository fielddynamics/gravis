# SPARC sources (175 galaxies)

To build all 175 galaxy JSON files used by "Add observation from SPARC":

1. **Get table2.dat** (mass models: rotation curves and photometry per galaxy).
   - Download from VizieR: https://cdsarc.cds.unistra.fr/ftp/J/AJ/152/157/table2.dat
   - Save as `data/table2.dat`.
   - Or use a zip that contains both `table1.dat` and `table2.dat` and run with `--zip path/to/zip`.

2. **Run the build script** (from repo root):
   ```
   python data/build_sparc_sources.py
   ```
   The script uses `data/sparc_table1.dat` (or `table1.dat`) and `data/table2.dat`. It will try to download table2 if missing (may fail on some networks due to SSL). Each galaxy is validated to have photometric data and must pass the app schema before being written.

3. **Output**: `data/sparc_sources/<id>.json` for each of the 175 galaxies. The import list in the app will then show all 175 and allow import by id.

ASCII only; no unicode in scripts or output.
