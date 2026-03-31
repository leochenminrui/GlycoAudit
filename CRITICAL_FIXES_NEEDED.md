# Critical Fixes Needed Before Public Release

**Generated:** 2026-03-31
**Status:** BLOCKING - Must fix before GitHub push

---

## 🔴 CRITICAL: Hardcoded Paths (MUST FIX)

### Issue
Multiple files contain hardcoded absolute paths to `/home/minrui/glyco/public_glyco_mirror/`

### Affected Files

1. **src/glycoaudit/analysis/splits.py** (3 occurrences)
   ```python
   glytoucan_path: Path = Path("/home/minrui/glyco/public_glyco_mirror/data/raw/glytoucan/bulk_export.json")
   ```

2. **src/glycoaudit/analysis/structure_keys.py** (1 occurrence)
   ```python
   glytoucan_path: Path = Path("/home/minrui/glyco/public_glyco_mirror/data/raw/glytoucan/bulk_export.json")
   ```

3. **src/glycoaudit/analysis/findings_v1/*.py** (4 files)
   ```python
   BASE_PATH = Path("/home/minrui/glyco/public_glyco_mirror")
   ```
   - `make_findings_figures.py`
   - `run_level3_hypothesis_tests.py`
   - `compute_level1_preferences.py`
   - `run_level2_ablation.py`

4. **scripts/release/build_summary.py**
   ```python
   BENCHMARK_PATH = Path("/home/minrui/glyco/public_glyco_mirror/benchmark_release_v1")
   ```

5. **scripts/evaluate/e7_preferences.py**
   ```python
   OUTPUT_DIR = Path('/home/minrui/glyco/public_glyco_mirror/results/RELEASE_RUN_V3_INTERPRETABILITY/E7_ssv_preferences')
   ```

### Fix Strategy

Replace all hardcoded paths with **relative paths** or **configurable base paths**:

```python
# BEFORE (BAD)
BASE_PATH = Path("/home/minrui/glyco/public_glyco_mirror")

# AFTER (GOOD - Option 1: Relative to script)
import os
SCRIPT_DIR = Path(__file__).parent
BASE_PATH = SCRIPT_DIR.parent.parent  # Adjust as needed

# AFTER (GOOD - Option 2: Environment variable)
BASE_PATH = Path(os.getenv("GLYCOAUDIT_BASE", Path.cwd()))

# AFTER (GOOD - Option 3: Config file)
from glycoaudit.config import get_base_path
BASE_PATH = get_base_path()
```

### Action Items

- [ ] Create `src/glycoaudit/config.py` with path resolution logic
- [ ] Replace all hardcoded paths in affected files
- [ ] Test all scripts work from different directories
- [ ] Add path tests to test suite

---

## ⚠️ HIGH PRIORITY: Authentication Tokens

### Issue
`src/glycoaudit/mirror/sources/glycam_web.py` references authentication tokens

### Current Code
```python
self.auth_token = config.glycam_auth_token
# References: GLYCAM_AUTH_TOKEN environment variable
```

### Status
✅ **GOOD** - Uses environment variable, not hardcoded
⚠️ **WARNING** - Ensure no actual tokens are committed

### Action Items

- [ ] Verify no `.env` files with tokens in repository
- [ ] Add `.env.example` template
- [ ] Document required environment variables in README

---

## 🟡 MEDIUM PRIORITY: Version Suffixes

### Issue
Some copied files still have version suffixes (`_v0`, `_v1`, `_v2`, `expanded_v1`)

### Affected Areas
- `benchmark_results/E1_ensemble_sensitivity_v2/`
- `benchmark_results/E5_split_robustness_v2/`
- File references to `expanded_v1` in scripts

### Recommendation
**KEEP AS-IS** for now - These indicate data provenance and experiment versions. Only rename if it causes confusion.

### Action Items

- [ ] Document version naming convention in `docs/benchmark_schema.md`
- [ ] Clarify that `_v2` = "version 2 with improvements", not "draft 2"

---

## 🟡 MEDIUM PRIORITY: Missing Data Files

### Issue
Repository structure created but actual data files not present (by design, but needs docs)

### Expected Behavior
- `data/raw/`, `data/ssv/`, etc. are empty (ignored by `.gitignore`)
- Users download data via scripts or pre-computed archive

### Action Items

- [x] Created `data/README.md` explaining download process
- [ ] Test download scripts work on clean checkout
- [ ] Create pre-computed data archive for Zenodo
- [ ] Add download links to main README

---

## 🟢 LOW PRIORITY: Import Path Updates

### Issue
Code still imports from old module names (e.g., `from mirror.sources import ...`)

### Fix Strategy
With new package structure `glycoaudit`, update imports:

```python
# OLD
from mirror.sources import glytoucan

# NEW
from glycoaudit.mirror.sources import glytoucan
```

### Action Items

- [ ] Update all imports in `src/glycoaudit/`
- [ ] Update all imports in `scripts/`
- [ ] Test imports work: `python -c "import glycoaudit"`

---

## 🟢 LOW PRIORITY: Script Naming Consistency

### Issue
Scripts renamed but may still reference old paths internally

### Action Items

- [ ] Check all scripts for internal path references
- [ ] Update any README/docs that mention old script names
- [ ] Create wrapper scripts if needed for backward compatibility

---

## Testing Checklist

Before declaring "publication ready":

### Basic Functionality
- [ ] `python -m glycoaudit` works
- [ ] All scripts run without import errors
- [ ] Download script works from clean directory
- [ ] Feature computation works on sample data
- [ ] Benchmark evaluation runs on sample data

### Path Independence
- [ ] Clone repo to different directory, test still works
- [ ] Run scripts from different working directories
- [ ] No hardcoded `/home/minrui/` paths trigger

### Security
- [ ] No credentials committed
- [ ] No private paths exposed
- [ ] `.gitignore` properly excludes sensitive files

### Documentation
- [ ] README has correct installation instructions
- [ ] All docs reference correct script names
- [ ] Example commands actually work

---

## Estimated Fix Time

| Priority | Task | Time |
|----------|------|------|
| 🔴 CRITICAL | Fix hardcoded paths | 2-3 hours |
| ⚠️ HIGH | Verify no secrets | 30 minutes |
| 🟡 MEDIUM | Document versions | 1 hour |
| 🟡 MEDIUM | Test data download | 1 hour |
| 🟢 LOW | Update imports | 1-2 hours |
| 🟢 LOW | Test suite | 2-3 hours |
| **TOTAL** | **Before initial release** | **8-11 hours** |

---

## Immediate Next Steps (Priority Order)

1. **Fix hardcoded paths** (CRITICAL)
   - Create `src/glycoaudit/config.py`
   - Replace paths in 10 affected files
   - Test from different directory

2. **Verify security** (HIGH)
   - Grep for secrets/tokens
   - Check `.env` files
   - Verify `.gitignore` works

3. **Test basic workflow** (HIGH)
   - Fresh clone
   - Install dependencies
   - Run sample benchmark
   - Verify no crashes

4. **Update documentation** (MEDIUM)
   - Fix any broken links
   - Update script references
   - Add troubleshooting section

5. **Create release checklist** (MEDIUM)
   - Pre-submission checklist
   - Post-publication TODO
   - Maintenance plan

---

## Files Generated in This Cleanup

### ✅ Created
- `GlycoAudit_clean/` - New repository structure
- `README.md` - Main documentation
- `LICENSE` - MIT + CC-BY for data
- `CITATION.cff` - Citation metadata
- `.gitignore` - Proper exclusions
- `docs/quickstart.md` - User guide
- `data/README.md` - Data documentation
- **THIS FILE** - Critical issues tracker

### 📁 Copied (with cleanup)
- `src/glycoaudit/` - Core modules
- `scripts/` - Essential scripts (organized into subdirs)
- `benchmark_results/` - All paper results
- `configs/` - Configuration files

### 🗑️ Excluded (not copied)
- 19+ internal development `.md` files
- 10+ root-level summary docs
- Old version scripts (v0, v1, v2)
- Obsolete `results/` and `reports/` directories
- Test/debug scripts
- LaTeX manuscript drafts
- `__pycache__/` and cache files

---

## Next Session TODO

1. Create `src/glycoaudit/config.py`
2. Run search-and-replace for hardcoded paths
3. Test installation on fresh clone
4. Create sample data for testing
5. Write `tests/test_basic.py`
6. Document remaining TODOs in GitHub Issues

---

**Status:** Repository structure complete, but CRITICAL path fixes needed before public release.
