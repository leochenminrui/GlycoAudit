# Pre-Push Checklist for GlycoAudit

**Complete these tasks before pushing to GitHub**

---

## 🔴 CRITICAL (Must Complete)

### 1. Fix Hardcoded Paths (~3 hours)

#### Step 1.1: Create config module
- [ ] Create `src/glycoaudit/config.py`:
```python
from pathlib import Path
import os

def get_base_path() -> Path:
    """Get project base path (repository root)."""
    # Option 1: From environment variable
    if "GLYCOAUDIT_BASE" in os.environ:
        return Path(os.environ["GLYCOAUDIT_BASE"])

    # Option 2: Relative to this file
    return Path(__file__).parent.parent.parent

def get_data_path() -> Path:
    """Get data directory path."""
    return get_base_path() / "data"

def get_glytoucan_path() -> Path:
    """Get GlyTouCan bulk export path."""
    return get_data_path() / "raw" / "glytoucan" / "bulk_export.json"

def get_benchmark_path() -> Path:
    """Get benchmark results path."""
    return get_base_path() / "benchmark_results"
```

#### Step 1.2: Fix affected files
- [ ] `src/glycoaudit/analysis/splits.py` (3 occurrences)
  - Replace: `Path("/home/minrui/glyco/public_glyco_mirror/data/raw/glytoucan/bulk_export.json")`
  - With: `from glycoaudit.config import get_glytoucan_path; ... get_glytoucan_path()`

- [ ] `src/glycoaudit/analysis/structure_keys.py` (1 occurrence)
  - Same fix as above

- [ ] `src/glycoaudit/analysis/findings_v1/make_findings_figures.py`
  - Replace: `BASE_PATH = Path("/home/minrui/glyco/public_glyco_mirror")`
  - With: `from glycoaudit.config import get_base_path; BASE_PATH = get_base_path()`

- [ ] `src/glycoaudit/analysis/findings_v1/run_level3_hypothesis_tests.py`
  - Same fix as above

- [ ] `src/glycoaudit/analysis/findings_v1/compute_level1_preferences.py`
  - Same fix as above

- [ ] `src/glycoaudit/analysis/findings_v1/run_level2_ablation.py`
  - Same fix as above

- [ ] `scripts/release/build_summary.py`
  - Replace: `BENCHMARK_PATH = Path("/home/minrui/glyco/public_glyco_mirror/benchmark_release_v1")`
  - With: `from glycoaudit.config import get_benchmark_path; BENCHMARK_PATH = get_benchmark_path()`

- [ ] `scripts/evaluate/e7_preferences.py`
  - Replace: `OUTPUT_DIR = Path('/home/minrui/glyco/public_glyco_mirror/results/...')`
  - With: `OUTPUT_DIR = get_base_path() / "benchmark_results" / "ssv_preferences"`

#### Step 1.3: Test from different directory
```bash
# Test 1: From root
cd /home/minrui/glyco/GlycoAudit_clean
python -c "from glycoaudit.config import get_base_path; print(get_base_path())"

# Test 2: From subdirectory
cd /home/minrui/glyco/GlycoAudit_clean/scripts
python -c "from glycoaudit.config import get_base_path; print(get_base_path())"

# Test 3: From different location entirely
cd /tmp
python -c "import sys; sys.path.insert(0, '/home/minrui/glyco/GlycoAudit_clean/src'); from glycoaudit.config import get_base_path; print(get_base_path())"
```

---

### 2. Update Package Imports (~2 hours)

- [ ] Update imports in `src/glycoaudit/`:
```bash
# Find all imports that need updating
cd src/glycoaudit
grep -r "^from mirror" . --include="*.py"
grep -r "^from analysis" . --include="*.py"
grep -r "^import mirror" . --include="*.py"
```

- [ ] Fix import patterns:
```python
# OLD (wrong)
from mirror.sources import glytoucan
from analysis.splits import generate_scaffold_split

# NEW (correct)
from glycoaudit.mirror.sources import glytoucan
from glycoaudit.analysis.splits import generate_scaffold_split
```

- [ ] Test all imports:
```bash
python -c "import glycoaudit"
python -c "from glycoaudit.mirror.sources import glytoucan"
python -c "from glycoaudit.analysis import splits"
python -c "from glycoaudit.config import get_base_path"
```

---

## ⚠️ HIGH PRIORITY (Should Complete)

### 3. Test Clean Checkout (~2 hours)

#### Step 3.1: Simulate fresh clone
```bash
# Create test directory
mkdir -p /tmp/glycoaudit_test
cd /tmp/glycoaudit_test

# Copy clean repo
cp -r /home/minrui/glyco/GlycoAudit_clean/* .
```

#### Step 3.2: Test installation
```bash
# Test conda install
conda env create -f environment.yml
conda activate glycoaudit

# Test package import
python -c "import glycoaudit"
```

#### Step 3.3: Test sample workflow (if sample data exists)
```bash
# Test feature computation (if sample data available)
python scripts/compute/compute_ssv.py --help

# Test evaluation (if sample data available)
python scripts/evaluate/run_benchmark.py --help
```

- [ ] No import errors
- [ ] No path errors
- [ ] Help messages work

---

### 4. Security Audit (~30 min)

- [ ] No credentials:
```bash
cd /home/minrui/glyco/GlycoAudit_clean
grep -r "password\|token\|secret\|api_key" . --include="*.py" | grep -v "# "
```

- [ ] No private paths:
```bash
grep -r "/home/minrui" . --include="*.py"
grep -r "minrui" . --include="*.py"
```

- [ ] `.gitignore` works:
```bash
# Check what would be committed
git init
git add .
git status

# Should NOT include:
# - data/raw/
# - __pycache__/
# - *.pyc
# - logs/
```

- [ ] No large files:
```bash
find . -type f -size +1M
# Should only be benchmark result CSVs
```

---

### 5. Documentation Review (~1 hour)

- [ ] README.md accurate:
  - [ ] Installation instructions work
  - [ ] Quick start commands work
  - [ ] All links valid
  - [ ] Statistics up-to-date
  - [ ] GitHub URLs updated (not placeholder)

- [ ] docs/quickstart.md tested:
  - [ ] Commands copy-paste-able
  - [ ] Examples run
  - [ ] Troubleshooting section helpful

- [ ] Script paths updated in all docs:
  - [ ] No references to old script names
  - [ ] All `scripts/*/*.py` paths correct

---

## 🟡 MEDIUM PRIORITY (Nice to Have)

### 6. Create Sample Data (~1 hour)

- [ ] Extract 10 glycans as sample:
```bash
mkdir -p data/sample_outputs
head -11 data/ssv/expanded_v1/ssv_features.csv > data/sample_outputs/ssv_sample.csv
```

- [ ] Test scripts work with sample data

---

### 7. Add Basic Tests (~2 hours)

- [ ] Create `tests/test_imports.py`:
```python
def test_import_glycoaudit():
    import glycoaudit

def test_import_submodules():
    from glycoaudit import mirror, analysis, evaluation

def test_config_paths():
    from glycoaudit.config import get_base_path
    assert get_base_path().exists()
```

- [ ] Create `tests/test_config.py`:
```python
def test_base_path():
    from glycoaudit.config import get_base_path
    base = get_base_path()
    assert base.name == "GlycoAudit_clean" or "GlycoAudit" in str(base)

def test_data_path():
    from glycoaudit.config import get_data_path
    data = get_data_path()
    assert data.name == "data"
```

- [ ] Run tests:
```bash
pip install pytest
pytest tests/
```

---

### 8. Setup Version Control (~30 min)

- [ ] Initialize git:
```bash
cd /home/minrui/glyco/GlycoAudit_clean
git init
git add .
git status  # Review what's being added
```

- [ ] Check repo size:
```bash
du -sh .git
# Should be <10MB
```

- [ ] Create first commit:
```bash
git commit -m "Initial commit: GlycoAudit v1.0.0

Publication-ready repository for glycan-binding benchmark.

- Core modules: mirror, features, evaluation, analysis
- Scripts: 20 essential scripts organized by function
- Benchmark results: E1-E11, cross-source, case studies
- Documentation: README, quickstart, data guide
- Configuration: environment.yml, requirements.txt

Closes #1 (repository restructuring)
"
```

---

## 🟢 LOW PRIORITY (Future)

### 9. Advanced Features

- [ ] CI/CD setup (GitHub Actions)
- [ ] Comprehensive test suite
- [ ] Tutorial notebooks
- [ ] API documentation (Sphinx)
- [ ] Contribution guidelines
- [ ] Issue templates
- [ ] Pre-commit hooks

---

## Final Verification

### Before Pushing to GitHub

- [ ] All 🔴 CRITICAL tasks complete
- [ ] All ⚠️ HIGH PRIORITY tasks complete
- [ ] Repo size <10MB
- [ ] No hardcoded paths found: `grep -r "/home/minrui" . --include="*.py"` returns empty
- [ ] No secrets found: `grep -r "password\|token" . --include="*.py" | grep -v "#"` returns only docs
- [ ] README tested: All commands work
- [ ] Sample workflow tested: At least one script runs
- [ ] `.gitignore` verified: No data files in git status

### Create GitHub Repository

```bash
# On GitHub:
# 1. Create new repository "GlycoAudit"
# 2. Make it PUBLIC (after verification) or PRIVATE (for review)
# 3. Do NOT initialize with README (we have one)

# Locally:
cd /home/minrui/glyco/GlycoAudit_clean
git remote add origin https://github.com/YOUR_USERNAME/GlycoAudit.git
git branch -M main
git push -u origin main
```

### Post-Push Verification

- [ ] Repository loads on GitHub
- [ ] README renders correctly
- [ ] Files organized properly
- [ ] No large files warning
- [ ] Clone from GitHub works:
```bash
cd /tmp
git clone https://github.com/YOUR_USERNAME/GlycoAudit.git
cd GlycoAudit
conda env create -f environment.yml
```

---

## Estimated Time

| Priority | Tasks | Time |
|----------|-------|------|
| 🔴 CRITICAL | Fix paths, imports | 5 hours |
| ⚠️ HIGH | Testing, security, docs | 3-4 hours |
| 🟡 MEDIUM | Sample data, tests | 3 hours |
| 🟢 LOW | Advanced features | Future |
| **TOTAL (for push)** | **8-9 hours** | |

---

## Current Status

- [x] Audit complete
- [x] Repository structure created
- [x] Files migrated
- [x] Documentation written
- [ ] Paths fixed 🔴
- [ ] Imports updated 🔴
- [ ] Testing complete ⚠️
- [ ] Ready to push ⏳

---

## Questions / Decisions Needed

1. **Repository name:** `GlycoAudit` or `glycan-binding-benchmark`?
2. **License:** MIT confirmed? (yes, MIT + CC-BY for data)
3. **Author info:** Update CITATION.cff with real names/ORCIDs
4. **GitHub org:** Personal account or organization?
5. **Public vs Private:** Start private for review, then public?

---

**Track progress:** Check off items as completed
**Questions?** See `CRITICAL_FIXES_NEEDED.md` or `GLYCOAUDIT_RELEASE_REPORT.md`
