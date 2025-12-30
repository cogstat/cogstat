# Contributing to CogStat

Developer guide for CogStat contributors. For detailed architecture and component interactions, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Project Overview

CogStat is a statistical analysis package designed for researchers. It provides a PyQt6-based GUI for statistical analysis with support for multiple data formats, localization, and comprehensive statistical methods. The project emphasizes automatic statistical decision-making to simplify analysis for researchers.

**Key Technologies:**
- Python 3.7+
- PyQt6 for GUI
- Statistical libraries: scipy, statsmodels, pingouin, scikit-posthocs
- Data handling: pandas, numpy
- Visualization: matplotlib
- R integration via rpy2 (optional)

## Development Commands

### Running the Application

```bash
# Run GUI from command line
python run_cogstat_gui.py

# Run as module
python -m cogstat
```

### Testing

Tests use Python's unittest framework:

```bash
# Run calculation validation tests
python -m unittest cogstat/test/validate_calculations.py

# Run specific test class
python -m unittest cogstat.test.validate_calculations.CogStatTestCase

# Run specific test method
python -m unittest cogstat.test.validate_calculations.CogStatTestCase.test_explore_variables
```

**Important:** Tests validate statistical calculations against other software (SPSS, jamovi, JASP, R). When fixing calculation bugs, always add warnings to release notes ([changelog.md](changelog.md)).

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install as package (development mode)
pip install -e .

# Install as package
python setup.py install
```

### Building Platform-Specific Versions

```bash
# macOS application bundle
python setup-mac.py py2app

# Windows installer (requires Inno Setup)
# Use cogstat_win.iss with Inno Setup Compiler

# Code signing (macOS Intel with Python 3.11)
./codesign-cogstat-intel-python3_11.sh
```

## File Organization

```
cogstat/
├── cogstat.py           # Core engine: CogStatData class
├── cogstat_gui.py       # Main window: StatMainWindow
├── cogstat_dialogs.py   # All analysis dialogs
├── cogstat_stat.py      # Descriptive statistics
├── cogstat_hyp_test.py  # Hypothesis tests
├── cogstat_chart.py     # Matplotlib visualizations
├── cogstat_config.py    # Settings, themes, version info
├── cogstat_util.py      # Output conversion utilities
├── cogstat_stat_num.py  # Low-level numerical functions
├── ui/                  # Qt Designer .ui files and generated Python
├── locale/              # Translations (gettext .po/.mo files)
├── resources/           # Icons, splash screen
├── demo_data/           # Example datasets
├── docs/                # Jupyter notebook tutorials
└── test/                # Unit tests and validation
```

## Development Guidelines

### Data Import
- All import methods funnel through `CogStatData._import_data()`
- Supports: pandas DataFrame, clipboard, files (CSV, Excel, SPSS, R, JASP, jamovi, SAS, STATA, ODS)
- First row must be variable names
- Optional second row for measurement levels (`nom`, `ord`, `int`, `unk`)

### Measurement Levels

Variables are classified as:
- `'nom'` - Nominal (categorical, unordered)
- `'ord'` - Ordinal (categorical, ordered)
- `'int'` - Interval/ratio (continuous)
- `'unk'` - Unknown (treated as interval for analysis)

Priority for assignment: default (`unk`) → file metadata → second row → parameter → constraints (strings → `nom`)

### Statistical Validation
- All calculated values must be tested in `validate_calculations.py`
- Validate against popular software (SPSS, jamovi, JASP, R packages)
- Document validation in test comments: `# software version result`
- Use 3-digit decimal precision for test data

### Error Handling
- Broken analyses display standardized error message with bug report link
- Set `detailed_error_message = True` in preferences for full tracebacks

### Chart Output
- Charts return matplotlib Figure objects
- Image format: PNG (default) or SVG (experimental)
- Size controlled by `fig_size_x` and `fig_size_y` in config

### Filtering and Outliers
- Default method: median ± 2.5 × MAD
- Filtering status tracked in `CogStatData.filtering_status`
- Always display filtering status in analysis output

## Code Conventions

### Analysis Method Structure

```python
def analysis_method(self, var_name, ...):
    # 1. Initialize results with expected keys
    results = {key: None for key in ['analysis info', 'warning', ...]}
    
    # 2. Add main heading
    results['analysis info'] = '<cs_h1>' + _('Analysis Name') + '</cs_h1>'
    
    # 3. Check preconditions
    if not precondition:
        results['warning'] = _('Error message')
        return cs_util.convert_output(results)
    
    # 4. Add filtering status
    results['analysis info'] += self._filtering_status()
    
    # 5. Populate sections
    results['raw data info'] = '<cs_h2>' + _('Raw data') + '</cs_h2>'
    results['raw data chart'] = cs_chart.create_chart(...)
    # ... more sections
    
    # 6. Return processed results
    return cs_util.convert_output(results)
```

### Custom HTML Tags

Use these tags (converted to HTML by `cogstat_config.cs_tags`):
- `<cs_h1>` through `<cs_h4>` - Headings
- `<cs_decision>` - Statistical decision explanations
- `<cs_warning>` - Warning messages
- `<cs_fix_width_font>` - Monospace text

### Localization

All user-facing strings must use gettext:
```python
from . import cogstat_config as csc
_ = csc._  # Translation function

message = _('This text will be translated')
```

## UI Development

### Creating/Modifying Dialogs

1. Edit `.ui` file in Qt Designer
2. Generate Python wrapper:
   ```bash
   pyuic6 cogstat/ui/dialog_name.ui -o cogstat/ui/dialog_name.py
   ```
3. Dialog class in `cogstat_dialogs.py` inherits from both `QDialog` and generated UI class

### Dialog Pattern

```python
class my_dialog(QtWidgets.QDialog, my_dialog_ui.Ui_Dialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        # Connect signals
        
    def init_vars(self, names):
        # Populate variable lists from data columns
        
    def read_parameters(self):
        # Return tuple of user selections
        return (selected_vars, option1, option2)
```

## Known Issues & Considerations

- **R integration:** May require manual `R_HOME` configuration on Windows
- **macOS:** Requires code signing for distribution
- **High DPI:** `app_devicePixelRatio` set from PyQt6
- **Validation gaps:** Some statistics not available in popular software - document when validation impossible
- **Excluded versions:** scipy 1.10, matplotlib 2.0.1/2.0.2 (compatibility bugs)

## Quick Reference

| Task | Location |
|------|----------|
| Add new analysis | `cogstat.py` (method) → `cogstat_dialogs.py` (dialog) → `cogstat_gui.py` (menu) |
| Add new chart type | `cogstat_chart.py` |
| Add statistical test | `cogstat_hyp_test.py` |
| Modify settings | `cogstat_config.py` |
| Add translation | `cogstat/locale/` |
| Add test | `cogstat/test/validate_calculations.py` |

## Related Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) - Detailed component interactions and data flows
- [README.md](README.md) - User documentation
- [changelog.md](changelog.md) - Release notes
