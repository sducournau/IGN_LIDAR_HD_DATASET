# Project Consolidation Workflow

## Trigger Command
When the user types **"consolidate project"**, execute the following comprehensive workflow:

---

## Phase 1: Codebase Analysis
1. **Analyze project structure**
   - List all modules in `ign_lidar/`
   - Identify core vs. auxiliary modules
   - Map dependencies between modules

2. **Scan for duplicate implementations**
   - Search for similar function names across modules
   - Identify redundant class implementations
   - Check for parallel features/utilities

3. **Check for naming anti-patterns**
   - Files/classes with prefixes: `enhanced_`, `unified_`, `new_`, `improved_`, `v2_`
   - Functions with redundant prefixes
   - Variables with unclear naming

---

## Phase 2: Quality Audit
1. **Code quality assessment**
   - Check for unused imports
   - Identify deprecated functions
   - Find TODO/FIXME comments
   - Check for inconsistent error handling

2. **Documentation audit**
   - Missing docstrings
   - Outdated documentation
   - Inconsistent formatting

3. **Test coverage analysis**
   - Identify untested modules
   - Check for obsolete tests
   - Verify integration test completeness

---

## Phase 3: Consolidation Actions

### 3.1 Merge Duplicate Features
- **Strategy**: Keep the most recent/complete implementation
- **Process**:
  1. Identify all duplicates
  2. Compare implementations (feature completeness, performance)
  3. Migrate any unique functionality from old to new
  4. Update all references to point to consolidated version
  5. Delete old implementation
  6. Update tests

### 3.2 Remove Redundant Naming
- **Files to rename**:
  - `enhanced_*.py` → base name
  - `unified_*.py` → base name
  - `new_*.py` → base name
  
- **Functions/Classes to rename**:
  - `enhanced_process_*` → `process_*`
  - `unified_compute_*` → `compute_*`
  - `new_classifier_*` → `classifier_*`

- **Process**:
  1. Use `mcp_oraios_serena_rename_symbol` for each rename
  2. Verify all references updated
  3. Update documentation
  4. Run tests to confirm no breakage

### 3.3 Clean Repository Structure

#### Root Directory Cleanup
- **Archive documentation files**:
  - Move `.md` files (except README.md, CHANGELOG.md, LICENSE) to `docs/archive/`
  - Move old API docs to `docs/archive/api/`
  
- **Remove obsolete files**:
  - Old configuration examples (keep only latest in `examples/`)
  - Temporary test files
  - Outdated scripts in `scripts/`

#### Code Cleanup
- **Remove deprecated code**:
  - Delete functions/classes marked as deprecated for >2 versions
  - Remove commented-out code blocks
  - Clean up unused utility functions

- **Consolidate utilities**:
  - Merge scattered utility functions into proper modules
  - Remove duplicate utility implementations

---

## Phase 4: Improvements

### 4.1 Code Organization
- Ensure all modules follow single responsibility principle
- Move misplaced functions to appropriate modules
- Create missing `__init__.py` imports for cleaner API

### 4.2 Performance Optimization
- Identify performance bottlenecks from profiling
- Consolidate redundant computations
- Add caching where appropriate

### 4.3 Error Handling
- Standardize exception types
- Add missing error handling
- Improve error messages for user clarity

---

## Phase 5: Verification & Documentation

### 5.1 Testing
- Run full test suite: `pytest tests/ -v`
- Run integration tests: `pytest tests/ -v -m integration`
- Check coverage: `pytest tests/ -v --cov=ign_lidar --cov-report=html`
- Fix any broken tests

### 5.2 Documentation Update
- Update `README.md` with consolidated structure
- Refresh `CHANGELOG.md` with consolidation notes
- Update API documentation
- Refresh examples to use consolidated APIs

### 5.3 Final Verification
- Check imports in all modules
- Verify no broken references
- Ensure backward compatibility warnings in place
- Review git diff for sanity check

---

## Execution Order
1. **Analysis first** - Never delete/modify before full analysis
2. **One module at a time** - Complete consolidation per module before moving on
3. **Test after each change** - Run relevant tests after each consolidation
4. **Document as you go** - Update documentation immediately after changes
5. **Use symbolic tools** - Always use Serena's symbolic editing tools (replace_symbol_body, rename_symbol, etc.)

---

## Key Principles
- ✅ **Modify existing > Create new** - Always upgrade existing code first
- ✅ **Merge duplicates** - One canonical implementation per feature
- ✅ **Clear naming** - No redundant prefixes (enhanced, unified, new, etc.)
- ✅ **Archive, don't delete** - Move old docs to archives before deletion
- ✅ **Test everything** - Verify after each consolidation step
- ✅ **Backward compatibility** - Add deprecation warnings for removed APIs

---

## Tools to Use
- `mcp_oraios_serena_find_symbol` - Locate code elements
- `mcp_oraios_serena_find_referencing_symbols` - Find all usages
- `mcp_oraios_serena_rename_symbol` - Safe renaming across codebase
- `mcp_oraios_serena_replace_symbol_body` - Update implementations
- `mcp_oraios_serena_search_for_pattern` - Find anti-patterns
- `mcp_oraios_serena_think_about_collected_information` - Pause to analyze
- `mcp_oraios_serena_think_about_task_adherence` - Verify on track

---

## Output Format
After consolidation, provide:
1. **Summary of changes**
   - Files renamed/deleted
   - Functions/classes consolidated
   - Duplicates removed
   
2. **Quality metrics**
   - Test coverage before/after
   - Number of deprecations removed
   - Code duplication percentage reduction

3. **Action items**
   - Remaining TODOs
   - Suggested follow-up improvements
   - Breaking changes (if any)
