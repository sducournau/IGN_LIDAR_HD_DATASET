# Task Completion Checklist

When you complete a coding task, follow this checklist:

## 1. Code Quality
- [ ] **Format code:** Run `black ign_lidar tests` to format
- [ ] **Sort imports:** Run `isort ign_lidar tests --profile=black`
- [ ] **Lint code:** Run `flake8 ign_lidar --max-line-length=88 --ignore=E203,W503`
- [ ] **Type check:** Run `mypy ign_lidar --ignore-missing-imports`
- [ ] **Pre-commit:** Run `pre-commit run --all-files` (optional but recommended)

## 2. Testing
- [ ] **Run unit tests:** `pytest tests -v -m unit`
- [ ] **Run integration tests:** `pytest tests -v -m integration` (if applicable)
- [ ] **Check coverage:** `pytest tests -v --cov=ign_lidar --cov-report=term`
- [ ] **Verify specific tests:** Run tests related to your changes

## 3. Documentation
- [ ] **Update docstrings:** Ensure all new/modified functions have Google-style docstrings
- [ ] **Type hints:** Add type annotations to all function signatures
- [ ] **Update README:** If adding major features, update README.md
- [ ] **Update CHANGELOG:** Add entry to CHANGELOG.md for significant changes
- [ ] **Example configs:** Update/add example YAML configs if changing configuration

## 4. Code Review Self-Check
- [ ] **No duplication:** Verify you didn't create duplicate functionality
- [ ] **Refactored existing:** Check if you modified existing code instead of creating new files
- [ ] **Backward compatible:** Ensure changes don't break existing functionality
- [ ] **Error handling:** Added proper exception handling
- [ ] **Logging:** Added appropriate log messages (info, warning, error)
- [ ] **Memory efficient:** Verified no memory leaks for large datasets

## 5. Git & Version Control
- [ ] **Stage changes:** `git add <modified_files>`
- [ ] **Commit:** `git commit -m "Clear descriptive message"`
- [ ] **Check status:** `git status` to verify all changes are committed
- [ ] **Push:** `git push origin <branch>` (if working on a branch)

## 6. Performance Considerations
- [ ] **GPU compatibility:** If modifying feature computation, test with both CPU and GPU
- [ ] **Memory usage:** For large dataset operations, verify memory management
- [ ] **Benchmark:** Run performance tests if changing core algorithms

## 7. Optional (for major changes)
- [ ] **Integration test:** Test full pipeline with example configs
- [ ] **Documentation site:** Update docs/ if needed
- [ ] **Version bump:** Update version in pyproject.toml for releases

## Quick Command Sequence (Common Workflow)
```bash
# 1. Format and lint
black ign_lidar tests
isort ign_lidar tests --profile=black
flake8 ign_lidar --max-line-length=88 --ignore=E203,W503

# 2. Run tests
pytest tests -v -m unit
pytest tests -v --cov=ign_lidar --cov-report=term

# 3. Commit
git add .
git commit -m "Your descriptive commit message"
git push origin main
```

## Windows-Specific Notes
- Use forward slashes or double backslashes in file paths
- Git Bash provides Unix-like commands
- PowerShell and Git Bash have different syntax for some commands
