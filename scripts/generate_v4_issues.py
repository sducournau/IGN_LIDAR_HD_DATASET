#!/usr/bin/env python3
"""
GitHub Issue Generator for v4.0.0 Release

Creates individual GitHub issues for all v4.0.0 tasks.

Usage:
    python scripts/generate_v4_issues.py --dry-run  # Preview issues
    python scripts/generate_v4_issues.py --execute  # Create issues (requires gh CLI)
"""

import argparse
import json
import subprocess
from dataclasses import dataclass
from typing import List
from pathlib import Path


@dataclass
class Issue:
    """GitHub issue details"""
    title: str
    body: str
    labels: List[str]
    milestone: str = "v4.0.0"
    assignee: str = ""


class IssueGenerator:
    """Generates GitHub issues for v4.0.0 tasks"""
    
    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.issues: List[Issue] = []
        
    def create_phase1_issues(self):
        """Create Phase 1 issues"""
        
        # Issue 1: Deprecation Audit Complete
        self.issues.append(Issue(
            title="[Phase 1] Complete Comprehensive Deprecation Audit",
            body="""## Task
Complete comprehensive audit of all deprecated code in preparation for v4.0.0.

## Deliverables
- [x] Run audit script: `python scripts/audit_deprecations_v4.py`
- [x] Review deprecation report (416 items found)
- [ ] Categorize by removal priority
- [ ] Create removal plan for each high-priority item
- [ ] Document in V4_IMPLEMENTATION_CHECKLIST.md

## Context
Automated audit found:
- Total: 416 deprecations
- High Priority: 59 items
- Files Affected: 74

Full report: `deprecation_audit_report.json`

## Acceptance Criteria
- All deprecated items documented
- Removal priority assigned
- Plan created for systematic removal

## Related
- Part of Phase 1 (Preparation)
- Blocks Phase 2 implementation tasks
- See: VERSION_4.0.0_RELEASE_PLAN.md
""",
            labels=["v4.0", "phase-1", "audit", "documentation"]
        ))
        
        # Issue 2: v3.7.0 Transitional Release
        self.issues.append(Issue(
            title="[Phase 1] Create v3.7.0 Transitional Release",
            body="""## Task
Create v3.7.0 as final v3.x release with enhanced deprecation warnings.

## Steps
1. [ ] Create v3.7.0 branch from main
2. [ ] Add comprehensive deprecation warnings to all deprecated code
3. [ ] Update deprecation messages with v4.0 timeline
4. [ ] Add migration guide links to warnings
5. [ ] Test all deprecation warnings fire correctly
6. [ ] Update CHANGELOG.md for v3.7.0
7. [ ] Create release notes
8. [ ] Tag and release to PyPI

## Enhanced Warnings Needed
- Configuration v3.x structure warnings
- FeatureComputer deprecation warning
- Deprecated normal function warnings
- Legacy import path warnings
- GPU helper deprecation warnings

## Timeline
- Start: Week 3 (Dec 9, 2025)
- Release: Mid-January 2026
- Support: Through Q2 2026

## Acceptance Criteria
- v3.7.0 released to PyPI
- All deprecated code has clear warnings
- Migration guide available
- Community announced

## Related
- Prepares users for v4.0.0
- Reduces migration friction
- See: V4_BREAKING_CHANGES.md
""",
            labels=["v4.0", "phase-1", "release", "v3.7.0"]
        ))
        
        # Issue 3: Migration Tool Testing
        self.issues.append(Issue(
            title="[Phase 1] Test Config Migration Tool on Real Configs",
            body="""## Task
Thoroughly test the config migration tool on 20+ real-world configurations.

## Test Configs Needed
- [ ] User-submitted configs (collect from issues/discussions)
- [ ] Example configs from examples/ directory
- [ ] Preset configs
- [ ] Edge cases (empty configs, complex nested)
- [ ] Production configs (sanitized)

## Test Scenarios
1. Simple LOD2 config
2. Complex LOD3 config
3. ASPRS classification config
4. Multi-scale adaptive config
5. GPU-enabled configs
6. CPU-only configs
7. Configs with custom settings
8. Configs with deprecated parameters

## Validation
- [ ] All configs migrate successfully
- [ ] Output validates against v4.0 schema
- [ ] No data loss
- [ ] Warnings for ambiguous cases
- [ ] Dry-run mode works correctly

## Success Metrics
- >95% migration success rate
- <5% requiring manual intervention
- Clear error messages for failures

## Command
```bash
ign-lidar migrate-config <config> --dry-run --verbose
```

## Related
- Critical for user migration
- Reduces v4.0 adoption friction
- See: V4_BREAKING_CHANGES.md
""",
            labels=["v4.0", "phase-1", "testing", "migration"]
        ))
        
    def create_phase2_issues(self):
        """Create Phase 2 issues"""
        
        # Issue 4: Delete Old Config Files
        self.issues.append(Issue(
            title="[Phase 2] Delete schema.py and schema_simplified.py",
            body="""## Task
Remove old configuration files as part of v4.0 finalization.

## Files to Delete
- `ign_lidar/config/schema.py` (415 lines)
- `ign_lidar/config/schema_simplified.py` (~300 lines)

## Steps
1. [ ] Run finalization script: `python scripts/finalize_config_v4.py --dry-run`
2. [ ] Review planned changes
3. [ ] Execute deletion: `python scripts/finalize_config_v4.py --execute`
4. [ ] Update imports in __init__.py
5. [ ] Remove test cases for old configs
6. [ ] Update documentation

## Backups
Script automatically creates .v3backup files before deletion.

## Testing
- [ ] All tests pass after deletion
- [ ] No import errors
- [ ] Config loading works with v4.0 only

## Breaking Change
⚠️ This is a BREAKING CHANGE
- All v3.x configs will no longer load
- Users must migrate before upgrading

## Related
- Blocks: API stabilization tasks
- See: V4_BREAKING_CHANGES.md section 1
""",
            labels=["v4.0", "phase-2", "breaking-change", "config"]
        ))
        
        # Issue 5: Remove FeatureComputer
        self.issues.append(Issue(
            title="[Phase 2] Remove FeatureComputer Class",
            body="""## Task
Delete the deprecated FeatureComputer class.

## File to Modify
- `ign_lidar/features/feature_computer.py`

## Steps
1. [ ] Delete entire `feature_computer.py` file
2. [ ] Remove import from `features/__init__.py`
3. [ ] Update all internal usage to FeatureOrchestrator
4. [ ] Remove test cases for FeatureComputer
5. [ ] Update documentation and examples

## Migration Pattern
```python
# OLD (v3.x) - REMOVED
from ign_lidar.features import FeatureComputer
computer = FeatureComputer(use_gpu=True)

# NEW (v4.0) - REQUIRED
from ign_lidar.features import FeatureOrchestrator
from ign_lidar import Config
config = Config(use_gpu=True)
orchestrator = FeatureOrchestrator(config)
```

## Testing
- [ ] All feature tests pass
- [ ] No import errors
- [ ] Documentation updated

## Breaking Change
⚠️ BREAKING CHANGE
- FeatureComputer no longer available
- Users must use FeatureOrchestrator

## Related
- See: V4_BREAKING_CHANGES.md section 2
- Deprecation since: v3.7.0
""",
            labels=["v4.0", "phase-2", "breaking-change", "features"]
        ))
        
        # Issue 6: Remove Deprecated Normal Functions
        self.issues.append(Issue(
            title="[Phase 2] Remove Deprecated Normal Computation Functions",
            body="""## Task
Remove deprecated normal computation functions from numba_accelerated.py and gpu_processor.py.

## Functions to Remove
From `features/numba_accelerated.py`:
- `compute_normals_from_eigenvectors()`
- `compute_normals_from_eigenvectors_numpy()`
- `compute_normals_from_eigenvectors_numba()`

From `features/gpu_processor.py`:
- `_compute_normals_cpu()`

## Migration Pattern
```python
# OLD - REMOVED
from ign_lidar.features.numba_accelerated import compute_normals_from_eigenvectors
normals = compute_normals_from_eigenvectors(eigenvectors)

# NEW - REQUIRED
from ign_lidar.features.compute.normals import compute_normals
normals = compute_normals(points, k_neighbors=30)
```

## Steps
1. [ ] Remove function definitions
2. [ ] Remove tests for deprecated functions
3. [ ] Update all internal call sites
4. [ ] Verify canonical implementations work
5. [ ] Update documentation

## Testing
- [ ] All normal computation tests pass
- [ ] GPU fallback works correctly
- [ ] No import errors

## Related
- See: V4_BREAKING_CHANGES.md section 3
- Canonical API: `features/compute/normals.py`
""",
            labels=["v4.0", "phase-2", "breaking-change", "features"]
        ))
        
    def create_phase3_issues(self):
        """Create Phase 3 issues"""
        
        # Issue 7: Comprehensive Testing
        self.issues.append(Issue(
            title="[Phase 3] Achieve >85% Test Coverage",
            body="""## Task
Expand test suite to achieve >85% code coverage.

## Current Status
- Coverage: 78%
- Target: 85%
- Gap: 7 percentage points

## Areas Needing Tests
1. [ ] Config v4.0 validation
2. [ ] Migration tool edge cases
3. [ ] GPU error handling
4. [ ] Memory management
5. [ ] Classification edge cases
6. [ ] IO operations
7. [ ] CLI commands

## Test Types Needed
- Unit tests
- Integration tests
- Performance tests
- Memory leak tests
- GPU tests (with fallback)

## Success Metrics
- Coverage >85%
- All critical paths tested
- Edge cases covered
- Performance benchmarks passing

## Commands
```bash
pytest tests/ -v --cov=ign_lidar --cov-report=html
pytest tests/ -v --cov=ign_lidar --cov-report=term-missing
```

## Related
- Required for v4.0 release
- See: VERSION_4.0.0_RELEASE_PLAN.md
""",
            labels=["v4.0", "phase-3", "testing", "quality"]
        ))
        
    def generate_all_issues(self):
        """Generate all issues"""
        self.create_phase1_issues()
        self.create_phase2_issues()
        self.create_phase3_issues()
        
    def save_to_json(self, output_path: Path):
        """Save issues to JSON file"""
        issues_data = [
            {
                'title': issue.title,
                'body': issue.body,
                'labels': issue.labels,
                'milestone': issue.milestone,
                'assignee': issue.assignee
            }
            for issue in self.issues
        ]
        
        with open(output_path, 'w') as f:
            json.dump(issues_data, f, indent=2)
        
        print(f"✅ Saved {len(issues_data)} issues to {output_path}")
    
    def create_issues_via_gh(self):
        """Create issues using GitHub CLI"""
        if self.dry_run:
            print("🔵 DRY RUN - Would create the following issues:")
            for i, issue in enumerate(self.issues, 1):
                print(f"\n{i}. {issue.title}")
                print(f"   Labels: {', '.join(issue.labels)}")
                print(f"   Milestone: {issue.milestone}")
            return
        
        print("⚠️  Creating issues via GitHub CLI...")
        
        for issue in self.issues:
            cmd = [
                'gh', 'issue', 'create',
                '--title', issue.title,
                '--body', issue.body,
                '--label', ','.join(issue.labels),
            ]
            
            if issue.milestone:
                cmd.extend(['--milestone', issue.milestone])
            
            if issue.assignee:
                cmd.extend(['--assignee', issue.assignee])
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                print(f"✅ Created: {issue.title}")
                print(f"   URL: {result.stdout.strip()}")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to create: {issue.title}")
                print(f"   Error: {e.stderr}")
    
    def print_summary(self):
        """Print summary of issues"""
        print("\n" + "="*80)
        print("GitHub Issues Summary for v4.0.0")
        print("="*80)
        print(f"\nTotal Issues: {len(self.issues)}")
        
        # Group by phase
        phases = {}
        for issue in self.issues:
            phase_label = next((l for l in issue.labels if l.startswith('phase-')), 'other')
            if phase_label not in phases:
                phases[phase_label] = []
            phases[phase_label].append(issue)
        
        print("\n📋 By Phase:")
        for phase, issues in sorted(phases.items()):
            print(f"  {phase}: {len(issues)} issues")
        
        print("\n🏷️  Labels Used:")
        all_labels = set()
        for issue in self.issues:
            all_labels.update(issue.labels)
        for label in sorted(all_labels):
            count = sum(1 for i in self.issues if label in i.labels)
            print(f"  {label}: {count}")
        
        print("\n" + "="*80)
        print("Next Steps:")
        print("1. Review issues (saved to v4_issues.json)")
        print("2. Create milestone in GitHub: v4.0.0")
        print("3. Run with --execute to create issues (requires gh CLI)")
        print("4. Assign issues to team members")
        print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Generate GitHub issues for v4.0.0')
    parser.add_argument('--dry-run', action='store_true', help='Preview without creating')
    parser.add_argument('--execute', action='store_true', help='Create issues via gh CLI')
    parser.add_argument('--output', default='v4_issues.json', help='JSON output file')
    args = parser.parse_args()
    
    if not args.dry_run and not args.execute:
        print("❌ Error: Must specify either --dry-run or --execute")
        parser.print_help()
        return 1
    
    generator = IssueGenerator(dry_run=args.dry_run)
    generator.generate_all_issues()
    
    # Save to JSON
    output_path = Path(args.output)
    generator.save_to_json(output_path)
    
    # Print summary
    generator.print_summary()
    
    # Create issues if requested
    if args.execute:
        print("\n⚠️  About to create issues in GitHub...")
        print("This requires the GitHub CLI (gh) to be installed and authenticated.")
        response = input("Continue? (yes/no): ")
        if response.lower() == 'yes':
            generator.create_issues_via_gh()
        else:
            print("Cancelled.")
    
    return 0


if __name__ == '__main__':
    exit(main())
