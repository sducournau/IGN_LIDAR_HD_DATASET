#!/usr/bin/env python3
"""
Root Directory Consolidation Script
Cleans and organizes the root directory by moving scripts to appropriate locations
and removing duplicates/temporary files.
"""

import shutil
from pathlib import Path
import sys

# Color codes
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(msg):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{msg}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}\n")


def print_action(action, path, dest=None):
    if dest:
        print(f"{Colors.BLUE}  [{action}]{Colors.ENDC} {path} → {dest}")
    else:
        print(f"{Colors.BLUE}  [{action}]{Colors.ENDC} {path}")


def print_success(msg):
    print(f"{Colors.GREEN}✓ {msg}{Colors.ENDC}")


def print_warning(msg):
    print(f"{Colors.YELLOW}⚠ {msg}{Colors.ENDC}")


def print_error(msg):
    print(f"{Colors.RED}✗ {msg}{Colors.ENDC}")


def consolidate_root(dry_run=False):
    """Consolidate and clean root directory."""
    
    root = Path(__file__).parent.parent
    scripts_dir = root / "scripts"
    
    print_header("🧹 ROOT DIRECTORY CONSOLIDATION")
    
    actions = []
    
    # 1. Move workflow scripts to scripts/workflows/
    workflows_dir = scripts_dir / "workflows"
    workflow_scripts = [
        "complete_workflow.py",
    ]
    
    print_header("📦 Phase 1: Move Workflow Scripts")
    for script in workflow_scripts:
        src = root / script
        if src.exists():
            dest = workflows_dir / script
            actions.append(("move", src, dest))
            print_action("MOVE", script, f"scripts/workflows/{script}")
    
    # 2. Move utility scripts to scripts/
    print_header("🔧 Phase 2: Move Utility Scripts")
    utility_scripts = [
        "check_progress.sh",
        "consolidate_repo.py",
    ]
    
    for script in utility_scripts:
        src = root / script
        if src.exists():
            dest = scripts_dir / script
            actions.append(("move", src, dest))
            print_action("MOVE", script, f"scripts/{script}")
    
    # 3. Consolidate metadata regeneration scripts
    print_header("🔄 Phase 3: Consolidate Metadata Scripts")
    
    # Keep regen_metadata_simple.py as the canonical version
    # Move regenerate_metadata.py to legacy
    metadata_scripts = [
        ("regenerate_metadata.py", scripts_dir / "legacy" / "regenerate_metadata.py"),
    ]
    
    for script, dest in metadata_scripts:
        src = root / script
        if src.exists():
            actions.append(("move", src, dest))
            print_action("ARCHIVE", script, f"scripts/legacy/{script}")
    
    # Rename regen_metadata_simple.py to regenerate_metadata.py
    src = root / "regen_metadata_simple.py"
    if src.exists():
        dest = scripts_dir / "regenerate_metadata.py"
        actions.append(("move", src, dest))
        print_action("MOVE+RENAME", "regen_metadata_simple.py", "scripts/regenerate_metadata.py")
    
    # 4. Clean up temporary/log files
    print_header("🗑️  Phase 4: Clean Temporary Files")
    temp_files = [
        "workflow_output.log",
    ]
    
    for file in temp_files:
        path = root / file
        if path.exists():
            actions.append(("remove", path, None))
            print_action("REMOVE", file)
    
    # 5. Update .gitignore
    print_header("📝 Phase 5: Update .gitignore")
    gitignore = root / ".gitignore"
    if gitignore.exists():
        print_action("UPDATE", ".gitignore")
        actions.append(("update_gitignore", gitignore, None))
    
    # Summary
    print_header("📊 SUMMARY")
    print(f"Total actions: {len(actions)}")
    print(f"  - Moves: {sum(1 for a in actions if a[0] == 'move')}")
    print(f"  - Removals: {sum(1 for a in actions if a[0] == 'remove')}")
    print(f"  - Updates: {sum(1 for a in actions if a[0] == 'update_gitignore')}")
    
    if dry_run:
        print_warning("\n🔍 DRY RUN MODE - No changes made")
        return
    
    # Execute actions
    print_header("⚡ EXECUTING ACTIONS")
    
    # Create directories if needed
    workflows_dir.mkdir(parents=True, exist_ok=True)
    (scripts_dir / "legacy").mkdir(parents=True, exist_ok=True)
    
    errors = []
    
    for action_type, src, dest in actions:
        try:
            if action_type == "move":
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dest))
                print_success(f"Moved {src.name}")
            
            elif action_type == "remove":
                src.unlink()
                print_success(f"Removed {src.name}")
            
            elif action_type == "update_gitignore":
                update_gitignore(src)
                print_success("Updated .gitignore")
        
        except Exception as e:
            errors.append((src, str(e)))
            print_error(f"Failed: {src.name} - {e}")
    
    # Final summary
    print_header("✅ CONSOLIDATION COMPLETE")
    
    if errors:
        print_error(f"{len(errors)} errors occurred:")
        for path, error in errors:
            print(f"  - {path}: {error}")
    else:
        print_success("All actions completed successfully!")
    
    # Next steps
    print(f"\n{Colors.CYAN}📋 Next Steps:{Colors.ENDC}")
    print(f"  1. Review changes: {Colors.CYAN}git status{Colors.ENDC}")
    print(f"  2. Update imports if needed")
    print(f"  3. Test scripts in new locations")
    print(f"  4. Commit changes: {Colors.CYAN}git add -A && git commit -m 'chore: consolidate root directory'{Colors.ENDC}")


def update_gitignore(gitignore_path):
    """Update .gitignore with log file patterns."""
    content = gitignore_path.read_text()
    
    # Check if workflow_output.log is already ignored
    if "workflow_output.log" not in content and "*.log" not in content:
        # Add to log files section
        lines = content.split('\n')
        
        # Find log files section
        for i, line in enumerate(lines):
            if "# Log files" in line or "*.log" in line:
                # Already has log pattern
                return
        
        # Add at the end
        if not content.endswith('\n'):
            content += '\n'
        
        content += "\n# Workflow outputs\nworkflow_output.log\n"
        gitignore_path.write_text(content)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Consolidate and clean root directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/consolidate_root.py --dry-run    # Preview changes
  python scripts/consolidate_root.py              # Execute consolidation
        """
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without making them"
    )
    
    args = parser.parse_args()
    
    try:
        consolidate_root(dry_run=args.dry_run)
    except KeyboardInterrupt:
        print_warning("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
