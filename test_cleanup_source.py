#!/usr/bin/env python3
"""
Test Script: Remove .py source files from PyInstaller distributions

⚠️  WARNING: This is EXPERIMENTAL and may break your application!
    Always test on a COPY of your dist folders first!

Usage:
    # Dry run (shows what would be removed)
    python test_cleanup_source.py --dry-run

    # Actually remove files (creates backup first)
    python test_cleanup_source.py --execute

    # Clean specific distribution
    python test_cleanup_source.py --execute --target dist_app
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from datetime import datetime

# Critical modules that should keep their .py files
# These modules use runtime introspection, dynamic imports, or source reading
CRITICAL_MODULES = {
    # Deep Learning Frameworks
    'torch',
    'transformers',
    'lmdeploy',
    'mmengine',
    'triton',

    # Data Validation & APIs
    'pydantic',
    'fastapi',
    'gradio',
    'uvicorn',

    # Dynamic Loading
    'pkg_resources',
    'importlib',
    'setuptools',

    # Type Systems
    'typing_extensions',
    'typing_inspect',

    # Configuration
    'yaml',
    'toml',
    'configparser',
    
    # ADD THESE NEW ONES:
    'cv2',              # ← OpenCV needs its .py files
    'ultralytics',      # ← YOLO needs its .py files
}

# Your application source files (ALWAYS keep these)
YOUR_SOURCE_FILES = {
    'main.py',
    'app.py',
    'models.py',
    'utils.py',
    'prompts.py',
    'internvl2.py',
    'pixtral.py',
    'gr_app.py',
}


class SourceCleaner:
    def __init__(self, target_dir, dry_run=True):
        self.target_dir = Path(target_dir)
        self.internal_dir = self.target_dir / "my_app" / "_internal"
        self.dry_run = dry_run
        self.stats = {
            'total_py_files': 0,
            'removed': 0,
            'kept_critical': 0,
            'kept_source': 0,
            'errors': 0,
        }
        self.removed_files = []
        self.kept_files = []

    def is_critical_module(self, py_file):
        """Check if file belongs to a critical module"""
        file_str = str(py_file)
        return any(f"/{module}/" in file_str or f"/{module}.py" in file_str
                   for module in CRITICAL_MODULES)

    def is_your_source(self, py_file):
        """Check if this is your application source code"""
        return py_file.name in YOUR_SOURCE_FILES

    def backup_directory(self):
        """Create a backup before making changes"""
        if self.dry_run:
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.target_dir.parent / f"{self.target_dir.name}_backup_{timestamp}"

        print(f"\n📦 Creating backup: {backup_path}")
        shutil.copytree(self.target_dir, backup_path)
        print(f"✓ Backup created successfully")

        return backup_path

    def scan_and_clean(self):
        """Scan for .py files and remove non-critical ones"""
        if not self.internal_dir.exists():
            print(f"❌ Directory not found: {self.internal_dir}")
            return False

        print(f"\n{'🔍 Scanning' if self.dry_run else '🧹 Cleaning'}: {self.internal_dir}")
        print(f"Mode: {'DRY RUN (no files will be deleted)' if self.dry_run else 'EXECUTE (files will be deleted)'}\n")

        # Find all .py files
        py_files = list(self.internal_dir.rglob("*.py"))
        self.stats['total_py_files'] = len(py_files)

        print(f"Found {len(py_files)} .py files\n")

        # Process each file
        for py_file in py_files:
            rel_path = py_file.relative_to(self.internal_dir)

            # Check if it's your source code
            if self.is_your_source(py_file):
                self.stats['kept_source'] += 1
                self.kept_files.append((str(rel_path), "Your source code"))
                continue

            # Check if it's a critical module
            if self.is_critical_module(py_file):
                self.stats['kept_critical'] += 1
                self.kept_files.append((str(rel_path), "Critical module"))
                continue

            # Safe to remove
            if not self.dry_run:
                try:
                    py_file.unlink()
                    self.stats['removed'] += 1
                    self.removed_files.append(str(rel_path))
                except Exception as e:
                    print(f"❌ Error removing {rel_path}: {e}")
                    self.stats['errors'] += 1
            else:
                self.stats['removed'] += 1
                self.removed_files.append(str(rel_path))

        return True

    def print_report(self):
        """Print summary report"""
        print("\n" + "="*70)
        print("CLEANUP REPORT")
        print("="*70)

        print(f"\nTarget: {self.target_dir}")
        print(f"Mode: {'DRY RUN' if self.dry_run else 'EXECUTED'}\n")

        print(f"Total .py files found:        {self.stats['total_py_files']}")
        print(f"Files removed:                {self.stats['removed']}")
        print(f"Files kept (critical modules): {self.stats['kept_critical']}")
        print(f"Files kept (your source):      {self.stats['kept_source']}")
        print(f"Errors:                        {self.stats['errors']}")

        # Calculate size savings (approximate)
        total_kept = self.stats['kept_critical'] + self.stats['kept_source']
        removal_percentage = (self.stats['removed'] / self.stats['total_py_files'] * 100) if self.stats['total_py_files'] > 0 else 0

        print(f"\nRemoval rate: {removal_percentage:.1f}%")

        # Show samples
        if self.removed_files:
            print(f"\n📋 Sample removed files (first 20):")
            for f in self.removed_files[:20]:
                print(f"    ✗ {f}")
            if len(self.removed_files) > 20:
                print(f"    ... and {len(self.removed_files) - 20} more")

        if self.kept_files:
            print(f"\n📌 Sample kept files (first 20):")
            for f, reason in self.kept_files[:20]:
                print(f"    ✓ {f} ({reason})")
            if len(self.kept_files) > 20:
                print(f"    ... and {len(self.kept_files) - 20} more")

        print("\n" + "="*70)

        if self.dry_run:
            print("\n💡 This was a DRY RUN. No files were actually deleted.")
            print("   Run with --execute to actually remove files.")
        else:
            print("\n✅ Cleanup completed!")
            print("   ⚠️  IMPORTANT: Test your application now:")
            print("   ./run_all.sh")

        print("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Remove .py source files from PyInstaller distributions"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be removed without actually deleting'
    )
    parser.add_argument(
        '--execute',
        action='store_true',
        help='Actually remove the files (creates backup first)'
    )
    parser.add_argument(
        '--target',
        type=str,
        default=None,
        help='Target specific distribution (e.g., dist_app, dist_pixtral, dist_lmdeploy)'
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.dry_run and not args.execute:
        print("❌ Error: You must specify either --dry-run or --execute")
        parser.print_help()
        sys.exit(1)

    if args.dry_run and args.execute:
        print("❌ Error: Cannot use both --dry-run and --execute")
        sys.exit(1)

    # Determine targets
    if args.target:
        targets = [args.target]
    else:
        targets = ['dist_app', 'dist_pixtral', 'dist_lmdeploy']

    print("\n" + "="*70)
    print("PyInstaller Source Code Cleanup Tool")
    print("="*70)
    print(f"\n⚠️  WARNING: This tool removes .py source files from your distribution.")
    print(f"   This may break your application if not configured correctly!")
    print(f"\n   Mode: {'DRY RUN (safe)' if args.dry_run else 'EXECUTE (will delete files)'}")
    print(f"   Targets: {', '.join(targets)}")

    if not args.dry_run:
        print(f"\n   A backup will be created before removing files.")
        response = input("\n   Continue? (yes/no): ")
        if response.lower() != 'yes':
            print("   Cancelled.")
            sys.exit(0)

    # Process each target
    all_stats = []

    for target in targets:
        if not Path(target).exists():
            print(f"\n⚠️  Skipping {target}: directory not found")
            continue

        cleaner = SourceCleaner(target, dry_run=args.dry_run)

        # Create backup if executing
        if not args.dry_run:
            backup_path = cleaner.backup_directory()

        # Scan and clean
        success = cleaner.scan_and_clean()

        if success:
            cleaner.print_report()
            all_stats.append((target, cleaner.stats))

    # Overall summary
    if len(all_stats) > 1:
        print("\n" + "="*70)
        print("OVERALL SUMMARY")
        print("="*70)

        total_removed = sum(stats['removed'] for _, stats in all_stats)
        total_kept = sum(stats['kept_critical'] + stats['kept_source'] for _, stats in all_stats)

        print(f"\nTotal files removed: {total_removed}")
        print(f"Total files kept: {total_kept}")

        for target, stats in all_stats:
            removal_pct = (stats['removed'] / stats['total_py_files'] * 100) if stats['total_py_files'] > 0 else 0
            print(f"  {target}: {stats['removed']} removed ({removal_pct:.1f}%)")

    print("\n" + "="*70)
    print("\n🧪 TESTING CHECKLIST:")
    print("   1. Run: ./run_all.sh")
    print("   2. Check all logs for errors: tail -f logs/*.log")
    print("   3. Test all API endpoints")
    print("   4. Test OCR functionality end-to-end")
    print("   5. If broken, restore from backup")
    print("\n")


if __name__ == "__main__":
    main()
