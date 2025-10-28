"""
Project Cleanup Script
Removes duplicate/unnecessary files before testing branch
"""

import os
import shutil
from pathlib import Path

# Get project root
PROJECT_ROOT = Path(__file__).parent

print("🧹 Code-Mixed NLP Project Cleanup")
print("=" * 60)

# Files to remove (duplicates and standalone test scripts)
files_to_remove = [
    "app/tests/test_core.py.skip",  # Bad design with sys.exit
    # Standalone test scripts (not proper pytest files)
    # Keeping test_api_integration.py and test_v2_new_endpoints.py
]

# Directories to clean
dirs_to_clean = [
    "__pycache__",
    ".pytest_cache",
    "*.pyc",
    "*.pyo",
    "*.egg-info"
]

removed_count = 0
freed_space = 0

# Remove specific files
print("\n📂 Removing duplicate/unnecessary files...")
for file_path in files_to_remove:
    full_path = PROJECT_ROOT / file_path
    if full_path.exists():
        size = full_path.stat().st_size
        full_path.unlink()
        freed_space += size
        removed_count += 1
        print(f"  ❌ Removed: {file_path} ({size / 1024:.1f} KB)")

# Clean __pycache__ directories
print("\n🗑️  Cleaning __pycache__ directories...")
pycache_count = 0
for root, dirs, files in os.walk(PROJECT_ROOT):
    # Skip venv
    if 'venv' in root:
        continue
    
    for dir_name in dirs:
        if dir_name == '__pycache__':
            dir_path = os.path.join(root, dir_name)
            try:
                # Calculate size before deletion
                for f in os.listdir(dir_path):
                    f_path = os.path.join(dir_path, f)
                    if os.path.isfile(f_path):
                        freed_space += os.path.getsize(f_path)
                
                shutil.rmtree(dir_path)
                pycache_count += 1
                print(f"  ❌ Removed: {dir_path}")
            except Exception as e:
                print(f"  ⚠️  Failed to remove {dir_path}: {e}")

# Clean .pyc files
print("\n🗑️  Cleaning .pyc files...")
pyc_count = 0
for root, dirs, files in os.walk(PROJECT_ROOT):
    if 'venv' in root:
        continue
    
    for file in files:
        if file.endswith('.pyc') or file.endswith('.pyo'):
            file_path = os.path.join(root, file)
            try:
                size = os.path.getsize(file_path)
                os.remove(file_path)
                freed_space += size
                pyc_count += 1
            except Exception as e:
                print(f"  ⚠️  Failed to remove {file_path}: {e}")

print(f"  ✅ Removed {pyc_count} .pyc/.pyo files")

# Clean pytest cache
pytest_cache = PROJECT_ROOT / ".pytest_cache"
if pytest_cache.exists():
    print("\n🗑️  Cleaning pytest cache...")
    try:
        shutil.rmtree(pytest_cache)
        print("  ✅ Removed .pytest_cache")
    except Exception as e:
        print(f"  ⚠️  Failed to remove pytest cache: {e}")

# Summary
print("\n" + "=" * 60)
print("✅ Cleanup Complete!")
print(f"📊 Files removed: {removed_count}")
print(f"📊 Pycache dirs removed: {pycache_count}")
print(f"💾 Space freed: {freed_space / 1024 / 1024:.2f} MB")

# List remaining test files
print("\n📋 Remaining Test Files:")
test_dir = PROJECT_ROOT / "app" / "tests"
if test_dir.exists():
    test_files = sorted(test_dir.glob("test_*.py"))
    for test_file in test_files:
        size = test_file.stat().st_size / 1024
        print(f"  ✅ {test_file.name} ({size:.1f} KB)")

print("\n🎯 Next Steps:")
print("  1. Review PROJECT_STATUS.md for known issues")
print("  2. Fix CM-BERT label mapping (CRITICAL)")
print("  3. Run: pytest app/tests/test_api_integration.py app/tests/test_v2_new_endpoints.py -v")
print("  4. Create testing branch: git checkout -b v2-testing")
print("  5. Stage and commit V2 changes")
print("\n" + "=" * 60)
