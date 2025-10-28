"""
Pytest configuration for test discovery and imports.
"""
import sys
from pathlib import Path

# Add the project root to the Python path
# From app/tests/ we need to go up two levels to reach root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
