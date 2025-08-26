"""
Pipeline Results Viewer Launcher
Simple script to launch the Streamlit viewer
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Launch the Streamlit viewer"""
    viewer_path = Path(__file__).parent / "streamlit_viewer.py"

    print("🎬 Launching Pipeline Results Viewer...")
    print(f"📂 Viewer path: {viewer_path}")
    print("🌐 Opening in browser...")

    # Launch streamlit
    cmd = [sys.executable, "-m", "streamlit", "run", str(viewer_path)]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching viewer: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Viewer closed by user")


if __name__ == "__main__":
    main()
