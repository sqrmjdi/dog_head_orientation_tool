"""
Build script to create executable for Manual Labeling Tool
Run this script to generate the .exe file
"""

import subprocess
import sys

def install_pyinstaller():
    """Install PyInstaller if not already installed."""
    try:
        import PyInstaller
        print("PyInstaller is already installed.")
    except ImportError:
        print("Installing PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        print("PyInstaller installed successfully.")

def build_exe():
    """Build the executable using PyInstaller."""
    import PyInstaller.__main__
    
    PyInstaller.__main__.run([
        'head_orientation/manual_labeling_ui.py',
        '--name=ManualLabelingTool',
        '--onefile',
        '--windowed',
        '--icon=NONE',
        '--add-data=head_orientation/data;data',
        '--add-data=head_orientation/output;output',
        '--hidden-import=PIL._tkinter_finder',
        '--hidden-import=openpyxl',
        '--hidden-import=xlrd',
        '--collect-all=cv2',
        '--noconfirm',
        '--clean',
    ])

if __name__ == "__main__":
    print("=" * 50)
    print("Building Manual Labeling Tool Executable")
    print("=" * 50)
    
    install_pyinstaller()
    build_exe()
    
    print("\n" + "=" * 50)
    print("Build complete!")
    print("The executable is located in: dist/ManualLabelingTool.exe")
    print("=" * 50)
