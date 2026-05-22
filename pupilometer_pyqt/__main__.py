#!/usr/bin/env python3
"""
Entry point for Pupilometer PyQt application
"""

import sys
from pupilometer_pyqt.ui.main_window import PupilometerApp

def main():
    app = PupilometerApp()
    sys.exit(app.run())

if __name__ == "__main__":
    main()
