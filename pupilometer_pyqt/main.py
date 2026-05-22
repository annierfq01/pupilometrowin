#!/usr/bin/env python3
"""
Main entry point - Alternative to __main__.py
"""

from pupilometer_pyqt.ui.main_window import PupilometerApp
import sys

if __name__ == "__main__":
    app = PupilometerApp()
    sys.exit(app.run())
