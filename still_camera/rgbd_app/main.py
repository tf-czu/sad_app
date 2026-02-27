import sys
# This is a workaround to prevent unwanted parsing of command-line arguments
# by libraries like Tkinter or pyrealsense2.
# It must be done before other imports that might parse argv.
sys.argv = [sys.argv[0]]

import tkinter as tk
from .gui import App

def main():
    """Main function to run the application."""
    root = tk.Tk()
    app = App(master=root)
    root.mainloop()


if __name__ == "__main__":
    main()
