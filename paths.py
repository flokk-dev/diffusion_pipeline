"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
import os

# ----- ROOT ----- #
ROOT = os.path.dirname(os.path.abspath(__file__))

# ----- RESOURCES ----- #
RESOURCES = os.path.join(ROOT, "resources")

FAVICON = os.path.join(RESOURCES, "favicon.ico")
REQUIREMENTS = os.path.join(RESOURCES, "requirements.txt")

# CONFIG
CONFIG = os.path.join(RESOURCES, "config")

LANGUAGES = os.path.join(CONFIG, "languages.json")

# DATA
DATA = os.path.join(RESOURCES, "data")

IMAGES = os.path.join(DATA, "images")
MASKS = os.path.join(DATA, "masks")


# ----- SRC ----- #
SRC = os.path.join(ROOT, "src")

# Pages
PAGES = os.path.join(SRC, "app", "pages")
