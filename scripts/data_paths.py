"""Configuration file for dataset paths."""
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_DIR = os.path.join(BASE_DIR, "data")

TCGA_EXPRESSION_FILE = os.path.join(DATA_DIR, "tcga.survival.fcptac.csv")
TCGA_SURVIVAL_FILE = os.path.join(DATA_DIR, "survival_tcga.csv")
CPTAC_EXPRESSION_FILE = os.path.join(DATA_DIR, "cptac.survival.fcptac.csv")
CPTAC_SURVIVAL_FILE = os.path.join(DATA_DIR, "survival_cptac.csv")

