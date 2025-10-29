from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent

TRAIN_CLINICAL_DATA_PATH = ROOT_DIR / "data/clinical_train.csv"
TRAIN_MOLECULAR_DATA_PATH = ROOT_DIR / "data/molecular_train.csv"
TRAIN_TARGET_PATH = ROOT_DIR / "data/target_train.csv"

TEST_CLINICAL_DATA_PATH = ROOT_DIR / "data/clinical_test.csv"
TEST_MOLECULAR_DATA_PATH = ROOT_DIR / "data/molecular_test.csv"