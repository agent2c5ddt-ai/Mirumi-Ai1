from pathlib import Path

from mirumi.app import MirumiApp
from mirumi.config import Settings


def main():
    project_root = Path(__file__).resolve().parent
    MirumiApp(settings=Settings.from_environment(project_root)).run()

if __name__ == "__main__":
    main()
