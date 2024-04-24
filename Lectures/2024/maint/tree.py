"""Generate tree structure based on course from last year."""

import shutil
from pathlib import Path

template_dir = Path(__file__).parents[2] / "2023"

output_dir = Path(__file__).parents[1]

skip_list = ["12-High_Performance_Computing"]


def main():
    for dir in template_dir.iterdir():
        if dir.name in skip_list:
            continue

        for sub_dir in ["code", "exervises", "lecture"]:
            dir_to_create = output_dir / dir.name.replace("-", "_").lower() / sub_dir
            dir_to_create.mkdir(exist_ok=True, parents=True)

            with open(dir_to_create / ".gitkeep", "w") as f:
                ...

        if (dir / "README.md").exists():
            shutil.copy(dir / "README.md", dir_to_create / "..")


if __name__ == "__main__":
    # main()
    pass
