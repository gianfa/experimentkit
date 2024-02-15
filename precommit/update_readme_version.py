from pathlib import Path
import re
import toml

ROOT = Path(__file__).parent.parent.resolve()


def update_readme_version():
    with open(ROOT / "pyproject.toml", "r") as f:
        toml_data = toml.load(f)
        version = toml_data["tool"]["poetry"]["version"]

    with open(ROOT / "README.md", "r") as f:
        readme_content = f.read()

    updated_readme_content = re.sub(
        r"(https://img.shields.io/badge/v).*?(-blue)",
        r"\g<1>" + version + r"\g<2>",
        readme_content,
    )
    with open(ROOT / "README.md", "w") as f:
        f.write(updated_readme_content)


if __name__ == "__main__":
    update_readme_version()
