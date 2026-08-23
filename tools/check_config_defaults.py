"""Fail if any function takes a default argument read from the configuration.

A default expression is evaluated once, when the module is imported, so a
parameter written as

    def train_model(num_epochs: int = config.NUM_EPOCHS) -> ...:

is frozen at the value config held at import time. Setting config.NUM_EPOCHS
from a notebook afterwards changes nothing, silently: the run trains for the
old number of epochs while every log and every record says otherwise.

This bug reached the experiments three times — set_seed kept seeding 42 through
two runs meant to measure seed-to-seed variation, create_model built networks
for the wrong channel count, and train_model would have ignored a longer epoch
budget. It is invisible in review because the code reads correctly.

The rule is therefore mechanical: no default argument may read config. Resolve
it in the body instead.

    def train_model(num_epochs: Optional[int] = None) -> ...:
        if num_epochs is None:
            num_epochs = config.NUM_EPOCHS

Run before uploading the modules for a campaign:

    python tools/check_config_defaults.py
"""

import ast
import sys
from pathlib import Path
from typing import List, Tuple

SOURCE_DIR: Path = Path(__file__).resolve().parents[1] / "src"

# Module-level constants that describe what is true by construction, not
# settings a notebook overrides. Freezing these at import is harmless.
ALLOWED: frozenset = frozenset(
    {
        "DEFAULT_EPSILON",
        "DEFAULT_POOL_SIZES",
        "DEFAULT_COVERAGE_KERNELS",
        "COMPARISON_COLUMNS",
    }
)


def reads_configuration(expression: str) -> bool:
    """Tell whether a default expression reads a configurable value.

    Args:
        expression: Source of the default, e.g. "config.NUM_EPOCHS".

    Returns:
        True if the expression reads the config module or a tunable constant.
    """
    if expression.startswith("config."):
        return True

    root = expression.split(".")[0].split("[")[0]

    return root.isidentifier() and root.isupper() and root not in ALLOWED


def find_offenders(source_dir: Path = SOURCE_DIR) -> List[Tuple[str, int, str, str, str]]:
    """Collect every default argument that reads the configuration.

    Args:
        source_dir: Folder holding the modules to check.

    Returns:
        Tuples of (module, line, function, parameter, default expression).
    """
    offenders: List[Tuple[str, int, str, str, str]] = []

    for path in sorted(source_dir.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            arguments = node.args
            positional = (
                arguments.args[-len(arguments.defaults):] if arguments.defaults else []
            )
            pairs = list(zip(positional, arguments.defaults))
            pairs += [
                (argument, default)
                for argument, default in zip(arguments.kwonlyargs, arguments.kw_defaults)
                if default is not None
            ]

            for argument, default in pairs:
                expression = ast.get_source_segment(text, default) or ast.unparse(default)
                if reads_configuration(expression):
                    offenders.append(
                        (path.name, node.lineno, node.name, argument.arg, expression)
                    )

    return offenders


def main() -> int:
    """Report offenders and return the process exit code.

    Returns:
        0 when no default reads the configuration, 1 otherwise.
    """
    offenders = find_offenders()

    if not offenders:
        print(f"OK - no default argument in {SOURCE_DIR.name}/ reads the configuration")
        return 0

    print(f"FAIL - {len(offenders)} default argument(s) read the configuration:\n")
    for module, line, function, parameter, expression in offenders:
        print(f"  {module}:{line}  {function}({parameter}={expression})")
    print("\nResolve them inside the function body, not in the signature.")

    return 1


if __name__ == "__main__":
    sys.exit(main())
