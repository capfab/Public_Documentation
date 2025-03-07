"""
Recursively runs sphinx-apidoc on all subdirectories within the specified base folder.
"""

__version__ = "0.1.0"

import os
import subprocess


def run_sphinx_apidoc_recursively(base_folder, output_folder):
    """
    Recursively runs sphinx-apidoc on all subdirectories within the specified base folder.

    :param base_folder: The root directory containing the Python modules.
    :type base_folder: str
    :param output_folder: The directory where the generated rst files should be placed.
    :type output_folder: str
    """
    # Run sphinx-apidoc for the current directory
    subprocess.run(
        ["sphinx-apidoc", "-o", output_folder, base_folder],
        check=True,
    )

    # Recursively process subdirectories
    for root, dirs, _ in os.walk(base_folder):
        for directory in dirs:
            sub_path = os.path.join(root, directory)
            subprocess.run(
                ["sphinx-apidoc", "-o", output_folder, sub_path],
                check=True,
            )


if __name__ == "__main__":
    # BASE_FOLDER = "sample_code"
    BASE_FOLDER = "vr_learning_algorithms"
    OUTPUT_FOLDER = "docs/source"

    run_sphinx_apidoc_recursively(BASE_FOLDER, OUTPUT_FOLDER)
    print("Sphinx-apidoc generation complete.")
