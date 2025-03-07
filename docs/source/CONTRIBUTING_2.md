# Contributing to LfD Algorithms

This document outlines the guidelines for contributing to our project, including coding style, pull request requirements, documentation guidelines, and more.

## Table of Contents

- [Getting Started](#getting-started)
- [Code Style Guide](#code-style-guide)
- [Branching and Commit Guidelines](#branching-and-commit-guidelines)
- [Pull Request Process](#pull-request-process)
- [Testing](#testing)
- [Documentation Guidelines](#documentation-guidelines)
- [Issue Reporting](#issue-reporting)
- [Contact](#contact)

## Getting Started

1. If you have been added to the project, you can clone the repository directly:

   ```sh
   git clone https://github.com/VinRobotics/vr_learning_algorithms.git
   ```

   Otherwise, if you are contributing externally, fork the repository first, then clone your fork:

   ```sh
   git clone https://github.com/VinRobotics/vr_learning_algorithms.git
   ```

2. Navigate to the project directory:
   ```sh
   cd localLfD
   ```
3. Install dependencies:
   ```sh
   pip install -e .[docs,tests,extra]
   ```
4. Run tests to ensure everything is set up correctly:
   ```sh
   WIP
   ```

## Code Style Guide

We follow the [Black](https://black.readthedocs.io/en/stable/) code style for Python in this project. Please adhere to the following:

- Use descriptive variable names following standard conventions (e.g., `snake_case` for variables and functions, `PascalCase` for classes).
- Since we plan to maintain API documentation, all code must follow a strict docstring format compatible with `Sphinx`.
  ```python
  """[Summary]
  :param [ParamName]: [ParamDescription], defaults to [DefaultParamVal]
  :type [ParamName]: [ParamType](, optional)
  ...
  :raises [ErrorType]: [ErrorDescription]
  ...
  :return: [ReturnDescription]
  :rtype: [ReturnType]
  """
  ```
- If you are using VS code, the Python Docstring extension can be used to auto-generate a docstring snippet once a function/class has been written. If you want the extension to generate docstrings in `Sphinx` format, you must set the `"autoDocstring.docstringFormat": "sphinx"` setting, under File > Preferences > Settings.
- Follow [PEP 257](https://peps.python.org/pep-0257/) standards and [Sphinx and RST syntax guide](https://thomas-cokelaer.info/tutorials/sphinx/docstring_python.html):

  ```python
  def test_function(arg1, arg2):
    """This function calculates the mean of two arguments.

    :param arg1: argument 1
    :type arg1: int, float, ...
    :param arg2: argument 2
    :type arg2: int, float, ...
    :return: mean of arg1 and arg2
    :rtype: int, float, ...

    :Example:

      >>> test_function(3, 5)
      4
    """
    return (arg1 + arg2) / 2
  ```

## Branching and Commit Guidelines

- Use meaningful branch names: `feature/feature-name`, `bugfix/issue-number`, `hotfix/issue-number`.
- Write concise and descriptive commit messages:

  ```
  [type] Short summary (max 50 chars)

  Optional detailed description, if necessary.
  ```

  Example:

  ```
  feat: Add ppo wrapper
  fix: Resolve issue with mujoco
  ```

- Prefix commit messages with:
  - `feat`: New feature
  - `fix`: Bug fix
  - `docs`: Documentation update
  - `style`: Formatting and style changes
  - `refactor`: Code refactoring
  - `test`: Adding or updating tests
  - `chore`: Maintenance tasks

## Pull Request Process

1. Ensure your code follows the style guide and passes all tests.
2. Provide a clear description of the changes.
3. Reference related issues if applicable.
4. If you know the right people or team that should approve your PR (and you have the required permissions to do so), add them to the Reviewers list.
5. Each PR need to be reviewed and accepted by at least one of the maintainers (@baotruyenthach, @capfab, TBD).
6. Wait for a review and address any requested changes.

**Note: Please limit to fewer than 300 lines of code per PR. 😊**

## Testing

- Ensure all new features include unit and integration tests.
- Run tests before submitting a pull request:
  ```sh
  WIP
  ```

## Documentation Guidelines

- Update the `README.md` if necessary.
- Use inline comments for complex logic.
- Keep documentation up to date with code changes.
- **We are working on an API documentation pipeline.**

## Issue Reporting

- Search existing issues before opening a new one.
- Provide a clear title and description.
- Include steps to reproduce the issue, if applicable.
- Add relevant labels (`bug`, `enhancement`, `documentation`, etc.).

## Contact

For any questions, reach out to the maintainers.

Thank you for your contributions!

Credits: this contributing guide is based on [sbx](https://github.com/araffin/sbx/blob/master/CONTRIBUTING.md), following [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).
