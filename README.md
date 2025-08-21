# StoryEvals
evals for your video generations

## Development Setup

### Pre-commit Hooks

This project uses pre-commit hooks to ensure code quality and consistency. The hooks will run automatically on every commit and include:

- **Ruff**: Fast Python linting and formatting (primary tool)
- **MyPy**: Type checking
- **Various checks**: Merge conflicts, YAML validation, large files, etc.

#### Quick Setup

1. **Automatic setup** (recommended):
   ```bash
   python setup_precommit.py
   ```

2. **Manual setup**:
   ```bash
   # Install pre-commit
   uv add --dev pre-commit

   # Install the hooks
   pre-commit install

   # Install additional dependencies
   uv add --dev mypy types-requests types-PyYAML

   # Run on all files
   pre-commit run --all-files
   ```

#### Usage

- **Automatic**: Hooks run automatically on every commit
- **Manual run on staged files**: `pre-commit run`
- **Manual run on all files**: `pre-commit run --all-files`
- **Run specific hook**: `pre-commit run <hook-id>`
- **Update hook versions**: `pre-commit autoupdate`

#### Configuration

All tool configurations are in `pyproject.toml`:
- Ruff: Python 3.11+, line length 88, comprehensive rule set (handles both linting and formatting)
- MyPy: Strict type checking enabled

#### Troubleshooting

If hooks fail:
1. Fix the issues automatically: `pre-commit run --all-files`
2. Some issues may require manual fixes
3. Check the output for specific error messages
4. Run individual hooks to isolate problems: `pre-commit run <hook-id>`
