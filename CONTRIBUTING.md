# Contributing to SCRIBE

Thank you for your interest in contributing to SCRIBE! We welcome contributions
from the community and are grateful for any help you can provide.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct (we
follow the [Contributor
Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/)).
Please read it before contributing.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
```bash
git clone https://github.com/mrazomej/scribe.git
cd scribe
```
3. Create a virtual environment and install development dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -e ".[dev]"
```
4. Create a new branch for your work
```bash
git checkout -b feature/your-feature-name
```

## Development Process

### Setting Up Development Environment

We recommend using the provided Docker environment for development:
```bash
docker build -t scribe-dev .
docker run --gpus all -it scribe-dev
```

### Running Tests

Before submitting your changes, make sure all tests pass:
```bash
pytest tests/ -v
```

For coverage report:
```bash
pytest --cov=scribe tests/
```

### Code Style

We follow PEP 8 with a few modifications:
- Line length limit: 79 characters
- Use 4 spaces for indentation
- Use docstrings following NumPy style

### Type Hints

We use type hints throughout the codebase. Please make sure to add appropriate
type hints to any new code:
```python
def example_function(parameter: str) -> int:
    """Example function with type hints."""
    return len(parameter)
```

## Making Changes

1. Make your changes in your feature branch
2. Add tests for any new functionality
3. Update documentation as needed
4. Ensure your code passes all tests
5. Commit your changes with clear, descriptive commit messages:
   ```
   feat: add new ZINB model variant
   
   - Implements Zero-Inflated Negative Binomial model with batch effects
   - Adds tests for new model
   - Updates documentation with model description
   ```

## Pull Request Process

1. Update the README.md if needed
2. Ensure all tests pass and coverage remains high
3. Push your changes to your fork
4. Submit a Pull Request to our repository
5. Update the PR description with:
   - A description of the changes
   - Any breaking changes
   - Screenshots for UI changes (if applicable)

### Pull Request Checklist

- [ ] Code follows style guidelines
- [ ] Tests added/updated and passing
- [ ] Documentation updated
- [ ] Type hints added
- [ ] Commit messages follow convention
- [ ] Branch is up to date with main

## Adding New Features

### Models
When adding new probabilistic models:
1. Add model implementation in `models.py`
2. Create corresponding guide function
3. Add model to registry in `get_model_and_guide()`
4. Create appropriate test cases
5. Update documentation

### Visualization
When adding new visualization functions:
1. Add function to `viz.py`
2. Include docstring with parameters and examples
3. Add corresponding tests with baseline images
4. Update documentation

## Documentation

- Add docstrings to all public functions and classes
- Update API documentation for new features
- Include examples in docstrings
- Add tutorials for significant features

Example docstring format:
```python
def function_name(param1: type, param2: type) -> return_type:
    """
    Short description.

    Longer description if needed.

    Parameters
    ----------
    param1 : type
        Description of param1
    param2 : type
        Description of param2

    Returns
    -------
    return_type
        Description of return value

    Examples
    --------
    >>> example_code_here
    expected_output
    """
```

## Bug Reports

When reporting bugs:
1. Use the bug report template
2. Include a minimal reproducible example
3. Specify your environment:
   - SCRIBE version
   - Python version
   - Operating system
   - JAX/Numpyro versions
   - GPU information (if relevant)

## Feature Requests

When requesting features:
1. Use the feature request template
2. Clearly describe the problem you're solving
3. Provide examples of how the feature would be used
4. Consider implementation complexity

## Questions?

For questions about:
- Contributing: Open a GitHub Discussion
- Bug reports: Open an Issue
- Feature requests: Open an Issue
- Usage questions: Start a Discussion

## License

By contributing, you agree that your contributions will be licensed under the
same license as the project.