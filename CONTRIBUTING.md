# Contributing to LSTMFund

This file outlines the guidelines for contributing to the LSTMFund project. Is meant to help developers understand how to work with the codebase.

## Using uv

This project uses [uv](https://docs.astral.sh/uv/) as the dependency managar, build tool, task runner, among other things. Before contributing, ensure you have `uv` installed. You can find installation instructions in the [uv documentation](https://docs.astral.sh/uv/getting-started/installation/).

### Adding New Dependencies

To add new dependencies, run `uv add <package-name>` in the terminal, for example:

```bash
uv add requests
```
