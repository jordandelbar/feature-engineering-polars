# Contributing guide

Once you have installed the package using poetry, do not forget to
install the pre-commit hooks by running:

```bash
poetry run pre-commit install
```
When you are ready to contribute, open a pull request to the dev branch and ensure that it:

- Clearly describes what the contribution does.
- References any related issue numbers (e.g., Closes #123).

We follow the Conventional Commits specification for commit messages:

- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation changes
- `style`: Code style changes (e.g., formatting, removing whitespace)
- `refactor`: Code changes that neither fix a bug nor add a feature.
- `perf`: A code change that improves performance
- `test`: Adding or correcting tests
- `build`: Changes that affect the build system or external dependencies
- `ci`: Changes to continuous integration configuration files or scripts
- `chore`: Maintenance tasks (e.g., build scripts)
- `revert`: Reverts a previous commit

example commit message:

```
feat: Add a new feature
```

When the commit introduces a breaking change, add an exclamation point (`!`) to the
type and include a description of the breaking change in the commit message. For example:

```
feat!: Add a breaking change new feature
```
