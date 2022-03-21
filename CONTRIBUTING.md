# Contributing

Contributions are welcome and greatly appreciated!

## Types of Contributions

### Report Bugs

Report bugs at https://github.com/Jammy2211/PyAutoCTI/issues

If you are playing with the PyAutoCTI library and find a bug, please
reporting it including:

* Your operating system name and version.
* Any details about your Python environment.
* Detailed steps to reproduce the bug.

### Propose New Features

The best way to send feedback is to open an issue at
https://github.com/Jammy2211/PyAutoCTI
with tag *enhancement*.

If you are proposing a nnew feature:

* Explain in detail how it should work.
* Keep the scope as narrow as possible, to make it easier to implement.

### Implement Features
Look through the Git issues for operator or feature requests.
Anything tagged with *enhancement* is open to whoever wants to
implement it.

### Add Examples or improve Documentation
Writing new features is not the only way to get involved and
contribute. Create examples with existing features as well 
as improving the documentation of existing operators is as important
as making new non-linear searches and very much encouraged.


## Getting Started to contribute

Ready to contribute?

1. Follow the installation instructions for installing **PyAutoCTI** (and parent projects) from source root 
on our [readthedocs](https://pyautocti.readthedocs.io/en/latest/installation/source.html).

2. Create a feature branch for local development (for **PyAutoCTI** and every parent project where changes are implemented):
    ```
    git checkout -b feature/name-of-your-branch
    ```
    Now you can make your changes locally.

3. When you're done making changes, check that old and new tests pass succesfully:
    ```
    cd PyAutoCTI/test_autocti
    python3 -m pytest
    ```

4. Commit your changes and push your branch to GitHub::
    ```
    git add .
    git commit -m "Your detailed description of your changes."
    git push origin feature/name-of-your-branch
    ```
    Remember to add ``-u`` when pushing the branch for the first time.

5. Submit a pull request through the GitHub website.


### Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include new tests for all the core routines that have been developed.
2. If the pull request adds functionality, the docs should be updated accordingly.
