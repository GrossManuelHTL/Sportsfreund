# Python package installation guide

1. Make sure Python *3.11* is installed on your system.
2. If you don't know how to setup Python 3.11, check out the [Python version guide](python-version.md).
3. Select the [correct interpreter](python-interpreter.md) in your IDE.
4. Make sure you have the latest version of `pip` installed:
   ```bash
   python -m pip install --upgrade pip
   ```
5. If you only want to download the packages that are _used_ in the projekt, run:
```bash
   pip install pipreqs
   pipreqs [--force] --[ignore] <path_to_your_project>
   ```
6. If you want to install all the packages that are written down in the requirements.txt file for the project, run:
```bash
   pip install -r requirements.txt
   ```
