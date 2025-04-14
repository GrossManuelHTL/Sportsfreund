# Installing the correct Python version for the project

## Steps:

1. (Windows) Visit [python.org/3.11](https://www.python.org/downloads/release/python-3110/), scroll down and 
download the Windows installer 64-bit executable installer.
2. Open the executable
3. Before clicking "Install", tick the "Add Python 3.11 to PATH" checkbox
4. When the setup is complete, open a new terminal and run `python --version` to verify the installation.
5. If you something went wrong + you previously installed another version of Python that is not 3.11 but the installation setup was successfull, do the following:
    1. Open the Windows environment variables
    2. Click on "Environment Variables"
    3. Click on the "Path" variable in the "System variables" section
    4. Click on the entries that are related to the previous Python installation and click "Delete" (2 entries)
    5. Add the following entries to the Path variable:
        1. `C:\Users\<your_username>\AppData\Local\Programs\Python\Python311`
        2. `C:\Users\<your_username>\AppData\Local\Programs\Python\Python311\Scripts`
    6. Click "OK" to save the changes
    7. Open a new terminal and run `python --version` to verify the installation.
   