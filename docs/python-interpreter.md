# Selecting the right Python interpreter

## Jetbrains IDEs (PyCharm, Webstorm, etc.)
When you open a project in a Jetbrains IDE, it will automatically detect the Python interpreter that is used in the project. If you want to change the interpreter, follow these steps:
1. Open the project in the Jetbrains IDE
2. If a .venv exists in the project, click on it, then open the Scripts folder, if there is any version not equal to 3.11, delete the .venv 
3. Click on "File" in the top menu 
4. Click on "Settings"
5. Click on Build, Execution, Deployment 
6. Click on "Python Interpreter"
7. If the .venv is valid, click on the dropdown menu and select the Python 3.11 interpreter 
8. If the .venv is not valid and has been deleted, click on the "Add Interpreter" button, then click on "Add Local Interpreter"Leave the default settings and click on "OK" (Type should be "Virtualenv Environment", Base python should be a Path containing "Python 3.11")
9. Click on "OK" to save the changes