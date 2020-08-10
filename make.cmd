:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: VARIABLES                                                                    :
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

SETLOCAL
SET PROJECT_DIR=%cd%
SET CONDA_PARENT=arcgispro-py3
SET PROJECT_NAME=ba-tools
SET ENV_NAME=ba-tools

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: COMMANDS                                                                     :
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

:: Jump to command
GOTO %1

:: Build the local environment from the environment file
:env
    ENDLOCAL & (

        :: Run this from the ArcGIS Python Command Prompt
        :: Clone and activate the new environment
        CALL conda create --name "%ENV_NAME%" --clone "%CONDA_PARENT%" -y
        CALL activate "%ENV_NAME%"

        :: Install nodejs so it does not throw an error later
        CALL conda install -y nodejs

        :: Install additional packages
        CALL conda env update -f environment.yml

        :: Install the local package in development mode
        CALL python -m pip install -e .

        :: Additional steps for the map widget to work in Jupyter Lab
        CALL jupyter labextension install @jupyter-widgets/jupyterlab-manager -y
        CALL jupyter labextension install arcgis-map-ipywidget@1.8.2 -y

        :: Set the ArcGIS Pro Python environment
        proswap "%ENV_NAME%"
    )
    EXIT /B

:: Update the current environment with resources needed to publish the package
:env_dev
    ENDLOCAL & (

        :: Install additional packages
        CALL conda env update -f environment_dev.yml

    )
    EXIT /B

:: Make the package for uploading
:build
    ENDLOCAL & (

        :: Build the pip package
        CALL python setup.py sdist

        :: Build conda package
        CALL conda build ./conda-recipe --output-folder ./conda-recipe/conda-build

    )
    EXIT /B

:build_upload
    ENDLOCAL & (

        :: Build the pip package
        CALL python setup.py sdist bdist_wheel
        CALL twine upload ./dist/*

        :: Build conda package
        CALL conda build ./conda-recipe --output-folder ./conda-recipe/conda-build
        CALL anaconda upload ./conda-recipe/conda-build/win-64/ba-tools*.tar.bz2

    )
    EXIT /B

:: Run all tests in module
:test
	ENDLOCAL & (
		pytest
	)
	EXIT /B

EXIT /B