:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: VARIABLES                                                                    :
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

SETLOCAL
SET PROJECT_DIR=%cd%
SET PROJECT_NAME=ba-tools
SET ENV_NAME=ba-tools

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: COMMANDS                                                                     :
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

:: Jump to command
GOTO %1

:: Update the current environment with resources needed to publish the package
:env_update
    ENDLOCAL & (

        :: Install additional packages
        CALL conda env update -f environment.yml

    )
    EXIT /B

:: Make the package for uploading
:build
    ENDLOCAL & (

        :: Build the pip package
        CALL python setup.py sdist

        :: Build conda package
        CALL conda build ./conda-recipe

    )
    EXIT /B

:build_upload
    ENDLOCAL & (

        :: Build the pip package
        CALL python setup.py sdist
        CALL twine upload ./dist

        :: Build conda package
        CALL conda build ./conda-recipe
        CALL anaconda upload ./conda-recipe/conda-build\win-64\ba-tools*.tar.bz2

    )
    EXIT /B

:: Run all tests in module
:test
	ENDLOCAL & (
		pytest
	)
	EXIT /B

EXIT /B