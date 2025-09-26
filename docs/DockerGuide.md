## Guide for the Docker and DB setup
We need this so that the backend has a functional db in the background.
Before you start with this it is important, that you already have docker installed.

If the container does not exist already use the following command to do so
``
docker run --name sportsfreunddb -e POSTGRES_USER=spdb -e POSTGRES_PASSWORD=1234 -e POSTGRES_DB=SportsfreundDB -p 5432:5432 -d postgres
``

If that is done you can start and stop the db with these commands
``docker start sportsfreunddb``
``docker stop sportsfreunddb``


For changes in the backend a new migration as to be done with that. This only has to be done if YOU change the db, because the migrationfiles will get pushed on git.
``python manage.py makemigrations``

If you are not sure of the current state of your local db, update it with the migrationfiles wich is done by the following command
``python manage.py migrate``
