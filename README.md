# Template for the inference api

## Setting up (for local development)

First install docker (e.g. `brew install --cask docker`) 
and also git-lfs (e.g. `brew install git-lfs` and then `git lfs install`).

Clone the repository.
Then go to `inference-api/inference-api/models` 
(where the first `inference-api` is the root directory of the local repo),
and clone the `textattack/albert-base-v2-imdb` from Huggingface.

Go back to root `inference-api` and run `docker-compose build` to build the docker image(s).

Run `docker-compose up` and you should see the service(s) running in the terminal session.

Now from a web browser go to `http://localhost:12306/docs` you should see a fastapi web service.


## How to use for development

By default the `inference-api` directory is watched for changes from the running session for hot reload.
So any changes in the code will trigger the fastapi service to rerun -- be careful when to do this as
init time of transformer model loading is non-trivial.

When the service session is running in a terminal session, open up another terminal session and run 
`docker-compose exec -it inference-api bash` and you will be inside the running docker container.

Run `make fmt` will trigger autoformat of the codebase using black.

Run `make lint` will trigger linting of the codebase using flake8.


## Deployment

Do everything from the setting up section before the `docker-compose up` step.
Run `docker-compose up -d` instead.

## Other technical details

- The [conda environment](./inference-api/environment.yml) is created from micromamba
- [Makefile](./inference-api/Makefile) should be used for inferfacing with the code infrastructure
- Inside the docker image / container, the working directory is `/inference-api`
- docs for fastapi is https://fastapi.tiangolo.com/
