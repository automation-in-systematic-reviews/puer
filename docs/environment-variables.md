# Environment variables

This is the canonical reference for PUER API keys, environment loading and precedence, Docker Compose environment inputs, and task-runner environment checks.
The application reads its configuration in `inference-api/app/resources/globals.py`.

## Application variables

### WCRF_API_KEY

- Type: string.
- Default: none.
- Scope: API process and authenticated PUER requests.
- Purpose: Supplies the key that the server compares with the `X-API-Key` header for the authenticated legacy screening, prompt extraction, prompt prediction, and risk-of-bias endpoints.
- Constraints: Required when the API starts. This is the key that a PUER client sends to the PUER server, not an OpenAI credential.

### OPENAI_API_KEY

- Type: string or null.
- Default: null.
- Scope: Server-side provider calls from prompt endpoints and `/risk_of_bias/assess`.
- Purpose: Authenticates outbound OpenAI requests made by the server.
- Constraints: Optional at startup but required when invoking either prompt endpoint or `/risk_of_bias/assess`. Clients must not send this credential to PUER. Missing provider configuration is reported when the relevant request reaches the provider boundary.

## LLM variable ownership

The LLM model and reasoning variable names are listed here for discoverability, but [LLM configuration](llm-configuration.md) owns their defaults, endpoint scope, provider compatibility, lazy initialization, and response provenance.

- Prompt endpoints use `OPENAI_STUDY_SCREENING_MODEL` and `OPENAI_STUDY_SCREENING_REASONING_EFFORT`.
- `/risk_of_bias/assess` uses `OPENAI_MODEL` and `OPENAI_REASONING_EFFORT` independently.

## Example configuration file

- Template: [`.env.example`](../.env.example), containing placeholders and current nonsecret defaults.
- Local copy: `.env` at the PUER repository root, beside `docker-compose.yml`; ignored by Git.
- Required edit: Replace `WCRF_API_KEY` with a private value before starting the API.
- Live provider calls: Fill in `OPENAI_API_KEY` locally; leave it blank for offline work.
- Model settings: See [LLM configuration](llm-configuration.md) for compatibility and endpoint scope.
- Docker limitation: Screening-specific overrides in the template are not forwarded by the current Compose service.
- Security: Keep the populated `.env` private; never commit it or copy its values into logs or documentation.

From the PUER repository root, copy the template without overwriting an existing configuration:

```sh
cp -n .env.example .env
chmod 600 .env
```

Edit the local copy yourself, then start the API using the existing development or Docker workflow.
For native use, the application's `Env.read_env()` searches for `.env`; an already exported variable takes precedence over the file.
For Docker, Compose reads the root `.env` for interpolation and forwards only the variables listed below.
Restart the native API or recreate the container after configuration changes.

## Native setup

Run `just run-local` from `inference-api` in the same shell that holds the exported variables.
The following Bash commands hide the required inbound key and do not put its literal value in command history.

```bash
read -r -s -p 'WCRF_API_KEY: ' WCRF_API_KEY
printf '\n'
export WCRF_API_KEY
```

Set `OPENAI_API_KEY` only when invoking an OpenAI-backed endpoint.
Use the same hidden-input pattern for that server-side key when it is needed.

```bash
read -r -s -p 'OPENAI_API_KEY: ' OPENAI_API_KEY
printf '\n'
export OPENAI_API_KEY
```

For nonsecret model and reasoning overrides, see [LLM configuration](llm-configuration.md).
Keep real key values in the private local `.env` or process environment, never in tracked files or documentation.

## Loading and restart behavior

- Loading: The application explicitly calls `Env.read_env()` before reading these variables, so its implemented `.env` loading behavior can supply values found by that call.
- Precedence: Existing process environment values are retained because the application does not request `.env` values to override them.
- Import timing: The application reads configuration while its globals module is imported.
- Restart: Restart the API after changing a value.
- Task runner: Exported variables are inherited by the child process started by `just`.
- Docker interpolation: Docker Compose separately uses the project-level `.env` for `${...}` interpolation.

## Docker Compose input

### INFERENCE_API_PORT

- Type: string interpreted as a host port number.
- Default: `12306`.
- Scope: Docker Compose host port publishing.
- Purpose: Changes the host port published for the API container.
- Constraints: It is not read by the API and is not passed into the container as application configuration.

### Docker Compose forwarding

- Type: environment-variable mapping.
- Default: The Compose defaults apply only to variables shown below.
- Scope: `docker-compose.yml` interpolation and the `inference-api` container.
- Purpose: Forwards selected server configuration into the container.
- Constraints: The current Compose service forwards only `WCRF_API_KEY`, `OPENAI_API_KEY`, `OPENAI_MODEL`, and `OPENAI_REASONING_EFFORT` into the container. It does not currently forward the screening-specific model or reasoning overrides, so those variables are not documented as Docker overrides here.

## Task-runner environment

### CONDA_DEFAULT_ENV

- Type: string.
- Default: none supplied by PUER.
- Scope: `just` recipe preflight checks.
- Purpose: Identifies the active Conda environment.
- Constraints: The recipes require the value `inference-api`.

### CONDA_PREFIX

- Type: non-empty path string.
- Default: none supplied by PUER.
- Scope: `just` recipe preflight checks and uv environment selection.
- Purpose: Identifies the active Conda prefix.
- Constraints: The recipes require a non-empty value and expect `python`, `uv`, and `just` under this prefix.

### UV_PROJECT_ENVIRONMENT

- Type: path string.
- Default: set by the `just` recipes from `CONDA_PREFIX`.
- Scope: uv commands in the task runner.
- Purpose: Directs `uv` to install into and run from the active Conda prefix.
- Constraints: It controls the development tool environment and is not PUER application configuration.
