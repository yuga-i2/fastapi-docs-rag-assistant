"""
List of source documents that make up the knowledge base.

We use the official FastAPI documentation (tutorial + advanced + deployment
sections) as the corpus for this project. It's real, well-structured,
technical documentation -- a good stand-in for "internal docs" that a
support/engineering team might want a Q&A assistant for.

Each entry is a path inside the tiangolo/fastapi GitHub repo, under
docs/en/docs/. We fetch the raw markdown for each at ingestion time.
"""

REPO_RAW_BASE = "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs"

DOC_PATHS = [
    "index.md",
    "features.md",
    "async.md",
    "tutorial/first-steps.md",
    "tutorial/path-params.md",
    "tutorial/query-params.md",
    "tutorial/body.md",
    "tutorial/query-params-str-validations.md",
    "tutorial/path-params-numeric-validations.md",
    "tutorial/body-multiple-params.md",
    "tutorial/body-fields.md",
    "tutorial/body-nested-models.md",
    "tutorial/schema-extra-example.md",
    "tutorial/extra-data-types.md",
    "tutorial/cookie-params.md",
    "tutorial/header-params.md",
    "tutorial/response-model.md",
    "tutorial/extra-models.md",
    "tutorial/response-status-code.md",
    "tutorial/request-forms.md",
    "tutorial/request-files.md",
    "tutorial/request-forms-and-files.md",
    "tutorial/handling-errors.md",
    "tutorial/path-operation-configuration.md",
    "tutorial/body-updates.md",
    "tutorial/dependencies/index.md",
    "tutorial/security/index.md",
    "tutorial/middleware.md",
    "tutorial/cors.md",
    "tutorial/sql-databases.md",
    "tutorial/bigger-applications.md",
    "tutorial/background-tasks.md",
    "tutorial/testing.md",
    "tutorial/debugging.md",
    "deployment/index.md",
    "deployment/docker.md",
    "deployment/concepts.md",
    "advanced/index.md",
    "advanced/path-operation-advanced-configuration.md",
    "advanced/additional-status-codes.md",
    "advanced/response-directly.md",
    "advanced/custom-response.md",
    "advanced/additional-responses.md",
    "advanced/response-cookies.md",
    "advanced/response-headers.md",
    "advanced/response-change-status-code.md",
    "advanced/websockets.md",
    "advanced/events.md",
    "advanced/settings.md",
]
