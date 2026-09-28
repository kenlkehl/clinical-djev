class DjevError(Exception):
    status = 500
    code = "internal_error"


class InvalidRequest(DjevError):
    status = 422
    code = "invalid_request"


class ContextTooLong(DjevError):
    status = 413
    code = "context_too_long"


class BackendError(DjevError):
    status = 502
    code = "backend_error"


class BackendUnavailable(DjevError):
    status = 503
    code = "backend_unavailable"
