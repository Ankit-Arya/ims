class AppError(Exception):
    pass


class NotFoundError(AppError):
    pass


class ForbiddenError(AppError):
    pass


class ConfigurationError(AppError):
    pass
