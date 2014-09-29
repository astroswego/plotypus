from pkg_resources import resource_filename as _rf

__all__ = [
    'matplotlibrc'
]

matplotlibrc = _rf('plotypus.resources', 'matplotlibrc')
