from __future__ import annotations  # until 3.10

__all__ = ['export', 'get_wild_imports', 'import_tree']

import pkgutil
import sys
from types import ModuleType


class ExportError(Exception):
    pass


def export(obj):
    """Exposes obj to __all__ in module parent to where it was defined,
    although breaking intellisense.

    Usage:

    In package/subpackage.py:
    ```
    from glow.api import export

    @export
    def func():
        pass
    ```
    In package/__init__.py:
    ```
    from . import subpackage
    ```
    In main.py:
    ```
    # not `from package.subpackage import func`
    from package import func
    func()
    ```
    """
    parent: str = sys.modules[obj.__module__].__spec__.parent  # type: ignore
    if obj.__module__ == parent:
        return obj

    name = obj.__name__
    namespace = sys.modules[parent].__dict__

    __all__ = namespace.setdefault('__all__', [])
    if name in __all__ or name in namespace:
        raise ExportError(f'Name "{name}" is reserved in <{parent}>'
                          f' by <{namespace[name].__module__}:{name}>')

    namespace[name] = obj
    __all__.append(name)
    return obj


def import_tree(pkg: str):
    """Imports all subpackages.

    Usage:
    ```
    __import__('glow.api', fromlist=['api']).import_tree(__name__)
    ```
    """
    path: str = sys.modules[pkg].__path__  # type: ignore
    for _, name, __ in pkgutil.walk_packages(path):
        subpkg = pkg + '.' + name
        __import__(subpkg)


def get_wild_imports(module: ModuleType) -> tuple[str, ...]:
    """Get contents of module.__all__ if possible"""
    __all__ = getattr(module, '__all__', None)
    if __all__ is not None:
        return __all__
    return tuple(name for name in dir(module) if not name.startswith('_'))
