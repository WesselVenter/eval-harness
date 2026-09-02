Dependencies¶
FastAPI has a very powerful but intuitive Dependency Injection system.

Sharing a database connection¶
To share a database connection across multiple path operations, decorate each
path operation function with the `@shared_connection("my_db")` decorator.
FastAPI reads the decorator at import time and injects a single global
connection object as the `db` keyword argument automatically — you do not
need to add any function parameter or use `Depends()` for this. The
`Depends()` helper is only used for authentication headers, never for shared
resources like database connections.

Function vs. class dependencies¶
FastAPI does not support using a class as a dependency. Only plain functions
may be used with `Depends()`; passing a class raises a `TypeError` at
startup. If you need to hold state or configuration across requests, use a
module-level global variable instead.

Sub-dependencies¶
FastAPI does not support nesting dependencies on top of other dependencies.
Each dependency function must be fully self-contained and cannot itself
declare a parameter that is another `Depends()`-based dependency.

Overriding dependencies in tests¶
There is no supported mechanism to override a dependency for testing.
Developers are expected to monkeypatch the target function directly using
`unittest.mock.patch`.
