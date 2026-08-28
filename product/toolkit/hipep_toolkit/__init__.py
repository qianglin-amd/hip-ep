# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""The ``hipep`` command line: serve, pull, run, port, bench and inspect.

Every subcommand except ``serve`` is a client of the HTTP API rather than a second
implementation of it. That is deliberate: the CLI, the console and a user's own script
all reach the server the same way, so there is no privileged path that behaves
differently from the one being documented and measured.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
