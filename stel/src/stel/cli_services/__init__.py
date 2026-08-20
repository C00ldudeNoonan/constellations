"""Internal CLI services (issue #190, Workstream D).

Logic behind the Click commands — the shared project/profile bootstrap, watch
execution, and other operations — factored out of ``cli.py`` so it is
importable and independently testable. The Click declarations and user-facing
formatting stay at the CLI edge in ``cli.py``; these modules never define
commands.
"""
