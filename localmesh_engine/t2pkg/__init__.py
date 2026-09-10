"""Namespace that makes the vendored trellis2 importable as a *subpackage*.

Two constraints have to be satisfied at once, and getting either wrong fails
in a way that points somewhere else entirely.

1. trellis2 must sit one level down. Its MoGe loader does
   `from ...moge.model.v2 import MoGeModel`, and from `trellis2.pipelines`
   three levels is one more than the package has - "attempted relative import
   beyond top-level package". As `t2pkg.trellis2.pipelines` it resolves to
   `t2pkg.moge`, which is the sibling layout the checkout actually has.

2. It must be imported under exactly ONE name. Aliasing `sys.modules["trellis2"]`
   to this package looks harmless, but any later `from trellis2.modules import x`
   re-imports the submodule under the other name and builds a second set of
   classes. The symptom is remote from the cause: an isinstance() check between
   the two copies fails, and the model dies with
   "linear(): argument must be Tensor, not SparseTensor".

So: no aliases. Everything - engine included - imports through `t2pkg`.
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))


def _checkout() -> str:
    from .. import config
    return str(config.TRELLIS_SRC)


# Search this directory first, then the checkout: `t2pkg.trellis2`, `t2pkg.moge`
# and `t2pkg.projection` all resolve into the vendored tree without copying it,
# so it stays updatable with a git pull.
__path__ = [_HERE, _checkout()]
