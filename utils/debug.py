"""IPython debugger"""
from IPython.core import debugger


# usage: import debug and use debug() wherever you want to stop
debug = debugger.Pdb().set_trace
