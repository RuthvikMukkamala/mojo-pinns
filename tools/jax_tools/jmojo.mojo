'''
This module contains a collection of tools for working with Jax written for project.
'''
from python import Python


fn use_jnp_dtype(dtype: String) raises -> PythonObject:

    var jnp = Python.import_module("jax.numpy")
    return jnp.dtype(dtype)


