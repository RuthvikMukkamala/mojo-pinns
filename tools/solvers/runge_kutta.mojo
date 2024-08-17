
from python import Python
import os
from collections.list import List
from collections.dict import Dict, KeyElement
from jax_tools import jmojo

struct RungeKutta:

    var initial_time: Int
    var final_time: Int
    var num_steps: Int
    var time_range: List[Int]
    var dtype: PythonObject
    var dt: PythonObject
    var q: Int


    fn __init__(inout self,
    initial_time: Int,
    final_time: Int, 
    num_steps: Int, 
    time_range: List[Int],
    q: Int,
    dtype: String) raises:

        var jnp = Python.import_module("jax.numpy")

        self.initial_time = initial_time
        self.final_time = final_time
        self.num_steps = num_steps
        self.time_range = time_range
        self.q = q
        self.dtype = jnp.dtype(dtype)

        self.dt = jnp.array(time_range[final_time] - time_range[initial_time]).astype(self.dtype) 
        if self.q == 0:
            self.q = int(jnp.ceil(0.5 * jnp.log(jnp.finfo(self.dtype).eps) / jnp.log(self.dt)))
        
    
    fn load_weights(inout self, data_file: PythonObject, q: Int) raises -> PythonObject:
        
        var jnp = Python.import_module("jax.numpy")
        
        var weights = jnp.reshape(data_file[0 : q**2 + q], (q + 1, q)).astype(self.dtype)

        weights = jmojo.use_file_dtype(self.dtype)
        

        return weights
        
    

    





   
    