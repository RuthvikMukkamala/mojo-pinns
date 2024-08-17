# Mojo adaptation of the module
struct JaxNumPy:
    fn import_jax_numpy() -> PythonObject:
        from python import Python
        return Python.import_module("jax.numpy")

    fn use_jnp_dtype(dtype: String) raises -> PythonObject:
        let jnp = self.import_jax_numpy()
        return jnp.dtype(dtype)


struct LossFunctions:

    # Sum of Squared Errors (SSE)
    fn sse(loss: Float, preds: Dictionary[String, PythonObject], target: Optional[Dictionary[String, PythonObject]] = none, keys: Optional[List[String]] = none, mid: Optional[Int] = none) -> Float:
        let jnp = JaxNumPy().import_jax_numpy()

        if keys.is_none():
            return loss

        for key in keys!:
            if target.is_none() and mid.is_none():
                loss = loss + jnp.sum(jnp.square(preds[key]))
            elif target.is_none() and !mid.is_none():
                loss = loss + jnp.sum(jnp.square(preds[key][:mid!] - preds[key][mid!:]))
            elif !target.is_none():
                loss = loss + jnp.sum(jnp.square(preds[key] - target![key]))
        return loss

    # Mean Squared Error (MSE)
    fn mse(loss: Float, preds: Dictionary[String, PythonObject], target: Optional[Dictionary[String, PythonObject]] = none, keys: Optional[List[String]] = none, mid: Optional[Int] = none) -> Float:
        let jnp = JaxNumPy().import_jax_numpy()

        if keys.is_none():
            return loss

        for key in keys!:
            if target.is_none() and mid.is_none():
                loss = loss + jnp.mean(jnp.square(preds[key]))
            elif target.is_none() and !mid.is_none():
                loss = loss + jnp.mean(jnp.square(preds[key][:mid!] - preds[key][mid!:]))
            elif !target.is_none():
                loss = loss + jnp.mean(jnp.square(preds[key] - target![key]))
        return loss

    # Relative L2 Error
    fn relative_l2_error(preds: PythonObject, target: PythonObject) -> Float:
        let jnp = JaxNumPy().import_jax_numpy()
        return jnp.sqrt(jnp.mean(jnp.square(preds - target)) / jnp.mean(jnp.square(target)))


struct ModelFunctions:

    # Fix extra variables for optimization purposes
    fn fix_extra_variables(trainable_variables: Dictionary[String, PythonObject], extra_variables: Optional[Dictionary[String, PythonObject]] = none) -> (Dictionary[String, PythonObject], Optional[Dictionary[String, PythonObject]]):
        if extra_variables.is_none():
            return (trainable_variables, none)

        var extra_variables_dict = Dictionary[String, PythonObject]()
        for key in extra_variables!:
            let variable = JaxNumPy().import_jax_numpy().array(extra_variables![key])
            extra_variables_dict[key] = variable
            trainable_variables[key] = variable
        return (trainable_variables, extra_variables_dict)

    # Make model functional
    fn make_functional(net: PythonObject, params: PythonObject, n_dim: Int, discrete: Bool, output_fn: Optional[PythonObject] = none) -> PythonObject:
        def _execute_model(net: PythonObject, params: PythonObject, inputs: List[PythonObject], time: PythonObject, output_c: Optional[PythonObject] = none) -> PythonObject:
            outputs_dict = net(params, inputs, time)
            if output_c.is_none():
                return outputs_dict
            else:
                return outputs_dict[output_c!].squeeze()

        def functional_model_1d(params: PythonObject, x: PythonObject, time: PythonObject, output_c: Optional[PythonObject] = none) -> PythonObject:
            return _execute_model(net, params, [x], time, output_c)

        def functional_model_2d(params: PythonObject, x: PythonObject, y: PythonObject, time: PythonObject, output_c: Optional[PythonObject] = none) -> PythonObject:
            return _execute_model(net, params, [x, y], time, output_c)

        def functional_model_3d(params: PythonObject, x: PythonObject, y: PythonObject, z: PythonObject, time: PythonObject, output_c: Optional[PythonObject] = none) -> PythonObject:
            return _execute_model(net, params, [x, y, z], time, output_c)

        let models = Dictionary[Int, PythonObject]()
        models[2] = functional_model_1d
        models[3] = functional_model_2d
        models[4] = functional_model_3d

        if models.contains_key(n_dim):
            return models[n_dim]!
        else:
            raise ValueError(f"{n_dim} number of dimensions is not supported.")
