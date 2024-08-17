from collections.dict import Dict

struct LossFunctions:

    # Sum of Squared Errors (SSE)
    fn sse(loss: Float32, preds: Dict[String, PythonObject], target: Optional[Dict[String, PythonObject]] = None, keys: Optional[List[String]] = None, mid: Optional[Int] = None) -> Float32:
        let jnp = JaxNumPy().import_jax_numpy()

        if keys.is_none():
            return loss

        for key in keys.get_value_or([]):
            if target.is_none() and mid.is_none():
                loss = loss + jnp.sum(jnp.square(preds[key]))
            elif target.is_none() and mid.is_some():
                loss = loss + jnp.sum(jnp.square(preds[key][:mid.get_value()] - preds[key][mid.get_value():]))
            elif target.is_some():
                loss = loss + jnp.sum(jnp.square(preds[key] - target.get_value()[key]))
        return loss

    # Mean Squared Error (MSE)
    fn mse(loss: Float32, preds: Dict[String, PythonObject], target: Optional[Dict[String, PythonObject]] = None, keys: Optional[List[String]] = None, mid: Optional[Int] = None) -> Float32:
        let jnp = JaxNumPy().import_jax_numpy()

        if keys.is_none():
            return loss

        for key in keys.get_value_or([]):
            if target.is_none() and mid.is_none():
                loss = loss + jnp.mean(jnp.square(preds[key]))
            elif target.is_none() and mid.is_some():
                loss = loss + jnp.mean(jnp.square(preds[key][:mid.get_value()] - preds[key][mid.get_value():]))
            elif target.is_some():
                loss = loss + jnp.mean(jnp.square(preds[key] - target.get_value()[key]))
        return loss

    # Relative L2 Error
    fn relative_l2_error(preds: PythonObject, target: PythonObject) -> Float32:
        let jnp = JaxNumPy().import_jax_numpy()
        return jnp.sqrt(jnp.mean(jnp.square(preds - target)) / jnp.mean(jnp.square(target)))


struct ModelFunctions:

    # Fix extra variables for optimization purposes
    fn fix_extra_variables(trainable_variables: Dict[String, PythonObject], extra_variables: Optional[Dict[String, PythonObject]] = None) -> (Dict[String, PythonObject], Optional[Dict[String, PythonObject]]):
        if extra_variables.is_none():
            return (trainable_variables, None)

        var extra_variables_dict = Dict[String, PythonObject]()
        for key in extra_variables.get_value_or([]):
            let variable = JaxNumPy().import_jax_numpy().array(extra_variables.get_value()[key])
            extra_variables_dict[key] = variable
            trainable_variables[key] = variable
        return (trainable_variables, extra_variables_dict)

    # Make model functional
    fn make_functional(net: PythonObject, params: PythonObject, n_dim: Int, discrete: Bool, output_fn: Optional[PythonObject] = None) -> PythonObject:
        def _execute_model(net: PythonObject, params: PythonObject, inputs: List[PythonObject], time: PythonObject, output_c: Optional[PythonObject] = None) -> PythonObject:
            outputs_dict = net(params, inputs, time)
            if output_c.is_none():
                return outputs_dict
            else:
                return outputs_dict[output_c.get_value()].squeeze()

        def functional_model_1d(params: PythonObject, x: PythonObject, time: PythonObject, output_c: Optional[PythonObject] = None) -> PythonObject:
            return _execute_model(net, params, [x], time, output_c)

        def functional_model_2d(params: PythonObject, x: PythonObject, y: PythonObject, time: PythonObject, output_c: Optional[PythonObject] = None) -> PythonObject:
            return _execute_model(net, params, [x, y], time, output_c)

        def functional_model_3d(params: PythonObject, x: PythonObject, y: PythonObject, z: PythonObject, time: PythonObject, output_c: Optional[PythonObject] = None) -> PythonObject:
            return _execute_model(net, params, [x, y, z], time, output_c)

        let models = Dict[Int, PythonObject]()
        models[2] = functional_model_1d
        models[3] = functional_model_2d
        models[4] = functional_model_3d

        if models.contains_key(n_dim):
            return models[n_dim]
        else:
            raise ValueError(f"{n_dim} number of dimensions is not supported.")
