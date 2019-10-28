def transform_observations(observations, instruction_sensor_uuid):
    # only take instruction tokens from instruction sensor
    for i in range(len(observations)):
        observations[i][instruction_sensor_uuid] = observations[i][
            instruction_sensor_uuid
        ]["tokens"]
    return observations
