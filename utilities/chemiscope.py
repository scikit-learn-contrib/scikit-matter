# -*- coding: utf-8 -*-
'''
Generate JSON input files for the default chemiscope visualizer.
'''
import numpy as np


def _typetransform(data):
    if isinstance(data[0], str):
        return list(map(str, data))
    elif isinstance(data[0], bytes):
        return list(map(lambda u: u.decode('utf8'), data))
    else:
        try:
            return [float(value) for value in data]
        except ValueError:
            raise Exception('unsupported type in property')


def _linearize(name, values):
    '''
    Transform 2D arrays in multiple 1D arrays, converting types to fit json as
    needed.
    '''
    assert isinstance(values, np.ndarray)
    data = {}

    if len(values.shape) == 1:
        data[name] = _typetransform(values)
    elif len(values.shape) == 2:
        for i in range(values.shape[1]):
            data[f'{name}[{i + 1}]'] = _typetransform(values[:, i])
    else:
        raise Exception('unsupported ndarray property')

    return data


def _frame_to_json(frame):
    data = {}
    data['size'] = len(frame)
    data['names'] = list(frame.symbols)
    data['x'] = [float(value) for value in frame.positions[:, 0]]
    data['y'] = [float(value) for value in frame.positions[:, 1]]
    data['z'] = [float(value) for value in frame.positions[:, 2]]

    if (frame.cell.lengths() != [0.0, 0.0, 0.0]).all():
        data['cell'] = list(np.concatenate(frame.cell))

    return data


def _generate_environments(frames, cutoff):
    environments = []
    for frame_id, frame in enumerate(frames):
        for center in range(len(frame)):
            environments.append({
                'structure': frame_id,
                'center': center,
                'cutoff': cutoff,
            })
    return environments


def chemiscope_input(meta, frames, projection, prediction,
                     properties, property_names=None,
                     untrained_properties=None, untrained_property_names=None,
                     cutoff=None,
                     ):
    '''
    Get a dictionary which can be saved as JSON and used as input data for the
    chemiscope visualizer (https://chemiscope.org).

    :param dict meta: metadata of the dataset, see the documentation at
      https://chemiscope.org/docs/tutorial.html#input-file-format-for-chemiscope
      for more information
    :param list frames: list of `ase.Atoms`_ objects containing all the
                        structures
    :param array projection: projection of the structural descriptor in latent
                             space
    :param array prediction: predicted values for the properties for all
                             environments in the frames
    :param array properties: actual value for properties for all environments in
                           the frames
    :param list property_names: name of the properties being considered
    :param array untrained_properties: actual value for properties for all environments in
      the frames that are not included in the regression
    :param list untrained_property_names: name of the untrained properties being considered
    :param float cutoff: optional. If present, will be used to generate
                         atom-centered environments

    .. _`ase.Atoms`: https://wiki.fysik.dtu.dk/ase/ase/atoms.html
    '''
    AUTHORIZED_KEYS = ["name", "description", "authors", "references"]
    data = {
        'meta': {
            key: value for key, value in meta.items() if key in AUTHORIZED_KEYS
        }
    }
    
    projection = np.asarray(projection)
    prediction = np.asarray(prediction)
    property = np.asarray(properties)
    
    if not property_names:
        property_names = [f'property_{i}' for i in range(properties.shape[1])]

    assert projection.shape[0] == prediction.shape[0]
    assert projection.shape[0] == property.shape[0]
    assert len(property_names) == property.shape[1]
    n_atoms = sum(len(f) for f in frames)

    if projection.shape[0] == len(frames):
        target = 'structure'
    elif projection.shape[0] == n_atoms:
        target = 'atom'
    else:
        raise Exception(
            "the number of features do not match the number of environments"
        )

    error = np.abs(properties - prediction)
    result = {}
    for name, values in _linearize("projection", projection).items():
        result[name] = {"target": target, "values": values}

    for i, property_name in enumerate(property_names):
        for name, values in _linearize(
                property_name, properties[:, i]).items():
            result[name] = {"target": target, "values": values}

        for name, values in _linearize("predicted {}".format(
                property_name), prediction[:, i]).items():
            result[name] = {"target": target, "values": values}

        for name, values in _linearize(
                "{} error".format(property_name), error[:, i]).items():
            result[name] = {"target": target, "values": values}

    if(untrained_property_names is not None):
        for i, property_name in enumerate(untrained_property_names):
            for name, values in _linearize(
                    property_name, untrained_properties[:, i]).items():
                result[name] = {"target": target, "values": values}

    data['properties'] = result
    data['structures'] = [_frame_to_json(frame) for frame in frames]

    if cutoff is not None:
        data['environments'] = _generate_environments(frames, cutoff)

    return data
