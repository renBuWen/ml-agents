# # Unity ML-Agents Toolkit

import logging

import os
import multiprocessing
import numpy as np
import yaml
from docopt import docopt

from .trainer_controller import TrainerController
from . import MetaCurriculumError, MetaCurriculum
from .exception import TrainerError
from mlagents.envs import UnityEnvironment
from mlagents.envs.exception import UnityEnvironmentException


def run_training(sub_id, run_seed, run_options):
    """
    Launches training session.
    :param sub_id: Unique id for training session.
    :param run_seed: Random seed used for training.
    :param run_options: Command line arguments for training.
    """
    # Docker Parameters
    docker_target_name = (run_options['--docker-target-name']
        if run_options['--docker-target-name'] != 'None' else None)

    # General parameters
    env_path = (run_options['--env']
        if run_options['--env'] != 'None' else None)
    run_id = run_options['--run-id']
    load_model = run_options['--load']
    train_model = run_options['--train']
    save_freq = int(run_options['--save-freq'])
    keep_checkpoints = int(run_options['--keep-checkpoints'])
    worker_id = int(run_options['--worker-id'])
    curriculum_folder = (run_options['--curriculum']
        if run_options['--curriculum'] != 'None' else None)
    lesson = int(run_options['--lesson'])
    fast_simulation = not bool(run_options['--slow'])
    no_graphics = run_options['--no-graphics']
    trainer_config_path = run_options['<trainer-config-path>']

    # Recognize and use docker volume if one is passed as an argument
    if not docker_target_name:
        model_path = './models/{run_id}'.format(run_id=run_id)
        summaries_dir = './summaries'
    else:
        trainer_config_path = \
            '/{docker_target_name}/{trainer_config_path}'.format(
                docker_target_name=docker_target_name,
                trainer_config_path = trainer_config_path)
        if curriculum_folder is not None:
            curriculum_folder = \
                '/{docker_target_name}/{curriculum_folder}'.format(
                    docker_target_name=docker_target_name,
                    curriculum_folder=curriculum_folder)
        model_path = '/{docker_target_name}/models/{run_id}'.format(
            docker_target_name=docker_target_name,
            run_id=run_id)
        summaries_dir = '/{docker_target_name}/summaries'.format(
            docker_target_name=docker_target_name)

    env = init_environment(env_path, docker_target_name, no_graphics, worker_id + sub_id, fast_simulation, run_seed)

    if curriculum_folder is None:
        meta_curriculum = None
    else:
        meta_curriculum = MetaCurriculum(curriculum_folder, env._resetParameters)

    if meta_curriculum:
        for brain_name in meta_curriculum.brains_to_curriculums.keys():
            if brain_name not in env.external_brain_names:
                raise MetaCurriculumError('One of the curriculums '
                                          'defined in ' +
                                          curriculum_folder + ' '
                                                                   'does not have a corresponding '
                                                                   'Brain. Check that the '
                                                                   'curriculum file has the same '
                                                                   'name as the Brain '
                                                                   'whose curriculum it defines.')
    if env_path is None:
        env_name = 'editor_' + env.academy_name
    else:
        # Extract out name of environment
        env_name = os.path.basename(os.path.normpath(env_path))

    external_brains = {}
    for brain_name in env.external_brain_names:
        external_brains[brain_name] = env.brains[brain_name]
    trainer_config = load_config(trainer_config_path)

    # Create controller and begin training.
    tc = TrainerController(env_name, model_path, summaries_dir, run_id + '-' + str(sub_id),
                           save_freq, meta_curriculum,
                           load_model, train_model,
                           keep_checkpoints, lesson, external_brains, run_seed)
    tc.start_learning(env, trainer_config)


def load_config(trainer_config_path):
    try:
        with open(trainer_config_path) as data_file:
            trainer_config = yaml.load(data_file)
            return trainer_config
    except IOError:
        raise UnityEnvironmentException('Parameter file could not be found '
                                        'at {}.'
                                        .format(trainer_config_path))
    except UnicodeDecodeError:
        raise UnityEnvironmentException('There was an error decoding '
                                        'Trainer Config from this path : {}'
                                        .format(trainer_config_path))


def init_environment(env_path, docker_target_name, no_graphics, worker_id, fast_simulation, seed):
    if env_path is not None:
        # Strip out executable extensions if passed
        env_path = (env_path.strip()
                    .replace('.app', '')
                    .replace('.exe', '')
                    .replace('.x86_64', '')
                    .replace('.x86', ''))
    docker_training = docker_target_name is not None
    if docker_training and env_path is not None:
        env_path = '/{docker_target_name}/{env_name}'.format(
            docker_target_name=docker_target_name, env_name=env_path)
    return UnityEnvironment(file_name=env_path,
                                worker_id=worker_id,
                                seed=seed,
                                docker_training=docker_training,
                                no_graphics=no_graphics,
                                train_mode=fast_simulation
                            )


def main():
    try:
        print('''
    
                        ▄▄▄▓▓▓▓
                   ╓▓▓▓▓▓▓█▓▓▓▓▓
              ,▄▄▄m▀▀▀'  ,▓▓▓▀▓▓▄                           ▓▓▓  ▓▓▌
            ▄▓▓▓▀'      ▄▓▓▀  ▓▓▓      ▄▄     ▄▄ ,▄▄ ▄▄▄▄   ,▄▄ ▄▓▓▌▄ ▄▄▄    ,▄▄
          ▄▓▓▓▀        ▄▓▓▀   ▐▓▓▌     ▓▓▌   ▐▓▓ ▐▓▓▓▀▀▀▓▓▌ ▓▓▓ ▀▓▓▌▀ ^▓▓▌  ╒▓▓▌
        ▄▓▓▓▓▓▄▄▄▄▄▄▄▄▓▓▓      ▓▀      ▓▓▌   ▐▓▓ ▐▓▓    ▓▓▓ ▓▓▓  ▓▓▌   ▐▓▓▄ ▓▓▌
        ▀▓▓▓▓▀▀▀▀▀▀▀▀▀▀▓▓▄     ▓▓      ▓▓▌   ▐▓▓ ▐▓▓    ▓▓▓ ▓▓▓  ▓▓▌    ▐▓▓▐▓▓
          ^█▓▓▓        ▀▓▓▄   ▐▓▓▌     ▓▓▓▓▄▓▓▓▓ ▐▓▓    ▓▓▓ ▓▓▓  ▓▓▓▄    ▓▓▓▓`
            '▀▓▓▓▄      ^▓▓▓  ▓▓▓       └▀▀▀▀ ▀▀ ^▀▀    `▀▀ `▀▀   '▀▀    ▐▓▓▌
               ▀▀▀▀▓▄▄▄   ▓▓▓▓▓▓,                                      ▓▓▓▓▀
                   `▀█▓▓▓▓▓▓▓▓▓▌
                        ¬`▀▀▀█▓

        ''')
    except:
        print('\n\n\tUnity Technologies\n')

    logger = logging.getLogger('mlagents.trainers')
    _USAGE = '''
    Usage:
      mlagents-learn <trainer-config-path> [options]
      mlagents-learn --help

    Options:
      --env=<file>               Name of the Unity executable [default: None].
      --curriculum=<directory>   Curriculum json directory for environment [default: None].
      --keep-checkpoints=<n>     How many model checkpoints to keep [default: 5].
      --lesson=<n>               Start learning from this lesson [default: 0].
      --load                     Whether to load the model or randomly initialize [default: False].
      --run-id=<path>            The directory name for model and summary statistics [default: ppo].
      --num-runs=<n>             Number of concurrent training sessions [default: 1]. 
      --save-freq=<n>            Frequency at which to save model [default: 50000].
      --seed=<n>                 Random seed used for training [default: -1].
      --slow                     Whether to run the game at training speed [default: False].
      --train                    Whether to train model, or only run inference [default: False].
      --worker-id=<n>            Number to add to communication port (5005) [default: 0].
      --docker-target-name=<dt>  Docker volume to store training-specific files [default: None].
      --no-graphics              Whether to run the environment in no-graphics mode [default: False].
    '''

    options = docopt(_USAGE)
    logger.info(options)
    num_runs = int(options['--num-runs'])
    seed = int(options['--seed'])

    if options['--env'] == 'None' and num_runs > 1:
        raise TrainerError('It is not possible to launch more than one concurrent training session '
                           'when training from the editor.')

    jobs = []
    run_seed = seed
    for i in range(num_runs):
        if seed == -1:
            run_seed = np.random.randint(0, 10000)
        p = multiprocessing.Process(target=run_training, args=(i, run_seed, options))
        jobs.append(p)
        p.start()
