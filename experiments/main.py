'''A main script to run attack for LLMs.'''
import time
import importlib
import numpy as np
import torch.multiprocessing as mp
from absl import app
from ml_collections import config_flags
import os

from rpo import get_goals_and_targets, get_workers

_CONFIG = config_flags.DEFINE_config_file('config')

# Function to import module at the runtime
def dynamic_import(module):
    return importlib.import_module(module)

# Function to load jailbreaks
def load_jailbreaks_from_files(folder_path):
    jailbreaks = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                jailbreaks.append(file.read().strip())
    return jailbreaks

def main(_):

    mp.set_start_method('spawn')

    params = _CONFIG.value

    attack_lib = dynamic_import(f'rpo')

    print(params)

    train_goals, train_targets, test_goals, test_targets = get_goals_and_targets(params)

    process_fn = lambda s: s.replace('Sure, h', 'H')
    process_fn2 = lambda s: s.replace("Sure, here is", "Sure, here's")
    train_targets_adv = [process_fn(t) if np.random.random() < 0.5 else process_fn2(t) for t in train_targets]

    train_targets = ["I cannot" for i in train_targets]
    test_targets = ["I cannot" for i in test_targets]
    
    workers, test_workers = get_workers(params)

    managers = {
        "AP": attack_lib.AttackPrompt,
        "PM": attack_lib.PromptManager,
        "MPA": attack_lib.MultiPromptAttack,
    }

    jailbreaks_dir = '/home/andyz3/rpo/experiments/jailbreaks'
    jailbreaks = load_jailbreaks_from_files(jailbreaks_dir)

    timestamp = time.strftime("%Y%m%d-%H:%M:%S")
    attack = attack_lib.ProgressiveMultiPromptAttack(
        jailbreaks,
        train_goals,
        train_targets,
        train_targets_adv,
        workers,
        progressive_models=params.progressive_models,
        progressive_goals=params.progressive_goals,
        control_init=params.control_init,
        safe_init=params.safe_init,
        logfile=f"{params.result_prefix}_{timestamp}.json",
        managers=managers,
        test_goals=test_goals,
        test_targets=test_targets,
        test_workers=test_workers,
        mpa_deterministic=params.gbda_deterministic,
        mpa_lr=params.lr,
        mpa_batch_size=params.batch_size,
        mpa_n_steps=params.n_steps,
    )
    attack.run(
        n_steps=params.n_steps,
        batch_size=params.batch_size, 
        topk=params.topk,
        temp=params.temp,
        target_weight=params.target_weight,
        control_weight=params.control_weight,
        test_steps=getattr(params, 'test_steps', 1),
        anneal=params.anneal,
        incr_control=params.incr_control,
        stop_on_success=params.stop_on_success,
        verbose=params.verbose,
        filter_cand=params.filter_cand,
        allow_non_ascii=params.allow_non_ascii,
        selection_interval=params.selection_interval
    )

    for worker in workers + test_workers:
        worker.stop()

if __name__ == '__main__':
    app.run(main)