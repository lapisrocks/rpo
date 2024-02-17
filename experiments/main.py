'''A main script to run attack for LLMs.'''
import time
import importlib
import numpy as np
import torch.multiprocessing as mp
from absl import app
from ml_collections import config_flags
import os

from llm_attacks import get_goals_and_targets, get_workers

_CONFIG = config_flags.DEFINE_config_file('config')

# Function to import module at the runtime
def dynamic_import(module):
    return importlib.import_module(module)

def create_jailbreak_map(jailbreaks_dir):
    jailbreak_map = {}
    for filename in os.listdir(jailbreaks_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(jailbreaks_dir, filename), 'r', encoding='utf-8') as file:
                jailbreak = filename[:-4] 
                adv_target = file.read().strip()
                jailbreak_map[jailbreak] = adv_target
    return jailbreak_map

def main(_):

    mp.set_start_method('spawn')

    params = _CONFIG.value

    attack_lib = dynamic_import(f'llm_attacks.{params.attack}')

    manual_prefix = """""""

    print(params)

    train_goals, train_targets, adv_goals, adv_targets = get_goals_and_targets(params)

    train_targets = ["I cannot" for i in train_targets]
    test_targets = ["I cannot" for i in test_targets]

    process_fn = lambda s: s.replace('Sure, h', 'H')
    process_fn2 = lambda s: s.replace("Sure, here is", "Sure, here's")
    train_targets_adv = [process_fn(t) if np.random.random() < 0.5 else process_fn2(t) for t in adv_targets]
    
    workers, test_workers = get_workers(params)

    managers = {
        "AP": attack_lib.AttackPrompt,
        "PM": attack_lib.PromptManager,
        "MPA": attack_lib.MultiPromptAttack,
    }

    jailbreaks_dir = 'jailbreaks'
    jailbreak_map = create_jailbreak_map(jailbreaks_dir)

    timestamp = time.strftime("%Y%m%d-%H:%M:%S")
    attack = attack_lib.ProgressiveMultiPromptAttack(
        jailbreak_map,
        train_goals,
        train_targets,
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