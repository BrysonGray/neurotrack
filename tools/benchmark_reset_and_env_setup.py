import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.optim.adamw import AdamW
from torch.optim.adam import Adam

from neurotrack.data import NeuronPatchDataset
from neurotrack.environments import NeuronTrackingEnvironment
from neurotrack.training import PrioritizedReplayBuffer
from neurotrack.models import ConvNet
from neurotrack.training import sac

cfg = Path('/home/brysongray/neurotrack/configs/training/train_sac_convex_composite.json')
params = json.loads(cfg.read_text())


def run_once(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32

    img_dir = params['img_dir']
    swc_dir = params['swc_dir']
    target_step_len = params.get('target_step_len', 1.0)
    step_width = params.get('step_width', 1.0)
    batch_size = params.get('batchsize', params.get('batch_size', 256))
    gamma = params.get('gamma', 0.99)
    tau = params.get('tau', 0.005)
    lr = params.get('lr', params.get('learning_rate', 0.001))
    update_alpha = bool(params.get('update_alpha', True))
    repeat_starts = bool(params.get('repeat_starts', True))
    branching = params.get('branching', 0)
    rng_seed = params.get('rng_seed', 1)
    start_complexity = params.get('start_complexity', 0.0)
    start_idx = params.get('start_idx', 0)
    seeds_path = params.get('seeds_path')
    root_sampling_probability = params.get('root_sampling_probability')
    soma_sample_radius = float(params.get('soma_sample_radius', 0.0))
    random_offset = float(params.get('random_offset', 0.0))
    init_temperature = params.get('init_temperature', 0.005)
    target_entropy = params.get('target_entropy', 0.0)

    patch_radius = 17
    in_channels = 2
    input_size = 2 * patch_radius + 1

    rng = np.random.default_rng(rng_seed)
    dataset = NeuronPatchDataset(
        img_dir=img_dir,
        swc_dir=swc_dir,
        crop_size=128,
        patches_per_image=10,
        alpha=start_complexity,
        step_width=step_width,
        rng=rng,
        seeds_path=seeds_path,
        root_sampling_probability=root_sampling_probability,
        soma_sample_radius=soma_sample_radius,
        random_offset=random_offset,
    )

    env = NeuronTrackingEnvironment(
        dataset=dataset,
        radius=patch_radius,
        target_step_len=target_step_len,
        step_width=step_width,
        max_len=1000,
        repeat_starts=repeat_starts,
        branching=branching,
        start_idx=start_idx,
    )

    actor = ConvNet(chin=in_channels, chout=4).to(device=DEVICE, dtype=dtype)
    Q1 = ConvNet(chin=in_channels + 3, chout=1).to(device=DEVICE, dtype=dtype)
    Q2 = ConvNet(chin=in_channels + 3, chout=1).to(device=DEVICE, dtype=dtype)
    Q1_target = ConvNet(chin=in_channels + 3, chout=1).to(device=DEVICE, dtype=dtype)
    Q2_target = ConvNet(chin=in_channels + 3, chout=1).to(device=DEVICE, dtype=dtype)
    Q1_target.load_state_dict(Q1.state_dict())
    Q2_target.load_state_dict(Q2.state_dict())

    log_alpha = torch.log(torch.tensor(init_temperature, device=DEVICE))
    log_alpha.requires_grad = True

    Q1_optimizer = AdamW(Q1.parameters(), lr=lr)
    Q2_optimizer = AdamW(Q2.parameters(), lr=lr)
    actor_optimizer = AdamW(actor.parameters(), lr=lr)
    log_alpha_optimizer = Adam([log_alpha], lr=lr)

    memory = PrioritizedReplayBuffer(
        10000,
        obs_shape=(in_channels, input_size, input_size, input_size),
        action_shape=(3,),
        alpha=0.8,
    )

    update_after = 256
    update_every = 1
    updates_per_step = 1
    steps_target = 280

    policy_device = next(actor.parameters()).device

    stats = {
        'step_total_s': 0.0,
        'reset_s': 0.0,
        'action_select_s': 0.0,
        'env_step_s': 0.0,
        'memory_push_s': 0.0,
        'sample_s': 0.0,
        'update_actor_s': 0.0,
        'update_q_s': 0.0,
        'obs_recover_s': 0.0,
    }
    counts = {
        'steps': 0,
        'resets': 0,
        'samples': 0,
        'actor_updates': 0,
        'q_updates': 0,
        'obs_recovers': 0,
    }

    t0 = time.perf_counter()
    obs = env.reset(return_state=True)
    stats['reset_s'] += time.perf_counter() - t0
    counts['resets'] += 1

    steps_done = 0
    while counts['steps'] < steps_target:
        step_t0 = time.perf_counter()

        learning_started = steps_done >= update_after
        env.branching = int(learning_started) * branching

        t0 = time.perf_counter()
        if not learning_started:
            action_for_env = torch.randn(3, dtype=torch.float32, device=obs.device) * 3
        else:
            with torch.no_grad():
                actor_out = actor(obs.to(device=policy_device, dtype=dtype))
                direction_dist = sac.sample_from_output(actor_out)
                sampled_action = direction_dist.rsample()[0]
            action_for_env = sampled_action.to(device=obs.device)
        stats['action_select_s'] += time.perf_counter() - t0

        steps_done += 1

        t0 = time.perf_counter()
        next_obs, reward, terminated, truncated, info = env.step(action_for_env)
        stats['env_step_s'] += time.perf_counter() - t0

        current_target_vectors = info['current_target_vectors']
        next_target_vectors = info['next_target_vectors']
        if next_target_vectors is None:
            next_target_vectors = current_target_vectors

        t0 = time.perf_counter()
        memory.push(obs, action_for_env, next_obs, reward, current_target_vectors, next_target_vectors, terminated)
        stats['memory_push_s'] += time.perf_counter() - t0

        if learning_started and steps_done % update_every == 0:
            for _ in range(updates_per_step):
                t0 = time.perf_counter()
                (
                    batch_obs,
                    batch_actions,
                    batch_next_obs,
                    batch_rewards,
                    batch_target_vecs,
                    batch_target_masks,
                    batch_next_target_vecs,
                    batch_next_target_masks,
                    batch_dones,
                    weights,
                    tree_idxs,
                ) = memory.sample(batch_size, transform=True)
                stats['sample_s'] += time.perf_counter() - t0
                counts['samples'] += 1

                if gamma > 0:
                    t0 = time.perf_counter()
                    td_error = sac.update_Q(
                        actor,
                        Q1,
                        Q1_target,
                        Q2,
                        Q2_target,
                        batch_obs,
                        batch_actions,
                        batch_next_target_vecs,
                        batch_next_target_masks,
                        batch_next_obs,
                        batch_dones,
                        Q1_optimizer,
                        Q2_optimizer,
                        gamma,
                        log_alpha,
                        weights=weights,
                    )
                    sac.target_update(Q1, Q2, Q1_target, Q2_target, tau)
                    memory.update_priorities(tree_idxs, td_error.cpu().numpy())
                    stats['update_q_s'] += time.perf_counter() - t0
                    counts['q_updates'] += 1

                t0 = time.perf_counter()
                sac.update_actor(
                    batch_obs,
                    batch_target_vecs,
                    batch_target_masks,
                    batch_dones,
                    gamma,
                    actor,
                    actor_optimizer,
                    Q1,
                    Q2,
                    log_alpha,
                    log_alpha_optimizer,
                    target_entropy,
                    update_alpha=update_alpha,
                )
                stats['update_actor_s'] += time.perf_counter() - t0
                counts['actor_updates'] += 1

        if info.get('terminate_episode', False):
            t0 = time.perf_counter()
            obs = env.reset(return_state=True)
            stats['reset_s'] += time.perf_counter() - t0
            counts['resets'] += 1
        elif not terminated:
            obs = next_obs
        else:
            t0 = time.perf_counter()
            obs = env.get_state()
            stats['obs_recover_s'] += time.perf_counter() - t0
            counts['obs_recovers'] += 1

        stats['step_total_s'] += time.perf_counter() - step_t0
        counts['steps'] += 1

    return {
        'device': str(DEVICE),
        'steps': counts['steps'],
        'resets': counts['resets'],
        'mean_step_total_ms': (stats['step_total_s'] / counts['steps']) * 1000.0,
        'mean_reset_ms': (stats['reset_s'] / counts['resets']) * 1000.0,
        'mean_env_step_ms': (stats['env_step_s'] / counts['steps']) * 1000.0,
        'mean_sample_ms': (stats['sample_s'] / max(1, counts['samples'])) * 1000.0,
        'mean_update_actor_ms': (stats['update_actor_s'] / max(1, counts['actor_updates'])) * 1000.0,
    }


if __name__ == "__main__":
    results = []
    for rep_seed in (0, 1):
        r = run_once(rep_seed)
        results.append(r)
        print(f"rep_seed={rep_seed} {r}")

    mean_step = sum(r['mean_step_total_ms'] for r in results) / len(results)
    mean_reset = sum(r['mean_reset_ms'] for r in results) / len(results)
    print({'mean_of_reps_step_total_ms': mean_step, 'mean_of_reps_reset_ms': mean_reset})
