import sys
import gym
sys.path.append('/Users/syl/Code/ElegantRL')
from elegantrl.train.run import train_agent, train_agent_multiprocessing
from elegantrl.train.config import Config, get_gym_env_args, build_env
from elegantrl.agents import AgentPPO

'''train'''


def train_ppo_a2c_for_order_execution_vec_env():
    from order_execution_env import OrderExecutionVecEnv
    num_envs = 10

    gamma = 0.999
    n_stack = 8

    agent_class = AgentPPO
    env_class = OrderExecutionVecEnv
    env_args = {'env_name': 'OrderExecutionVecEnv-v2',
                'num_envs': num_envs,
                'max_step': 5000,
                'state_dim': 48 * n_stack,
                'action_dim': 2,
                'if_discrete': False,

                'share_name': '000768.SZ',
                'beg_date': '20220601',
                'end_date': '20220909',
                'if_random': False}
    if not env_args:
        get_gym_env_args(env=OrderExecutionVecEnv(), if_print=True)  # return env_args

    args = Config(agent_class, env_class, env_args)  # see `config.py Arguments()` for hyperparameter explanation
    args.break_step = int(1e6)  # break training if 'total_step > break_step'
    args.net_dims = (256, 128, 64)  # the middle layer dimension of MultiLayer Perceptron
    args.gamma = gamma  # discount factor of future rewards
    args.horizon_len = 2 ** 9

    args.batch_size = args.horizon_len * num_envs // 32
    args.repeat_times = 4  # repeatedly update network using ReplayBuffer to keep critic's loss small
    args.learning_rate = 1e-4
    args.state_value_tau = 0.01

    eval_num_envs = 16
    args.save_gap = int(8)
    args.if_keep_save = True
    args.if_over_write = False
    args.eval_per_step = int(4e3)
    args.eval_times = eval_num_envs
    from order_execution_env import OrderExecutionVecEnvForEval
    args.eval_env_class = OrderExecutionVecEnvForEval
    args.eval_env_args = env_args.copy()
    args.eval_env_args['num_envs'] = eval_num_envs
    args.eval_env_args['max_step'] = 4000 * 22
    args.eval_env_args['beg_date'] = '20220910'
    args.eval_env_args['end_date'] = '20221010'

    args.gpu_id = GPU_ID
    args.eval_gpu_id = GPU_ID
    args.random_seed = GPU_ID
    args.num_workers = 2

    if_check = False
    if if_check:
        train_agent(args)
    else:
        train_agent_multiprocessing(args)
    """
    0% < 100% < 120%
    ################################################################################
    ID     Step    Time |    avgR   stdR   avgS  stdS |    expR   objC   etc.
    6  1.64e+04     559 |  100.75    0.3  88000     0 |   -2.81   0.44  -0.03  -0.03
    6  1.64e+04     559 |  100.75
    6  1.64e+05    1025 |  101.19    0.5  88000     0 |   -2.58   0.37  -0.10  -0.13
    6  1.64e+05    1025 |  101.19
    6  1.72e+05    1471 |  101.21    0.5  88000     0 |   -2.58   0.50   0.01  -0.12
    6  1.72e+05    1471 |  101.21
    6  1.80e+05    1916 |  101.20    0.5  88000     0 |   -2.60   0.27  -0.14  -0.11
    6  1.88e+05    2362 |  101.21    0.5  88000     0 |   -2.63   0.63  -0.19  -0.10
    6  1.88e+05    2362 |  101.21
    6  1.97e+05    2807 |  101.22    0.5  88000     0 |   -2.64   0.58  -0.18  -0.10
    6  1.97e+05    2807 |  101.22
    6  2.05e+05    3253 |  101.24    0.5  88000     0 |   -2.64   0.25   0.04  -0.09
    6  2.05e+05    3253 |  101.24
    6  2.13e+05    3698 |  101.24    0.5  88000     0 |   -2.67   0.46  -0.05  -0.08
    6  2.13e+05    3698 |  101.24
    6  2.21e+05    4143 |  101.25    0.5  88000     0 |   -2.68   0.33  -0.01  -0.07
    6  2.21e+05    4143 |  101.25
    6  2.29e+05    4589 |  101.26    0.5  88000     0 |   -2.69   0.50   0.08  -0.06
    6  2.29e+05    4589 |  101.26
    6  2.38e+05    5034 |  101.27    0.5  88000     0 |   -2.71   0.26   0.05  -0.05
    6  2.38e+05    5034 |  101.27
    """


if __name__ == '__main__':
    GPU_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # >=0 means GPU ID, -1 means CPU

    train_ppo_a2c_for_order_execution_vec_env()
