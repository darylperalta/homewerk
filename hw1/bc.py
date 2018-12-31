#!/usr/bin/env python

"""
Code for behavioral cloning
"""

import os
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
from keras.models import Sequential
from keras.layers import Dense, Dropout

def bc_model(env):
    # input_len, output_len = env_dims(env)
    # return (env.observation_space.shape[0], env.action_space.shape[0])
    input_len = env.observation_space.shape[0]
    output_len = env.action_space.shape[0]
    print('input', input_len)
    print('output', output_len)
    model = Sequential()
    model.add(Dense(units=64, input_dim=input_len, activation='relu'))

    model.add(Dense(units=output_len))

    model.compile(loss='mse', optimizer='adam')
    return model

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    parser.add_argument('--train',action='store_true', default=False,
                        help='Flag if Train or Test')
    parser.add_argument('--weights', default='bc.hdf5',
                        help='path to model weights')
    parser.add_argument('--dagger',action='store_true', default=False,
                        help='Flag if DAgger will be used')
    parser.add_argument('--dagger_iter', default=5,
                        help='Dagger Iterations')
    args = parser.parse_args()

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    with tf.Session():
        tf_util.initialize()


        #
        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []
        # observations = []
        # actions = []
        # for i in range(args.num_rollouts):
        #     print('iter', i)
        #     obs = env.reset()
        #     done = False
        #     totalr = 0.
        #     steps = 0
        #     while not done:
        #         action = policy_fn(obs[None,:])
        #         observations.append(obs)
        #         actions.append(action)
        #         obs, r, done, _ = env.step(action)
        #         totalr += r
        #         steps += 1
        #         if args.render:
        #             env.render()
        #         if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
        #         if steps >= max_steps:
        #             break
        #     returns.append(totalr)

        # print('returns', returns)
        # print('mean return', np.mean(returns))
        # print('std of return', np.std(returns))
        #
        # expert_data = {'observations': np.array(observations),
        #                'actions': np.array(actions)}

        # with open(os.path.join('expert_data', args.envname + '.pkl'), 'wb') as f:
        #     pickle.dump(expert_data, f, pickle.HIGHEST_PROTOCOL)
        with open(os.path.join('expert_data', args.envname + '.pkl'), 'rb') as f:
            expert_data = pickle.load(f)
        # print(expert_data)
        print(type(expert_data))
        observations = expert_data['observations']
        print(observations.shape)
        actions = expert_data['actions'][:,0,:]
        print(actions.shape)

        our_model = bc_model(env)
        # our_model.train()
        print('args train', args.train)
        if args.train:
            print('train model')
            our_model.fit(observations, actions,
                      batch_size=32,
                      epochs=300)

            our_model.save_weights(args.weights)
            print('saving weights to ', args.weights)
        else:
            our_model.load_weights(args.weights)
            print('args dagger', args.dagger)
            if args.dagger:
                print('dagger')
                for i in range(args.dagger_iter):
                    new_observations = []
                    new_actions = []
                    print('iter', i)
                    obs = env.reset()
                    action = our_model.predict(obs[None,:])
                    obs, r, done, _ = env.step(action)
                    done = False
                    totalr = 0.
                    steps = 0
                    while not done:
                        action = policy_fn(obs[None,:])
                        new_observations.append(obs)
                        new_actions.append(action)
                        action_old = our_model.predict(obs[None,:])
                        obs, r, done, _ = env.step(action_old)
                        totalr += r
                        steps += 1
                        if args.render:
                            env.render()
                        if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                        if steps >= max_steps:
                            break
                    new_observations_np = np.array(new_observations)
                    new_actions_np = np.array(new_actions)

                our_model.fit(new_observations_np, new_actions_np[:,0,:],
                          batch_size=128,
                          epochs=300)
                    # returns.append(totalr)
            print('test')
            for i in range(args.num_rollouts):
                print('iter', i)
                obs = env.reset()
                done = False
                totalr = 0.
                steps = 0
                while not done:
                    action = our_model.predict(obs[None,:])
                    # action = policy_fn(obs[None,:])
                    # observations.append(obs)
                    # actions.append(action)
                    obs, r, done, _ = env.step(action)
                    totalr += r
                    steps += 1
                    if args.render:
                        env.render()
                    if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                    if steps >= max_steps:
                        break
                returns.append(totalr)

            print('returns', returns)
            print('mean return', np.mean(returns))
            print('std of return', np.std(returns))



if __name__ == '__main__':
    main()
