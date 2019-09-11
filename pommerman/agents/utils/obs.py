import numpy as np
import torch


class Observations:

    @staticmethod
    def process(obs, device):
        position = np.array(obs['position'], dtype=np.float32)
        ammo = np.float32(obs['ammo'])
        blast_strength = np.float32(obs['blast_strength'])
        can_kick = np.float32(obs['can_kick'])
        ints = np.array([ammo, blast_strength, can_kick], dtype=np.float32)
        pad = np.zeros((6,), dtype=np.float32)
        character = np.concatenate((position, ints, pad))
        enemies = np.array([e.value for e in obs["enemies"]], dtype=np.float32)

        board = np.array(obs['board'], dtype=np.float32)

        for i in range(len(enemies)):
            board[board == enemies[i]] = -1

        board = np.c_[board, character]

        bomb_blast = np.array(obs['bomb_blast_strength'], dtype=np.float32)
        bomb_blast = np.c_[bomb_blast, np.zeros((11,), dtype=np.float32)]
        bomb_life = np.array(obs['bomb_life'], dtype=np.float32)
        bomb_life = np.c_[bomb_life, np.zeros((11,), dtype=np.float32)]

        field = np.stack((board, bomb_blast, bomb_life), axis=-1)
        field_tensor = torch.from_numpy(field).float().to(device)
        # field_tensor = torch.from_numpy(field).long().to(device)
        state = field_tensor.reshape(1, -1)

        return state
