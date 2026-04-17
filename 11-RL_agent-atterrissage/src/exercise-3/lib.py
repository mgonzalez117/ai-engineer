import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random

# Héritage de nn.Module : réseau de neuronnes avec pytorch
class DQN(nn.Module):
    """
    Deep Q-Network : réseau de neurones pour approximer la fonction Q.
    """
    def __init__(self, state_size, action_size, hidden_size=128):
        """
        Initialise le réseau de neurones.
        
        Args:
            state_size (int): Dimension de l'espace d'état
            action_size (int): Nombre d'actions possibles
            hidden_size (int): Nombre de neurones dans les couches cachées
        """
        super(DQN, self).__init__()
        
        # Définition des couches
        self.fc1 = nn.Linear(state_size, hidden_size) # prend les 4 infos d'entrée et les projette vers la hidden layer
        self.fc2 = nn.Linear(hidden_size, hidden_size) # couche intermédiaire pour affiner la réflexion
        self.fc3 = nn.Linear(hidden_size, action_size) # couche finale de décision
    
    def forward(self, x):
        """
        Passage avant à travers le réseau.
        
        Args:
            x (torch.Tensor): État d'entrée
            
        Returns:
            torch.Tensor: Valeurs Q pour chaque action
        """
        x = F.relu(self.fc1(x)) # fonction d'activation ReLU
        x = F.relu(self.fc2(x)) # idem
        x = self.fc3(x)  # Pas d'activation sur la couche de sortie, on veut obtenir les scores bruts
        return x


class ReplayBuffer:
    """
    Buffer pour stocker les expériences et les échantillonner aléatoirement.
    """
    def __init__(self, capacity):
        """
        Initialise le buffer avec une capacité maximale.
        
        Args:
            capacity (int): Nombre maximum d'expériences à stocker
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """
        Ajoute une expérience au buffer.
        
        Args:
            state: État actuel (ex: position du chariot, angle de la barre, etc.)
            action: Action prise (ex: 0=gauche, 1=droite)
            reward: Récompense reçue (ex: +1 si le bâton tient encore)
            next_state: État suivant (ex: nouvelle position, nouvel angle)
            done: Booléen indiquant si l'épisode est terminé (ex: bâton tombé, a dépassé un certain angle)
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        Échantillonne un batch aléatoire d'expériences.
        
        Args:
            batch_size (int): Taille du batch à échantillonner
            
        Returns:
            tuple: Batch d'états, actions, récompenses, états suivants, et dones
        """

        # Pioche aléatoirement dans les expériences précédentes
        # on ne fait pas dans l'ordre pour éviter à l'agent d'oublier ce qu'il a appris dans le passé ("catastrophic forgetting")
        # on casse les corrélations temporelles ==> apprentissage + stable
        batch = random.sample(self.buffer, batch_size)
        
        # Crée des listes vides pour chaque composant
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        # Parcourt chaque expérience et sépare les composants
        for experience in batch:
            state, action, reward, next_state, done = experience
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """
        Retourne la taille actuelle du buffer.
        """
        return len(self.buffer)