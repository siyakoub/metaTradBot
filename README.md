# MetaTradBot

Ce projet a pour but de développer une intelligence artificielle capable d’investir à la place de l’utilisateur et de prendre les meilleures décisions d’investissement possible.

L’entraînement de ce modèle est basé sur des données récupérées dynamiquement via l’API de MetaTrader 5.

---

## Principe du modèle

Un **DQN Agent** (Deep Q‑Network Agent) est un agent d’apprentissage par renforcement qui combine :

1. **Q‑Learning**  
   Un algorithme off‑policy cherchant à estimer la fonction de valeur d’action :  
   $$
     Q(s,a) = \mathbb{E}\bigl[r_{t+1} + \gamma\,\max_{a'} Q(s_{t+1},a') \mid s_t = s,\; a_t = a\bigr]
   $$

2. **Approximation par réseau de neurones**  
   Au lieu d’une table de Q‑valeurs, on utilise un réseau de neurones \(Q_\theta(s,a)\) pour prédire \(Q(s,a)\), ce qui permet de gérer des espaces d’état continus ou de grande dimension.

3. **Experience Replay**  
   On stocke les transitions \((s_t, a_t, r_{t+1}, s_{t+1}, d_t)\) dans un buffer \(\mathcal{D}\), puis on entraîne le réseau à partir de mini‑lots aléatoires prélevés dans ce buffer, ce qui rompt la corrélation temporelle et stabilise l’apprentissage.

4. **Target Network**  
   On maintient deux réseaux :
   - **Réseau principal** \(Q_\theta\)  
   - **Réseau cible** \(Q_{\theta^-}\), copié périodiquement depuis \(Q_\theta\)  
   
   Les cibles de régression sont calculées avec \(Q_{\theta^-}\) pour plus de stabilité.

---

## Architecture du `DQNAgent`

| Composant                 | Détail                                                                                                                                         |
|---------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| **Network (Q‑valeurs)**   | - Couche d’entrée → dimension = nombre de features d’état  
                              - 2 couches cachées (64 → 128 neurones, activation ReLU)  
                              - Couche de sortie → taille = nombre d’actions possibles                                                                                   |
| **Replay Buffer**         | Mémoire circulaire (deque) stockant jusqu’à 100 000 transitions                                                                                |
| **Politique ε‑greedy**    | - ε décroît linéairement de 1.0 à 0.01 sur N étapes  
                              - Avec probabilité ε : action aléatoire, sinon argmax\_a Q(s,a)                                                                               |
| **Mise à jour du réseau** | Pour chaque mini‑batch \(\{(s_i,a_i,r_i,s'_i,d_i)\}_{i=1}^B\) tiré de \(\mathcal{D}\) :  
                              1. Cible :  
                                 $$
                                   y_i = r_i + \gamma\,\max_{a'} Q_{\theta^-}(s'_i, a')
                                 $$
                              2. Loss (MSE) :  
                                 $$
                                   L(\theta) = \frac{1}{B}\sum_{i=1}^B\bigl(y_i - Q_\theta(s_i,a_i)\bigr)^2
                                 $$
                              3. Descente de gradient (Adam, lr≈1e‑3)  
                              4. Décroissance de ε                                                                                                                         |
| **Target Update**         | Copie des poids : \(\theta^- \leftarrow \theta\) toutes les C étapes (ex. 1 000 pas)                                                             |

---

## Flux d’exécution

1. **`act(state)`**  
   - Prépare l’état  
   - Sélectionne l’action selon ε‑greedy  
2. **`remember(s,a,r,s',done)`**  
   - Ajoute la transition au buffer  
3. **`replay()`**  
   - Si \(\lvert\mathcal{D}\rvert\) ≥ batch_size  
   - Échantillonne un mini‑batch  
   - Met à jour \(Q_\theta\), décroit ε  
4. **`update_target_network()`**  
   - Copie périodiquement \(Q_\theta\to Q_{\theta^-}\)

---

## Installation

1. Clonez le dépôt :  
   ```bash
   git clone https://github.com/siyakoub/metaTradBot.git
   cd metaTradBot

Le projet est utilisable uniquement sur Windows car il nécessite d'avoir MetaTrader5 installé sur la machine.

2. Créez et activez un environnement virtuel (recommandé) :
   ```bash
   python -m venv venv
   venv\Scripts\activate       # Windows

3. Mettez à jour pip puis installez les dépendances :
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt

### Conseil
Pour mieux entrainer le modèle, il est conseillé de posséder une bonne carte graphique au lieu d'utiliser le CPU.
Pour ce faire utiliser la version 3.10 ou 3.11 de Python pour installez pytorch avec CUDA, ce qui réduira la durée de l'apprentissage et augmentera la performance (puissance de calcul) 
   
## Structure du projet
       metaTradBot/
        ├─ main.py
        ├─ model/
        │  └─ IA/
        │     ├─ Analyzer/MarketAnalyzer.py
        │     ├─ Dataprocessing/dataProcess.py
        │     ├─ DQN/dqn_agent.py
        │     ├─
        │     └─ trainModel/trainModel.py
        ├─ SQL/
        ├─ tests/
        ├─ requirements.txt
        └─ .github/
           └─ workflows/

