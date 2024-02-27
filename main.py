import numpy as np
import utils as ut
import matplotlib.pyplot as plt
import time

# Taille de la zone :
x_min = 0
x_max = 70
y_min = 0
y_max = 70

# Obstacles :
obs_1 = np.array([43, 25, 6, 20])
obs_2 = np.array([42.5, 24.5, 6.5, 20])

# Nombre de noeud dans l'arbre :
numNodes = 9500

# Paramètres du robot :
pivot1 = np.array([20, 0])
pivot6 = np.array([50, 0])
objet_center = np.array([6, 0])
L = 20
L0 = 30
L2 = L + objet_center[0]
list_pos = [1, 2, 3, 4]

# Paramètres initiaux :
x_start = np.array([25., 30., 0.])
sols_start = ut.FindPoint(x_start, pivot1, pivot6, L, L2)
q_start = ut.q(x_start, 0, 0, sols_start)
nodes = np.array([q_start])
x_goal = np.array([50., 20., 0.])
sols_g = ut.FindPoint(x_goal, pivot1, pivot6, L, L2)
q_goal = ut.q(x_goal, 0, 0, sols_g)

# Paramètres de l'algorithme :
dist_max = 1
r = 20
esp = 0.5

## Mise en place de l'affichage :

figure, axes = plt.subplots()
axes.set_xlim([x_min, x_max])
axes.set_ylim([y_min, y_max])
axes.set_title('RRTs')

# Affichage des obstacles :
#plt.plot([obs_1[0], obs_1[0] + obs_1[2]], [obs_1[1], obs_1[1]], color='black')
#plt.plot([obs_1[0], obs_1[0] + obs_1[2]], [obs_1[1] + obs_1[3], obs_1[1] + obs_1[3]], color='black')
#plt.plot([obs_1[0], obs_1[0]], [obs_1[1], obs_1[1] + obs_1[3]], color='black')
#plt.plot([obs_1[0] + obs_1[2], obs_1[0] + obs_1[2]], [obs_1[1], obs_1[1] + obs_1[3]], color='black')
plt.plot([obs_2[0], obs_2[0] + obs_2[2]], [obs_2[1], obs_2[1]], color='black')
plt.plot([obs_2[0], obs_2[0] + obs_2[2]], [obs_2[1] + obs_2[3], obs_2[1] + obs_2[3]], color='black')
plt.plot([obs_2[0], obs_2[0]], [obs_2[1], obs_2[1] + obs_2[3]], color='black')
plt.plot([obs_2[0] + obs_2[2], obs_2[0] + obs_2[2]], [obs_2[1], obs_2[1] + obs_2[3]], color='black')

# Affichage de l'arbre :
plt.scatter(q_start.coor[0], q_start.coor[1], color='red', marker='o')
plt.scatter(q_goal.coor[0], q_goal.coor[1], color='green', marker='o')

t0 = time.time()
# Mise en place de la boucle :
for i in range(numNodes):
    # Génération d'un point aléatoire :
    x_rand = np.array(
        [np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max), np.random.uniform(-np.pi, np.pi)])

    # On vérifie si on est dans le workspace du robot :
    if ut.InRobotWorkSpace(x_rand, L, L2, pivot1, pivot6):
        # On trouve le point le plus proche dans l'arbre :
        ndist = np.zeros(len(nodes))
        for j in range(len(nodes)):
            ndist[j] = ut.dist(nodes[j].coor[:2], x_rand[:2])
        dist_min = np.min(ndist)
        q_near = nodes[np.argmin(ndist)]

        # On trouve la position associée à ce point :
        q_new = ut.NewPoint(q_near.coor, x_rand, dist_max, dist_min)

        # On vérifie que le point ne correspond pas au point d'arrivée :
        if (abs(q_new.coor[0] - q_goal.coor[0]) <= esp) and (abs(q_new.coor[1] - q_goal.coor[1]) <= esp):
            break

        # On vérifie que le point est dans le workspace du robot :
        if ut.InRobotWorkSpace(q_new.coor, L, L2, pivot1, pivot6):
            # Pour un point on a au maximum 4 solutions et on vérifie la collision pour chacune d'entre elles :
            bool, pos_final, pos_not_ok = ut.Collision(obs_2, pivot1, pivot6, L, q_new.coor, L2)
            if bool:
                q_new.lpos = pos_final
                for val in pos_not_ok:
                    for val2 in list_pos:
                        if val == val2:
                            list_pos.remove(val2)
                # Affichage de la branche :
                plt.plot([q_near.coor[0], q_new.coor[0]], [q_near.coor[1], q_new.coor[1]], color='black')
                # On cherche maintenant le chemin le plus court :
                # On trouve le point le plus proche dans l'arbre : (paramètre r)

                q_new.cost = q_near.cost + ut.dist(q_new.coor[:2], q_near.coor[:2])

                # On trouve les points dans un rayon r :
                q_nearest = np.array([])
                for j in range(len(nodes)):
                    if ut.dist(nodes[j].coor[:2], q_new.coor[:2]) <= r:
                        q_nearest = np.append(q_nearest, nodes[j])

                # Initialisation du coût :
                q_min = q_near
                C_min = q_new.cost

                # On cherche un autre chemin le plus optimal :

                for gp in range(len(q_nearest)):
                    if q_nearest[gp].cost + ut.dist(q_new.coor[:2], q_nearest[gp].coor[:2]) < C_min:
                        q_min = q_nearest[gp]
                        C_min = q_nearest[gp].cost + ut.dist(q_new.coor[:2],q_nearest[gp].coor[:2])

                        # Affichage de la branche :
                        plt.plot([q_min.coor[0], q_new.coor[0]], [q_min.coor[1], q_new.coor[1]], color='blue')

                # On met à jour les parents :
                for j in range(len(nodes)):
                    if (nodes[j].coor == q_min.coor).all():
                        q_new.parent = j

                # On ajoute le nouveau point à l'arbre :
                nodes = np.append(nodes, q_new)
t1 = time.time()
print("Temps de calcul = ", t1 - t0)
D = np.zeros(len(nodes))
for j in range(len(nodes)):
    D[j] = ut.dist(nodes[j].coor[:2], q_goal.coor[:2])

# On remonte pour trouver le chemin le plus court :
q_goal.parent = np.argmin(D)
q_end = q_goal
nodes = np.append(nodes, q_goal)
good_path = np.array([q_end])

while q_end.parent != 0:
    start = q_end.parent
    plt.plot([q_end.coor[0], nodes[start].coor[0]], [q_end.coor[1], nodes[start].coor[1]], color='red', linewidth=4)
    good_path = np.append(good_path, nodes[start])
    q_end = nodes[start]

print("Position possible = ", list_pos)

# Affichage robot uniquement :
#### POS 1 ####
figure2, axes2 = plt.subplots()
axes2.set_xlim([x_min, x_max])
axes2.set_ylim([y_min, y_max])
axes2.set_title('Position 1')

plt.plot([obs_2[0], obs_2[0] + obs_2[2]], [obs_2[1], obs_2[1]], color='black')
plt.plot([obs_2[0], obs_2[0] + obs_2[2]], [obs_2[1] + obs_2[3], obs_2[1] + obs_2[3]], color='black')
plt.plot([obs_2[0], obs_2[0]], [obs_2[1], obs_2[1] + obs_2[3]], color='black')
plt.plot([obs_2[0] + obs_2[2], obs_2[0] + obs_2[2]], [obs_2[1], obs_2[1] + obs_2[3]], color='black')

for node in good_path:
    plt.plot([node.lpos[0].p1[0], node.lpos[0].p2[0], node.lpos[0].p3[0], node.lpos[0].p4[0], node.lpos[0].p5[0], node.lpos[0].p6[0]],
              [node.lpos[0].p1[1], node.lpos[0].p2[1], node.lpos[0].p3[1], node.lpos[0].p4[1], node.lpos[0].p5[1], node.lpos[0].p6[1]], marker='o')

#### POS 2 ####
figure3, axes3 = plt.subplots()
axes3.set_xlim([x_min, x_max])
axes3.set_ylim([y_min, y_max])
axes3.set_title('Position 2')

plt.plot([obs_2[0], obs_2[0] + obs_2[2]], [obs_2[1], obs_2[1]], color='black')
plt.plot([obs_2[0], obs_2[0] + obs_2[2]], [obs_2[1] + obs_2[3], obs_2[1] + obs_2[3]], color='black')
plt.plot([obs_2[0], obs_2[0]], [obs_2[1], obs_2[1] + obs_2[3]], color='black')
plt.plot([obs_2[0] + obs_2[2], obs_2[0] + obs_2[2]], [obs_2[1], obs_2[1] + obs_2[3]], color='black')

for node in good_path:
    plt.plot([node.lpos[1].p1[0], node.lpos[1].p2[0], node.lpos[1].p3[0], node.lpos[1].p4[0], node.lpos[1].p5[0], node.lpos[1].p6[0]],
              [node.lpos[1].p1[1], node.lpos[1].p2[1], node.lpos[1].p3[1], node.lpos[1].p4[1], node.lpos[1].p5[1], node.lpos[1].p6[1]], marker='o')

#### POS 3 ####
figure4, axes4 = plt.subplots()
axes4.set_xlim([x_min, x_max])
axes4.set_ylim([y_min, y_max])
axes4.set_title('Position 3')

plt.plot([obs_2[0], obs_2[0] + obs_2[2]], [obs_2[1], obs_2[1]], color='black')
plt.plot([obs_2[0], obs_2[0] + obs_2[2]], [obs_2[1] + obs_2[3], obs_2[1] + obs_2[3]], color='black')
plt.plot([obs_2[0], obs_2[0]], [obs_2[1], obs_2[1] + obs_2[3]], color='black')
plt.plot([obs_2[0] + obs_2[2], obs_2[0] + obs_2[2]], [obs_2[1], obs_2[1] + obs_2[3]], color='black')

for node in good_path:
    plt.plot([node.lpos[2].p1[0], node.lpos[2].p2[0], node.lpos[2].p3[0], node.lpos[2].p4[0], node.lpos[2].p5[0], node.lpos[2].p6[0]],
              [node.lpos[2].p1[1], node.lpos[2].p2[1], node.lpos[2].p3[1], node.lpos[2].p4[1], node.lpos[2].p5[1], node.lpos[2].p6[1]], marker='o')

#### POS 4 ####
figure5, axes5 = plt.subplots()
axes5.set_xlim([x_min, x_max])
axes5.set_ylim([y_min, y_max])
axes5.set_title('Position 4')

plt.plot([obs_2[0], obs_2[0] + obs_2[2]], [obs_2[1], obs_2[1]], color='black')
plt.plot([obs_2[0], obs_2[0] + obs_2[2]], [obs_2[1] + obs_2[3], obs_2[1] + obs_2[3]], color='black')
plt.plot([obs_2[0], obs_2[0]], [obs_2[1], obs_2[1] + obs_2[3]], color='black')
plt.plot([obs_2[0] + obs_2[2], obs_2[0] + obs_2[2]], [obs_2[1], obs_2[1] + obs_2[3]], color='black')

for node in good_path:
    plt.plot([node.lpos[3].p1[0], node.lpos[3].p2[0], node.lpos[3].p3[0], node.lpos[3].p4[0], node.lpos[3].p5[0], node.lpos[3].p6[0]],
              [node.lpos[3].p1[1], node.lpos[3].p2[1], node.lpos[3].p3[1], node.lpos[3].p4[1], node.lpos[3].p5[1], node.lpos[3].p6[1]], marker='o')
plt.show()
