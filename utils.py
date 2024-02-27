import numpy as np
from shapely import Polygon


class pos:
    def __init__(self):
        self.p1 = np.array([0., 0.])
        self.p2 = np.array([0., 0.])
        self.p3 = np.array([0., 0.])
        self.p4 = np.array([0., 0.])
        self.p5 = np.array([0., 0.])
        self.p6 = np.array([0., 0.])


class q:
    def __init__(self, coor, parent, cost, lpos):
        self.coor = coor
        self.parent = parent
        self.cost = cost
        self.lpos = lpos


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def dist(q1, q2):
    return np.sqrt((q1[0] - q2[0]) ** 2 + (q1[1] - q2[1]) ** 2)


def check(A, B, C1, C2, C3, C4, D1, D2, D3, D4):
    ints1 = ccw(A, C1, D1) != ccw(B, C1, D1) and ccw(A, B, C1) != ccw(A, B, D1)
    ints2 = ccw(A, C2, D2) != ccw(B, C2, D2) and ccw(A, B, C2) != ccw(A, B, D2)
    ints3 = ccw(A, C3, D3) != ccw(B, C3, D3) and ccw(A, B, C3) != ccw(A, B, D3)
    ints4 = ccw(A, C4, D4) != ccw(B, C4, D4) and ccw(A, B, C4) != ccw(A, B, D4)

    if ints1 == 0 and ints2 == 0 and ints3 == 0 and ints4 == 0:
        return True
    else:
        return False


def FindPosition(q, L, L0):
    alpha = q[0]
    beta = q[1]
    tau = q[2]
    omega = q[3]

    Cx = -L0 / 2 + L * (np.cos(alpha) + np.cos(beta))
    Cy = L * (np.sin(alpha) + np.sin(beta))
    Dx = L0 / 2 + L * (np.cos(tau) + np.cos(omega))
    Dy = L * (np.sin(tau) + np.sin(omega))
    xx = (Cx + Dx) / 2
    xy = (Cy + Dy) / 2
    phy = np.pi / 2 - np.arctan2(L * (np.cos(alpha) + np.cos(beta) + np.cos(tau) + np.cos(omega)),
                                 L0 + (L * (np.sin(alpha) + np.sin(beta) + np.sin(tau) + np.sin(omega))))

    return np.array([xx, xy, phy])


def NewPoint(xn, xr, dist_max, val):
    x_new = q(np.array([0., 0., 0.]), 0, 0, 0)
    if val >= dist_max:
        x_new.coor[0] = xn[0] + (xr[0] - xn[0]) * dist_max / val
        x_new.coor[1] = xn[1] + (xr[1] - xn[1]) * dist_max / val
    else:
        x_new.coor[0] = xn[0]
        x_new.coor[1] = xn[1]
    x_new.coor[2] = xn[2]
    return x_new


def NoCollision(n2, n1, o):
    A = np.array([n1[0], n1[1]])
    B = np.array([n2[0], n2[1]])
    obs = np.array([o[0], o[1], o[0] + o[2], o[1] + o[3]])

    C1 = np.array([obs[0], obs[1]])
    D1 = np.array([obs[0], obs[3]])
    C2 = np.array([obs[0], obs[2]])
    D2 = np.array([obs[2], obs[1]])
    C3 = np.array([obs[2], obs[3]])
    D3 = np.array([obs[2], obs[1]])
    C4 = np.array([obs[2], obs[3]])
    D4 = np.array([obs[0], obs[3]])

    # Check if path from n1 to n2 intersects any of the four edges of the obstacle

    return check(A, B, C1, C2, C3, C4, D1, D2, D3, D4)


def steer(qr, qn, val, eps):
    qnew = np.array([0, 0])
    if val >= eps:
        qnew[0] = qn[0] + (qr[0] - qn[0]) * eps / dist(qr, qn)
        qnew[1] = qn[1] + (qr[1] - qn[1]) * eps / dist(qr, qn)
    else:
        qnew[0] = qr[0]
        qnew[1] = qr[1]
    return qnew


def circcirc(x1, y1, r1, x2, y2, r2):
    dx, dy = x2 - x1, y2 - y1
    d = np.sqrt(dx * dx + dy * dy)
    if d > r1 + r2:
        return None  # no solutions, the circles are separate
    if d < abs(r1 - r2):
        return None  # no solutions because one circle is contained within the other
    if d == 0 and r1 == r2:
        return None  # circles are coincident and there are an infinite number of solutions

    a = (r1 * r1 - r2 * r2 + d * d) / (2 * d)
    h = np.sqrt(r1 * r1 - a * a)
    xm = x1 + a * dx / d
    ym = y1 + a * dy / d
    xs1 = xm + h * dy / d
    xs2 = xm - h * dy / d
    ys1 = ym - h * dx / d
    ys2 = ym + h * dx / d

    return np.array([xs1, xs2, ys1, ys2])


def FindPoint(x, pivot1, pivot6, L, L2):
    phi = x[2]

    pos1 = pos()
    pos2 = pos()
    pos3 = pos()
    pos4 = pos()

    pos1.p1 = np.array([pivot1[0], pivot1[1]])
    pos2.p1 = np.array([pivot1[0], pivot1[1]])
    pos3.p1 = np.array([pivot1[0], pivot1[1]])
    pos4.p1 = np.array([pivot1[0], pivot1[1]])

    pos1.p6 = np.array([pivot6[0], pivot6[1]])
    pos2.p6 = np.array([pivot6[0], pivot6[1]])
    pos3.p6 = np.array([pivot6[0], pivot6[1]])
    pos4.p6 = np.array([pivot6[0], pivot6[1]])

    pos1.p3 = np.array([x[0] - L2 / 2 * np.cos(phi), x[1] + L2 / 2 * np.sin(phi)])
    pos2.p3 = np.array([x[0] - L2 / 2 * np.cos(phi), x[1] + L2 / 2 * np.sin(phi)])
    pos3.p3 = np.array([x[0] - L2 / 2 * np.cos(phi), x[1] + L2 / 2 * np.sin(phi)])
    pos4.p3 = np.array([x[0] - L2 / 2 * np.cos(phi), x[1] + L2 / 2 * np.sin(phi)])

    pos1.p4 = np.array([x[0] + L2 / 2 * np.cos(phi), x[1] + L2 / 2 * np.sin(phi)])
    pos2.p4 = np.array([x[0] + L2 / 2 * np.cos(phi), x[1] + L2 / 2 * np.sin(phi)])
    pos3.p4 = np.array([x[0] + L2 / 2 * np.cos(phi), x[1] + L2 / 2 * np.sin(phi)])
    pos4.p4 = np.array([x[0] + L2 / 2 * np.cos(phi), x[1] + L2 / 2 * np.sin(phi)])

    Inter1 = circcirc(pos1.p1[0], pos1.p1[1], L, pos1.p3[0], pos1.p3[1], L)
    Inter2 = circcirc(pos1.p6[0], pos1.p6[1], L, pos1.p4[0], pos1.p4[1], L)

    if Inter1 is not None and Inter2 is not None:
        pos1.p2 = np.array([Inter1[0], Inter1[2]])
        pos2.p2 = np.array([Inter1[0], Inter1[2]])
        pos3.p2 = np.array([Inter1[1], Inter1[3]])
        pos4.p2 = np.array([Inter1[1], Inter1[3]])

        pos1.p5 = np.array([Inter2[0], Inter2[2]])
        pos2.p5 = np.array([Inter2[1], Inter2[3]])
        pos3.p5 = np.array([Inter2[0], Inter2[2]])
        pos4.p5 = np.array([Inter2[1], Inter2[3]])

        return np.array([pos1, pos2, pos3, pos4])
    else:
        return None


def InRobotWorkSpace(x, L, L2, pivot1, pivot6):
    phi = x[2]
    pos1 = pos()
    pos1.p1 = np.array([pivot1[0], pivot1[1]])
    pos1.p6 = np.array([pivot6[0], pivot6[1]])
    pos1.p3 = np.array([x[0] - L2 / 2 * np.cos(phi), x[1] + L2 / 2 * np.sin(phi)])
    pos1.p4 = np.array([x[0] + L2 / 2 * np.cos(phi), x[1] + L2 / 2 * np.sin(phi)])

    Inter1 = circcirc(pos1.p1[0], pos1.p1[1], L, pos1.p3[0], pos1.p3[1], L)
    Inter2 = circcirc(pos1.p6[0], pos1.p6[1], L, pos1.p4[0], pos1.p4[1], L)

    if Inter1 is None or Inter2 is None:
        return False
    else:
        return True


def Collision(obstacle, pivot1, pivot2, L, x, L2):
    list_pos = FindPoint(x, pivot1, pivot2, L, L2)
    pos_final = []
    list_iter = []
    if list_pos is None:
        return False, pos_final

    obs = Polygon([(obstacle[0], obstacle[1]), (obstacle[0], obstacle[1] + obstacle[3]),
                   (obstacle[0] + obstacle[2], obstacle[1] + obstacle[3]), (obstacle[0] + obstacle[2], obstacle[1])])

    i = 1

    for pos in list_pos:
        poly = Polygon([(pos.p1[0], pos.p1[1]), (pos.p2[0], pos.p2[1]), (pos.p3[0], pos.p3[1]), (pos.p4[0], pos.p4[1]),
                        (pos.p5[0], pos.p5[1]), (pos.p6[0], pos.p6[1])])
        pos_final.append(pos)
        if poly.intersects(obs):
            list_iter.append(i)
        i += 1

    if len(list_iter) == len(list_pos):
        return False, pos_final, list_iter
    else:
        return True, pos_final, list_iter
