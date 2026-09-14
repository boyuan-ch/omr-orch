"""
Memory-lean drop-in for zss.distance (zss 1.2.0), producing the identical distance.

Why this exists
---------------
zss 1.2.0's `distance()` always records the edit script, even when return_operations=False:
  - `operations` is a size_a x size_b grid of Python lists, and
  - every forest-distance cell does `partial_ops[x][y] = partial_ops[x-1][y] + [op]`,
    copying an ever-growing list.
Memory therefore scales like size_a * size_b * path_length, which is what blows up RAM on large
pages (whole-score comparisons reach ~1400 x 2100 nodes). The distance itself only ever reads
`treedists` / `fd`; the operation lists are never used for the returned value.

This module runs the exact same Zhang-Shasha recurrence -- same AnnotatedTree enumeration (imported
from zss itself), same keyroot order, same cost callbacks, same float arithmetic -- and simply does
not build the operation lists. Memory drops to O(size_a * size_b) floats.

Use `install()` to swap it in for `zss.distance`; `uninstall()` restores the original.
"""
import zss
from zss.compare import AnnotatedTree

_ORIGINAL_DISTANCE = zss.distance


def distance(A, B, get_children, insert_cost, remove_cost, update_cost, return_operations=False):
    if return_operations:
        # the edit script is exactly what this module avoids building
        return _ORIGINAL_DISTANCE(A, B, get_children, insert_cost, remove_cost, update_cost,
                                  return_operations=True)

    A, B = AnnotatedTree(A, get_children), AnnotatedTree(B, get_children)
    size_a = len(A.nodes)
    size_b = len(B.nodes)
    treedists = [[0.0] * size_b for _ in range(size_a)]

    Al = A.lmds
    Bl = B.lmds
    An = A.nodes
    Bn = B.nodes

    def treedist(i, j):
        m = i - Al[i] + 2
        n = j - Bl[j] + 2
        fd = [[0.0] * n for _ in range(m)]

        ioff = Al[i] - 1
        joff = Bl[j] - 1

        for x in range(1, m):
            fd[x][0] = fd[x - 1][0] + remove_cost(An[x + ioff])
        fd0 = fd[0]
        for y in range(1, n):
            fd0[y] = fd0[y - 1] + insert_cost(Bn[y + joff])

        Ali = Al[i]
        Blj = Bl[j]
        for x in range(1, m):
            node1 = An[x + ioff]
            rm1 = remove_cost(node1)
            fdx = fd[x]
            fdxm1 = fd[x - 1]
            td_row = treedists[x + ioff]
            a_is_anc = (Ali == Al[x + ioff])
            p = Al[x + ioff] - 1 - ioff
            for y in range(1, n):
                node2 = Bn[y + joff]
                if a_is_anc and Blj == Bl[y + joff]:
                    v = min(fdxm1[y] + rm1,
                            fdx[y - 1] + insert_cost(node2),
                            fdxm1[y - 1] + update_cost(node1, node2))
                    fdx[y] = v
                    td_row[y + joff] = v
                else:
                    q = Bl[y + joff] - 1 - joff
                    fdx[y] = min(fdxm1[y] + rm1,
                                 fdx[y - 1] + insert_cost(node2),
                                 fd[p][q] + td_row[y + joff])

    for i in A.keyroots:
        for j in B.keyroots:
            treedist(i, j)

    return treedists[-1][-1]


def install():
    zss.distance = distance


def uninstall():
    zss.distance = _ORIGINAL_DISTANCE
