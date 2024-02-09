from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Type
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import numpy.linalg as la
from dataclasses import dataclass


def overlaps(start1, end1, start2, end2, include_touches: bool):
    begin_overlap = max(start1, start2)
    end_overlap = min(end1, end2)
    if include_touches and np.allclose(begin_overlap, end_overlap):
        return True
    else:
        return begin_overlap < end_overlap


# {{{ box tools

@dataclass
class Box:
    lower_left: np.ndarray
    size: float
    children: list[Box]
    particles: Optional[np.ndarray]
    parent: Optional[Box] = None

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return hash(id(self))

    @property
    def center(self):
        return self.lower_left + self.size/2


def get_root_box(particles: Optional[np.ndarray]=None) -> Box:
    return Box(lower_left=np.zeros(2), size=1, children=[], particles=particles)


def split_box(box: Box):
    from pytools import generate_nonnegative_integer_tuples_below

    box.children = [
        Box(
            lower_left=box.lower_left + np.array(offset) * box.size * 0.5,
            size=box.size / 2,
            children=[],
            particles=None,
            parent=box,
        )
        for offset in generate_nonnegative_integer_tuples_below(2, len(box.lower_left))
    ]
    if box.particles is not None:
        for ch in box.children:
            in_box = (
                (box.particles >= ch.lower_left.reshape(-1, 1)).all(axis=0)
                & 
                (box.particles < ch.lower_left.reshape(-1, 1) + ch.size).all(axis=0)
            )
            ch.particles = box.particles[:, in_box]
        box.particles = None


def uniform_split_box(box: Box, levels: int):
    if not levels:
        return

    split_box(box)
    for ch in box.children:
        uniform_split_box(ch, levels - 1)


def split_boxes_by_mac(root: Box, tbox: Box) -> None:
    if root is tbox:
        return
    if are_neighbors(root, tbox):
        return
    
    if not root.children:
        dim = len(root.lower_left)

        rsource = root.size/2
        rtarget = tbox.size/2
        r = la.norm(root.center - tbox.center, 2)

        # multipole acceptance criterion
        if r < 3*max(rsource, rtarget):
            split_box(root)

    for ch in root.children:
        split_boxes_by_mac(ch, tbox)


class _NotProvided:
    pass


def plot_box(box: Box,
        particle_marker: str | None | Type[_NotProvided] = _NotProvided,
        particle_kwargs: Optional[Dict[str, Any]] = None,
        center_marker: str | None = None,
        center_kwargs: Optional[Dict[str, Any]] = None,
        box_boundaries: bool =True,
        box_boundaries_kwargs: Optional[Dict[str, Any]] = None):
    if box_boundaries:
        if box_boundaries_kwargs is None:
            box_boundaries_kwargs = {
                "edgecolor": "black",
                "facecolor": "none",
            }
        
        plt.gca().add_patch(
            patches.Rectangle(
                xy=tuple(box.lower_left),
                width=box.size,
                height=box.size,
                **box_boundaries_kwargs
            )
        )
    if box.particles is not None:
        if particle_marker is not None:
            if particle_marker is _NotProvided:
                particle_marker = "ro"
            if particle_kwargs is None:
                particle_kwargs = {"markersize": 1}
            plt.plot(box.particles[0], box.particles[1], particle_marker,
                     **particle_kwargs)

        if center_marker is not None:
            if center_marker is _NotProvided:
                center_marker = "ro"
            if center_kwargs is None:
                center_kwargs = {"markersize": 4}
            plt.plot(box.center[0], box.center[1], center_marker,
                     **center_kwargs)


def plot_boxtree(box: Box, *, except_boxes: Sequence[Box] = (), **kwargs):
    if box in except_boxes:
        return

    if box.children:
        for ch in box.children:
            plot_boxtree(ch, except_boxes=except_boxes, **kwargs)
    else:
        plot_box(box, **kwargs)
    


def find_box_at_opt(root: Box, point: np.ndarray) -> Optional[Box]:
    in_box = ((point >= root.lower_left).all(axis=0)
        &  (point < root.lower_left + root.size).all(axis=0))
    if not in_box:
        return None

    for ch in root.children:
        res = find_box_at_opt(ch, point)
        if res:
            return res

    return root


class BoxNotFoundError(Exception):
    pass


def find_box_at(root: Box, point: np.ndarray) -> Box:
    res = find_box_at_opt(root, point)
    if res is None:
        raise BoxNotFoundError()
    return res


def are_neighbors(box1: Box, box2: Box) -> bool:
    dim = len(box1.lower_left)
    touches_along_axis =  [
        np.isclose(box1.lower_left[axis] + box1.size, box2.lower_left[axis])
        or 
        np.isclose(box2.lower_left[axis] + box2.size, box1.lower_left[axis])
        for axis in range(dim)
    ]
    overlaps_along_axis = [
        overlaps(
                 box1.lower_left[axis], box1.lower_left[axis] + box1.size,
                 box2.lower_left[axis], box2.lower_left[axis] + box2.size,
                 include_touches=True)
        for axis in range(dim)
    ]

    for axis in range(dim):
        if (touches_along_axis[axis] 
                and all(overlaps_along_axis[:axis])
                and all(overlaps_along_axis[axis:])):
            return True

    return False


def find_box_neighbors(root: Box, box: Box) -> List[Box]:
    dim = len(root.lower_left)

    overlaps_along_axis = [
            overlaps(
                     root.lower_left[axis], root.lower_left[axis] + root.size,
                     box.lower_left[axis], box.lower_left[axis] + box.size,
                     include_touches=True)
            for axis in range(dim)
        ]
    if not all(overlaps_along_axis):
        return []

    res = [
        nb
        for ch in root.children
        for nb in find_box_neighbors(ch, box)
    ]

    if res:
        return res
    elif are_neighbors(root, box):
        return [root]
    else:
        return []

# }}}


def configure_plot(clear: bool = True) -> None:
    if clear:
        plt.clf()
    plt.gca().set_aspect("equal")
    plt.axis("off")
    plt.tight_layout()


def get_particles():
    return np.random.rand(2, 2000)


def plot_bhut_particles():
    configure_plot()
    plot_box(get_root_box(get_particles()))
    plt.savefig("media/bhut-01-particles.pdf")


def plot_bhut_boxes():
    configure_plot()
    b = get_root_box(get_particles())
    uniform_split_box(b, 3)
    plot_boxtree(b)

    plt.savefig("media/bhut-02-boxes.pdf")


def plot_bhut_boxes_target():
    configure_plot()
    b = get_root_box(get_particles())
    uniform_split_box(b, 3)
    tbox = find_box_at(b, np.array([0.3, 0.6]))
    plot_boxtree(b, except_boxes=[tbox])
    plot_boxtree(tbox, particle_marker="bo")

    plt.savefig("media/bhut-03-boxes-tgt.pdf")


def plot_bhut_boxes_mpole():
    configure_plot()
    b = get_root_box(get_particles())
    uniform_split_box(b, 3)
    tbox = find_box_at(b, np.array([0.3, 0.6]))
    nbs = find_box_neighbors(b, tbox)
    plot_boxtree(b, except_boxes=nbs + [tbox],
             center_marker="ro",
             particle_marker=None)
    for nb in nbs:
        plot_box(nb)
    plot_box(tbox, particle_marker="bo")

    plt.savefig("media/bhut-04-boxes-mpole.pdf")


def plot_bhut_levels():
    plt.clf()

    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.title(f"Level {i}")
        configure_plot(clear=False)
        b = get_root_box()
        uniform_split_box(b, i)
        plot_boxtree(b)

    plt.savefig("media/bhut-05-levels.pdf")


def plot_bhut_box_sizes():
    configure_plot()
    b = get_root_box(get_particles())
    uniform_split_box(b, 4)
    tbox = find_box_at(b, np.array([0.3, 0.6]))
    plot_box(b, particle_marker=None)
    plot_boxtree(b, except_boxes=[tbox], box_boundaries=False)
    plot_boxtree(tbox, particle_marker="bo")

    plt.savefig("media/bhut-06-box-sizes.pdf")


def plot_bhut_particle_tree():
    configure_plot()
    tpoint = np.array([0.42, 0.6])
    b = get_root_box(get_particles())
    for i in range(5):
        tbox = find_box_at(b, tpoint)
        split_box(tbox)

    tbox = find_box_at(b, tpoint)
    split_boxes_by_mac(b, tbox)

    plot_boxtree(b, except_boxes=[tbox])
    plot_box(tbox, particle_marker="bo")

    plt.savefig("media/bhut-07-particle-tree.pdf")


    
def plot_bhut_mpole_tree():
    configure_plot()
    tpoint = np.array([0.42, 0.6])
    b = get_root_box(get_particles())
    for i in range(5):
        tbox = find_box_at(b, tpoint)
        split_box(tbox)

    tbox = find_box_at(b, tpoint)
    split_boxes_by_mac(b, tbox)

    nbs = find_box_neighbors(b, tbox)
    plot_boxtree(b, except_boxes=[tbox] + nbs, 
                 particle_marker=None, center_marker="ro")
    for nb in nbs:
        plot_box(nb)
    plot_box(tbox, particle_marker="bo")

    plt.savefig("media/bhut-08-mpole-tree.pdf")


if __name__ == "__main__":
    plot_bhut_particles()
    plot_bhut_boxes()
    plot_bhut_boxes_target()
    plot_bhut_boxes_mpole()
    plot_bhut_levels()
    plot_bhut_box_sizes()
    plot_bhut_particle_tree()
    plot_bhut_mpole_tree()
