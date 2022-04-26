import os

import numpy as np
import vedo as vp


def loader(dir, fn):
    fmsh = os.path.join(dir, "meshes", fn + ".ply")
    fgeo = os.path.join(dir, "geodesic_matrices", fn + ".npy")
    mesh = vp.load(fmsh)
    mesh.normalize()
    centre = mesh.points(copy=True).mean(axis=0)
    mesh.shift(*centre)
    dg = np.load(fgeo)
    return mesh, dg


class PartitionViewer(vp.plotter.Plotter):
    def __init__(self, dir):
        super().__init__()
        self.__dir = dir
        self.keyPressFunction = self.keyfunc
        self.mouseLeftClickFunction = self.onLeftClick
        self.mouseRightClickFunction = self.onRightClick

    def addMesh(self, fn):
        self.clear()
        mesh, g = loader(self.__dir, fn)
        self.__fout = os.path.join(self.__dir, "partitions", fn + ".npy")
        self.__mesh = mesh
        self.add([self.__mesh])
        self.__pts = mesh.points(copy=True)
        self.__g = g
        self.__left = []
        self.__right = []
        self.__spheres = []
        self.__last = None
        self.__partition = np.zeros(self.__pts.shape[0])
        self.__partitions = []

    def onLeftClick(self, mesh):
        self.__left.append(self.picked3d)
        self.custom_update()
        self.__last = -1

    def onRightClick(self, mesh):
        self.__right.append(self.picked3d)
        self.custom_update()
        self.__last = 1

    def custom_update(self):
        self.resetcam = True
        self.remove(self.__spheres)
        self.__spheres = []
        if 0 < len(self.__left):
            self.__spheres.append(
                vp.shapes.Spheres(self.__left, r=0.04, c="r")
            )
        if 0 < len(self.__right):
            self.__spheres.append(
                vp.shapes.Spheres(self.__right, r=0.04, c="g")
            )
        self.add(self.__spheres)
        if len(self.__left) == len(self.__right):
            self.partition_mesh()
        self.__mesh.addPointArray(self.__partition, "partition")

    def partition_mesh(self):

        if len(self.__left) == 0:
            self.__partition = np.zeros(self.__partition.shape)
            return

        left = [
            np.argmin(np.linalg.norm(self.__pts - l, axis=-1))
            for l in self.__left
        ]
        right = [
            np.argmin(np.linalg.norm(self.__pts - r, axis=-1))
            for r in self.__right
        ]
        N = self.__g.shape[0]
        dg = (self.__g[left] - self.__g[right]).sum(axis=0)
        ind = np.argsort(dg)[: N // 2]
        part = np.ones(N)
        part[ind] = -1
        self.__partition = part

    def keyfunc(self, key):
        if key == "c":
            self.clear_last()
        elif key == "s":
            if len(self.__partitions) == 0:
                print("Not saved. Press n first")
                return
            elif 0 < len(self.__left) or 0 < len(self.__right):
                self.keyfunc("n")
            np.save(self.__fout, np.stack(self.__partitions, axis=0))
        elif key == "n":
            self.__left.clear()
            self.__right.clear()
            self.__last = None
            self.__partitions.append(self.__partition.copy())
            self.custom_update()
        else:
            print(len(self.__partitions))
        return

    def clear_last(self):
        if self.__last == 1:
            self.__right.pop()
            self.__last = -1
        elif self.__last == -1:
            self.__left.pop()
            self.__last = 1
        else:
            return
        self.custom_update()


if __name__ == "__main__":

    dir = "/home/jamesklatzow/Documents/EBI/Preprocess/data/TOSCA"

    files = os.listdir(os.path.join(dir, "meshes"))
    fig = PartitionViewer(dir)
    for f in files:
        fn, _ = f.split(".ply")
        if os.path.isfile(os.path.join(dir, "partitions", fn + ".npy")):
            print(f"skipping {fn}")
            continue
        print(fn)
        try:
            fig.addMesh(fn)
            fig.show()
        except Exception:
            continue

    # vp.mouseMiddleClickFunction = onMiddleClick
    # vp.mouseRightClickFunction  = onRightClick

    # printc("Click object to trigger function call", invert=1, box="-")

    # vp += __doc__
    # vp.show()

    # saveOBJ(data, fn[:-3] + 'obj')
