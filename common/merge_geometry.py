from klampt import Geometry3D
from klampt.io import numpy_convert
import numpy as np

def merge_triangle_meshes(*items):
    merged_geom = Geometry3D()
    verts = []
    tris = []
    nverts = 0
    for i,item in enumerate(items):
        if isinstance(item,Geometry3D):
            xform,(iverts,itris) = numpy_convert.to_numpy(item)
        elif hasattr(item,'geometry'):
            xform,(iverts,itris) = numpy_convert.to_numpy(item.geometry())
        else:
            raise ValueError("Don't know how to merge trimesh from item of type "+item.__class__.__name__)
        verts.append(np.dot(np.hstack((iverts,np.ones((len(iverts),1)))),xform.T)[:,:3])
        tris.append(itris+nverts)
        nverts += len(iverts)
    verts = np.vstack(verts)
    tris = np.vstack(tris)
    for t in tris:
        assert all(v >= 0 and v < len(verts) for v in t)
    mesh = numpy_convert.from_numpy((verts,tris),'TriangleMesh')
    merged_geom.setTriangleMesh(mesh)
    return merged_geom