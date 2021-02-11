import numpy as np
from klampt import TriangleMesh,PointCloud,Geometry3D
from klampt.io import numpy_convert
from klampt.model import geometry

def get_object_normals(obj):
    """General-purpose object normal getter"""
    if isinstance(obj,TriangleMesh):
        verts,tris = numpy_convert.to_numpy(obj)
        return get_triangle_normals(verts,tris)
    elif isinstance(obj,PointCloud):
        return geometry.point_cloud_normals(obj)
    elif isinstance(obj,Geometry3D):
        if obj.type() == 'TriangleMesh':
            return get_object_normals(obj.getTriangleMesh())
        elif obj.type() == 'PointCloud':
            return get_object_normals(obj.getPointCloud())
        else:
            raise ValueError("Can only get normals for triangle mesh and point cloud")
    elif hasattr(obj,'geometry'):
        return get_object_normals(obj.geometry())
    else:
        raise ValueError("Invalid object sent to get_object_normals, can only be a TriangleMesh or PointCloud")

def get_triangle_normals(verts,tris):
    """
    Returns a list or numpy array of (outward) triangle normals for the
    triangle mesh defined by vertices verts and triangles tris.
    
    Args:
        verts: a Numpy array with shape (numPoints,3)
        tris: a Numpy int array with shape (numTris,3)
    """
    normals = np.zeros(tris.shape)
    dba = verts[tris[:,1]]-verts[tris[:,0]]
    dca = verts[tris[:,2]]-verts[tris[:,0]]
    n = np.cross(dba,dca)
    norms = np.linalg.norm(n,axis=1)[:, np.newaxis]
    n = np.divide(n,norms,where=norms!=0)
    return n
