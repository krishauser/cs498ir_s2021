from klampt import RigidObjectModel
from klampt.math import so3
from klampt.io import numpy_convert
import numpy as np
from collections import deque
from scipy.spatial import ConvexHull
import math
from normals import get_triangle_normals

def stable_faces(obj,com=None,stability_tol=0.0,merge_tol=0.0):
    """
    Returns a list of support polygons on the object that are
    likely to be stable on a planar surface.
    
    Args:
        obj (RigidObjectModel or Geometry3D): the object.
        com (3-list, optional): sets the local center of mass. If
            not given, the default RigidObjectModel's COM will be used,
            or (0,0,0) will be used for a Geometry3D.
        stability_tol (float,optional): if > 0, then only faces that
            are stable with all perturbed "up" directions (dx,dy,1) with
            ||(dx,dy)||<= normal_tol will be outputted (robust stability). 
            If < 0, then all faces that are stable from some "up" direction
            (dx,dy,1) with ||(dx,dy)||<= |normal_tol| will be outputted
            (non-robust stability)
        merge_tol (float, optional): if > 0, then adjacent faces with
            normals whose angles are within this tolerance (in rads) will
            be merged together.
    
    Returns:
        list of list of 3-vectors: The set of all polygons that could
        form stable sides. Each polygon is convex and listed in
        counterclockwise order (i.e., the outward normal can be obtained
        via:
        
            (poly[2]-poly[0]) x (poly[1]-poly[0])
        
    """
    if isinstance(obj,RigidObjectModel):
        geom = obj.geometry()
        if com is None:
            com = obj.getMass().getCom()
    else:
        geom = obj
        if com is None:
            com = (0,0,0)
    assert len(com) == 3,"Need to provide a 3D COM"
    ch_trimesh = geom.convert('ConvexHull').convert('TriangleMesh')
    xform, (verts, tris) = numpy_convert.to_numpy(ch_trimesh)
    trinormals = get_triangle_normals(verts,tris)
    
    edges = dict()
    tri_neighbors = np.full(tris.shape,-1,dtype=np.int32)
    for ti,tri in enumerate(tris):
        for ei,e in enumerate([(tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])]):
            if (e[1],e[0]) in edges:
                tn,tne = edges[(e[1],e[0])]
                if tri_neighbors[tn][tne] >= 0:
                    print("Warning, triangle",ti,"neighbors two triangles on edge",tne,"?")
                tri_neighbors[ti][ei] = tn
                tri_neighbors[tn][tne] = ti
            else:
                edges[e] = ti,ei
    num_empty_edges = 0
    for ti,tri in enumerate(tris):
        for e in range(3):
            if tri_neighbors[tn][e] < 0:
                num_empty_edges += 1
    if num_empty_edges > 0:
        print("Info: boundary of mesh has",num_empty_edges,"edges")
    visited = [False]*len(tris)
    cohesive_faces = dict()
    for ti,tri in enumerate(tris):
        if visited[ti]:
            continue
        face = [ti]
        visited[ti] = True
        myvisit = set()
        myvisit.add(ti)
        q = deque()
        q.append(ti)
        while q:
            tvisit = q.popleft()
            for tn in tri_neighbors[tvisit]:
                if tn >= 0 and tn not in myvisit:
                    if math.acos(trinormals[ti].dot(trinormals[tn])) <= merge_tol:
                        face.append(tn)
                        myvisit.add(tn)
                        q.append(tn)
        for t in myvisit:
            visited[t] = True
        cohesive_faces[ti] = face
    output = []
    for t,face in cohesive_faces.items():
        n = trinormals[t]
        R = so3.canonical(n)
        if len(face) > 1:
            #project face onto the canonical basis
            faceverts = set()
            for t in face:
                faceverts.add(tris[t][0])
                faceverts.add(tris[t][1])
                faceverts.add(tris[t][2])
            faceverts = list(faceverts)
            xypts = [so3.apply(so3.inv(R),verts[v])[1:3] for v in faceverts]
            try:
                ch = ConvexHull(xypts)
                face = [faceverts[v] for v in ch.vertices]
            except Exception:
                print("Error computing convex hull of",xypts)
                print("Vertex indices",faceverts)
                print("Vertices",[verts[v] for v in faceverts])
        else:
            face = tris[face[0]]
        comproj = np.array(so3.apply(so3.inv(R),com)[1:3])
        
        stable = True
        for vi in range(len(face)):
            vn = (vi+1)%len(face)
            a,b = face[vi],face[vn]
            pa = np.array(so3.apply(so3.inv(R),verts[a])[1:3])
            pb = np.array(so3.apply(so3.inv(R),verts[b])[1:3])
            #check distance from com
            elen = np.linalg.norm(pb-pa)
            if elen == 0:
                continue
            sign = np.cross(pb - pa,comproj-pa)/elen
            if sign < stability_tol:
                stable = False
                break
        if stable:
            output.append([verts[i] for i in face])
    return output

def debug_stable_faces(obj,faces):
    from klampt import vis,Geometry3D,GeometricPrimitive
    from klampt.math import se3
    import random
    vis.createWindow()
    obj.setTransform(*se3.identity())
    vis.add("object",obj)
    for i,f in enumerate(faces):
        gf = GeometricPrimitive()
        gf.setPolygon(np.stack(f).flatten())
        color = (1,0.5+0.5*random.random(),0.5+0.5*random.random(),0.5)
        vis.add("face{}".format(i),Geometry3D(gf),color=color,hide_label=True)
    vis.setWindowTitle("Stable faces")
    vis.dialog()