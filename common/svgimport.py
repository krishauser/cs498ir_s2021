from klampt import vis
from klampt.math import vectorops
from klampt import GeometricPrimitive
from klampt.model.trajectory import Trajectory,HermiteTrajectory

def svg_import_trajectories(fn,scale='auto',center=False,dims=3,
                            fit=1,want_attributes=False):
    """Reads one or more trajectories from a SVG file.  The svgpathtools
    library must be installed.

    To debug, try::

        trajs,attrs = svg_to_trajectories('test.svg',center=True,want_attributes=True)
        for i,(traj,attr) in enumerate(zip(trajs,attrs)):
            name = attr.get('name',"path %d"%i)
            vis.add(name,traj)
            for a,v in attr.items():
                if a != 'name':
                    vis.setAttribute(name,a,v)
        vis.loop()
    
    Args:
        fn (str): the filename
        scale (float, optional): if given, scales the units in fn to world
            units.  'auto' will fit the drawing to the box [0,fit]x[0,fit].
        center (bool or tuple, optional): if True, sets the drawing's origin
            to its center. If a 2-tuple, this is a center in world units.
            If false, no centering is done.
        fit (float, optional): if scale = 'auto', the drawing will be resized
            to fit into a box this size.
        dims (int, optional): either 2 or 3, depending on whether you want
            a 2D or 3D trajectory.
        want_attributes (bool, optional): if given, also tries parsing the
            attributes of the paths in the SVG file.  The attributes can
            be passed directly to the vis module.

    Returns:
        list of Trajectory: zero or more trajectories from the file.  If
        want_attributes = True, this also returns a list of dicts giving attributes
        parsed from the SVG file that can be given to the vis module.
    """
    try:
        from svgpathtools import svg2paths
    except ImportError:
        print('Need the svgpathtools package. Try "python -m pip install svgpathtools".')
        raise
    from klampt.model.collide import bb_create,bb_union
    paths, attributes = svg2paths(fn)
    trajs = []
    attrs = []
    for i,p in enumerate(paths):
        traj = svg_path_to_trajectory(p)
        traj.checkValid()
        if len(traj.milestones) == 0:
            print("Path",i,"is invalid, has",len(traj.milestones),"milestones")
            continue
        trajs.append(traj)
        attrs.append(attributes[i])        
    print("Read",len(trajs),"paths")
    if scale == False:
        scale = 1
    shift = [0,0]
    if isinstance(center,(tuple,list)):
        shift = [-center[0],-center[1]]
    if center == True or scale == 'auto':
        bounds = bb_create()
        for traj in trajs:
            if isinstance(traj,HermiteTrajectory):
                traj = traj.discretize(10)
            bounds = bb_union(bounds,bb_create(*traj.milestones))
        print("Bounds:",bounds)
        if center == True and scale == 'auto':
            scale = fit / max(bounds[1][0]-bounds[0][0],bounds[1][1]-bounds[0][1])
            shift = [-0.5*(bounds[1][0]+bounds[0][0])*scale,-0.5*(bounds[1][1]+bounds[0][1])*scale]
        elif center == True:
            shift = [-0.5*(bounds[1][0]+bounds[0][0])*scale,-0.5*(bounds[1][1]+bounds[0][1])*scale]
        elif scale == 'auto':
            scale = fit / max(bounds[1][0],bounds[1][1])
    print("Shift",shift,"scale",scale)
    for traj in trajs:
        if len(traj.milestones[0]) == 2:
            for j,m in enumerate(traj.milestones):
                traj.milestones[j] = vectorops.add(shift,vectorops.mul(m,scale))
        else:
            for j,m in enumerate(traj.milestones):
                traj.milestones[j] = vectorops.add(shift,vectorops.mul(m[:2],scale)) + vectorops.mul(m[2:],scale)
    if dims == 3:
        for traj in trajs:
            if len(traj.milestones[0]) == 2:
                for j,m in enumerate(traj.milestones):
                    traj.milestones[j] = m + [0.0]
            else:
                for j,m in enumerate(traj.milestones):
                    traj.milestones[j] = m[:2] + [0.0] + m[2:] + [0.0]
    if want_attributes:
        parsed_attrs = []
        for a in attrs:
            parsed = dict()
            styledict = a
            if 'id' in a:
                parsed['name'] = a['id']
            if 'style' in a:
                styledict = dict(v.split(':') for v in a['style'].split(';'))
            else:
                print(a)
            if 'stroke' in styledict:
                rgb = styledict['stroke'].strip().strip('#')
                a = 1
                if 'opacity' in styledict:
                    a = float(styledict['opacity'])
                if 'stroke-opacity' in styledict:
                    a = float(styledict['stroke-opacity'])
                if len(rgb)==3:
                    r,g,b = rgb
                    r = int(r,16)/0xf
                    g = int(g,16)/0xf
                    b = int(b,16)/0xf
                    parsed["color"] = (r,g,b,a)
                elif len(rgb)==6:
                    r,g,b = rgb[0:2],rgb[2:4],rgb[4:6]
                    r = int(r,16)/0xff
                    g = int(g,16)/0xff
                    b = int(b,16)/0xff
                    parsed["color"] = (r,g,b,a)
            if 'stroke-width' in styledict:
                parsed['width'] = float(styledict['stroke-width'].strip('px'))
            parsed_attrs.append(parsed)
        return trajs,parsed_attrs
    return trajs


def svg_path_to_trajectory(p):
    """Produces either a Trajectory or HermiteTrajectory according to an SVG
    Path.  The path must be continuous.
    """
    from svgpathtools import Path, Line, QuadraticBezier, CubicBezier, Arc
    if not p.iscontinuous():
        raise ValueError("Can't do discontinuous paths")
    pwl = not any(isinstance(seg,(CubicBezier,QuadraticBezier)) for seg in p)
    if pwl:
        milestones = []
        for seg in p:
            a,b = seg.start,seg.end
            if not milestones:
                milestones.append([a.real,a.imag])
            milestones.append([b.real,b.imag])
        return Trajectory(list(range(len(milestones))),milestones)
    times = []
    milestones = []
    for i,seg in enumerate(p):
        if isinstance(seg,CubicBezier):
            a,c1,c2,b = seg.start,seg.control1,seg.control2,seg.end
            vstart = (c1.real-a.real)*3,(c1.imag-a.imag)*3
            vend = (b.real-c2.real)*3,(b.imag-c2.imag)*3
            if not milestones:
                milestones.append([a.real,a.imag,vstart[0],vstart[1]])
                times.append(i)
            elif vectorops.distance(milestones[-1][2:],vstart) > 1e-4:
                milestones.append([a.real,a.imag,0,0])
                times.append(i)
                milestones.append([a.real,a.imag,vstart[0],vstart[1]])
                times.append(i)
            milestones.append([b.real,b.imag,vend[0],vend[1]])
            times.append(i+1)
        elif isinstance(seg,Line):
            a,b = seg.start,seg.end
            if not milestones:
                milestones.append([a.real,a.imag,0,0])
                times.append(i)
            elif vectorops.norm(milestones[-1][2:]) > 1e-4:
                milestones.append([a.real,a.imag,0,0])
                times.append(i)
            milestones.append([b.real,b.imag,0,0])
            times.append(i+1)
        else:
            raise NotImplementedError("Can't handle pieces of type {} yet".format(seg.__class.__name__))
    return HermiteTrajectory(times,milestones)

def svg_path_to_polygon(p,dt=0.1):
    """Produces a Klampt polygon geometry from an SVG Path."""
    traj = svg_path_to_trajectory(p)
    if isinstance(traj,HermiteTrajectory):
        traj = traj.discretize(dt)
    verts = sum(traj.milestones,[])
    g = GeometricPrimitive()
    g.setPolygon(verts)
    return g

if __name__ == '__main__':
    import sys
    fn = 'test.svg'
    if len(sys.argv) > 1:
        fn = sys.argv[1]

    trajs,attrs = svg_import_trajectories(fn,center=True,want_attributes=True)
    for i,(traj,attr) in enumerate(zip(trajs,attrs)):
        name = attr.get('name',"path %d"%i)
        vis.add(name,traj)
        for a,v in attr.items():
            if a != 'name':
                vis.setAttribute(name,a,v)
    vis.loop()