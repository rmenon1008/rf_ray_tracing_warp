import http.server
import trimesh as tm
from trimesh import viewer
import numpy as np

def visualize(mesh=None, tx_pos=None, rx_pos=None, paths=None, points=None, point_color_pairs=None):
    # Visualize the result
    scene = tm.Scene()

    # # Show the normals of the mesh
    # vec = np.column_stack((mesh.faces, mesh.faces + (mesh.face_normals * 2)))
    # path = tm.load_path(vec.reshape((-1, 2, 3)))
    # scene.add_geometry(path)

    if mesh is not None:
        mesh.visual.face_colors = [100, 100, 100, 255]
        scene.add_geometry(mesh)

    if tx_pos is not None:
        tx_vis = tm.primitives.Sphere(radius=1, center=tx_pos)
        tx_vis.visual.face_colors = [255, 0, 0, 255]
        scene.add_geometry(tx_vis)

    if rx_pos is not None:
        rx_vis = tm.primitives.Sphere(radius=1, center=rx_pos)
        rx_vis.visual.face_colors = [0, 255, 0, 255]
        scene.add_geometry(rx_vis)

    if points is not None:
        scene.add_geometry(tm.points.PointCloud(points, colors=np.tile([255, 255, 255], (points.shape[0], 1))))

    if paths is not None:
        print(f"Adding {len(paths)} paths to the scene...")
        for path in paths:
            path = tm.load_path(path)
            path.colors = np.tile([200, 200, 200], (path.entities.shape[0], 1))
            scene.add_geometry(path)

    if point_color_pairs is not None:
        for point_color in point_color_pairs:
            point = tm.primitives.Sphere(radius=0.1, center=point_color[0])
            point.visual.face_colors = point_color[1]
            scene.add_geometry(point)

    with open("tmp/scene.html", "w") as f:
        f.write(tm.viewer.scene_to_html(scene))

    class Handler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/index.html' or self.path == '/':
                self.path = '/tmp/scene.html'
                return http.server.SimpleHTTPRequestHandler.do_GET(self)
    httpd = http.server.HTTPServer(('', 8000), Handler)
    print("Serving visualization at localhost:8000")
    httpd.serve_forever()
