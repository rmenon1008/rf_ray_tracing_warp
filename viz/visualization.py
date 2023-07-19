import http.server
import trimesh as tm
from trimesh import viewer
import numpy as np

def visualize(paths, tx_pos, rx_pos, mesh, points=None):
    # Visualize the result
    scene = tm.Scene()
    mesh.visual.face_colors = [100, 100, 100, 255]
    scene.add_geometry(mesh)

    tx_vis = tm.primitives.Sphere(radius=0.25, center=tx_pos)
    tx_vis.visual.face_colors = [255, 0, 0, 255]
    scene.add_geometry(tx_vis)

    rx_vis = tm.primitives.Sphere(radius=0.25, center=rx_pos)
    rx_vis.visual.face_colors = [0, 255, 0, 255]
    scene.add_geometry(rx_vis)

    if points:
        scene.add_geometry(tm.points.PointCloud(points, colors=np.tile([255, 255, 255], (points.shape[0], 1))))

    # Add the paths to the scene
    print(f"Adding {paths.shape[0]} paths to the scene...")
    for path in paths:
        path = tm.load_path(path)
        path.colors = np.tile([200, 200, 200], (path.entities.shape[0], 1))
        scene.add_geometry(path)
    print("Done!", end="\n\n")

    with open("viz/scene.html", "w") as f:
        f.write(tm.viewer.scene_to_html(scene))

    class Handler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/index.html' or self.path == '/':
                self.path = '/viz/scene.html'
                return http.server.SimpleHTTPRequestHandler.do_GET(self)
    httpd = http.server.HTTPServer(('', 8000), Handler)
    print("Serving visualization at localhost:8000")
    httpd.serve_forever()
