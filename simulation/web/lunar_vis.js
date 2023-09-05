function convert_to_threejs_mesh(mesh) {
  const face_normals = mesh.face_normals;
  const vertices = mesh.vertices;
  const faces = mesh.faces;

  const geometry = new THREE.BufferGeometry();
  const material = new THREE.MeshLambertMaterial({ color: 0x666666 });

  const positions = new Float32Array(faces.length * 3 * 3);
  const normals = new Float32Array(faces.length * 3 * 3);

  for (let i = 0; i < faces.length; i++) {
    const face = faces[i];
    const normal = face_normals[i];
    for (let j = 0; j < 3; j++) {
      const vertex = vertices[face[j]];
      positions[i * 9 + j * 3 + 0] = vertex[0];
      positions[i * 9 + j * 3 + 1] = vertex[1];
      positions[i * 9 + j * 3 + 2] = vertex[2];
      normals[i * 9 + j * 3 + 0] = normal[0];
      normals[i * 9 + j * 3 + 1] = normal[1];
      normals[i * 9 + j * 3 + 2] = normal[2];
    }
  }

  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3));

  geometry.computeVertexNormals();

  three_mesh = new THREE.Mesh(geometry, material);
  three_mesh.receiveShadow = true;

  return three_mesh;
}

function getCenterPoint(mesh) {
  var middle = new THREE.Vector3();
  var geometry = mesh.geometry;

  geometry.computeBoundingBox();

  middle.x = (geometry.boundingBox.max.x + geometry.boundingBox.min.x) / 2;
  middle.y = (geometry.boundingBox.max.y + geometry.boundingBox.min.y) / 2;
  middle.z = (geometry.boundingBox.max.z + geometry.boundingBox.min.z) / 2;

  return middle;
}

function arrToChars(arr) {
  maxVal = Math.max(...arr);
  minVal = Math.min(...arr);
  const chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
  const charArr = [];
  for (let i = 0; i < arr.length; i++) {
    const val = arr[i];
    const char = chars[Math.round((val - minVal) / (maxVal - minVal) * (chars.length - 1))];
    charArr.push(char);
  }
  return charArr.join("");
}

function toDbm(arr) {
  const dbm = [];
  for (let i = 0; i < arr.length; i++) {
    const val = arr[i];
    dbm.push(10 * Math.log10(val));
  }
  return dbm;
}


const LunarVis = function (mesh) {
  const elements = document.getElementById("elements");
  let width = elements.getBoundingClientRect().width;
  let height = elements.getBoundingClientRect().height;
  let aspectRatio = width / height;

  // Create the scene, camera and renderer
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x444444);
  const camera = new THREE.PerspectiveCamera(75, aspectRatio, 0.1, 1000);
  camera.up.set(0, 0, 1);
  camera.position.set(0, 0, 20);
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.shadowMap.enabled = true;
  renderer.setSize(width, height);
  renderer.setPixelRatio(window.devicePixelRatio);

  // Add it to the page
  const viewerContainer = document.createElement('div');
  viewerContainer.appendChild(renderer.domElement);
  elements.appendChild(viewerContainer);

  // Create a listener for resize events
  window.addEventListener('resize', function () {
    width = elements.getBoundingClientRect().width;
    height = elements.getBoundingClientRect().height;
    aspectRatio = width / height;
    camera.aspect = aspectRatio;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
  });

  // Add the main mesh to the scene
  scene.add(convert_to_threejs_mesh(mesh));

  // Add controls
  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.target.set(getCenterPoint(convert_to_threejs_mesh(mesh)).x, getCenterPoint(convert_to_threejs_mesh(mesh)).y, getCenterPoint(convert_to_threejs_mesh(mesh)).z);

  // Add lights
  const ambient = new THREE.AmbientLight(0xffffff, 0.3); // soft white light
  scene.add(ambient);
  pointLight = new THREE.PointLight(0xffffff, 0.9);
  pointLight.position.set(getCenterPoint(convert_to_threejs_mesh(mesh)).x, getCenterPoint(convert_to_threejs_mesh(mesh)).y, 100);
  scene.add(pointLight);

  // Keep track of the pointer
  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();

  // Add a listener for mouse move events
  renderer.domElement.addEventListener('mousedown', function (event) {
    console.log(event);
    pointer.x = ( event.offsetX / width ) * 2 - 1;
	  pointer.y = - ( event.offsetY / height ) * 2 + 1;
  }, false);

  // Keep track of the scene objects so they can be removed later
  const sceneLines = new THREE.Object3D();
  const sceneAgents = new THREE.Object3D();
  scene.add(sceneLines);
  scene.add(sceneAgents);

  // Set up the tooltip
  const tooltip = document.createElement('div');
  tooltip.classList.add('tooltip');
  elements.appendChild(tooltip);

  // Render the scene
  const animate = function () {
    raycaster.setFromCamera( pointer, camera );
    const intersects = raycaster.intersectObjects(sceneAgents.children);
    // Make the tooltip invisible if there are no intersections
    if (intersects.length === 0) {
      tooltip.classList.remove('show');
    } else {
      // Otherwise, update the tooltip position and make it visible
      const x = pointer.x * width / 2 + width / 2;
      const y = - pointer.y * height / 2 + height / 2;
      tooltip.style.left = x + 'px';
      tooltip.style.top = y + 'px';
      tooltip.classList.add('show');

      // Update the tooltip text
      const agent = intersects[0].object.userData;
      tooltip.innerHTML = `
        <h1>Agent ${agent.id}</h1>
        <p>Position: ${agent.pos}</p>
        <div class="graph">${agent.csi_mag ? arrToChars(agent.csi_mag) : ""}</div>
      `;
      console.log(agent.csi_mag);
    }
    renderer.render(scene, camera);
    controls.update();
    requestAnimationFrame(animate);
  };
  animate();

  // Clears the canvas
  // Called on initialization by mesa
  this.reset = function () {
    sceneLines.children = [];
    sceneAgents.children = [];
  };

  // Renders the current simulation state
  // Called every frame by mesa
  this.render = function (modelState) {
    this.reset();
    console.log(modelState);

    const agents = modelState.agents;
    agents.forEach(agent => {
      // Place a sphere at the agent's position
      const geometry = new THREE.SphereGeometry(0.5, 32, 32);
      const color = agent.type === "fixed" ? 0x00ff00 : 0xff0000;
      const material = new THREE.MeshBasicMaterial({ color: color });
      const sphere = new THREE.Mesh(geometry, material);
      sphere.position.set(agent.pos[0], agent.pos[1], agent.pos[2]);
      sphere.userData = agent;
      sceneAgents.add(sphere);
    });

    const paths = [];
    const path_dict = modelState.paths
    for (p in path_dict) {
      paths.push(...path_dict[p]);
    };

    paths.forEach(path => {
      const points = [];
      path.forEach(point => {
        points.push(new THREE.Vector3(point[0], point[1], point[2]));
      });
      const geometry = new THREE.BufferGeometry().setFromPoints(points);
      const material = new THREE.LineBasicMaterial({ color: 0xffffff });
      const line = new THREE.Line(geometry, material);
      sceneLines.add(line);
    });
  };
};