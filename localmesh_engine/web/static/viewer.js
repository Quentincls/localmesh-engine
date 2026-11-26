import {
  ACESFilmicToneMapping, Box3, DirectionalLight, DoubleSide, Mesh, MeshBasicMaterial,
  MeshStandardMaterial, PCFShadowMap, PerspectiveCamera, PlaneGeometry, PMREMGenerator,
  Scene, ShadowMaterial, Sphere, Spherical, Vector3, WebGLRenderer,
} from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

function disposeModel(model, originalMaterials = new Map()) {
  const geometries = new Set();
  const materials = new Set();
  const textures = new Set();
  const images = new Set();
  model.traverse((object) => {
    if (!object.isMesh) return;
    geometries.add(object.geometry);
    const original = originalMaterials.get(object) ?? object.material;
    for (const material of [original].flat()) materials.add(material);
  });
  for (const material of materials) {
    for (const value of Object.values(material)) {
      if (value?.isTexture) textures.add(value);
    }
    material.dispose();
  }
  for (const texture of textures) {
    const image = texture.source.data;
    if (typeof image?.close === 'function') images.add(image);
    texture.dispose();
  }
  for (const image of images) image.close();
  for (const geometry of geometries) geometry.dispose();
}

export class MeshViewer {
  constructor(surface) {
    this.surface = surface;
    this.renderer = new WebGLRenderer({ alpha: true, antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
    this.renderer.toneMapping = ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 0.8;
    this.renderer.shadowMap.type = PCFShadowMap;
    this.renderer.shadowMap.autoUpdate = false;
    const canvas = this.renderer.domElement;
    canvas.setAttribute('aria-label', 'Interactive 3D model. Use the arrow keys to orbit and plus or minus to zoom.');
    canvas.setAttribute('role', 'img');
    canvas.tabIndex = 0;
    surface.replaceChildren(canvas);

    this.scene = new Scene();
    this.camera = new PerspectiveCamera(40, 1, 0.01, 100);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.enablePan = false;
    this.controls.autoRotateSpeed = 1;
    this.controls.addEventListener('change', () => this.invalidate());
    canvas.addEventListener('keydown', (event) => this.moveCamera(event));

    const room = new RoomEnvironment();
    const pmrem = new PMREMGenerator(this.renderer);
    this.scene.environment = pmrem.fromScene(room, 0.035).texture;
    this.scene.environmentIntensity = 0.8;
    room.dispose();
    pmrem.dispose();

    this.light = new DirectionalLight(0xfff6e8, 1.9);
    this.light.shadow.mapSize.set(1024, 1024);
    this.light.shadow.bias = -0.0004;
    this.scene.add(this.light, this.light.target);

    this.floor = new Mesh(new PlaneGeometry(), new ShadowMaterial({ opacity: 0.65 }));
    this.floor.rotation.x = -Math.PI / 2;
    this.floor.receiveShadow = true;
    this.floor.visible = false;
    this.scene.add(this.floor);

    this.replacementMaterials = {
      solid: new MeshStandardMaterial({ color: 0xb4bdc9, roughness: 0.85, side: DoubleSide }),
      wireframe: new MeshBasicMaterial({
        color: 0x9cc5ff, wireframe: true, toneMapped: false, transparent: true, opacity: 0.55,
      }),
    };
    this.originalMaterials = new Map();
    this.loader = new GLTFLoader();
    this.mode = 'textured';
    this.frame = null;
    this.lastFrameTime = 0;
    this.sizeObserver = new ResizeObserver(() => this.resize());
    this.sizeObserver.observe(surface);
    this.resize();
  }

  resize() {
    const { clientWidth, clientHeight } = this.surface;
    if (!clientWidth || !clientHeight) return;
    this.renderer.setSize(clientWidth, clientHeight, false);
    this.camera.aspect = clientWidth / clientHeight;
    this.camera.updateProjectionMatrix();
    this.invalidate();
  }

  invalidate() {
    if (this.frame !== null) return;
    this.frame = requestAnimationFrame((time) => {
      this.frame = null;
      const elapsed = Math.min((time - this.lastFrameTime) / 1000, 0.05);
      this.lastFrameTime = time;
      if (this.model) this.controls.update(elapsed);
      this.renderer.render(this.scene, this.camera);
      if (this.controls.autoRotate && this.model) this.invalidate();
    });
  }

  moveCamera(event) {
    if (!this.model || !['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', '+', '=', '-'].includes(event.key)) return;
    event.preventDefault();
    const target = this.controls.target;
    const view = new Spherical().setFromVector3(new Vector3().subVectors(this.camera.position, target));
    const orbitStep = Math.PI / 24;
    const horizontal = { ArrowLeft: -orbitStep, ArrowRight: orbitStep };
    const vertical = { ArrowUp: -orbitStep, ArrowDown: orbitStep };
    view.theta += horizontal[event.key] ?? 0;
    view.phi += vertical[event.key] ?? 0;
    const zoom = { '+': 0.9, '=': 0.9, '-': 1.1 };
    view.radius *= zoom[event.key] ?? 1;
    this.camera.position.setFromSpherical(view.makeSafe()).add(target);
    this.controls.update(0);
  }

  async load(url, fitCamera) {
    this.pending?.abort();
    const request = new AbortController();
    this.pending = request;
    try {
      const response = await fetch(url, { signal: request.signal });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const modelData = await response.arrayBuffer();
      const baseUrl = new URL('.', new URL(url, window.location.href)).href;
      const { scene: model } = await this.loader.parseAsync(modelData, baseUrl);
      if (request.signal.aborted) {
        disposeModel(model);
        return false;
      }
      this.setModel(model, fitCamera);
      return true;
    } catch (error) {
      if (request.signal.aborted) return false;
      throw error;
    } finally {
      if (this.pending === request) this.pending = null;
    }
  }

  clear() {
    this.pending?.abort();
    this.removeModel();
    this.floor.visible = false;
    this.invalidate();
  }

  removeModel() {
    if (!this.model) return;
    this.model.removeFromParent();
    disposeModel(this.model, this.originalMaterials);
    this.originalMaterials.clear();
    this.model = null;
  }

  setModel(model, fitCamera) {
    const bounds = new Box3().setFromObject(model);
    const { center, radius } = bounds.getBoundingSphere(new Sphere());
    if (!Number.isFinite(radius) || radius <= 0) {
      disposeModel(model);
      throw new Error('The GLB contains no displayable geometry.');
    }
    if (fitCamera || !this.model) this.fit(radius);
    this.removeModel();
    this.model = model;
    model.position.sub(center);
    model.traverse((object) => {
      if (object.isMesh) this.originalMaterials.set(object, object.material);
    });
    this.scene.add(model);

    this.floor.position.y = bounds.min.y - center.y - radius * 0.012;
    this.floor.scale.setScalar(radius * 9);
    this.light.position.set(3 * radius, 5 * radius, 2.4 * radius);
    const shadowCamera = this.light.shadow.camera;
    shadowCamera.left = shadowCamera.bottom = -1.7 * radius;
    shadowCamera.right = shadowCamera.top = 1.7 * radius;
    shadowCamera.near = 0.1 * radius;
    shadowCamera.far = 10 * radius;
    shadowCamera.updateProjectionMatrix();
    this.light.shadow.normalBias = 0.012 * radius;
    this.setMode(this.mode);
  }

  fit(radius) {
    const verticalFov = (this.camera.fov * Math.PI) / 360;
    const horizontalFov = Math.atan(Math.tan(verticalFov) * this.camera.aspect);
    const distance = (radius * 1.13) / Math.sin(Math.min(verticalFov, horizontalFov));
    this.camera.near = radius * 0.005;
    this.camera.far = Math.max(radius * 100, distance * 5);
    this.camera.updateProjectionMatrix();
    this.controls.minDistance = radius * 0.4;
    this.controls.maxDistance = Math.max(radius * 14, distance * 2);
    this.controls.target.set(0, 0, 0);
    this.camera.position.set(0.55, 0.3, 1).setLength(distance);
    this.controls.update(0);
  }

  setMode(mode) {
    this.mode = mode;
    const shadows = mode === 'shadows';
    this.floor.visible = shadows && Boolean(this.model);
    this.light.castShadow = shadows;
    this.renderer.shadowMap.enabled = shadows;
    this.renderer.shadowMap.needsUpdate = shadows;
    this.light.shadow.needsUpdate = shadows;
    for (const [mesh, original] of this.originalMaterials) {
      mesh.material = this.replacementMaterials[mode] ?? original;
      mesh.castShadow = shadows;
      mesh.receiveShadow = shadows;
    }
    this.invalidate();
  }

  setAutoRotate(enabled) {
    this.controls.autoRotate = enabled;
    this.controls.enableDamping = !enabled;
    this.invalidate();
  }
}
