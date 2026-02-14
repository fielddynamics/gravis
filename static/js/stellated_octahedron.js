// =========================================================================
// Stellated Octahedron - Three.js Visualization
// Extracted from the standalone DFP Gravity Field Dynamics Visual
//
// Copyright (c) 2026 Stephen Nelson / Field Dynamics
// MIT License
//
// Dependencies: Three.js r128 (must be loaded before this script)
// =========================================================================

// =========================================================================
// CONSTANTS
// =========================================================================

const CONSTANTS = {
    // =====================================================================
    // TOPOLOGY
    // =====================================================================
    topology: {
        k: 4,           // Coupling ports per tetrahedron
        d: 3,           // Spatial dimensions
        vertices: 8,    // Total vertices (2^d)
    },

    // =====================================================================
    // COLORS - Blue (Tet A) and Yellow (Tet B)
    // =====================================================================
    colors: {
        tetA: 0x4488ff,          // Blue
        tetB: 0xffcc00,          // Yellow/Gold
        background: 0x181818,    // Dark background
        highlight: 0xffffff,     // White highlight
        fieldOrigin: 0xffffff,  // White glow at center
    },

    // =====================================================================
    // VERTEX ENCODING
    // Binary encoding: XYZ where each bit is 0 or 1
    // Parity determines tetrahedron: ODD -> Tet A, EVEN -> Tet B
    // =====================================================================
    vertices: {
        encoding: {
            '000': { x: -1, y: -1, z: -1, tet: 'B', index: 0 },
            '001': { x: -1, y: -1, z: +1, tet: 'A', index: 1 },
            '010': { x: -1, y: +1, z: -1, tet: 'A', index: 2 },
            '011': { x: -1, y: +1, z: +1, tet: 'B', index: 3 },
            '100': { x: +1, y: -1, z: -1, tet: 'A', index: 4 },
            '101': { x: +1, y: -1, z: +1, tet: 'B', index: 5 },
            '110': { x: +1, y: +1, z: -1, tet: 'B', index: 6 },
            '111': { x: +1, y: +1, z: +1, tet: 'A', index: 7 },
        },
        // Tet A: ODD parity (1 or 3 bits set) - apex at 111
        tetA: ['001', '010', '100', '111'],
        // Tet B: EVEN parity (0 or 2 bits set) - apex at 000
        tetB: ['000', '011', '101', '110'],
    },

    // =====================================================================
    // PI DIGITS (Base 8) - 300 digits for 10+ closure cycles
    // Pi (octal) = 3.1103755242102643021514230630505600670163211220111...
    // Verified source: OEIS A006941 (https://oeis.org/A006941)
    // =====================================================================
    pi: {
        // OEIS A006941 - First 300 octal digits of Pi (verified)
        digits_base8: [
            3, 1, 1, 0, 3, 7, 5, 5, 2, 4, 2, 1, 0, 2, 6, 4, 3, 0, 2, 1,  // 1-20
            5, 1, 4, 2, 3, 0, 6, 3, 0, 5, 0, 5, 6, 0, 0, 6, 7, 0, 1, 6,  // 21-40
            3, 2, 1, 1, 2, 2, 0, 1, 1, 1, 6, 0, 2, 1, 0, 5, 1, 4, 7, 6,  // 41-60
            3, 0, 7, 2, 0, 0, 2, 0, 2, 7, 3, 7, 2, 4, 6, 1, 6, 6, 1, 1,  // 61-80
            6, 3, 3, 1, 0, 4, 5, 0, 5, 1, 2, 0, 2, 0, 7, 4, 6, 1, 6, 1,  // 81-100
            5, 0, 0, 2, 3, 3, 5, 7, 3, 7, 1, 2, 4, 3, 1, 5, 4, 7, 4, 6,  // 101-120
            4, 7, 2, 2, 0, 6, 1, 5, 4, 6, 0, 1, 2, 6, 0, 5, 1, 5, 5, 7,  // 121-140
            4, 4, 5, 7, 4, 2, 4, 1, 5, 6, 4, 7, 7, 4, 1, 1, 5, 2, 6, 6,  // 141-160
            5, 5, 5, 2, 4, 3, 4, 1, 1, 0, 5, 7, 1, 1, 0, 2, 6, 6, 5, 3,  // 161-180
            5, 4, 6, 1, 1, 3, 6, 3, 7, 5, 4, 3, 3, 6, 4, 2, 3, 0, 4, 1,  // 181-200
            3, 5, 1, 5, 1, 4, 3, 3, 7, 5, 5, 3, 2, 6, 0, 5, 7, 7, 7, 2,  // 201-220
            7, 1, 3, 3, 3, 6, 4, 0, 1, 5, 3, 3, 7, 5, 5, 7, 3, 4, 3, 4,  // 221-240
            1, 5, 3, 7, 6, 6, 5, 5, 2, 1, 1, 4, 7, 7, 2, 2, 6, 5, 6, 4,  // 241-260
            7, 6, 2, 2, 0, 2, 1, 3, 7, 0, 4, 5, 4, 3, 7, 7, 1, 4, 4, 4,  // 261-280
            4, 5, 0, 3, 1, 4, 5, 0, 7, 5, 4, 7, 1, 0, 5, 5, 4, 7, 5, 6   // 281-300
        ],
        // Checksum: length=300, sum=1032, weightedSum=167584, polyHash=1687952315
    },

    // =====================================================================
    // FACE DEFINITIONS - Which 3 vertices make up each face
    // A face completes when ALL 3 of its vertices have been visited
    // Face names use OPPOSITE vertex convention (e.g., FACE_A7 is opposite V7)
    // =====================================================================
    faceVertices: {
        'FACE_A1': { vertices: new Set([2, 4, 7]), tetId: 'A', role: 'SPATIAL', faceIdx: 1 },
        'FACE_A2': { vertices: new Set([1, 4, 7]), tetId: 'A', role: 'SPATIAL', faceIdx: 2 },
        'FACE_A4': { vertices: new Set([1, 2, 7]), tetId: 'A', role: 'SPATIAL', faceIdx: 0 },
        'FACE_A7': { vertices: new Set([1, 2, 4]), tetId: 'A', role: 'TIME', faceIdx: 3 },    // TIME face
        'FACE_B0': { vertices: new Set([3, 5, 6]), tetId: 'B', role: 'TIME', faceIdx: 3 },    // TIME face
        'FACE_B3': { vertices: new Set([0, 5, 6]), tetId: 'B', role: 'SPATIAL', faceIdx: 1 },
        'FACE_B5': { vertices: new Set([0, 3, 6]), tetId: 'B', role: 'SPATIAL', faceIdx: 2 },
        'FACE_B6': { vertices: new Set([0, 3, 5]), tetId: 'B', role: 'SPATIAL', faceIdx: 0 },
    },

    // =====================================================================
    // VERTEX TO FACES MAPPING - Which faces contain each vertex
    // When a vertex is visited, it contributes to completing these faces
    // =====================================================================
    vertexToFaces: {
        0: ['FACE_B3', 'FACE_B5', 'FACE_B6'],  // V0 is in 3 Tet B faces
        1: ['FACE_A2', 'FACE_A4', 'FACE_A7'],  // V1 is in 3 Tet A faces
        2: ['FACE_A1', 'FACE_A4', 'FACE_A7'],  // V2 is in 3 Tet A faces
        3: ['FACE_B0', 'FACE_B5', 'FACE_B6'],  // V3 is in 3 Tet B faces
        4: ['FACE_A1', 'FACE_A2', 'FACE_A7'],  // V4 is in 3 Tet A faces
        5: ['FACE_B0', 'FACE_B3', 'FACE_B6'],  // V5 is in 3 Tet B faces
        6: ['FACE_B0', 'FACE_B3', 'FACE_B5'],  // V6 is in 3 Tet B faces
        7: ['FACE_A1', 'FACE_A2', 'FACE_A4'],  // V7 is in 3 Tet A faces
    },

    // =====================================================================
    // VERTEX INFO - Tetrahedron assignment by parity
    // =====================================================================
    vertexInfo: {
        0: { tetId: 'B', binary: '000' },  // Even parity -> Tet B
        1: { tetId: 'A', binary: '001' },  // Odd parity -> Tet A
        2: { tetId: 'A', binary: '010' },  // Odd parity -> Tet A
        3: { tetId: 'B', binary: '011' },  // Even parity -> Tet B
        4: { tetId: 'A', binary: '100' },  // Odd parity -> Tet A
        5: { tetId: 'B', binary: '101' },  // Even parity -> Tet B
        6: { tetId: 'B', binary: '110' },  // Even parity -> Tet B
        7: { tetId: 'A', binary: '111' },  // Odd parity -> Tet A
    },
};

// =========================================================================
// PI BASE-8 PREFLIGHT VALIDATION
// Scientific verification of pi digits before field closure execution
// =========================================================================

const PiBase8Validator = {
    /**
     * PREFLIGHT VALIDATION FOR PI BASE-8 DIGITS
     * 
     * Uses a cryptographic-style hash of the full sequence to verify
     * data integrity. Any single digit error will cause validation to fail.
     * 
     * Reference: OEIS A006941 (https://oeis.org/A006941)
     * 
     * Why this matters:
     * - Pi base-8 digits drive Field Closure vertex selection (0-7)
     * - Incorrect digits would produce wrong traversal patterns
     * - Pi is irrational/non-repeating; the hash validates the full sequence
     */

    // Expected values for the full 300-digit sequence (OEIS A006941)
    EXPECTED: {
        length: 300,
        sum: 1032,              // Sum of all 300 digits
        weightedSum: 167584,    // Position-weighted sum: sum(digit[i] * (i+1))
        polyHash: 1687952315,   // Polynomial hash for collision resistance
    },

    /**
     * Compute polynomial hash of digit sequence
     * hash = sum(digit[i] * BASE^i) mod (2^31 - 1)
     * This is a standard string hashing technique with good collision resistance
     */
    computePolyHash(digits) {
        const PRIME = 2147483647;  // 2^31 - 1 (Mersenne prime)
        const BASE = 31;
        let hash = 0;
        let basePower = 1;

        for (let i = 0; i < digits.length; i++) {
            hash = (hash + digits[i] * basePower) % PRIME;
            basePower = (basePower * BASE) % PRIME;
        }

        return hash;
    },

    /**
     * Compute all checksums for the digit sequence
     */
    computeChecksums(digits) {
        let sum = 0;
        let weightedSum = 0;

        for (let i = 0; i < digits.length; i++) {
            sum += digits[i];
            weightedSum += digits[i] * (i + 1);
        }

        return {
            length: digits.length,
            sum: sum,
            weightedSum: weightedSum,
            polyHash: this.computePolyHash(digits),
        };
    },

    /**
     * Validate the pi base-8 sequence against expected checksums
     * @returns {Object} Validation result with status and details
     */
    validate() {
        const digits = CONSTANTS.pi.digits_base8;
        const computed = this.computeChecksums(digits);
        const expected = this.EXPECTED;

        // Check all values match
        const lengthOK = computed.length === expected.length;
        const sumOK = computed.sum === expected.sum;
        const weightedOK = computed.weightedSum === expected.weightedSum;
        const hashOK = computed.polyHash === expected.polyHash;

        const passed = lengthOK && sumOK && weightedOK && hashOK;

        return {
            passed,
            computed,
            expected,
            details: {
                length: { ok: lengthOK, computed: computed.length, expected: expected.length },
                sum: { ok: sumOK, computed: computed.sum, expected: expected.sum },
                weightedSum: { ok: weightedOK, computed: computed.weightedSum, expected: expected.weightedSum },
                polyHash: { ok: hashOK, computed: computed.polyHash, expected: expected.polyHash },
            },
        };
    },

    /**
     * Log validation results to console
     */
    logResult(result) {
        const status = result.passed ? 'PASSED' : 'FAILED';
        const color = result.passed ? '#4CAF50' : '#f44336';

        console.log(`%c[Pi Base-8 Validation] ${status}`, `color: ${color}; font-weight: bold;`);
        console.log(`  Length:      ${result.details.length.computed} (expected ${result.details.length.expected}) ${result.details.length.ok ? '✓' : '✗'}`);
        console.log(`  Sum:         ${result.details.sum.computed} (expected ${result.details.sum.expected}) ${result.details.sum.ok ? '✓' : '✗'}`);
        console.log(`  WeightedSum: ${result.details.weightedSum.computed} (expected ${result.details.weightedSum.expected}) ${result.details.weightedSum.ok ? '✓' : '✗'}`);
        console.log(`  PolyHash:    ${result.details.polyHash.computed} (expected ${result.details.polyHash.expected}) ${result.details.polyHash.ok ? '✓' : '✗'}`);

        return result;
    },

    /**
     * Run validation and throw error if failed
     * Call this before starting field closure animation
     */
    assertValid() {
        const result = this.validate();
        this.logResult(result);

        if (!result.passed) {
            const failures = Object.entries(result.details)
                .filter(([k, v]) => !v.ok)
                .map(([k, v]) => `${k}: got ${v.computed}, expected ${v.expected}`);
            throw new Error(`Pi Base-8 Validation FAILED: ${failures.join('; ')}`);
        }

        return true;
    }
};

// =========================================================================
// STELLATED OCTAHEDRON RENDERER CLASS
// =========================================================================

class StellatedOctahedronRenderer {
    /**
     * Core 3D rendering class for the stellated octahedron
     * (two interpenetrating tetrahedra forming the Merkaba)
     */
    constructor(container) {
        this.container = container;
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.clock = null;

        // Main container group - all objects rotate together
        this.containerGroup = null;

        // Groups for organized management
        this.groups = {
            tetA: null,
            tetB: null,
            vertices: null,
            sphere: null,
            effects: null,
        };

        // Mesh references
        this.vertexMeshes = {};
        this.faceMeshes = { tetA: [], tetB: [] };
        this.edgeLines = { tetA: [], tetB: [] };
        this.portApertureData = null;
        this.fieldOrigin = null;

        // Tetrahedron vertices
        this.tetAVertices = null;
        this.tetBVertices = null;
        this.vertexPositions = {};

        // Animation state
        this.animationId = null;
        this.isDragging = false;
        this.prevMouse = { x: 0, y: 0 };
        this.autoRotate = true;
        this.rotationSpeed = 0.06;
        this.rotationVelocity = { x: 0, y: 0 };
        this.dampingFactor = 0.95;

        // Axis lock
        this.axisLocked = false;
        this._lockedSpinAngle = 0;

        // Scale
        this.scale = 2.0;

        // Colors
        this.tetAColor = CONSTANTS.colors.tetA;
        this.tetBColor = CONSTANTS.colors.tetB;
    }

    // =====================================================================
    // INITIALIZATION
    // =====================================================================

    init() {
        this._createClock();
        this._createScene();
        this._createCamera();
        this._createRenderer();

        // If WebGL failed, bail out gracefully.
        if (this._webglFailed) return this;

        this._createLights();
        this._createContainerGroup();
        this._createGroups();
        this._createStellatedOctahedron();
        this._createEnclosingSphere();
        this._createPortApertureCircles();
        this._createFieldOrigin();
        this._createTriforceLines();
        this._setupControls();
        this._setupResizeHandler();
        this._startAnimation();

        return this;
    }

    _createClock() {
        this.clock = new THREE.Clock();
    }

    _createScene() {
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(CONSTANTS.colors.background);
    }

    _createCamera() {
        const aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera = new THREE.PerspectiveCamera(60, aspect, 0.1, 100);
        // Camera position (closer for tighter framing)
        this.camera.position.set(4.61, 3.46, 4.61);
        this.camera.lookAt(0, 0, 0);
    }

    _createRenderer() {
        // Try with full quality first, fall back to minimal settings if the
        // browser has run out of WebGL contexts (limit is ~8-16 per browser).
        var opts = [
            { antialias: true, alpha: true, powerPreference: 'default' },
            { antialias: false, alpha: false, powerPreference: 'low-power' }
        ];

        for (var i = 0; i < opts.length; i++) {
            try {
                this.renderer = new THREE.WebGLRenderer(opts[i]);
                break;
            } catch (e) {
                if (i === opts.length - 1) {
                    // All attempts failed.  Show a message instead of crashing.
                    console.error('[Stellated] WebGL not available:', e.message);
                    this.container.innerHTML =
                        '<div style="display:flex;align-items:center;justify-content:center;' +
                        'height:100%;color:#888;font-family:sans-serif;text-align:center;padding:24px;">' +
                        '<p>WebGL is not available. Please close other browser tabs and reload.</p></div>';
                    this._webglFailed = true;
                    return;
                }
                console.warn('[Stellated] WebGL context creation failed with options', i, ', retrying...');
            }
        }

        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
        this.container.appendChild(this.renderer.domElement);

        // Handle WebGL context loss (browsers may reclaim GPU resources
        // when the tab is backgrounded or under memory pressure).
        var self = this;
        this.renderer.domElement.addEventListener('webglcontextlost', function(e) {
            e.preventDefault();
            console.warn('[Stellated] WebGL context lost, pausing render loop.');
            if (self.animationId) cancelAnimationFrame(self.animationId);
        }, false);
        this.renderer.domElement.addEventListener('webglcontextrestored', function() {
            console.log('[Stellated] WebGL context restored, resuming.');
            self._startAnimation();
        }, false);
    }

    _createLights() {
        const ambient = new THREE.AmbientLight(0xffffff, 0.6);
        this.scene.add(ambient);

        const directional = new THREE.DirectionalLight(0xffffff, 0.4);
        directional.position.set(5, 10, 5);
        this.scene.add(directional);
    }

    _createContainerGroup() {
        this.containerGroup = new THREE.Group();
        this.containerGroup.name = 'stellated-octahedron-container';
        this.scene.add(this.containerGroup);
    }

    _createGroups() {
        Object.keys(this.groups).forEach(key => {
            this.groups[key] = new THREE.Group();
            this.groups[key].name = key;
            this.containerGroup.add(this.groups[key]);
        });
    }

    // =====================================================================
    // STELLATED OCTAHEDRON GEOMETRY
    // =====================================================================

    _createStellatedOctahedron() {
        const s = this.scale;

        // Build vertex position map from binary encoding
        Object.entries(CONSTANTS.vertices.encoding).forEach(([binary, data]) => {
            this.vertexPositions[binary] = new THREE.Vector3(
                data.x * s,
                data.y * s,
                data.z * s
            );
        });

        // Tet A vertices (ODD parity) - apex at 111
        this.tetAVertices = [
            this.vertexPositions['111'],  // APEX
            this.vertexPositions['001'],
            this.vertexPositions['010'],
            this.vertexPositions['100']
        ];

        // Tet B vertices (EVEN parity) - apex at 000
        this.tetBVertices = [
            this.vertexPositions['000'],  // APEX
            this.vertexPositions['011'],
            this.vertexPositions['101'],
            this.vertexPositions['110']
        ];

        // Create edges
        this._createTetrahedronEdges(this.tetAVertices, this.tetAColor, this.groups.tetA, 'tetA');
        this._createTetrahedronEdges(this.tetBVertices, this.tetBColor, this.groups.tetB, 'tetB');

        // Create faces
        this._createTetrahedronFaces(this.tetAVertices, this.tetAColor, 0.15, this.groups.tetA, 'A');
        this._createTetrahedronFaces(this.tetBVertices, this.tetBColor, 0.10, this.groups.tetB, 'B');

        // Create clock ticks on TIME faces
        this._createTimeFaceClockTicks(this.tetAVertices, this.tetAColor, this.groups.tetA);
        this._createTimeFaceClockTicks(this.tetBVertices, this.tetBColor, this.groups.tetB);
    }

    _createTetrahedronEdges(vertices, color, group, groupId) {
        const edges = [
            [0, 1], [0, 2], [0, 3],  // apex to base
            [1, 2], [2, 3], [3, 1]   // base triangle
        ];

        edges.forEach(([i, j]) => {
            const geometry = new THREE.BufferGeometry().setFromPoints([
                vertices[i], vertices[j]
            ]);
            const material = new THREE.LineBasicMaterial({
                color: color,
                transparent: true,
                opacity: 1.0
            });
            const line = new THREE.Line(geometry, material);
            group.add(line);
            this.edgeLines[groupId].push(line);
        });
    }

    _createTetrahedronFaces(vertices, color, opacity, group, tetId) {
        const faces = [
            [0, 1, 2],  // Side face
            [0, 2, 3],  // Side face
            [0, 3, 1],  // Side face
            [1, 3, 2]   // Base face (TIME)
        ];

        faces.forEach((faceIndices, faceIndex) => {
            const geometry = new THREE.BufferGeometry();
            const positions = new Float32Array([
                vertices[faceIndices[0]].x, vertices[faceIndices[0]].y, vertices[faceIndices[0]].z,
                vertices[faceIndices[1]].x, vertices[faceIndices[1]].y, vertices[faceIndices[1]].z,
                vertices[faceIndices[2]].x, vertices[faceIndices[2]].y, vertices[faceIndices[2]].z,
            ]);
            geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            geometry.computeVertexNormals();

            const material = new THREE.MeshBasicMaterial({
                color: color,
                transparent: true,
                opacity: opacity,
                side: THREE.DoubleSide,
                depthWrite: false
            });

            const mesh = new THREE.Mesh(geometry, material);
            mesh.userData = { tetId, faceIndex, originalOpacity: opacity };
            group.add(mesh);

            if (tetId === 'A') {
                this.faceMeshes.tetA.push(mesh);
            } else {
                this.faceMeshes.tetB.push(mesh);
            }
        });
    }

    _createTimeFaceClockTicks(vertices, color, group) {
        const v1 = vertices[1];
        const v2 = vertices[2];
        const v3 = vertices[3];

        const centroid = new THREE.Vector3()
            .add(v1).add(v2).add(v3).divideScalar(3);

        const edge1 = new THREE.Vector3().subVectors(v2, v1);
        const edge2 = new THREE.Vector3().subVectors(v3, v1);
        const normal = new THREE.Vector3().crossVectors(edge1, edge2).normalize();

        const clockCenter = centroid.clone().add(normal.clone().multiplyScalar(0.02));
        const distToVertex = centroid.distanceTo(v1);
        const clockRadius = distToVertex * 0.16;
        const innerRadius = clockRadius * 0.7;
        const outerRadius = clockRadius * 0.95;

        const xAxis = edge1.clone().normalize();
        const yAxis = new THREE.Vector3().crossVectors(normal, xAxis).normalize();

        // 12 tick marks
        for (let i = 0; i < 12; i++) {
            const angle = (i / 12) * Math.PI * 2;
            const cos = Math.cos(angle);
            const sin = Math.sin(angle);

            const innerPoint = clockCenter.clone()
                .add(xAxis.clone().multiplyScalar(cos * innerRadius))
                .add(yAxis.clone().multiplyScalar(sin * innerRadius));
            const outerPoint = clockCenter.clone()
                .add(xAxis.clone().multiplyScalar(cos * outerRadius))
                .add(yAxis.clone().multiplyScalar(sin * outerRadius));

            const tickGeo = new THREE.BufferGeometry().setFromPoints([innerPoint, outerPoint]);
            const tickMat = new THREE.LineBasicMaterial({
                color: color,
                transparent: true,
                opacity: 0.9
            });
            group.add(new THREE.Line(tickGeo, tickMat));
        }

        // Circle outline
        const circlePoints = [];
        for (let i = 0; i <= 32; i++) {
            const angle = (i / 32) * Math.PI * 2;
            circlePoints.push(clockCenter.clone()
                .add(xAxis.clone().multiplyScalar(Math.cos(angle) * clockRadius))
                .add(yAxis.clone().multiplyScalar(Math.sin(angle) * clockRadius)));
        }
        const circleGeo = new THREE.BufferGeometry().setFromPoints(circlePoints);
        const circleMat = new THREE.LineBasicMaterial({
            color: color,
            transparent: true,
            opacity: 0.5
        });
        group.add(new THREE.Line(circleGeo, circleMat));
    }

    // =====================================================================
    // ENCLOSING SPHERE
    // =====================================================================

    _createEnclosingSphere() {
        const sphereRadius = this.scale * Math.sqrt(3);
        const geometry = new THREE.SphereGeometry(sphereRadius, 24, 16);
        const material = new THREE.MeshBasicMaterial({
            color: 0x333333,
            wireframe: true,
            transparent: true,
            opacity: 0.08
        });
        const sphere = new THREE.Mesh(geometry, material);
        this.groups.sphere.add(sphere);
    }

    // =====================================================================
    // PORT APERTURE CIRCLES
    // =====================================================================

    _createPortApertureCircles() {
        this.portApertureData = {
            tetA: { circles: [] },
            tetB: { circles: [] },
            allMeshes: []
        };

        this._createPortCirclesForTet('A', this.tetAColor, 0.7, 0.1);
        this._createPortCirclesForTet('B', this.tetBColor, 0.35, 0.04);
    }

    _createPortCirclesForTet(tetId, color, lineOpacity, discOpacity) {
        const vertices = tetId === 'A' ? this.tetAVertices : this.tetBVertices;
        const sphereRadius = this.scale * Math.sqrt(3);
        const portAngularRadius = Math.acos(-1 / 3) / 2;

        const faceIndices = [
            [0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]
        ];

        for (let faceIndex = 0; faceIndex < 4; faceIndex++) {
            const face = faceIndices[faceIndex];
            const v0 = vertices[face[0]];
            const v1 = vertices[face[1]];
            const v2 = vertices[face[2]];

            const centroid = new THREE.Vector3().add(v0).add(v1).add(v2).divideScalar(3);
            const edge1 = new THREE.Vector3().subVectors(v1, v0);
            const edge2 = new THREE.Vector3().subVectors(v2, v0);
            const normal = new THREE.Vector3().crossVectors(edge1, edge2).normalize();

            if (normal.dot(centroid) < 0) normal.negate();

            const tangent1 = new THREE.Vector3();
            if (Math.abs(normal.x) < 0.9) {
                tangent1.set(1, 0, 0);
            } else {
                tangent1.set(0, 1, 0);
            }
            tangent1.cross(normal).normalize();
            const tangent2 = new THREE.Vector3().crossVectors(normal, tangent1).normalize();

            const circleRadius = sphereRadius * Math.sin(portAngularRadius);
            const offsetFromCenter = sphereRadius * Math.cos(portAngularRadius);
            const circleCenter = normal.clone().multiplyScalar(offsetFromCenter);

            // Circle line
            const circlePoints = [];
            for (let i = 0; i <= 48; i++) {
                const angle = (i / 48) * Math.PI * 2;
                circlePoints.push(circleCenter.clone()
                    .add(tangent1.clone().multiplyScalar(Math.cos(angle) * circleRadius))
                    .add(tangent2.clone().multiplyScalar(Math.sin(angle) * circleRadius)));
            }

            const circleGeo = new THREE.BufferGeometry().setFromPoints(circlePoints);
            const circleMat = new THREE.LineBasicMaterial({
                color: color,
                transparent: true,
                opacity: lineOpacity
            });
            const circleLine = new THREE.Line(circleGeo, circleMat);
            this.groups.sphere.add(circleLine);
            this.portApertureData.allMeshes.push(circleLine);

            // Filled disc
            const discGeo = new THREE.CircleGeometry(circleRadius * 0.95, 48);
            const discMat = new THREE.MeshBasicMaterial({
                color: color,
                transparent: true,
                opacity: discOpacity,
                side: THREE.DoubleSide,
                depthWrite: false
            });
            const disc = new THREE.Mesh(discGeo, discMat);
            disc.position.copy(circleCenter);
            disc.lookAt(circleCenter.clone().add(normal));
            this.groups.sphere.add(disc);
            this.portApertureData.allMeshes.push(disc);

            const tetData = tetId === 'A' ? this.portApertureData.tetA : this.portApertureData.tetB;
            tetData.circles.push({
                faceIndex,
                line: circleLine,
                disc: disc,
                originalColor: color,
                baseLineOpacity: lineOpacity,
                baseDiscOpacity: discOpacity,
                tetId
            });
        }
    }

    // =====================================================================
    // FIELD ORIGIN (Vortex Eye / Coupling Center)
    // =====================================================================
    // 
    // The Field Origin is the central point where both tetrahedra share
    // their geometric center - the coupling point of the bimetric field.
    //
    // Size proportions based on the 4% ratio derived from geometric analysis:
    // - Vertex distance from center = sqrt(3) * scale
    // - Field Origin radius = 4% of vertex distance
    //
    // =====================================================================

    _createFieldOrigin() {
        // Field origin is ~4% of the characteristic dimension
        const FIELD_ORIGIN_RATIO = 0.04;
        const vertexDistance = Math.sqrt(3) * this.scale;
        const originRadius = vertexDistance * FIELD_ORIGIN_RATIO;

        const geometry = new THREE.SphereGeometry(originRadius, 32, 32);
        const material = new THREE.MeshBasicMaterial({
            color: CONSTANTS.colors.fieldOrigin,
            transparent: true,
            opacity: 0.5,
            depthWrite: false
        });
        this.fieldOrigin = new THREE.Mesh(geometry, material);
        this.fieldOrigin.position.set(0, 0, 0);
        this.containerGroup.add(this.fieldOrigin);
    }

    // =====================================================================
    // TRIFORCE SUBDIVISION LINES
    // =====================================================================

    _createTriforceLines() {
        const faceVertexIndices = [
            [0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]
        ];

        // Tet A
        faceVertexIndices.forEach(indices => {
            const v0 = this.tetAVertices[indices[0]];
            const v1 = this.tetAVertices[indices[1]];
            const v2 = this.tetAVertices[indices[2]];
            this._addHeartTriangleLines(v0, v1, v2, 0x2255aa, 0.5);
        });

        // Tet B
        faceVertexIndices.forEach(indices => {
            const v0 = this.tetBVertices[indices[0]];
            const v1 = this.tetBVertices[indices[1]];
            const v2 = this.tetBVertices[indices[2]];
            this._addHeartTriangleLines(v0, v1, v2, 0xaa8800, 0.5);
        });
    }

    _addHeartTriangleLines(v0, v1, v2, color, opacity) {
        const m01 = new THREE.Vector3().addVectors(v0, v1).multiplyScalar(0.5);
        const m12 = new THREE.Vector3().addVectors(v1, v2).multiplyScalar(0.5);
        const m20 = new THREE.Vector3().addVectors(v2, v0).multiplyScalar(0.5);

        const edges = [[m01, m12], [m12, m20], [m20, m01]];

        edges.forEach(([a, b]) => {
            const geometry = new THREE.BufferGeometry().setFromPoints([a, b]);
            const material = new THREE.LineBasicMaterial({
                color: color,
                opacity: opacity,
                transparent: true
            });
            this.groups.effects.add(new THREE.Line(geometry, material));
        });
    }

    // =====================================================================
    // ANIMATION CONTROLS
    // =====================================================================

    _setupControls() {
        const domElement = this.renderer.domElement;

        domElement.addEventListener('mousedown', (e) => {
            this.isDragging = true;
            this.prevMouse.x = e.clientX;
            this.prevMouse.y = e.clientY;
            this.rotationVelocity.x = 0;
            this.rotationVelocity.y = 0;
        });

        domElement.addEventListener('mousemove', (e) => {
            if (!this.isDragging) return;

            const dx = e.clientX - this.prevMouse.x;
            const dy = e.clientY - this.prevMouse.y;

            if (this.axisLocked) {
                this._applyLockedYRotation(dx * 0.01);
                this.rotationVelocity.y = dx * 0.01;
            } else {
                this.containerGroup.rotation.y += dx * 0.01;
                this.containerGroup.rotation.x += dy * 0.01;
                this.rotationVelocity.x = dy * 0.01;
                this.rotationVelocity.y = dx * 0.01;
            }

            this.prevMouse.x = e.clientX;
            this.prevMouse.y = e.clientY;
        });

        domElement.addEventListener('mouseup', () => { this.isDragging = false; });
        domElement.addEventListener('mouseleave', () => { this.isDragging = false; });

        // No wheel zoom: camera position is fixed.  Scroll events pass
        // through to the page so the user can scroll normally.

        // Touch support
        domElement.addEventListener('touchstart', (e) => {
            if (e.touches.length === 1) {
                this.isDragging = true;
                this.prevMouse.x = e.touches[0].clientX;
                this.prevMouse.y = e.touches[0].clientY;
            }
        }, { passive: true });

        domElement.addEventListener('touchmove', (e) => {
            if (!this.isDragging || e.touches.length !== 1) return;
            e.preventDefault();

            const dx = e.touches[0].clientX - this.prevMouse.x;
            const dy = e.touches[0].clientY - this.prevMouse.y;

            if (this.axisLocked) {
                this._applyLockedYRotation(dx * 0.01);
            } else {
                this.containerGroup.rotation.y += dx * 0.01;
                this.containerGroup.rotation.x += dy * 0.01;
            }

            this.prevMouse.x = e.touches[0].clientX;
            this.prevMouse.y = e.touches[0].clientY;
        }, { passive: false });

        domElement.addEventListener('touchend', () => { this.isDragging = false; }, { passive: true });
    }

    _setupResizeHandler() {
        window.addEventListener('resize', () => {
            const width = this.container.clientWidth;
            const height = this.container.clientHeight;
            if (width === 0 || height === 0) return;

            this.camera.aspect = width / height;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(width, height);
        });
    }

    _startAnimation() {
        // Track visibility so the render loop pauses when scrolled off-screen.
        // This prevents the GPU from burning cycles on a hidden canvas and
        // starving the rest of the page (accordion clicks, scrolling, etc.).
        this._isVisible = true;

        if (typeof IntersectionObserver !== 'undefined') {
            var self = this;
            this._visibilityObserver = new IntersectionObserver(function(entries) {
                self._isVisible = entries[0].isIntersecting;
                // Resume the loop when the canvas scrolls back into view.
                if (self._isVisible && !self._loopRunning) {
                    self._loopRunning = true;
                    self.clock.getDelta(); // flush stale delta
                    requestAnimationFrame(self._animateFrame);
                }
            }, { threshold: 0.0 });
            this._visibilityObserver.observe(this.container);
        }

        var self = this;
        this._loopRunning = true;

        this._animateFrame = function() {
            // Stop the loop when the canvas is not visible.
            if (!self._isVisible) {
                self._loopRunning = false;
                return;
            }

            self.animationId = requestAnimationFrame(self._animateFrame);
            var deltaTime = self.clock.getDelta();

            if (!self.isDragging) {
                self.rotationVelocity.x *= self.dampingFactor;
                self.rotationVelocity.y *= self.dampingFactor;

                if (self.axisLocked) {
                    if (Math.abs(self.rotationVelocity.y) > 0.0001) {
                        self._applyLockedYRotation(self.rotationVelocity.y);
                    }
                    if (self.autoRotate && Math.abs(self.rotationVelocity.y) < 0.001) {
                        self._applyLockedYRotation(self.rotationSpeed * deltaTime);
                    }
                } else {
                    if (Math.abs(self.rotationVelocity.x) > 0.0001) {
                        self.containerGroup.rotation.x += self.rotationVelocity.x;
                    }
                    if (Math.abs(self.rotationVelocity.y) > 0.0001) {
                        self.containerGroup.rotation.y += self.rotationVelocity.y;
                    }
                    if (self.autoRotate &&
                        Math.abs(self.rotationVelocity.x) < 0.001 &&
                        Math.abs(self.rotationVelocity.y) < 0.001) {
                        self.containerGroup.rotation.y += self.rotationSpeed * deltaTime;
                    }
                }
            }

            self.renderer.render(self.scene, self.camera);
        };

        requestAnimationFrame(this._animateFrame);
    }

    // =====================================================================
    // AXIS LOCK
    // =====================================================================

    lockTimeFloor() {
        this.axisLocked = true;
        this._lockedSpinAngle = 0;
        this._updateLockedOrientation();
        this.rotationVelocity.x = 0;
        this.rotationVelocity.y = 0;
    }

    unlockAxis() {
        this.axisLocked = false;
        this._lockedSpinAngle = 0;
    }

    toggleAxisLock() {
        if (this.axisLocked) {
            this.unlockAxis();
        } else {
            this.lockTimeFloor();
        }
        return this.axisLocked;
    }

    _updateLockedOrientation() {
        const apexDirection = new THREE.Vector3(1, 1, 1).normalize();
        const upDirection = new THREE.Vector3(0, 1, 0);
        const alignQuat = new THREE.Quaternion().setFromUnitVectors(apexDirection, upDirection);
        const spinQuat = new THREE.Quaternion().setFromAxisAngle(
            new THREE.Vector3(0, 1, 0),
            this._lockedSpinAngle
        );
        this.containerGroup.quaternion.copy(alignQuat).premultiply(spinQuat);
    }

    _applyLockedYRotation(deltaY) {
        this._lockedSpinAngle += deltaY;
        this._updateLockedOrientation();
    }

    resetCamera() {
        this.camera.position.set(4.61, 3.46, 4.61);
        this.camera.lookAt(0, 0, 0);
        this.containerGroup.rotation.set(0, 0, 0);
        this.containerGroup.quaternion.set(0, 0, 0, 1);
        this.rotationVelocity.x = 0;
        this.rotationVelocity.y = 0;
        this._lockedSpinAngle = 0;
        if (this.axisLocked) {
            this._updateLockedOrientation();
        }
    }

    // =====================================================================
    // VISIBILITY CONTROL
    // =====================================================================

    /**
     * Set visibility level for progressive reveal
     * Level 1: Tetrahedron A only
     * Level 2: Tetrahedron A + B
     * Level 3: Both Tetrahedra + Field Origin
     * Level 4: Everything (Tetrahedra + Origin + Ports)
     */
    setVisibilityLevel(level) {
        // Tetrahedron A - always visible at level 1+
        this.groups.tetA.visible = level >= 1;

        // Tetrahedron B - visible at level 2+
        this.groups.tetB.visible = level >= 2;

        // Field Origin - visible at level 3+
        if (this.fieldOrigin) {
            this.fieldOrigin.visible = level >= 3;
        }

        // Coupling Ports (aperture circles on sphere) - visible at level 4
        this.groups.sphere.visible = level >= 4;

        // Triforce lines follow the tetrahedra visibility
        this.groups.effects.visible = level >= 2;
    }

    setTetAVisible(visible) {
        this.groups.tetA.visible = visible;
    }

    setTetBVisible(visible) {
        this.groups.tetB.visible = visible;
    }

    setFieldOriginVisible(visible) {
        if (this.fieldOrigin) {
            this.fieldOrigin.visible = visible;
        }
    }

    setPortsVisible(visible) {
        this.groups.sphere.visible = visible;
    }

    // =====================================================================
    // FACE & PORT HIGHLIGHTING
    // =====================================================================

    pulseFace(tetId, faceIndex, duration = 1000) {
        const meshes = tetId === 'A' ? this.faceMeshes.tetA : this.faceMeshes.tetB;
        const mesh = meshes[faceIndex];
        if (!mesh) return;

        const baseColor = tetId === 'A' ? this.tetAColor : this.tetBColor;
        const origColor = new THREE.Color(baseColor);
        const targetColor = tetId === 'A' ? new THREE.Color(0x6da3cc) : new THREE.Color(0xccbe52);
        const baseOp = mesh.userData.originalOpacity;
        const maxOp = tetId === 'A' ? 0.64 : 0.48;

        const startTime = performance.now();
        const flashDur = duration * 0.25;
        const fadeDur = duration * 0.75;

        const animate = (t) => {
            const elapsed = t - startTime;

            if (elapsed < flashDur) {
                const p = elapsed / flashDur;
                mesh.material.color.lerpColors(origColor, targetColor, p);
                mesh.material.opacity = baseOp + (maxOp - baseOp) * p;
            } else if (elapsed < flashDur + fadeDur) {
                const p = (elapsed - flashDur) / fadeDur;
                mesh.material.color.lerpColors(targetColor, origColor, p);
                mesh.material.opacity = maxOp - (maxOp - baseOp) * p;
            } else {
                mesh.material.color.copy(origColor);
                mesh.material.opacity = baseOp;
                return;
            }
            requestAnimationFrame(animate);
        };
        requestAnimationFrame(animate);
    }

    pulsePortAperture(tetId, faceIndex, duration = 1000) {
        const tetData = tetId === 'A' ? this.portApertureData.tetA : this.portApertureData.tetB;
        if (!tetData || !tetData.circles[faceIndex]) return;

        const circleData = tetData.circles[faceIndex];
        const origColor = new THREE.Color(circleData.originalColor);
        const targetColor = tetId === 'A' ? new THREE.Color(0xcccccc) : new THREE.Color(0xcccc88);
        const maxLineOp = tetId === 'A' ? 1.0 : 0.7;
        const maxDiscOp = tetId === 'A' ? 0.6 : 0.3;

        const startTime = performance.now();
        const flashDur = duration * 0.25;
        const fadeDur = duration * 0.75;

        const animate = (t) => {
            const elapsed = t - startTime;

            if (elapsed < flashDur) {
                const p = elapsed / flashDur;
                circleData.line.material.color.lerpColors(origColor, targetColor, p);
                circleData.line.material.opacity = circleData.baseLineOpacity + (maxLineOp - circleData.baseLineOpacity) * p;
                circleData.disc.material.color.lerpColors(origColor, targetColor, p);
                circleData.disc.material.opacity = circleData.baseDiscOpacity + (maxDiscOp - circleData.baseDiscOpacity) * p;
            } else if (elapsed < flashDur + fadeDur) {
                const p = (elapsed - flashDur) / fadeDur;
                circleData.line.material.color.lerpColors(targetColor, origColor, p);
                circleData.line.material.opacity = maxLineOp - (maxLineOp - circleData.baseLineOpacity) * p;
                circleData.disc.material.color.lerpColors(targetColor, origColor, p);
                circleData.disc.material.opacity = maxDiscOp - (maxDiscOp - circleData.baseDiscOpacity) * p;
            } else {
                circleData.line.material.color.copy(origColor);
                circleData.line.material.opacity = circleData.baseLineOpacity;
                circleData.disc.material.color.copy(origColor);
                circleData.disc.material.opacity = circleData.baseDiscOpacity;
                return;
            }
            requestAnimationFrame(animate);
        };
        requestAnimationFrame(animate);
    }

    pulseFieldOrigin(duration = 800, color = 0xffffff) {
        if (!this.fieldOrigin) return;

        const startTime = performance.now();
        const pulseColor = new THREE.Color(color);

        const animate = (t) => {
            const elapsed = t - startTime;
            const progress = Math.min(elapsed / duration, 1);

            let intensity;
            if (progress < 0.3) {
                intensity = progress / 0.3;
            } else {
                intensity = 1 - ((progress - 0.3) / 0.7);
            }

            const opacity = 0.5 + 0.4 * intensity;

            // Only change opacity and color, NOT scale
            this.fieldOrigin.material.opacity = opacity;
            this.fieldOrigin.material.color.copy(pulseColor);

            if (progress < 1) {
                requestAnimationFrame(animate);
            } else {
                this.fieldOrigin.material.opacity = 0.5;
            }
        };
        requestAnimationFrame(animate);
    }
}

// =========================================================================
// FIELD CLOSURE ENGINE
// Computes closure cycles dynamically from Pi digits (base 8)
// Supports: Full closure mode, Step-by-step mode, Manual mode
// =========================================================================

class FieldClosureEngine {
    constructor(renderer) {
        this.renderer = renderer;
        this.isRunning = false;
        this.isManualMode = false;
        this.speedMultiplier = 1;
        this.portTimer = null;
        this.animationFrameId = null;

        // Timing
        this.pulseDuration = 1200;
        this.cyclePauseDelay = 800;

        // Pi traversal state
        this.piDigits = CONSTANTS.pi.digits_base8;
        this.faceVertices = CONSTANTS.faceVertices;
        this.vertexToFaces = CONSTANTS.vertexToFaces;
        this.vertexInfo = CONSTANTS.vertexInfo;
        this.currentPiIndex = 0;
        this.currentCycleIndex = 0;
        this.totalCycles = 10;

        // Cumulative sum for vertex calculation (vertex = cumsum % 8)
        this.cumsum = 0;

        // Track face completion state
        // Each face needs all 3 of its vertices visited to complete
        this.faceVisitedVertices = {};  // face -> Set of visited vertices
        this.completedFaces = new Set();

        // Track accumulated steps for display
        this.cycleStartPiIndex = 0;
        this.cycleStartCumsum = 0;
        this.cycleSteps = [];  // Array of step info
        this.cycleFaces = [];  // Array of completed face info
    }

    start() {
        if (this.isManualMode) return;
        this.isRunning = true;
        this.runFullClosure();
    }

    stop() {
        this.isRunning = false;
        if (this.portTimer) {
            clearTimeout(this.portTimer);
            this.portTimer = null;
        }
        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }
    }

    setManualMode(enabled) {
        this.isManualMode = enabled;
        if (enabled) {
            this.stop();
        }
    }

    setSpeed(multiplier) {
        const wasManual = this.isManualMode;
        this.isManualMode = (multiplier === 0);
        this.speedMultiplier = multiplier || 1;

        if (wasManual && !this.isManualMode) {
            // Switching from manual to auto - restart animation
            this.start();
        } else if (!wasManual && this.isManualMode) {
            // Switching to manual - stop animation
            this.stop();
        }
    }

    getAdjustedTime(baseTime) {
        return baseTime / this.speedMultiplier;
    }

    /**
     * Step through one Pi digit at a time
     * Updates cumsum and calculates vertex = cumsum % 8
     * Tracks which vertices each face has seen - face completes when all 3 visited
     * Returns true if closure completed, false otherwise
     * 
     * STEP MODE FIELD CLOSURE:
     * The traversal uses cumulative sum modulo 8 to determine the current vertex:
     *   cumsum += pi_digit
     *   vertex = cumsum % 8
     * 
     * A face COMPLETES when all 3 of its constituent vertices have been visited
     * in the current cycle. A closure completes when all 8 faces have completed.
     */
    stepOnce() {
        if (this.currentPiIndex >= this.piDigits.length) {
            // Reset to beginning when Pi digits exhausted
            this.resetCycle();
            this.currentPiIndex = 0;
            this.cumsum = 0;
            this.currentCycleIndex = 0;
        }

        const digit = this.piDigits[this.currentPiIndex];

        // CORRECT: Vertex is cumsum % 8, not the raw digit
        this.cumsum += digit;
        const vertex = this.cumsum % 8;
        const vertexInfo = this.vertexInfo[vertex];

        // Track step for display
        const stepInfo = {
            piIndex: this.currentPiIndex,
            digit: digit,
            cumsum: this.cumsum,
            vertex: vertex,
            tetId: vertexInfo.tetId,
            newlyCompletedFaces: []
        };
        this.cycleSteps.push(stepInfo);

        // ---------------------------------------------------------
        // FACE COMPLETION TRACKING
        // When a vertex is visited, mark it as seen for all faces
        // containing that vertex. A face completes when all 3 vertices visited.
        // ---------------------------------------------------------
        const facesContainingVertex = this.vertexToFaces[vertex];

        for (const faceName of facesContainingVertex) {
            // Initialize tracking for this face if needed
            if (!this.faceVisitedVertices[faceName]) {
                this.faceVisitedVertices[faceName] = new Set();
            }

            // Mark this vertex as visited for this face
            this.faceVisitedVertices[faceName].add(vertex);

            // Check if face is now complete (all 3 vertices visited)
            const faceData = this.faceVertices[faceName];
            const visitedCount = this.faceVisitedVertices[faceName].size;

            if (visitedCount === 3 && !this.completedFaces.has(faceName)) {
                // Face just completed!
                this.completedFaces.add(faceName);
                stepInfo.newlyCompletedFaces.push(faceName);

                this.cycleFaces.push({
                    name: faceName,
                    tetId: faceData.tetId,
                    role: faceData.role,
                    faceIdx: faceData.faceIdx,
                    completedAtStep: this.cycleSteps.length,
                    completedAtPiIndex: this.currentPiIndex,
                    triggerVertex: vertex
                });

                // Pulse the completed face
                this.pulseSingleFace({
                    name: faceName,
                    tetId: faceData.tetId,
                    faceIdx: faceData.faceIdx,
                    role: faceData.role
                }, 600);
            }
        }

        // Update display
        this.updateStepStatusDisplay(stepInfo);

        this.currentPiIndex++;

        // ---------------------------------------------------------
        // CLOSURE COMPLETION CHECK
        // When all 8 faces have completed, closure is complete
        // ---------------------------------------------------------
        if (this.completedFaces.size === 8) {
            // =====================================================
            // VALIDATE STEP MODE CLOSURE
            // =====================================================
            this.validateClosureCycle(
                this.cycleFaces,
                this.completedFaces,
                this.currentCycleIndex + 1
            );

            this.currentCycleIndex++;
            if (this.currentCycleIndex >= this.totalCycles) {
                this.currentCycleIndex = 0;
            }
            // Reset for next cycle after brief delay
            setTimeout(() => this.resetCycle(), 800);
            return true;
        }

        return false;
    }

    resetCycle() {
        this.cycleStartPiIndex = this.currentPiIndex;
        this.cycleStartCumsum = this.cumsum;
        this.faceVisitedVertices = {};
        this.completedFaces = new Set();
        this.cycleSteps = [];
        this.cycleFaces = [];
    }

    /**
     * Run full closure mode - all 8 faces pulse together
     */
    runFullClosure() {
        if (!this.isRunning || this.isManualMode) return;

        // Reset cycle tracking
        this.resetCycle();

        // Compute all faces for current closure
        const faces = this.computeCurrentClosureFaces();

        this.updateFullClosureStatusDisplay(faces);

        // Pulse all 8 faces simultaneously
        this.pulseAllFaces(faces, this.getAdjustedTime(this.pulseDuration));

        // Schedule next closure
        this.portTimer = setTimeout(() => {
            this.currentCycleIndex = (this.currentCycleIndex + 1) % this.totalCycles;
            if (this.currentCycleIndex === 0) {
                // Reset to beginning of pi sequence
                this.currentPiIndex = 0;
                this.cumsum = 0;
            }
            this.runFullClosure();
        }, this.getAdjustedTime(this.pulseDuration + this.cyclePauseDelay));
    }

    /**
     * Compute all faces for one complete closure cycle.
     * 
     * CLOSURE GUARANTEES:
     * This function ensures that each closure satisfies the fundamental
     * requirements:
     * 
     * 1. ALL 8 FACES ACTIVATED: Every closure must activate all 8 faces
     *    (4 from Tet A + 4 from Tet B) before completion.
     * 
     * 2. NO DUPLICATE COUNTING: Each face is counted exactly once per closure.
     *    If a vertex is visited multiple times, it is only counted on
     *    first activation. Duplicates advance the Pi sequence but don't
     *    add to the closure.
     * 
     * 3. BIMETRIC BALANCE: Must have exactly 4 Tet A and 4 Tet B faces.
     *    This ensures field balance between the two tetrahedra.
     * 
     * 4. TIME FACE INCLUSION: Both TIME faces (FACE_A7 at vertex 7, FACE_B0 at
     *    vertex 0) must be included, ensuring proper temporal coupling.
     * 
     * The validation runs on every closure and logs errors to console if
     * any guarantee is violated (which should never happen with valid Pi digits).
     */
    computeCurrentClosureFaces() {
        const completedFaces = [];
        const completed = new Set();
        const faceVisited = {};  // face -> Set of visited vertices
        let idx = this.currentPiIndex;
        let cumsum = this.cumsum;

        // =========================================================
        // CLOSURE COMPUTATION
        // Traverse Pi digits until all 8 faces have completed
        // A face completes when all 3 of its vertices have been visited
        // =========================================================
        while (completed.size < 8 && idx < this.piDigits.length) {
            const digit = this.piDigits[idx];
            cumsum += digit;
            const vertex = cumsum % 8;

            // Mark this vertex as visited for all faces containing it
            const facesContainingVertex = this.vertexToFaces[vertex];

            for (const faceName of facesContainingVertex) {
                if (!faceVisited[faceName]) {
                    faceVisited[faceName] = new Set();
                }
                faceVisited[faceName].add(vertex);

                // Check if face just completed
                if (faceVisited[faceName].size === 3 && !completed.has(faceName)) {
                    completed.add(faceName);
                    const faceData = this.faceVertices[faceName];
                    completedFaces.push({
                        name: faceName,
                        tetId: faceData.tetId,
                        role: faceData.role,
                        faceIdx: faceData.faceIdx,
                        vertex: vertex,  // Trigger vertex
                        piStep: idx
                    });
                }
            }
            idx++;
        }

        // Update state for next closure
        this.currentPiIndex = idx;
        this.cumsum = cumsum;
        this.cycleFaces = completedFaces;

        // =========================================================
        // FIELD CLOSURE VALIDATION
        // Verify all guarantees are satisfied for this closure
        // =========================================================
        this.validateClosureCycle(completedFaces, completed, this.currentCycleIndex + 1);

        return completedFaces;
    }

    /**
     * CLOSURE VALIDATION SYSTEM
     * =========================
     * 
     * This validation runs on EVERY closure cycle to guarantee the integrity
     * of the Pi-driven field closure mechanism. Any violation is logged to
     * the console for debugging.
     * 
     * VALIDATIONS:
     * 
     * 1. COMPLETENESS: All 8 faces must be activated
     *    - If fewer than 8 faces, closure is incomplete (error)
     * 
     * 2. UNIQUENESS: Each face appears exactly once
     *    - Set size must equal array length (no duplicates)
     * 
     * 3. BIMETRIC BALANCE: Exactly 4 Tet A faces + 4 Tet B faces
     *    - Ensures field balance between the two tetrahedra
     * 
     * 4. TIME FACES: Both TIME faces (FACE_A7, FACE_B0) must be present
     *    - Required for proper temporal coupling in the structure
     * 
     * 5. SPATIAL FACES: All 6 spatial faces must be present
     *    - Required for full 3D spatial coupling
     * 
     * @param {Array} faces - Array of face objects activated in this closure
     * @param {Set} activatedSet - Set of face names for uniqueness check
     * @param {number} cycleNum - Current cycle number for error reporting
     */
    validateClosureCycle(faces, activatedSet, cycleNum) {
        const errors = [];

        // ---------------------------------------------------------
        // GUARANTEE 1: All 8 faces must be activated (completeness)
        // ---------------------------------------------------------
        if (faces.length !== 8) {
            errors.push(`COMPLETENESS VIOLATION: Expected 8 faces, got ${faces.length}`);
        }

        // ---------------------------------------------------------
        // GUARANTEE 2: No duplicate faces (uniqueness)
        // The Set size should equal the array length
        // ---------------------------------------------------------
        if (activatedSet.size !== faces.length) {
            errors.push(`UNIQUENESS VIOLATION: Set size ${activatedSet.size} != array length ${faces.length}`);
        }

        // ---------------------------------------------------------
        // GUARANTEE 3: Bimetric balance (4 Tet A + 4 Tet B)
        // ---------------------------------------------------------
        const tetACounts = faces.filter(f => f.tetId === 'A').length;
        const tetBCounts = faces.filter(f => f.tetId === 'B').length;

        if (tetACounts !== 4) {
            errors.push(`BIMETRIC VIOLATION: Expected 4 Tet A faces, got ${tetACounts}`);
        }
        if (tetBCounts !== 4) {
            errors.push(`BIMETRIC VIOLATION: Expected 4 Tet B faces, got ${tetBCounts}`);
        }

        // ---------------------------------------------------------
        // GUARANTEE 4: Both TIME faces must be present
        // TIME faces are at vertices 7 (Tet A apex) and 0 (Tet B apex)
        // ---------------------------------------------------------
        const faceNames = new Set(faces.map(f => f.name));
        const hasTimeA = faceNames.has('FACE_A7');  // TIME face from Tet A
        const hasTimeB = faceNames.has('FACE_B0');  // TIME face from Tet B

        if (!hasTimeA) {
            errors.push('TIME FACE VIOLATION: Missing Tet A TIME face (A7)');
        }
        if (!hasTimeB) {
            errors.push('TIME FACE VIOLATION: Missing Tet B TIME face (B0)');
        }

        // ---------------------------------------------------------
        // GUARANTEE 5: All 6 SPATIAL faces must be present
        // SPATIAL faces: A1, A2, A4 (Tet A) and B3, B5, B6 (Tet B)
        // ---------------------------------------------------------
        const spatialFaces = ['FACE_A1', 'FACE_A2', 'FACE_A4', 'FACE_B3', 'FACE_B5', 'FACE_B6'];
        const missingSpatial = spatialFaces.filter(sf => !faceNames.has(sf));

        if (missingSpatial.length > 0) {
            errors.push(`SPATIAL FACE VIOLATION: Missing faces: ${missingSpatial.join(', ')}`);
        }

        // ---------------------------------------------------------
        // REPORT VALIDATION RESULTS
        // ---------------------------------------------------------
        if (errors.length > 0) {
            // Log all errors as a group
            console.error(`[FIELD CLOSURE VALIDATION FAILED] Cycle ${cycleNum}:`);
            errors.forEach(err => console.error(`  - ${err}`));
            console.error('  Faces activated:', Array.from(faceNames).sort().join(', '));

            // This should NEVER happen with valid Pi digits
            // If it does, there's a bug in the algorithm
        } else {
            // Validation passed - log success at debug level
            // Uncomment the next line to see successful validations:
            // console.log(`[FIELD CLOSURE VALIDATED] Cycle ${cycleNum}: All 8 faces (4A+4B), both TIME faces present`);
        }

        return errors.length === 0;
    }

    /**
     * Pulse a single face (for step mode)
     */
    pulseSingleFace(face, duration) {
        const tetId = face.tetId;
        const faceIdx = face.faceIdx;

        // Get mesh
        const meshes = tetId === 'A'
            ? this.renderer.faceMeshes.tetA
            : this.renderer.faceMeshes.tetB;
        const mesh = meshes?.[faceIdx];

        // Get port aperture
        const tetPortData = tetId === 'A'
            ? this.renderer.portApertureData?.tetA
            : this.renderer.portApertureData?.tetB;
        const circle = tetPortData?.circles?.[faceIdx];

        if (!mesh) return;

        const baseColor = tetId === 'A' ? this.renderer.tetAColor : this.renderer.tetBColor;
        const origColor = new THREE.Color(baseColor);
        const targetColor = tetId === 'A' ? new THREE.Color(0x6da3cc) : new THREE.Color(0xccbe52);
        const baseOp = mesh.userData.originalOpacity;
        const maxOp = tetId === 'A' ? 0.68 : 0.56;

        // Field Origin for TIME faces
        const fieldOrigin = this.renderer.fieldOrigin;
        const isTimeFace = face.role === 'TIME';

        const startTime = performance.now();
        const flashDur = duration * 0.3;
        const fadeDur = duration * 0.7;

        const animate = (now) => {
            const elapsed = now - startTime;
            let intensity;

            if (elapsed < flashDur) {
                intensity = elapsed / flashDur;
            } else if (elapsed < flashDur + fadeDur) {
                intensity = 1 - ((elapsed - flashDur) / fadeDur);
            } else {
                intensity = 0;
            }

            // Animate mesh
            mesh.material.color.lerpColors(origColor, targetColor, intensity);
            mesh.material.opacity = baseOp + (maxOp - baseOp) * intensity;

            // Animate port aperture
            if (circle) {
                const portOrig = new THREE.Color(circle.originalColor);
                const portTarget = tetId === 'A' ? new THREE.Color(0xcccccc) : new THREE.Color(0xcccc88);
                circle.line.material.color.lerpColors(portOrig, portTarget, intensity);
                circle.line.material.opacity = circle.baseLineOpacity + (1.0 - circle.baseLineOpacity) * intensity;
                circle.disc.material.color.lerpColors(portOrig, portTarget, intensity);
                circle.disc.material.opacity = circle.baseDiscOpacity + (0.5 - circle.baseDiscOpacity) * intensity;
            }

            // Field Origin for TIME faces - opacity only, no scale
            if (isTimeFace && fieldOrigin) {
                fieldOrigin.material.opacity = 0.5 + 0.4 * intensity;
            }

            if (elapsed < flashDur + fadeDur) {
                requestAnimationFrame(animate);
            } else {
                // Reset
                mesh.material.color.copy(origColor);
                mesh.material.opacity = baseOp;
                if (circle) {
                    circle.line.material.color.set(circle.originalColor);
                    circle.line.material.opacity = circle.baseLineOpacity;
                    circle.disc.material.color.set(circle.originalColor);
                    circle.disc.material.opacity = circle.baseDiscOpacity;
                }
                if (isTimeFace && fieldOrigin) {
                    fieldOrigin.material.opacity = 0.5;
                }
            }
        };

        requestAnimationFrame(animate);
    }

    /**
     * Pulse all faces for full closure simultaneously
     */
    pulseAllFaces(faces, duration) {
        const allMeshData = [];
        const allPortData = [];

        faces.forEach(face => {
            const tetId = face.tetId;
            const faceIdx = face.faceIdx;

            const meshes = tetId === 'A'
                ? this.renderer.faceMeshes.tetA
                : this.renderer.faceMeshes.tetB;
            const mesh = meshes?.[faceIdx];

            if (mesh) {
                const baseColor = tetId === 'A' ? this.renderer.tetAColor : this.renderer.tetBColor;
                allMeshData.push({
                    mesh,
                    origColor: new THREE.Color(baseColor),
                    targetColor: tetId === 'A' ? new THREE.Color(0x6da3cc) : new THREE.Color(0xccbe52),
                    baseOp: mesh.userData.originalOpacity,
                    maxOp: tetId === 'A' ? 0.68 : 0.56
                });
            }

            const tetPortData = tetId === 'A'
                ? this.renderer.portApertureData?.tetA
                : this.renderer.portApertureData?.tetB;
            const circle = tetPortData?.circles?.[faceIdx];

            if (circle) {
                allPortData.push({
                    line: circle.line,
                    disc: circle.disc,
                    origColor: new THREE.Color(circle.originalColor),
                    targetColor: tetId === 'A' ? new THREE.Color(0xcccccc) : new THREE.Color(0xcccc88),
                    baseLineOp: circle.baseLineOpacity,
                    baseDiscOp: circle.baseDiscOpacity,
                    maxLineOp: 1.0,
                    maxDiscOp: 0.5
                });
            }
        });

        const fieldOrigin = this.renderer.fieldOrigin;

        const startTime = performance.now();
        const flashDur = duration * 0.3;
        const fadeDur = duration * 0.7;

        const animate = (now) => {
            if (!this.isRunning) return;

            const elapsed = now - startTime;
            let intensity;

            if (elapsed < flashDur) {
                intensity = elapsed / flashDur;
            } else if (elapsed < flashDur + fadeDur) {
                intensity = 1 - ((elapsed - flashDur) / fadeDur);
            } else {
                intensity = 0;
            }

            allMeshData.forEach(m => {
                m.mesh.material.color.lerpColors(m.origColor, m.targetColor, intensity);
                m.mesh.material.opacity = m.baseOp + (m.maxOp - m.baseOp) * intensity;
            });

            allPortData.forEach(p => {
                p.line.material.color.lerpColors(p.origColor, p.targetColor, intensity);
                p.line.material.opacity = p.baseLineOp + (p.maxLineOp - p.baseLineOp) * intensity;
                p.disc.material.color.lerpColors(p.origColor, p.targetColor, intensity);
                p.disc.material.opacity = p.baseDiscOp + (p.maxDiscOp - p.baseDiscOp) * intensity;
            });

            // Field Origin - opacity only, no scale
            if (fieldOrigin) {
                fieldOrigin.material.opacity = 0.5 + 0.4 * intensity;
            }

            if (elapsed < flashDur + fadeDur) {
                this.animationFrameId = requestAnimationFrame(animate);
            } else {
                allMeshData.forEach(m => {
                    m.mesh.material.color.copy(m.origColor);
                    m.mesh.material.opacity = m.baseOp;
                });
                allPortData.forEach(p => {
                    p.line.material.color.copy(p.origColor);
                    p.line.material.opacity = p.baseLineOp;
                    p.disc.material.color.copy(p.origColor);
                    p.disc.material.opacity = p.baseDiscOp;
                });
                if (fieldOrigin) {
                    fieldOrigin.material.opacity = 0.5;
                }
            }
        };

        this.animationFrameId = requestAnimationFrame(animate);
    }

    /**
     * Update display for step mode - shows each step as it occurs
     * Now shows: step# | pi[idx]=digit | cumsum | vertex | faces touched/completed
     */
    updateStepStatusDisplay(stepInfo) {
        const container = document.getElementById('closure-steps');
        if (!container) return;

        const stepNum = this.cycleSteps.length;
        const tetClass = stepInfo.tetId === 'A' ? 'tet-a' : 'tet-b';

        // Build completed faces indicator
        let facesHtml = '';
        if (stepInfo.newlyCompletedFaces.length > 0) {
            for (const faceName of stepInfo.newlyCompletedFaces) {
                const faceData = this.faceVertices[faceName];
                const faceClass = faceData.tetId === 'A' ? 'tet-a' : 'tet-b';
                const isTime = faceData.role === 'TIME';
                facesHtml += `<span class="step-face ${faceClass}">${faceName}</span>`;
                if (isTime) facesHtml += `<span class="step-time">★</span>`;
            }
        }

        let stepHtml = `<div class="closure-step active">`;
        stepHtml += `<span class="step-num">${stepNum}.</span>`;
        stepHtml += `<span class="step-pi">π[${stepInfo.piIndex}]=${stepInfo.digit}</span>`;
        stepHtml += `<span class="step-arrow">→</span>`;
        stepHtml += `<span class="step-cumsum">Σ=${stepInfo.cumsum}</span>`;
        stepHtml += `<span class="step-arrow">→</span>`;
        stepHtml += `<span class="step-vertex ${tetClass}">V${stepInfo.vertex}</span>`;
        if (facesHtml) {
            stepHtml += `<span class="step-arrow">→</span>`;
            stepHtml += facesHtml;
        }
        stepHtml += `<span class="step-count">[${this.completedFaces.size}/8]</span>`;
        stepHtml += `</div>`;

        // If this is the first step in cycle, clear and add header
        if (stepNum === 1) {
            container.innerHTML = `<div class="closure-mode-label">Step Mode - Cycle ${this.currentCycleIndex + 1}</div>` + stepHtml;
        } else {
            // Remove active class from previous steps and add new step
            container.querySelectorAll('.closure-step').forEach(el => el.classList.remove('active'));
            container.innerHTML += stepHtml;
        }

        // Scroll to show latest step if needed
        container.scrollTop = container.scrollHeight;
    }

    /**
     * Update display for full closure mode - shows all 8 faces in order
     */
    updateFullClosureStatusDisplay(faces) {
        const container = document.getElementById('closure-steps');
        if (!container) return;

        let html = `<div class="closure-mode-label">Full Closure - Cycle ${this.currentCycleIndex + 1}</div>`;

        faces.forEach((face, idx) => {
            const tetClass = face.tetId === 'A' ? 'tet-a' : 'tet-b';
            const isTime = face.role === 'TIME';

            html += `<div class="closure-step">`;
            html += `<span class="step-num">${idx + 1}.</span>`;
            html += `<span class="step-pi">π${face.piStep}</span>`;
            html += `<span class="step-arrow">=</span>`;
            html += `<span class="step-pi">${face.vertex}</span>`;
            html += `<span class="step-arrow">→</span>`;
            html += `<span class="step-face ${tetClass}">${face.name}</span>`;
            if (isTime) html += `<span class="step-time">★</span>`;
            html += `</div>`;
        });

        container.innerHTML = html;
    }
}

