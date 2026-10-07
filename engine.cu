#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cuda_runtime.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#define CUDA_CHECK(x) do {                                              \
    cudaError_t _e = (x);                                               \
    if (_e != cudaSuccess) {                                            \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                       \
                __FILE__, __LINE__, cudaGetErrorString(_e));            \
        std::exit(1);                                                   \
    }                                                                   \
} while (0)

#define GL_CHECK(x) do { (x); } while (0)

struct Vec3 {
    float x, y, z;
    __host__ __device__ Vec3(float x_ = 0.f, float y_ = 0.f, float z_ = 0.f)
        : x(x_), y(y_), z(z_) {}
    __host__ __device__ Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    __host__ __device__ Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    __host__ __device__ Vec3 operator*(float s) const        { return {x * s, y * s, z * s}; }
    __host__ __device__ Vec3& operator+=(const Vec3& o)      { x += o.x; y += o.y; z += o.z; return *this; }
    __host__ __device__ float lengthSq() const { return x*x + y*y + z*z; }
    __host__ __device__ float length()   const { return sqrtf(lengthSq()); }
};

using Entity = uint32_t;
static Entity NEXT_ID = 1;
static Entity createEntity() { return NEXT_ID++; }
template <typename T>
struct ComponentStorage {
    std::unordered_map<Entity, T> data;
    void add(Entity e, const T& c) { data[e] = c; }
    T* find(Entity e) {
        auto it = data.find(e);
        return (it == data.end()) ? nullptr : &it->second;
    }

    T& get(Entity e) {
        auto it = data.find(e);
        if (it == data.end()) {
            fprintf(stderr, "ComponentStorage::get: missing entity %u\n", e);
            std::exit(1);
        }
        return it->second;
    }

    bool has(Entity e) const { return data.find(e) != data.end(); }
    void remove(Entity e) { data.erase(e); }
    size_t size() const { return data.size(); }
};

struct Transform { Vec3 pos; };
struct Velocity  { Vec3 vel; };
struct Mass      { float invMass = 1.f; };
struct XPBDConstraint {
    Entity a, b;
    float  restLength;
    float  compliance;
    float  lambda = 0.f;
};

static ComponentStorage<Transform>  g_transforms;
static ComponentStorage<Velocity>   g_velocities;
static ComponentStorage<Mass>       g_masses;
static std::vector<XPBDConstraint>  g_constraints;

__global__ void computeForces(const Vec3* __restrict__ pos,
                              const float* __restrict__ mass,
                              Vec3* __restrict__ acc,
                              int n,
                              float G,
                              float eps2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Vec3 a(0.f, -9.81f, 0.f);
    Vec3 pi = pos[i];
    float mi = mass[i];
    for (int j = 0; j < n; ++j) {
        if (i == j) continue;
        Vec3 d = pos[j] - pi;
        float r2 = d.lengthSq() + eps2;
        float invR = rsqrtf(r2);
        float invR3 = invR * invR * invR;
        float f = G * mass[j] * invR3;
        a += d * f;
    }

    acc[i] = a;
}

class GPUSystem {
    Vec3*  d_pos  = nullptr;
    Vec3*  d_acc  = nullptr;
    float* d_mass = nullptr;
    int    capacity = 0;

public:
    ~GPUSystem() { shutdown(); }
    void init(int maxParticles) {
        capacity = maxParticles;
        CUDA_CHECK(cudaMalloc(&d_pos,  capacity * sizeof(Vec3)));
        CUDA_CHECK(cudaMalloc(&d_acc,  capacity * sizeof(Vec3)));
        CUDA_CHECK(cudaMalloc(&d_mass, capacity * sizeof(float)));
    }

    void shutdown() {
        if (d_pos)  { cudaFree(d_pos);  d_pos  = nullptr; }
        if (d_acc)  { cudaFree(d_acc);  d_acc  = nullptr; }
        if (d_mass) { cudaFree(d_mass); d_mass = nullptr; }
        capacity = 0;
    }

    void compute(const std::vector<Vec3>&  pos,
                 const std::vector<float>& mass,
                 std::vector<Vec3>&        acc,
                 float G = 0.5f,
                 float softening = 0.05f)
    {
        int n = static_cast<int>(pos.size());
        if (n == 0) return;
        if (n > capacity) {
            fprintf(stderr, "GPUSystem::compute: n=%d > capacity=%d\n", n, capacity);
            std::exit(1);
        }

        CUDA_CHECK(cudaMemcpy(d_pos,  pos.data(),  n * sizeof(Vec3),  cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_mass, mass.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        int threads = 256;
        int blocks  = (n + threads - 1) / threads;
        float eps2  = softening * softening;
        computeForces<<<blocks, threads>>>(d_pos, d_mass, d_acc, n, G, eps2);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        acc.resize(n);
        CUDA_CHECK(cudaMemcpy(acc.data(), d_acc, n * sizeof(Vec3), cudaMemcpyDeviceToHost));
    }
};

static void solveXPBD(float dt, int iterations) {
    if (g_constraints.empty()) return;
    float invDt2 = 1.f / (dt * dt);
    for (auto& c : g_constraints) c.lambda = 0.f;
    for (int it = 0; it < iterations; ++it) {
        for (auto& c : g_constraints) {
            auto* tA = g_transforms.find(c.a);
            auto* tB = g_transforms.find(c.b);
            if (!tA || !tB) continue;
            Vec3& pa = tA->pos;
            Vec3& pb = tB->pos;
            Vec3 d = pa - pb;
            float len = d.length();
            if (len < 1e-6f) continue;
            Vec3 n = d * (1.f / len);
            float C = len - c.restLength;
            float wa = g_masses.has(c.a) ? g_masses.get(c.a).invMass : 1.f;
            float wb = g_masses.has(c.b) ? g_masses.get(c.b).invMass : 1.f;
            float wSum = wa + wb;
            if (wSum <= 0.f) continue;
            float alphaTilde = c.compliance * invDt2;
            float dLambda = -(C + alphaTilde * c.lambda) / (wSum + alphaTilde);
            pa += n * ( wa * dLambda);
            pb += n * (-wb * dLambda);
            c.lambda += dLambda;
        }
    }
}

static const char* kVS = R"GLSL(
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in float aMass;
uniform mat4 uMVP;
uniform float uPointSize;
out float vMass;
void main() {
    gl_Position  = uMVP * vec4(aPos, 1.0);
    gl_PointSize = uPointSize * (0.5 + aMass);
    vMass = aMass;
}
)GLSL";

static const char* kFS = R"GLSL(
#version 330 core
in float vMass;
out vec4 FragColor;
void main() {
    vec2 c = gl_PointCoord - vec2(0.5);
    if (dot(c, c) > 0.25) discard;
    vec3 col = mix(vec3(0.2, 0.6, 1.0), vec3(1.0, 0.4, 0.2), clamp(vMass, 0.0, 1.0));
    FragColor = vec4(col, 1.0);
}
)GLSL";

static GLuint compileShader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetShaderInfoLog(s, sizeof(log), nullptr, log);
        fprintf(stderr, "Shader compile error:\n%s\n", log);
        std::exit(1);
    }
    return s;
}

static GLuint makeProgram() {
    GLuint vs = compileShader(GL_VERTEX_SHADER,   kVS);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, kFS);
    GLuint p  = glCreateProgram();
    glAttachShader(p, vs);
    glAttachShader(p, fs);
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetProgramInfoLog(p, sizeof(log), nullptr, log);
        fprintf(stderr, "Program link error:\n%s\n", log);
        std::exit(1);
    }
    glDeleteShader(vs);
    glDeleteShader(fs);
    return p;
}

struct Mat4 { float m[16]; };

static Mat4 mat4Identity() {
    Mat4 r{};
    r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.f;
    return r;
}

static Mat4 mat4Mul(const Mat4& A, const Mat4& B) {
    Mat4 r{};
    for (int c = 0; c < 4; ++c)
        for (int row = 0; row < 4; ++row) {
            float s = 0.f;
            for (int k = 0; k < 4; ++k) s += A.m[k*4 + row] * B.m[c*4 + k];
            r.m[c*4 + row] = s;
        }
    return r;
}

static Mat4 mat4Perspective(float fovy, float aspect, float zn, float zf) {
    float t = 1.f / tanf(fovy * 0.5f);
    Mat4 r{};
    r.m[0]  = t / aspect;
    r.m[5]  = t;
    r.m[10] = (zf + zn) / (zn - zf);
    r.m[11] = -1.f;
    r.m[14] = (2.f * zf * zn) / (zn - zf);
    return r;
}

static Mat4 mat4LookAt(Vec3 eye, Vec3 center, Vec3 up) {
    Vec3 f = (center - eye); float fl = f.length(); f = f * (1.f/fl);
    Vec3 s = { f.y*up.z - f.z*up.y, f.z*up.x - f.x*up.z, f.x*up.y - f.y*up.x };
    float sl = sqrtf(s.lengthSq()); s = s * (1.f/sl);
    Vec3 u = { s.y*f.z - s.z*f.y, s.z*f.x - s.x*f.z, s.x*f.y - s.y*f.x };

    Mat4 r = mat4Identity();
    r.m[0]=s.x; r.m[4]=s.y; r.m[8] =s.z;
    r.m[1]=u.x; r.m[5]=u.y; r.m[9] =u.z;
    r.m[2]=-f.x;r.m[6]=-f.y;r.m[10]=-f.z;
    r.m[12]=-(s.x*eye.x + s.y*eye.y + s.z*eye.z);
    r.m[13]=-(u.x*eye.x + u.y*eye.y + u.z*eye.z);
    r.m[14]= (f.x*eye.x + f.y*eye.y + f.z*eye.z);
    return r;
}

class Renderer {
public:
    GLFWwindow* window = nullptr;
    GLuint program = 0, vao = 0, vbo = 0;
    int    capacity = 0;

    bool init(int maxParticles) {
        if (!glfwInit()) { fprintf(stderr, "glfwInit failed\n"); return false; }
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        window = glfwCreateWindow(1024, 768, "GPU XPBD Physics", nullptr, nullptr);
        if (!window) { fprintf(stderr, "glfwCreateWindow failed\n"); return false; }
        glfwMakeContextCurrent(window);
        glewExperimental = GL_TRUE;
        GLenum err = glewInit();
        if (err != GLEW_OK) {
            fprintf(stderr, "glewInit failed: %s\n", glewGetErrorString(err));
            return false;
        }
        while (glGetError() != GL_NO_ERROR) {}
        program = makeProgram();
        capacity = maxParticles;
        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER,
                     capacity * (3 + 1) * sizeof(float),
                     nullptr, GL_DYNAMIC_DRAW);

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
                              4 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE,
                              4 * sizeof(float), (void*)(3 * sizeof(float)));
        glEnable(GL_PROGRAM_POINT_SIZE);
        glEnable(GL_DEPTH_TEST);
        glClearColor(0.05f, 0.05f, 0.08f, 1.f);
        return true;
    }

    void draw(const std::vector<Vec3>& pos, const std::vector<float>& mass) {
        int n = static_cast<int>(pos.size());
        if (n > capacity) n = capacity;
        std::vector<float> packed(n * 4);
        for (int i = 0; i < n; ++i) {
            packed[i*4 + 0] = pos[i].x;
            packed[i*4 + 1] = pos[i].y;
            packed[i*4 + 2] = pos[i].z;
            packed[i*4 + 3] = (i < (int)mass.size()) ? mass[i] : 1.f;
        }

        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferSubData(GL_ARRAY_BUFFER, 0, packed.size() * sizeof(float), packed.data());
        int fbw, fbh;
        glfwGetFramebufferSize(window, &fbw, &fbh);
        glViewport(0, 0, fbw, fbh);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        Mat4 proj = mat4Perspective(1.0f, (float)fbw / (float)fbh, 0.1f, 500.f);
        Mat4 view = mat4LookAt({0.f, 0.f, 40.f}, {0.f, 0.f, 0.f}, {0.f, 1.f, 0.f});
        Mat4 mvp  = mat4Mul(proj, view);
        glUseProgram(program);
        GLint locMVP = glGetUniformLocation(program, "uMVP");
        GLint locPS  = glGetUniformLocation(program, "uPointSize");
        glUniformMatrix4fv(locMVP, 1, GL_FALSE, mvp.m);
        glUniform1f(locPS, 8.f);
        glBindVertexArray(vao);
        glDrawArrays(GL_POINTS, 0, n);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    bool running() const { return window && !glfwWindowShouldClose(window); }

    void shutdown() {
        if (vbo)     { glDeleteBuffers(1, &vbo); vbo = 0; }
        if (vao)     { glDeleteVertexArrays(1, &vao); vao = 0; }
        if (program) { glDeleteProgram(program); program = 0; }
        if (window)  { glfwDestroyWindow(window); window = nullptr; }
        glfwTerminate();
    }
};

int main() {
    const int   N          = 400;
    const float dt         = 1.f / 60.f;
    const int   subSteps   = 4;
    const int   xpbdIters  = 8;
    GPUSystem gpu;
    gpu.init(N);
    Renderer renderer;
    if (!renderer.init(N)) return 1;
    std::vector<Entity> entities;
    entities.reserve(N);
    int side = (int)ceilf(cbrtf((float)N));
    for (int i = 0; i < N; ++i) {
        Entity e = createEntity();
        entities.push_back(e);
        float x = (float)(i % side) - side * 0.5f;
        float y = (float)((i / side) % side) - side * 0.5f;
        float z = (float)(i / (side * side)) - side * 0.5f;
        g_transforms.add(e, { Vec3(x * 0.8f, y * 0.8f + 2.f, z * 0.8f) });
        g_velocities.add(e, { Vec3(0,0,0) });
        float m = 1.f;
        g_masses.add(e, { 1.f / m });
    }

    for (int i = 0; i + 1 < N; ++i) {
        g_constraints.push_back({ entities[i], entities[i+1], 1.2f, 0.0f, 0.f });
    }
    srand(1234);
    for (int k = 0; k < N / 2; ++k) {
        int a = rand() % N;
        int b = rand() % N;
        if (a == b) continue;
        float d = (g_transforms.get(entities[a]).pos -
                   g_transforms.get(entities[b]).pos).length();
        g_constraints.push_back({ entities[a], entities[b], d, 0.001f, 0.f });
    }

    std::vector<Vec3>  pos(N);
    std::vector<Vec3>  acc(N);
    std::vector<float> mass(N);
    while (renderer.running()) {
        float h = dt / (float)subSteps;
        for (int s = 0; s < subSteps; ++s) {
            for (int i = 0; i < N; ++i) {
                pos[i]  = g_transforms.get(entities[i]).pos;
                mass[i] = 1.f / g_masses.get(entities[i]).invMass;
            }

            gpu.compute(pos, mass, acc);
            for (int i = 0; i < N; ++i) {
                auto& v = g_velocities.get(entities[i]).vel;
                auto& p = g_transforms.get(entities[i]).pos;
                v += acc[i] * h;
                p += v * h;
            }

            solveXPBD(h, xpbdIters);
        }

        for (int i = 0; i < N; ++i) {
            pos[i]  = g_transforms.get(entities[i]).pos;
            mass[i] = 1.f;
        }

        renderer.draw(pos, mass);
    }

    renderer.shutdown();
    gpu.shutdown();
    return 0;
}
