#version 420
layout (points) in;
layout (line_strip, max_vertices = 2) out;

in vec3 positionV[];
in vec3 normalV[];

out vec3 positionG;
out vec3 normalG;

uniform mat4 pvm;

void main() {
    positionG = positionV[0];
    normalG = normalV[0];

    gl_Position = gl_in[0].gl_Position; 
    EmitVertex();

    gl_Position = gl_in[0].gl_Position + (pvm * vec4(normalV[0] * 0.2, 0.0));
    EmitVertex();

    EndPrimitive();
}  