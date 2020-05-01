#version 420

uniform mat4 pvm;
uniform vec3 primitiveColor;

out vec4 fragColor;

void main() {
	fragColor = vec4(primitiveColor, 1.0);
}     
