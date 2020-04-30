#version 420

in vec3 positionG;
in vec3 normalG;

out vec4 fragColor;

uniform vec3 primitiveColor;

void main() {
	fragColor = vec4(primitiveColor, 1.0);
}     
