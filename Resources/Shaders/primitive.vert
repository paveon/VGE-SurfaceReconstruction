#version 420
layout(location=0) in vec3 position;
layout(location=1) in vec3 normal;

uniform mat4 pvm;

out vec3 positionV;
out vec3 normalV;

void main(){
	positionV = position;
    normalV = normal;
    gl_Position = pvm*vec4(position, 1.0);
}