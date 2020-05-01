#version 420
layout(location=0) in vec3 position;

uniform mat4 pvm;
uniform vec3 primitiveColor;

void main(){
    gl_Position = pvm * vec4(position, 1.0);
}