#version 420
layout(location=0) in vec3 position;
layout(location=1) in vec3 color;

out vec3 colorF;

uniform mat4 pvm;

void main(){
    colorF = color;
    gl_Position = pvm * vec4(position,1);
}