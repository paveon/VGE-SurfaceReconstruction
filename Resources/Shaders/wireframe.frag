#version 420

in vec3 colorF;

out vec4 fragColor;

void main(){
	fragColor = vec4(colorF, 1.0);
}     
