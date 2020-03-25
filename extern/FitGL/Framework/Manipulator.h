#pragma once

#include <Camera.h>
#include <functional>

class BaseApp;

class Manipulator {
public:
	Manipulator(Camera* c) :camera(c) {}
	virtual ~Manipulator() {}

	void setCamera(Camera *c) { camera=c; }
	Camera* getCamera() { return camera;}

	virtual void update(float) {}
	virtual void setupCallbacks(BaseApp &) {}
	
protected:
	Camera *camera;
};


class OrbitManipulator: public Manipulator{
public:
	OrbitManipulator(Camera* c=0) :Manipulator(c) {}
	virtual ~OrbitManipulator() {}

	virtual void update(float dt);
	virtual void setupCallbacks(BaseApp &app);

	void setZoom(float z) { zoom = z; }
	void setRotationX(float x) { movex = x; }
	void setRotationY(float y) { movey = y; }
protected:
	bool mouseButton[6] = {};
	float movex = 0;
	float movey = 0;
	int dragx = 0;
	int dragy = 0;
	float zoom = 30;
	float rotSpeedX = 1;
	float rotSpeedY = 1;
	float moveSpeed = 0.01f;
	float dragSpeed = 0.001f;
	bool keymap[256] = {};
};