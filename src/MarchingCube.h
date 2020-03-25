#ifndef FITGL_MARCHINGCUBE_H
#define FITGL_MARCHINGCUBE_H

#include <cstdint>
#include <array>
#include <glm/vec3.hpp>


namespace CSG {
   class Tree;
}


struct MarchingCube {
   std::array<glm::vec3, 8> m_Vertices;

   MarchingCube(glm::vec3 origin, float sideLength) : m_Vertices{
           origin,
           glm::vec3(origin.x, origin.y + sideLength, origin.z),
           glm::vec3(origin.x + sideLength, origin.y + sideLength, origin.z),
           glm::vec3(origin.x + sideLength, origin.y, origin.z),
           glm::vec3(origin.x, origin.y, origin.z + sideLength),
           glm::vec3(origin.x, origin.y + sideLength, origin.z + sideLength),
           glm::vec3(origin.x + sideLength, origin.y + sideLength, origin.z + sideLength),
           glm::vec3(origin.x + sideLength, origin.y, origin.z + sideLength)
   } {}

   void CreatePolygons(CSG::Tree& csgTree) const;
};

#endif //FITGL_MARCHINGCUBE_H
