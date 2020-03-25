#include "CsgCube.h"
#include "CsgTree.h"


namespace CSG {
   std::unique_ptr<CSG::Tree> Cube::CreateTree(const glm::vec3& origin, float sideLength) {
      return std::make_unique<Tree>(std::make_unique<Cube>(origin, sideLength));
   }

   Cube::Cube(const glm::vec3& origin, float sideLength) : Box(origin, glm::vec3(sideLength)) {
      glm::vec3 diagonalVec(m_Center - m_Vertices[0]);
      m_DiagonalSq = glm::dot(diagonalVec, diagonalVec);
   }
}