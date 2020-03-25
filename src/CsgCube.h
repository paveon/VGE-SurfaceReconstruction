#ifndef PGR_CSGCUBE_H
#define PGR_CSGCUBE_H


#include "CsgBox.h"


namespace CSG {
   class Tree;

   class Cube : public Box {
   public:
      float m_DiagonalSq;

      Cube(const glm::vec3& origin, float sideLength);

      virtual ~Cube() = default;

      static std::unique_ptr<CSG::Tree> CreateTree(const glm::vec3& origin, float sideLength);

      bool VertexCheck(const glm::vec3& vertex) const {
         glm::vec3 pointToCenter(m_Center - vertex);
         float distanceToCenterSq = glm::dot(pointToCenter, pointToCenter);
         if (distanceToCenterSq < m_HalfSizesSq.x) {
            return true;
         }
         if (distanceToCenterSq > m_DiagonalSq) {
            return false;
         }
         return Box::VertexCheck(vertex);
      }
   };
}


#endif //PGR_CSGCUBE_H
