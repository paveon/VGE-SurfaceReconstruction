#ifndef PGR_CSGBOX_H
#define PGR_CSGBOX_H

#include "CsgPrimitive.h"

namespace CSG {
   class Tree;

   class Box : public Primitive {
   protected:
      std::array<glm::vec3, 8> m_Vertices;
      std::array<glm::vec3, 6> m_Normals;
      glm::mat4 m_BasisVectors = glm::mat4(1.0f);

   public:
      glm::vec3 m_Center;
      glm::vec3 m_Sizes;
      glm::vec3 m_HalfSizes;
      glm::vec3 m_HalfSizesSq;

      Box(const glm::vec3& origin, const glm::vec3& sideLengths);

      virtual ~Box() = default;

      static std::unique_ptr<CSG::Tree> CreateTree(const glm::vec3& origin, const glm::vec3& sideLengths);

      bool VertexCheck(const glm::vec3& vertex) const override;

      glm::vec3 AxisX() const { return glm::vec3(m_BasisVectors[0]); }

      glm::vec3 AxisY() const { return glm::vec3(m_BasisVectors[1]); }

      glm::vec3 AxisZ() const { return glm::vec3(m_BasisVectors[2]); }

      glm::mat4 BasisVectors() const {
         return m_BasisVectors;
      }

      void Translate(glm::vec3 delta) override {
         for (glm::vec3& vertex : m_Vertices) {
            vertex += delta;
         }
         m_Center += delta;
      }

      void Rotate(const glm::vec3& point, const glm::mat4& m) override;

      void ResetOrientation() override;

      bool Intersects(const Primitive& primitive) const override;

      BoundingBox CalculateBB() const override;

      std::optional<float> LineIntersection(const glm::vec3& lineStart, const glm::vec3& lineEnd) const override;
   };
}


#endif //PGR_CSGBOX_H
