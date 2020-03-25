#ifndef PGR_CSGCYLINDER_H
#define PGR_CSGCYLINDER_H

#include "CsgPrimitive.h"


namespace CSG {
   class Tree;

   class Cylinder : public Primitive {
   protected:
      glm::mat4 m_BasisVectors = glm::mat4(1.0f);
      glm::vec3 m_BasePoint;
      glm::vec3 m_Axis;
      float m_Radius;
      float m_HeightSq;
      float m_Height;
      float m_HalfHeight;
      float m_RadiusSq;
      glm::vec3 m_Center;

      virtual double RadiusSqAtY(double) const { return m_RadiusSq; }

   public:
      Cylinder(const glm::vec3& basePoint,
               const glm::vec3& axis,
               float radius) : m_BasePoint(basePoint), m_Axis(axis),
                               m_Radius(radius),
                               m_HeightSq(glm::dot(m_Axis, m_Axis)),
                               m_Height(std::sqrt(m_HeightSq)),
                               m_HalfHeight(m_Height / 2.0f),
                               m_RadiusSq(m_Radius * m_Radius),
                               m_Center(m_BasePoint + (m_Axis / 2.0f)) {};

      virtual ~Cylinder() = default;

      static std::unique_ptr<CSG::Tree> CreateTree(const glm::vec3& basePoint, const glm::vec3& axis, float radius);

      bool VertexCheck(const glm::vec3& vertex) const override;

      void Translate(glm::vec3 delta) override {
         m_BasePoint += delta;
         m_Center += delta;
      }

      void Rotate(const glm::vec3& point, const glm::mat4& m) override;

      void ResetOrientation() override {};

      bool Intersects(const Primitive& primitive) const override;

      BoundingBox CalculateBB() const override;

      std::optional<float> LineIntersection(const glm::vec3& lineStart, const glm::vec3& lineEnd) const override;
   };
}


#endif //PGR_CSGCYLINDER_H
