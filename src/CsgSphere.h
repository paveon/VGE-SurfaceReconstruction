#ifndef PGR_CSGSPHERE_H
#define PGR_CSGSPHERE_H

#include "CsgPrimitive.h"


namespace CSG {
   class Tree;

   class Sphere : public Primitive {
   public:
      float m_Radius;
      float m_RadiusSq;

      glm::vec3 m_Center;

      Sphere(glm::vec3 center, float radius) : m_Radius(radius), m_RadiusSq(radius * radius), m_Center(center) {}

      virtual ~Sphere() = default;

      static std::unique_ptr<CSG::Tree> CreateTree(glm::vec3 center, float radius);

      bool VertexCheck(const glm::vec3& vertex) const override {
         glm::vec3 distance(m_Center - vertex);
         return !(glm::dot(distance, distance) >= m_RadiusSq);
      }

      std::optional<float> LineIntersection(const glm::vec3& lineStart, const glm::vec3& lineEnd) const override;

      void Translate(glm::vec3 delta) override {
         m_Center += delta;
      }

      void Rotate(const glm::vec3& point, const glm::mat4& m) override {
         m_Center = glm::vec3(m * glm::vec4(m_Center - point, 1.0f)) + point;
      }

      void ResetOrientation() override {}

      bool Intersects(const Primitive& primitive) const override;

      BoundingBox CalculateBB() const override {
         return {
                 m_Center - m_Radius,
                 m_Center + m_Radius
         };
      }
   };
}


#endif //PGR_CSGSPHERE_H
