#ifndef PGR_CSGCONE_H
#define PGR_CSGCONE_H

#include "CsgCylinder.h"


namespace CSG {
   class Cone : public Cylinder {
   protected:
      float m_RadiusHeightRatio;
      float m_RatioSq;

      double RadiusSqAtY(double relativeY) const override {
         double ratio = (1.0 - relativeY / m_Height);
         return m_RadiusSq * ratio;
      }

   public:
      Cone(const glm::vec3& basePoint,
           const glm::vec3& axis,
           float radius) : Cylinder(basePoint, axis, radius),
                           m_RadiusHeightRatio(m_Radius / m_Height),
                           m_RatioSq(m_RadiusHeightRatio * m_RadiusHeightRatio){};

      virtual ~Cone() = default;

      static std::unique_ptr<CSG::Tree> CreateTree(const glm::vec3& basePoint, const glm::vec3& axis, float radius);

      bool Intersects(const Primitive& primitive) const override;

      std::optional<float> LineIntersection(const glm::vec3& lineStart, const glm::vec3& lineEnd) const override;
   };
}


#endif //PGR_CSGCONE_H
