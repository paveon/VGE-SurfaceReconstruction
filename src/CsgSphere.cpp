#include "CsgTree.h"
#include "CsgSphere.h"
#include "CsgCube.h"


namespace CSG {
   std::unique_ptr<Tree> Sphere::CreateTree(glm::vec3 center, float radius) {
      return std::make_unique<Tree>(std::make_unique<Sphere>(center, radius));
   }

   bool Sphere::Intersects(const Primitive& primitive) const {
      if (dynamic_cast<const Sphere*>(&primitive)) {
         return SphereSphereIntersection(*((const Sphere*) &primitive), *this);
      }
      if (dynamic_cast<const Box*>(&primitive)) {
         return SphereBoxIntersection(*this, *((const Box*) &primitive));
      }
      return false;
   }

   std::optional<float> Sphere::LineIntersection(const glm::vec3& lineStart, const glm::vec3& lineEnd) const {
      glm::vec3 distance(lineEnd - lineStart);
      glm::vec3 tmp((lineStart - m_Center) * distance);
      float a = glm::dot(distance, distance);
      float b = 2 * (tmp.x + tmp.y + tmp.z);
      float c = glm::dot(m_Center, m_Center) + glm::dot(lineStart, lineStart);
      c -= 2 * glm::dot(m_Center, lineStart);
      c -= m_RadiusSq;
      float bb4ac = (b * b) - (4 * a * c);
      if (std::abs(a) < std::numeric_limits<float>::epsilon() || bb4ac < 0) {
         return {};
      }
      float a2 = 2 * a;
      float sqrtValue = sqrt(bb4ac);
      float t1 = (-b + sqrtValue) / a2;
      float t2 = (-b - sqrtValue) / a2;

      // Return intersection point closest to the line start
      return std::min(t1, t2);
   }
}