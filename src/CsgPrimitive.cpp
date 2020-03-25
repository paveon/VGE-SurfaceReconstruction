#include "CsgSphere.h"
#include "CsgCube.h"
#include "CsgPrimitive.h"


#include <algorithm>
#include <glm/gtx/string_cast.hpp>


namespace CSG {
   bool SphereBoxIntersection(const Sphere& sphere, const Box& cube) {
      // Calculate rotation matrix for oriented cube
      glm::mat4 rotation(cube.BasisVectors());

      // Project sphere center into the basis of cube
      glm::vec3 sphereCenter(glm::vec3(rotation * glm::vec4(sphere.m_Center - cube.m_Center, 1.0f)) + cube.m_Center);

      float halfSide = cube.m_HalfSizes.x;
      glm::vec3 closestPoint(
              std::clamp(sphereCenter.x, cube.m_Center.x - halfSide, cube.m_Center.x + halfSide),
              std::clamp(sphereCenter.y, cube.m_Center.y - halfSide, cube.m_Center.y + halfSide),
              std::clamp(sphereCenter.z, cube.m_Center.z - halfSide, cube.m_Center.z + halfSide)
      );
      glm::vec3 delta(sphereCenter - closestPoint);
      return (glm::dot(delta, delta) < sphere.m_RadiusSq);
   }

   bool SphereSphereIntersection(const Sphere& s1, const Sphere& s2) {
      glm::vec3 dir(s2.m_Center - s1.m_Center);
      float distanceSq = glm::dot(dir, dir);
      float radiusSum = s1.m_Radius + s2.m_Radius;
      return (distanceSq <= (radiusSum * radiusSum));
   }

   bool getSeparatingPlane(const glm::vec3& rpos, const glm::vec3& plane, const Box& b1, const Box& b2) {
      return (std::fabs(glm::dot(rpos, plane)) >
              (std::fabs(glm::dot(b1.AxisX() * b1.m_HalfSizes.x, plane)) +
               std::fabs(glm::dot(b1.AxisY() * b1.m_HalfSizes.y, plane)) +
               std::fabs(glm::dot(b1.AxisZ() * b1.m_HalfSizes.z, plane)) +
               std::fabs(glm::dot(b2.AxisX() * b2.m_HalfSizes.x, plane)) +
               std::fabs(glm::dot(b2.AxisY() * b2.m_HalfSizes.y, plane)) +
               std::fabs(glm::dot(b2.AxisZ() * b2.m_HalfSizes.z, plane))));
   }

   bool BoxBoxIntersection(const Box& b1, const Box& b2) {
      glm::vec3 distance(b2.m_Center - b1.m_Center);
      std::array<glm::vec3, 3> b1Axis{b1.AxisX(), b1.AxisY(), b1.AxisZ()};
      std::array<glm::vec3, 3> b2Axis{b2.AxisX(), b2.AxisY(), b2.AxisZ()};
      for (const auto& axis : b1Axis) {
         if (getSeparatingPlane(distance, axis, b1, b2))
            return false;
      }
      for (const auto& axis : b2Axis) {
         if (getSeparatingPlane(distance, axis, b1, b2))
            return false;
      }
      for (const auto& x : b1Axis) {
         for (const auto& y : b2Axis) {
            if (getSeparatingPlane(distance, glm::cross(x, y), b1, b2))
               return false;
         }
      }
      return true;
   }
}