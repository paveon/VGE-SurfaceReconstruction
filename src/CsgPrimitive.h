#ifndef PGR_CSGPRIMITIVE_H
#define PGR_CSGPRIMITIVE_H

#include <optional>

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <array>
#include <memory>
#include "CsgUtils.h"

namespace CSG {
   class Box;

   class Sphere;

   class Primitive {
   public:
      virtual bool VertexCheck(const glm::vec3& vertex) const = 0;

      virtual void Translate(glm::vec3 delta) = 0;

      virtual void Rotate(const glm::vec3& point, const glm::mat4& m) = 0;

      void Rotate(const glm::vec3& point, const glm::vec3& axis, float radAngle) {
         Rotate(point, glm::rotate(glm::mat4(1.0f), radAngle, axis));
      }

      void RotateX(const glm::vec3& point, float radAngle) {
         Rotate(point, glm::vec3(1.0f, 0.0f, 0.0f), radAngle);
      }

      void RotateY(const glm::vec3& point, float radAngle) {
         Rotate(point, glm::vec3(0.0f, 1.0f, 0.0f), radAngle);
      }

      void RotateZ(const glm::vec3& point, float radAngle) {
         Rotate(point, glm::vec3(0.0f, 0.0f, 1.0f), radAngle);
      }

      virtual void ResetOrientation() = 0;

      virtual bool Intersects(const Primitive& primitive) const = 0;

      virtual BoundingBox CalculateBB() const = 0;

      virtual std::optional<float> LineIntersection(const glm::vec3& lineStart, const glm::vec3& lineEnd) const = 0;

      // Helper function for line intersection tests
      bool ParameterInBounds(float value) const {
         return (value >= -std::numeric_limits<float>::epsilon()) &&
                (value <= (1.0f + std::numeric_limits<float>::epsilon()));
      };
   };

   bool SphereBoxIntersection(const Sphere& sphere, const Box& cube);

   bool SphereSphereIntersection(const Sphere& s1, const Sphere& s2);

   bool BoxBoxIntersection(const Box& b1, const Box& b2);
}


#endif //PGR_CSGPRIMITIVE_H
