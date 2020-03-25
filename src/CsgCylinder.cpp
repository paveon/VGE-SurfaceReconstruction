#include "CsgCylinder.h"
#include "CsgTree.h"


namespace CSG {
   std::unique_ptr<CSG::Tree> Cylinder::CreateTree(const glm::vec3& basePoint, const glm::vec3& axis, float radius) {
      return std::make_unique<Tree>(std::make_unique<Cylinder>(basePoint, axis, radius));
   }


   void Cylinder::Rotate(const glm::vec3& point, const glm::mat4& m) {
      glm::mat4 rot = glm::translate(glm::mat4(1.0f), point) * m * glm::translate(glm::mat4(1.0f), -point);

      m_BasePoint = glm::vec3(rot * glm::vec4(m_BasePoint, 1.0f));
      m_Axis = glm::vec3(rot * glm::vec4(m_Axis, 0.0f));
      m_Center = m_BasePoint + (m_Axis / 2.0f);

      m_BasisVectors = rot * m_BasisVectors;
      m_BasisVectors[0] = glm::normalize(m_BasisVectors[0]);
      m_BasisVectors[1] = glm::normalize(m_BasisVectors[1]);
      m_BasisVectors[2] = glm::normalize(m_BasisVectors[2]);
      m_BasisVectors[3] = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
   }

   bool Cylinder::VertexCheck(const glm::vec3& vertex) const {
      glm::vec3 rotatedVertex(glm::vec3(m_BasisVectors * glm::vec4(vertex - m_Center, 1.0f)) + m_Center);
      glm::vec3 axis(0.0f, m_Height, 0.0f);
      glm::vec3 base = m_Center;
      base.y -= m_HalfHeight;

      glm::vec3 baseToVertex(rotatedVertex - base);
      float dot = glm::dot(baseToVertex, axis);
      if (dot < 0.0f || dot >= m_HeightSq) {
         return false;
      }
      else {
         // point in circle check
         glm::vec2 capCenter2D(base.x, base.z);
         glm::vec2 delta(capCenter2D - glm::vec2(rotatedVertex.x, rotatedVertex.z));
         if (glm::dot(delta, delta) > RadiusSqAtY(rotatedVertex.y - base.y)) return false;
      }
      return true;
   }

   bool Cylinder::Intersects(const Primitive&) const {
      return false;
   }

   BoundingBox Cylinder::CalculateBB() const {
      glm::vec3 e = m_Radius * glm::sqrt(glm::vec3(1.0) - (m_Axis * m_Axis) / glm::dot(m_Axis, m_Axis));
      return BoundingBox(glm::min(m_BasePoint - e, m_BasePoint + m_Axis - e),
                         glm::max(m_BasePoint + e, m_BasePoint + m_Axis + e));
   }

   std::optional<float> Cylinder::LineIntersection(const glm::vec3& lineStart, const glm::vec3& lineEnd) const {
      glm::vec3 pointA(glm::vec3(m_BasisVectors * glm::vec4(lineStart - m_Center, 1.0f)) + m_Center);
      glm::vec3 pointB(glm::vec3(m_BasisVectors * glm::vec4(lineEnd - m_Center, 1.0f)) + m_Center);
      glm::vec3 direction(pointB - pointA);
      glm::vec3 axis(0.0f, m_Height, 0.0f);
      glm::vec3 base(m_Center);
      base.y -= m_HalfHeight;

      glm::vec2 capCenter2D(base.x, base.z);

      // Test against the body of the cylinder. Use line-circle intersection
      // in projected XZ plane (dropped Y coord)
      auto bodyCheck = [&]() -> std::optional<float> {
         glm::vec2 lineStart2D(pointA.x, pointA.z);
         glm::vec2 lineEnd2D(pointB.x, pointB.z);
         glm::vec2 direction2D(lineEnd2D - lineStart2D);
         glm::vec2 tmp((lineStart2D - capCenter2D) * direction2D);
         float a = glm::dot(direction2D, direction2D);
         float b = 2 * (tmp.x + tmp.y);
         float c = glm::dot(capCenter2D, capCenter2D) + glm::dot(lineStart2D, lineStart2D);
         c -= 2 * glm::dot(capCenter2D, lineStart2D);
         c -= m_RadiusSq;
         float bb4ac = (b * b) - (4 * a * c);

         if (std::abs(a) < std::numeric_limits<float>::epsilon())
            return {};
         if (std::abs(bb4ac) < std::numeric_limits<float>::epsilon())
            bb4ac = 0.0f;
         if (bb4ac < 0.0f)
            return {};

         float t;
         float a2 = 2 * a;
         if (AlmostEqual(bb4ac, 0.0f)) {
            t = (-b) / a2;
            if (ParameterInBounds(t)) {
               glm::vec3 p(pointA + (direction * (float)t));
               if (p.y >= base.y && p.y <= (base.y + axis.y))
                  return t;
            }
         }
         else {
            float sqrtValue = sqrt(bb4ac);
            std::array<double, 2> tValues{(-b - sqrtValue) / a2, (-b + sqrtValue) / a2};
            t = tValues[0];
            // Closest intersection point might be out of bounds in Y coordinates, check both
            for (float t : tValues) {
               if (ParameterInBounds(t)) {
                  glm::vec3 p(pointA + (direction * t));
                  if (p.y >= base.y && p.y <= (base.y + axis.y))
                     return t;
               }
            }
         }

         return {};
      };



      // Line is parallel to the end caps
      if (AlmostEqual(direction.y, 0.0f)) {
         if (pointA.y < base.y || pointA.y > (base.y + axis.y)) {
            return {};
         }
         return bodyCheck();
      }
      else {
         float tBottomCap = (base.y - pointA.y) / direction.y;
         float tTopCap = ((base.y + m_Height) - pointA.y) / direction.y;
         float t = std::min(tBottomCap, tTopCap);
         if (ParameterInBounds(t)) {
            // Test end cap plane intersection point, basically point in circle test
            glm::vec3 p(pointA + (direction * t));
            glm::vec2 distance(capCenter2D - glm::vec2(p.x, p.z));
            if (glm::dot(distance, distance) <= m_RadiusSq)
               return t;
         }

         return bodyCheck();
      }
   }
}