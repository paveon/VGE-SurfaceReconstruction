#include "CsgCone.h"
#include "CsgTree.h"


namespace CSG {
   std::unique_ptr<CSG::Tree> Cone::CreateTree(const glm::vec3& basePoint, const glm::vec3& axis, float radius) {
      return std::make_unique<Tree>(std::make_unique<Cone>(basePoint, axis, radius));
   }

   bool Cone::Intersects(const Primitive&) const {
      return false;
   }

   std::optional<float> Cone::LineIntersection(const glm::vec3& lineStart, const glm::vec3& lineEnd) const {
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
//         glm::vec3 axisDown(-axis);
//         glm::vec3 theta(axisDown / m_Height);
//         glm::vec3 w(pointA - coneVertex);
//         float m = m_RadiusSq / m_LengthSq;
//
//         float tmp = pow(glm::dot(direction, theta), 2);
//         float a = glm::dot(direction, direction) - m * (tmp) - tmp;
//         if (std::abs(a) < std::numeric_limits<float>::epsilon())
//            return {};
//
//         float b = 2.0f * (glm::dot(direction, w) - m * glm::dot(direction, theta) * glm::dot(w, theta) -
//                           glm::dot(direction, theta) * glm::dot(w, theta));
//         float c = glm::dot(w, w) - m * pow(glm::dot(w, theta), 2) - pow(glm::dot(w, theta), 2);
//
//         float d = pow(b, 2) - (4.f * a * c);
//         if (std::abs(d) < std::numeric_limits<float>::epsilon())
//            d = 0.0f;
//
//         if (d >= 0) {
//            auto tValues = std::array<float, 2>{
//                    static_cast<float>(((-b) - sqrt(d)) / (2.0f * a)),
//                    static_cast<float>(((-b) + sqrt(d)) / (2.0f * a))
//            };
//
//            for (float t : tValues) {
//               if (ParameterInBounds(t)) {
//                  glm::vec3 p(pointA + (direction * t));
//                  if (p.y >= base.y && p.y <= (base.y + axis.y))
//                     return t;
//               }
//            }
//            return {};
//         }
//         return {};

         float yk = pointA.y - (base.y + m_Height);

         glm::vec3 tmp(direction* direction);
         float a = tmp.x - (m_RatioSq * tmp.y) + tmp.z;
         tmp = direction * pointA;
         float b = 2 * (tmp.x + tmp.z - (direction.y * m_RatioSq * yk));
         tmp = pointA * pointA;
         float c = tmp.x + tmp.z - (m_RatioSq * yk * yk);
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
               glm::vec3 p(pointA + (direction * (float) t));
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


      if (AlmostEqual(direction.y, 0.0f)) {
         // Horizontal line
         if (pointA.y<base.y || pointA.y>(base.y + axis.y)) {
            return {};
         }
         glm::vec2 a2D(pointA.x, pointA.z);
         glm::vec2 b2D(pointB.x, pointB.z);
         double dirX = b2D.x - a2D.x;
         double dirY = b2D.y - a2D.y;
         double tmpX = (a2D.x - capCenter2D.x) * dirX;
         double tmpY = (a2D.y - capCenter2D.y) * dirY;

         double a = dirX * dirX + dirY * dirY;
         double b = 2.0 * (tmpX + tmpY);
         double c = (capCenter2D.x * capCenter2D.x + capCenter2D.y * capCenter2D.y) + (a2D.x * a2D.x + a2D.y * a2D.y);
         c -= 2.0 * (capCenter2D.x * a2D.x + capCenter2D.y * a2D.y);
         double radiusSq = RadiusSqAtY(pointA.y - base.y);
         c -= radiusSq;
         double d = (b * b) - (4 * a * c);

         if (std::abs(a) < std::numeric_limits<double>::epsilon())
            return {};
         if (d < 0.0f)
            return {};

         double a2 = 2 * a;
         double dSqrt = std::sqrt(d);
         std::array<double, 2> tValues{std::abs((-b - dSqrt) / a2), std::abs((-b + dSqrt) / a2)};
         std::optional<float> result;
         // Closest intersection point might be out of bounds in Y coordinates, check both
         for (float t : tValues) {
            if (ParameterInBounds(t)) {
               if (!result.has_value() || result.value() > t)
                  result = t;
            }
         }
         if (!result.has_value())
            float tmp = 0;

         return result;

      }
      else {
         //Test bottom cap
         float t = (base.y - pointA.y) / direction.y;
         if (ParameterInBounds(t)) {
            // Test end cap plane intersection point, basically point in circle test
            glm::vec3 p(pointA + (direction * t));
            glm::vec2 distance(capCenter2D - glm::vec2(p.x, p.z));
            if (glm::dot(distance, distance) <= m_RadiusSq)
               return t;
         }

         if (AlmostEqual(direction.x, 0.0f) && AlmostEqual(direction.z, 0.0f)) {
            // Vertical line
            glm::vec2 lineStart2D(pointA.x, pointA.y);
            glm::vec2 lineEnd2D(pointB.x, pointB.y);
            double y = direction.y >= 0 ? base.y + m_Height : base.y;

            double deltaY = lineEnd2D.y - lineStart2D.y;
            t = (y - lineStart2D.y) / (deltaY - m_Height);
            if (ParameterInBounds(t)) {
               return t;
//               glm::vec3 point(lineStart + direction * (float)t);
//               if (VertexCheck(point))
//                  return t;
            }
            return {};
         }

         return bodyCheck();
      }
   }
}