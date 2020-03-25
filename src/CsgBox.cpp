#include <glm/gtx/string_cast.hpp>
#include "CsgTree.h"
#include "CsgBox.h"
#include "CsgSphere.h"


namespace CSG {
   std::unique_ptr<CSG::Tree> Box::CreateTree(const glm::vec3& origin, const glm::vec3& sideLengths) {
      return std::make_unique<Tree>(std::make_unique<Box>(origin, sideLengths));
   }

   Box::Box(const glm::vec3& origin, const glm::vec3& sideLengths) : m_Vertices{
           origin,
           glm::vec3(origin.x, origin.y + sideLengths.y, origin.z),
           glm::vec3(origin.x + sideLengths.x, origin.y + sideLengths.y, origin.z),
           glm::vec3(origin.x + sideLengths.x, origin.y, origin.z),
           glm::vec3(origin.x, origin.y, origin.z + sideLengths.z),
           glm::vec3(origin.x, origin.y + sideLengths.y, origin.z + sideLengths.z),
           glm::vec3(origin.x + sideLengths.x, origin.y + sideLengths.y, origin.z + sideLengths.z),
           glm::vec3(origin.x + sideLengths.x, origin.y, origin.z + sideLengths.z)
   }, m_Normals{
           glm::vec3(1.0f, 0.0f, 0.0f),
           glm::vec3(-1.0f, 0.0f, 0.0f),
           glm::vec3(0.0f, 1.0f, 0.0f),
           glm::vec3(0.0f, -1.0f, 0.0f),
           glm::vec3(0.0f, 0.0f, 1.0f),
           glm::vec3(0.0f, 0.0f, -1.0f),
   }, m_Center(origin + (sideLengths / 2.0f)),
                                                                     m_Sizes(sideLengths),
                                                                     m_HalfSizes(m_Sizes / 2.0f),
                                                                     m_HalfSizesSq(m_HalfSizes * m_HalfSizes) {}

   void Box::ResetOrientation() {
      glm::vec3 halfSides = m_Sizes / 2.0f;
      glm::vec3 origin(m_Center - halfSides);
      m_Vertices[0] = origin;
      m_Vertices[1] = glm::vec3(origin.x, origin.y + m_Sizes.y, origin.z);
      m_Vertices[2] = glm::vec3(origin.x + m_Sizes.x, origin.y + m_Sizes.y, origin.z);
      m_Vertices[3] = glm::vec3(origin.x + m_Sizes.x, origin.y, origin.z);
      m_Vertices[4] = glm::vec3(origin.x, origin.y, origin.z + m_Sizes.z);
      m_Vertices[5] = glm::vec3(origin.x, origin.y + m_Sizes.y, origin.z + m_Sizes.z);
      m_Vertices[6] = glm::vec3(origin.x + m_Sizes.x, origin.y + m_Sizes.y, origin.z + m_Sizes.z);
      m_Vertices[7] = glm::vec3(origin.x + m_Sizes.x, origin.y, origin.z + m_Sizes.z);

      m_Normals[0] = glm::vec3(1.0f, 0.0f, 0.0f);
      m_Normals[0] = glm::vec3(-1.0f, 0.0f, 0.0f);
      m_Normals[0] = glm::vec3(0.0f, 1.0f, 0.0f);
      m_Normals[0] = glm::vec3(0.0f, -1.0f, 0.0f);
      m_Normals[0] = glm::vec3(0.0f, 0.0f, 1.0f);
      m_Normals[0] = glm::vec3(0.0f, 0.0f, -1.0f);
   }

   void Box::Rotate(const glm::vec3& point, const glm::mat4& m) {
      glm::mat4 rot = glm::translate(glm::mat4(1.0f), point) * m * glm::translate(glm::mat4(1.0f), -point);

      m_Center = glm::vec3(rot * glm::vec4(m_Center, 1.0f));
      for (glm::vec3& v : m_Vertices) {
         v = glm::vec3(rot * glm::vec4(v, 1.0f));
      }

      // Rotate normals
      rot = glm::transpose(glm::inverse(rot));
      for (glm::vec3& v : m_Normals) {
         v = glm::normalize(glm::vec3(rot * glm::vec4(v, 0.0f)));
      }

      // Recalculate basis vectors
      m_BasisVectors = glm::mat4(
              glm::vec4(glm::normalize(m_Vertices[3] - m_Vertices[0]), 0.0f),
              glm::vec4(glm::normalize(m_Vertices[1] - m_Vertices[0]), 0.0f),
              glm::vec4(glm::normalize(m_Vertices[4] - m_Vertices[0]), 0.0f),
              glm::vec4(0.0f, 0.0f, 0.0f, 1.0f)
      );
   }

   bool Box::VertexCheck(const glm::vec3& vertex) const {
      glm::vec3 rotatedVertex(glm::vec3(m_BasisVectors * glm::vec4(vertex - m_Center, 1.0f)) + m_Center);
      BoundingBox bb{
              m_Center - m_HalfSizes,
              m_Center + m_HalfSizes
      };

      /* Check first three faces with dot product */
      glm::vec3 pointToFace(bb.min - rotatedVertex);
      glm::vec3 pointToFace2(bb.max - rotatedVertex);
      return !(glm::dot(pointToFace2, glm::vec3(1.0f, 0.0f, 0.0f)) <= 0 ||
               glm::dot(pointToFace, glm::vec3(-1.0f, 0.0f, 0.0f)) <= 0 ||
               glm::dot(pointToFace2, glm::vec3(0.0f, 1.0f, 0.0f)) <= 0 ||
               glm::dot(pointToFace, glm::vec3(0.0f, -1.0f, 0.0f)) <= 0 ||
               glm::dot(pointToFace2, glm::vec3(0.0f, 0.0f, 1.0f)) <= 0 ||
               glm::dot(pointToFace, glm::vec3(0.0f, 0.0f, -1.0f)) <= 0);
   }

   BoundingBox Box::CalculateBB() const {
      BoundingBox bb;
      bb.min = m_Vertices[0];
      bb.max = m_Vertices[0];
      for (size_t i = 1; i < m_Vertices.size(); ++i) {
         for (size_t component = 0; component < 3; ++component) {
            if (m_Vertices[i][component] > bb.max[component])
               bb.max[component] = m_Vertices[i][component];
            else if (m_Vertices[i][component] < bb.min[component])
               bb.min[component] = m_Vertices[i][component];
         }
      }
      return bb;
   }

   bool Box::Intersects(const Primitive& primitive) const {
      if (dynamic_cast<const Sphere*>(&primitive)) {
         return SphereBoxIntersection(*((const Sphere*) &primitive), *this);
      }
      if (dynamic_cast<const Box*>(&primitive)) {
         return BoxBoxIntersection(*this, *((const Box*) &primitive));
      }
      return false;
   }

   std::optional<float> Box::LineIntersection(const glm::vec3& lineStart, const glm::vec3& lineEnd) const {
      // Reduce the problem to line AABB intersection
      glm::mat4 rotation(BasisVectors());
      glm::vec3 pointA(glm::vec3(rotation * glm::vec4(lineStart - m_Center, 1.0f)) + m_Center);
      glm::vec3 pointB(glm::vec3(rotation * glm::vec4(lineEnd - m_Center, 1.0f)) + m_Center);
      glm::vec3 direction(pointB - pointA);

      BoundingBox bb{
              m_Center - m_HalfSizes,
              m_Center + m_HalfSizes
      };

      glm::vec3 dirfrac(1.0f / direction);
      glm::vec3 t135((bb.min - pointA) * dirfrac);
      glm::vec3 t246((bb.max - pointA) * dirfrac);

      /* Slight hacks necessary due to the floating point precision errors */
      float tminX = std::min(t135.x, t246.x);
      float tminY = std::min(t135.y, t246.y);
      float tminZ = std::min(t135.z, t246.z);
      if (ParameterInBounds(tminX))
         return tminX;
      if (ParameterInBounds(tminY))
         return tminY;
      if (ParameterInBounds(tminZ))
         return tminZ;

      float tminXY = std::max(tminX, tminY);
      float tminXYZ = std::max(tminXY, tminZ);

      float tmaxX = std::max(t135.x, t246.x);
      float tmaxY = std::max(t135.y, t246.y);
      float tmaxZ = std::max(t135.z, t246.z);
      if (ParameterInBounds(tmaxX))
         return tmaxX;
      if (ParameterInBounds(tmaxY))
         return tmaxY;
      if (ParameterInBounds(tmaxZ))
         return tmaxZ;

      float tmaxXY = std::min(tmaxX, tmaxY);
      float tmaxXYZ = std::min(tmaxXY, tmaxZ);

      // if tmax < 0, ray (line) is intersecting AABB, but the whole AABB is behind us
      if (tmaxXYZ < 0) {
//         std::cout << "tminX: " << tminX << "\n"
//                   << "tminY: " << tminY << "\n"
//                   << "tminZ: " << tminZ << "\n"
//                   << "tminXY: " << tminXY << "\n"
//                   << "tminXYZ: " << tminXYZ << "\n"
//                   << "tmaxX: " << tmaxX << "\n"
//                   << "tmaxY: " << tmaxY << "\n"
//                   << "tmaxZ: " << tmaxZ << "\n"
//                   << "tmaxXY: " << tmaxXY << "\n"
//                   << "tmaxXYZ: " << tmaxXYZ << std::endl;
         return {};
      }

      // if tmin > tmax, ray doesn't intersect AABB
      if (tminXYZ > tmaxXYZ) {
//         std::cout << "tminX: " << tminX << "\n"
//                   << "tminY: " << tminY << "\n"
//                   << "tminZ: " << tminZ << "\n"
//                   << "tminXY: " << tminXY << "\n"
//                   << "tminXYZ: " << tminXYZ << "\n"
//                   << "tmaxX: " << tmaxX << "\n"
//                   << "tmaxY: " << tmaxY << "\n"
//                   << "tmaxZ: " << tmaxZ << "\n"
//                   << "tmaxXY: " << tmaxXY << "\n"
//                   << "tmaxXYZ: " << tmaxXYZ << std::endl;
         return {};
      }

      // Out of line segment bounds
      if (ParameterInBounds(tminXYZ)) {
         return tminXYZ;
      }
      else if (ParameterInBounds(tmaxXYZ)) {
         return tmaxXYZ;
      }

//      std::cout << "tminX: " << tminX << "\n"
//                << "tminY: " << tminY << "\n"
//                << "tminZ: " << tminZ << "\n"
//                << "tminXY: " << tminXY << "\n"
//                << "tminXYZ: " << tminXYZ << "\n"
//                << "tmaxX: " << tmaxX << "\n"
//                << "tmaxY: " << tmaxY << "\n"
//                << "tmaxZ: " << tmaxZ << "\n"
//                << "tmaxXY: " << tmaxXY << "\n"
//                << "tmaxXYZ: " << tmaxXYZ << std::endl;
      return {};
   }
}