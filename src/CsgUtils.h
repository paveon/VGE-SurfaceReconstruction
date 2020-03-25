#ifndef PGR_CSGUTILS_H
#define PGR_CSGUTILS_H

#include <glm/vec3.hpp>
#include <array>
#include <glm/geometric.hpp>

namespace CSG {
   enum Operation {
      Union,
      Intersection,
      Difference,
      None
   };

   struct BoundingBox {
   private:
      std::array<glm::vec3, 8> vertices = {};

   public:
      glm::vec3 min, max;
      static const std::array<uint32_t, 26> s_Indices;

      BoundingBox() = default;

      BoundingBox(glm::vec3 min, glm::vec3 max) : min(min), max(max) {}

      BoundingBox operator+(const BoundingBox& bb) const;

      BoundingBox operator*(const BoundingBox& bb) const;

      BoundingBox operator-(const BoundingBox& bb) const;

      bool Intersects(const BoundingBox& bb) const {
         return !((max.x <= bb.min.x || min.x >= bb.max.x) &&
                  (max.y <= bb.min.y || min.y >= bb.max.y) &&
                  (max.z <= bb.min.z || min.z >= bb.max.z));
      }

      void Translate(const glm::vec3& delta) {
         min += delta;
         max += delta;
      }

      bool Empty() const { return min == max; }

      glm::vec3 Center() const { return (min + max) / 2.0f; }

      float DistanceSq(const glm::vec3& point) const { return glm::length(Center() - point); }

      const std::array<glm::vec3, 8>& GetVertices();

      static constexpr size_t VertexDataSize = 8 * sizeof(glm::vec3);

      static constexpr size_t IndexDataSize = sizeof(s_Indices);
   };

   BoundingBox MergeBB(const BoundingBox& bb1, const BoundingBox& bb2, Operation op);

   inline bool AlmostEqual(float x, float y) {
      return std::abs(x - y) <= std::numeric_limits<float>::epsilon() * std::abs(x);
   }
}


#endif //PGR_CSGUTILS_H
