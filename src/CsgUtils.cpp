#include "CsgUtils.h"
#include <algorithm>


const std::array<uint32_t, 26> CSG::BoundingBox::s_Indices = {
        0, 1, 1, 2, 2, 3, 3, 0,
        4, 5, 5, 6, 6, 7, 7, 4,
        0, 4, 1, 5, 2, 6, 3, 7, 0, 4
};


CSG::BoundingBox CSG::BoundingBox::operator+(const CSG::BoundingBox& bb) const {
   if (!Intersects(bb)) return BoundingBox();
   return BoundingBox(
           {
                   std::min(min.x, bb.min.x),
                   std::min(min.y, bb.min.y),
                   std::min(min.z, bb.min.z)},
           {
                   std::max(max.x, bb.max.x),
                   std::max(max.y, bb.max.y),
                   std::max(max.z, bb.max.z)}
   );
}

CSG::BoundingBox CSG::BoundingBox::operator*(const CSG::BoundingBox& bb) const {
   if (!Intersects(bb)) return BoundingBox();
   return {
           {
                   std::max(min.x, bb.min.x),
                   std::max(min.y, bb.min.y),
                   std::max(min.z, bb.min.z)},
           {
                   std::min(max.x, bb.max.x),
                   std::min(max.y, bb.max.y),
                   std::min(max.z, bb.max.z)}
   };
}

CSG::BoundingBox CSG::BoundingBox::operator-(const CSG::BoundingBox& bb) const {
   if (!Intersects(bb)) return BoundingBox();
   unsigned axisX = (bb.min[0] <= min[0] && bb.max[0] >= max[0]);
   unsigned axisY = (bb.min[1] <= min[1] && bb.max[1] >= max[1]);
   unsigned axisZ = (bb.min[2] <= min[2] && bb.max[2] >= max[2]);
   unsigned result = axisX + axisY + axisZ;
   if (result == 3) {
      // Completely overlapping bounding box, result is empty
      return BoundingBox();
   }
   else if (result == 2) {
      // Bounding box is sliced by a plane
      BoundingBox mergedBB = *this;
      if (axisX && axisZ && !(bb.min.y > min.y && bb.max.y < max.y)) {
         if (bb.min.y > min.y) mergedBB.max.y = bb.min.y;
         else mergedBB.min.y = bb.max.y;
      }
      else if (axisX && axisY && !(bb.min.z > min.z && bb.max.z < max.z)) {
         if (bb.min.z > min.z) mergedBB.max.z = bb.min.z;
         else mergedBB.min.z = bb.max.z;
      }
      else if (axisY && axisZ && !(bb.min.z > min.z && bb.max.z < max.z)) {
         if (bb.min.x > min.x) mergedBB.max.x = bb.min.x;
         else mergedBB.min.x = bb.max.x;
      }
      return mergedBB;
   } else {
      // Unchanged bounding box
      return *this;
   }
}

const std::array<glm::vec3, 8>& CSG::BoundingBox::GetVertices() {
   vertices = {
           //bottom
           glm::vec3(min.x, min.y, min.z),
           {min.x, min.y, max.z},
           {max.x, min.y, max.z},
           {max.x, min.y, min.z},

           //top
           {min.x, max.y, min.z},
           {min.x, max.y, max.z},
           {max.x, max.y, max.z},
           {max.x, max.y, min.z},
   };

   return vertices;
}

CSG::BoundingBox CSG::MergeBB(const CSG::BoundingBox& bb1, const CSG::BoundingBox& bb2, CSG::Operation op) {
   switch (op) {
      case Operation::Union:
         return bb1 + bb2;
      case Operation::Intersection:
         return bb1 * bb2;
      case Operation::Difference:
         return bb1 - bb2;
      default:
         return BoundingBox();
   }
}
