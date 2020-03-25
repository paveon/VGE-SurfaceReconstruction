#include "CsgTree.h"

#include <iostream>
#include <functional>

#include <glm/gtx/transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <Loader.h>
#include <Shader.h>
#include <glm/gtx/string_cast.hpp>

#include "CsgPrimitive.h"
#include "MarchingCube.h"

namespace CSG {
   GLuint CSG::Tree::m_WireframeEBO;


   Tree::Node::Node(std::unique_ptr<CSG::Primitive> object) : m_Primitive(std::move(object)),
                                                              m_BB(m_Primitive->CalculateBB()) {}

   Tree::Node::Node(Operation op) {
      m_Operation = op;
   }

   Tree::Tree() {
      InitBuffers();
   }

   Tree::Tree(std::unique_ptr<CSG::Primitive> object) {
      m_Root = std::make_unique<Node>(std::move(object));

      InitBuffers();
   }

   Tree::Tree(std::unique_ptr<Tree> left, std::unique_ptr<Tree> right, Operation op) : m_Root(
           std::make_unique<Node>(op)) {
      m_Root->m_Left = std::move(left->m_Root);
      m_Root->m_Right = std::move(right->m_Root);

      InitBuffers();
   }

   void Tree::SampleTexture() {
      switch (m_TextureType) {
         case CSG::TextureType::None:
            break;

         case CSG::TextureType::Checkerboard:
            const float checkSize = 2;
            for (Vertex& v : m_MeshVertices) {
               glm::vec3 temp(glm::floor(v.pos * checkSize));
               float mod = glm::mod(temp.x + temp.y + temp.z, 2.0f);
               float colorValue = glm::max(glm::sign(mod), 0.0f);
               v.color = glm::vec3(colorValue);
            }

            break;
      }
   }

   void Tree::Regenerate(size_t gridSize) {
      CreateMesh(gridSize);
      SampleTexture();
      BufferMesh();
      m_Model = glm::mat4(1.0f);
   }

   void Tree::BufferBB() {
      glNamedBufferSubData(m_VBO, MeshDataSize(), BoundingBox::VertexDataSize, m_Root->m_BB.GetVertices().data());
   }

   void Tree::InitBuffers() {
      glCreateBuffers(1, &m_VBO);
      glCreateVertexArrays(1, &m_VAO);
      glEnableVertexArrayAttrib(m_VAO, 0);
      glEnableVertexArrayAttrib(m_VAO, 1);
   }


   void Tree::InitGL(const std::string& shaderFolder) {
      auto vs = compileShader(GL_VERTEX_SHADER, Loader::text(shaderFolder + "primitive.vert"));
      auto fs = compileShader(GL_FRAGMENT_SHADER, Loader::text(shaderFolder + "primitive.frag"));
      s_ShaderProgram = createProgram(vs, fs);

      glCreateBuffers(1, &m_WireframeEBO);
      glNamedBufferData(m_WireframeEBO, BoundingBox::IndexDataSize + 4, BoundingBox::s_Indices.data(), GL_STATIC_DRAW);
   }


   bool Tree::VertexCheck(glm::vec3 vertex) const {
      std::function<float(const std::unique_ptr<Node>& node)> traverse;
      traverse = [&](const std::unique_ptr<Node>& node) -> bool {
         if (node->m_Operation == CSG::Operation::None)
            return node->m_Primitive->VertexCheck(vertex);

         bool leftIn = traverse(node->m_Left);
         bool rightIn = traverse(node->m_Right);
         switch (node->m_Operation) {
            case CSG::Operation::Union:
               return leftIn || rightIn;

            case CSG::Operation::Difference:
               return leftIn && !rightIn;

            case CSG::Operation::Intersection:
               return leftIn && rightIn;

            default:
               return false;
         }
      };

      return traverse(m_Root);
   }


   void Tree::Translate(glm::vec3 delta) {
      std::function<void(const std::unique_ptr<Node>& node)> preorder;
      preorder = [&](const std::unique_ptr<Node>& node) -> void {
         node->m_BB.Translate(delta);
         if (node->m_Operation == Operation::None) {
            node->m_Primitive->Translate(delta);
            return;
         }
         preorder(node->m_Left);
         preorder(node->m_Right);
      };

      preorder(m_Root);
      m_Model = glm::translate(glm::mat4(1.0f), delta) * m_Model;
      std::cout << "Model: " << glm::to_string(m_Model) << std::endl;
      BufferBB();
   }

   void Tree::ResetRotation() {
      std::function<void(const std::unique_ptr<Node>& node)> preorder;
      preorder = [&](const std::unique_ptr<Node>& node) -> void {
         if (node->m_Operation == Operation::None) {
            node->m_Primitive->ResetOrientation();
            node->m_BB = node->m_Primitive->CalculateBB();
            return;
         }
         preorder(node->m_Left);
         preorder(node->m_Right);
         node->m_BB = MergeBB(node->m_Left->m_BB, node->m_Right->m_BB, node->m_Operation);
      };

      preorder(m_Root);

      //TODO: dunno for now, too complicated

      BufferBB();
   }

   void Tree::Rotate(const glm::vec3& point, const glm::vec3& axis, float radAngle) {
      Rotate(point, glm::rotate(glm::mat4(1.0f), radAngle, axis));
   }

   void Tree::Rotate(const glm::vec3& point, const glm::mat4& m) {
      //auto rot25 = glm::rotate(glm::mat4(1.0f), glm::radians(-25.0f), glm::vec3(0.0f, 0.0f, 1.0f));
      //auto rot25Inv = glm::rotate(glm::mat4(1.0f), glm::radians(25.0f), glm::vec3(0.0f, 0.0f, 1.0f));

      std::function<void(const std::unique_ptr<Node>& node)> preorder;
      preorder = [&](const std::unique_ptr<Node>& node) -> void {
         if (node->m_Operation == Operation::None) {
            //node->m_Primitive->Rotate(point, m);
            node->m_Primitive->Rotate(point, glm::inverse(m));
            node->m_BB = node->m_Primitive->CalculateBB();
            return;
         }
         preorder(node->m_Left);
         preorder(node->m_Right);
         node->m_BB = MergeBB(node->m_Left->m_BB, node->m_Right->m_BB, node->m_Operation);
      };
      preorder(m_Root);

      glm::mat4 rot(glm::translate(glm::mat4(1.0f), point) * m * glm::translate(glm::mat4(1.0f), -point));
      m_Model = rot * m_Model;
      BufferBB();
   }

   std::optional<glm::vec3> Tree::LineIntersection(const glm::vec3& lineStart, const glm::vec3& lineEnd) const {
      std::function<std::optional<float>(const std::unique_ptr<Node>& node)> traverse;
      traverse = [&](const std::unique_ptr<Node>& node) -> std::optional<float> {
         if (node->m_Operation == CSG::Operation::None)
            return node->m_Primitive->LineIntersection(lineStart, lineEnd);

         std::optional<float> left = traverse(node->m_Left);
         std::optional<float> right = traverse(node->m_Right);
         switch (node->m_Operation) {
            case CSG::Operation::Union:
               if (left.has_value() && right.has_value()) return std::min(left.value(), right.value());
               else if (left.has_value()) return left.value();
               else if (right.has_value()) return right.value();
               return {};

            case CSG::Operation::Difference:
               if (left.has_value() && right.has_value()) return std::max(left.value(), right.value());
               else if (left.has_value()) return left.value();
               else if (right.has_value()) return right.value();
               return {};

            case CSG::Operation::Intersection:
               if (left.has_value() && right.has_value()) return std::max(left.value(), right.value());
               else if (left.has_value()) return left.value();
               else if (right.has_value()) return right.value();
               return {};

            default:
               return 0;
         }
      };

      std::optional<float> tValue = traverse(m_Root);
      if (tValue.has_value()) {
         return (lineStart + (lineEnd - lineStart) * tValue.value());
      }
      else if (!VertexCheck(lineStart) && VertexCheck(lineEnd)) {
         // Just for corner cases when intersection check fails due to FP precision.
//         std::cout << "Tree-Line [" << glm::to_string(lineStart) << ", "
//                   << glm::to_string(lineEnd) << "] failed, fallback to t=0.5" << std::endl;
         return (lineStart + (lineEnd - lineStart) * 0.5f);
      }

      return {};
   }


   void Tree::CreateMesh(size_t gridSize) {
      m_MeshVertices.clear();
      const BoundingBox bb = m_Root->m_BB;
      float deltaX = std::abs(bb.min.x - bb.max.x);
      float deltaY = std::abs(bb.min.y - bb.max.y);
      float deltaZ = std::abs(bb.min.z - bb.max.z);
      float maxDelta = std::max(std::max(deltaX, deltaY), deltaZ);
      const float gridResolution = maxDelta / gridSize;
      const size_t xCount = static_cast<size_t>(deltaX / gridResolution) + 2;
      const size_t yCount = static_cast<size_t>(deltaY / gridResolution) + 2;
      const size_t zCount = static_cast<size_t>(deltaZ / gridResolution) + 2;
      const size_t gridLayerCount = xCount * yCount;
      const size_t cubeCount = gridLayerCount * zCount;

      //#pragma omp parallel for default(none) schedule(guided, 8)
      glm::vec3 originOffset = (bb.min - glm::vec3(gridResolution));
      for (size_t i = 0; i < cubeCount; ++i) {
         glm::vec3 gridPosition((i % xCount) * gridResolution,
                                ((i / xCount) % yCount) * gridResolution,
                                (i / (gridLayerCount)) * gridResolution);
         MarchingCube cube(gridPosition + originOffset, gridResolution);
         cube.CreatePolygons(*this);
      }
   }


   void Tree::BufferMesh() {
      std::cout << "Vertex count: " << m_MeshVertices.size() << std::endl;

      size_t vboSize = MeshDataSize() + BoundingBox::VertexDataSize * 2;
      if (vboSize > m_VboSize) {
         glNamedBufferData(m_VBO, vboSize + 4, 0, GL_STATIC_DRAW);
         m_VboSize = vboSize;
      }
      glNamedBufferSubData(m_VBO, 0, MeshDataSize(), m_MeshVertices.data());
      BufferBB();
   }


   void Tree::Draw(PerspectiveCamera& cam, glm::vec3 color, glm::vec3 wireframeColor) {
      glm::mat4 modelMatrix = m_Model;
      glm::mat4 pv = cam.getProjection() * cam.getView();
      glm::mat4 pvm = pv * modelMatrix;

      s_ShaderProgram.use();
      s_ShaderProgram.set3fv("primitiveColor", glm::value_ptr(color));


      glBindVertexArray(m_VAO);

      if (m_RenderBB) {
         s_ShaderProgram.set1ui("textureType", 0);
         s_ShaderProgram.setMatrix4fv("pvm", value_ptr(pv));
         glVertexArrayElementBuffer(m_VAO, m_WireframeEBO);
         glVertexArrayVertexBuffer(m_VAO, 0, m_VBO, MeshDataSize(), sizeof(glm::vec3));
         glDrawElements(GL_LINES, BoundingBox::s_Indices.size(), GL_UNSIGNED_INT, nullptr);
         glVertexArrayElementBuffer(m_VAO, 0);
      }

      s_ShaderProgram.set1ui("textureSource", static_cast<uint>(m_TextureSource));
      s_ShaderProgram.set1ui("textureType", static_cast<uint>(m_TextureType));
      s_ShaderProgram.setMatrix4fv("pvm", value_ptr(pvm));

      glVertexArrayVertexBuffer(m_VAO, 0, m_VBO, offsetof(Vertex, pos), sizeof(Vertex));
      glVertexArrayVertexBuffer(m_VAO, 1, m_VBO, offsetof(Vertex, color), sizeof(Vertex));
      glDrawArrays(GL_TRIANGLES, 0, m_MeshVertices.size());

      glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
      s_ShaderProgram.set3fv("primitiveColor", glm::value_ptr(wireframeColor));
      glDrawArrays(GL_TRIANGLES, 0, m_MeshVertices.size());
      glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
   }


   bool Tree::Merge(std::unique_ptr<Tree> tree, CSG::Operation op) {
      auto newRoot = std::make_unique<Node>(op);
      newRoot->m_BB = MergeBB(m_Root->m_BB, tree->m_Root->m_BB, op);
      newRoot->m_Left = std::move(m_Root);
      newRoot->m_Right = std::move(tree->m_Root);
      m_Root = std::move(newRoot);

      Regenerate();
      return true;
   }

   std::unique_ptr<Tree> Tree::SplitAtRoot() {
      if (!m_Root || m_Root->m_Primitive)
         return nullptr;

      auto rightSubtree = std::make_unique<Tree>();
      rightSubtree->m_Root = std::move(m_Root->m_Right);
      rightSubtree->m_TextureType = m_TextureType;
      rightSubtree->m_TextureSource = m_TextureSource;
      m_Root = std::move(m_Root->m_Left);
      Regenerate();
      rightSubtree->Regenerate();

      return rightSubtree;
   }

   bool Tree::RayIntersects(const glm::vec3& origin, const glm::vec3& direction) const {
      const glm::vec3& min = m_Root->m_BB.min;
      const glm::vec3& max = m_Root->m_BB.max;

      float tmin = (min.x - origin.x) / direction.x;
      float tmax = (max.x - origin.x) / direction.x;

      if (tmin > tmax) std::swap(tmin, tmax);

      float tymin = (min.y - origin.y) / direction.y;
      float tymax = (max.y - origin.y) / direction.y;

      if (tymin > tymax) std::swap(tymin, tymax);

      if ((tmin > tymax) || (tymin > tmax))
         return false;

      if (tymin > tmin)
         tmin = tymin;

      if (tymax < tmax)
         tmax = tymax;

      float tzmin = (min.z - origin.z) / direction.z;
      float tzmax = (max.z - origin.z) / direction.z;

      if (tzmin > tzmax) std::swap(tzmin, tzmax);

      return !((tmin > tzmax) || (tzmin > tmax));
   }
}