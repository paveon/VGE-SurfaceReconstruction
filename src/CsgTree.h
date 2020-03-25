#ifndef PGR_CSGTREE_H
#define PGR_CSGTREE_H

#include <memory>
#include <vector>

#include <GL/glew.h>
#include <glm/glm.hpp>

#include <ProgramObject.h>
#include <Camera.h>

#include "CsgUtils.h"


namespace CSG {
   static ProgramObject s_ShaderProgram;

   class Primitive;

   enum class TextureType {
      None,
      Checkerboard
   };

   enum class TextureSource {
      VertexColor,
      ShaderFormula
   };

   struct Vertex {
      glm::vec3 pos;
      glm::vec3 color;

      Vertex() = default;

      Vertex(const glm::vec3& pos, const glm::vec3& color) : pos(pos), color(color) {}
   };

   class Tree {
   private:
      class Node {
      private:
         friend class Tree;

         std::unique_ptr<Node> m_Left;
         std::unique_ptr<Node> m_Right;
         std::unique_ptr<CSG::Primitive> m_Primitive;
         Operation m_Operation = Operation::None;

         BoundingBox m_BB;

      public:
         Node(std::unique_ptr<CSG::Primitive> object);

         Node(Operation op);
      };

      std::unique_ptr<Node> m_Root;

      size_t m_VboSize = 0;
      GLuint m_VBO = 0;
      GLuint m_VAO = 0;
      static GLuint m_WireframeEBO;

      TextureSource m_TextureSource = TextureSource::VertexColor;
      TextureType m_TextureType = TextureType::None;

      size_t MeshDataSize() const { return m_MeshVertices.size() * sizeof(Vertex); }

      void SampleTexture();

      void InitBuffers();

      void BufferBB();

   public:
      glm::mat4 m_Model = glm::mat4(1.0f);
      std::vector<Vertex> m_MeshVertices;

      bool m_RenderBB = true;

      Tree();

      Tree(std::unique_ptr<CSG::Primitive> object);

      Tree(std::unique_ptr<Tree> left, std::unique_ptr<Tree> right, Operation op);

      void SetTextureSource(TextureSource source) {
         m_TextureSource = source;
      }

      void SetTextureType(TextureType type, bool bufferMesh) {
         if (type == m_TextureType)
            return;
         m_TextureType = type;
         if (bufferMesh && type != TextureType::None) {
            SampleTexture();
            BufferMesh();
         }
      }

      static void InitGL(const std::string& shaderFolder);

      glm::vec3 Center() const { return m_Root->m_BB.Center(); }

      bool RayIntersects(const glm::vec3& origin, const glm::vec3& direction) const;

      bool DistanceSq(const glm::vec3& point) const { return m_Root->m_BB.DistanceSq(point); }

      void ResetRotation();

      void Translate(glm::vec3 delta);

      void Rotate(const glm::vec3& point, const glm::vec3& axis, float radAngle);

      void Rotate(const glm::vec3& point, const glm::mat4& m);

      void RotateX(glm::vec3 point, float radAngle) {
         Rotate(point, glm::vec3(1.0f, 0.0f, 0.0f), radAngle);
      }

      void RotateY(glm::vec3 point, float radAngle) {
         Rotate(point, glm::vec3(0.0f, 1.0f, 0.0f), radAngle);
      }

      void RotateZ(glm::vec3 point, float radAngle) {
         Rotate(point, glm::vec3(0.0f, 0.0f, 1.0f), radAngle);
      }

      bool VertexCheck(glm::vec3 vertex) const;

      std::optional<glm::vec3> LineIntersection(const glm::vec3& lineStart, const glm::vec3& lineEnd) const;

      void CreateMesh(size_t gridSize);

      void BufferMesh();

      void Regenerate(size_t gridSize = 32);

      void Draw(PerspectiveCamera& cam, glm::vec3 color, glm::vec3 wireframeColor);

      bool Merge(std::unique_ptr<Tree> tree, Operation op);

      std::unique_ptr<Tree> SplitAtRoot();
   };
}


#endif //PGR_CSGTREE_H
