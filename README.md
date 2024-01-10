Script to search objects on images.


🖼 Leveraging Meta's SAM model, the project achieved precise object extraction from images.

🔗 Subsequently, OpenAI's Clip model translated these objects into embeddings, enabling versatile representation for images, words, and videos.

💾 To optimize storage and retrieval of these embeddings, we implemented LanceDB, an open-source vector database, enhancing efficiency and scalability.

💬 The querying process involves translating a sentence into an embedding using the same Clip model. The vector database then facilitates searching for the closest matching entity.

🌐 This approach allows for identifying objects in images through natural language queries, demonstrating the project's practical applications.

More info: https://blog.lancedb.com/search-within-an-image-331b54e4285e
