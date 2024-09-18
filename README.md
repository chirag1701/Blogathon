# Comparing Vector Search vs Semantic Search vs Graphs

Retrieval-Augmented Generation (RAG) combines the strengths of information retrieval with language generation. The core idea behind RAG is to improve the quality and accuracy of generative models, like GPT, by retrieving relevant information from a large corpus of documents. Instead of relying solely on a model's internal knowledge, RAG enhances the generation process by incorporating external, up-to-date knowledge, making it much more dynamic and contextually aware.

## Why Do We Need Effective Searching Techniques?

Now that you've landed on this blog, you're probably curious about how these searching techniques work to accurately find matches in a database. But have you ever wondered why we even need advanced searching techniques for our use case? Let's break it down.

### 1. Limited Model Capacity

Large language models (LLMs) like GPT may have extensive knowledge, but their capacity is still limited by the data they were trained on. This limitation becomes evident when dealing with niche topics or recent information that wasn't part of the model's training data. Retrieval-based methods solve this problem by accessing external knowledge sources at inference time, retrieving relevant and fresh information, and feeding it into the generation process. This ensures that responses stay current and more accurate, even for less common topics.

### 2. Efficiency in Handling Large Knowledge Bases

LLMs alone are not designed to store or efficiently search through massive datasets, such as millions of documents or articles. Even if they could store such vast amounts of data, the retrieval process would be inefficient without the right techniques. Enter vector-based search methods, which allow systems to swiftly search through these massive datasets and find the most relevant information based on meaning, rather than just keywords. This retrieved information can then be passed to the generative model for further processing.

### 3. Improved Response Accuracy and Specificity

Generative models sometimes "hallucinate" — generating answers that sound plausible but are, in fact, incorrect or irrelevant. This can undermine user trust in the system. By integrating retrieval, we can ground the model's responses in factual, up-to-date content, ensuring greater accuracy. Retrieval gives the model specific documents to reference, reducing the likelihood of generating misleading or irrelevant responses.

## Vector Search

Imagine you're building a movie recommender system. One way to approach this is by assigning scores to each movie in your catalog based on various aspects, like comedy, drama, and romance. Similarly, you ask users to rate their preferences in these same genres. To generate a ranking of recommended movies, you could multiply the respective scores between the movies and users, resulting in a dot product that quantifies how well a particular movie matches a user's tastes.

This intuitive approach is the foundation of vector search. In this context, both the movies and the user's preferences are represented as vectors in a high-dimensional space. In more technical applications, such as document search, these vectors are generated through word embeddings. Word embeddings are created by models like transformers, which have been trained on massive amounts of data to capture the underlying meanings and relationships between words.

When a user inputs a query (for example, "action-comedy movies"), this query is also converted into a vector using the same embedding model. To measure how relevant the documents (or movies) are to the user's query, we calculate the similarity between the vectors. This can be done using:

- Dot product: Measures the alignment of two vectors.
- Cosine similarity: Measures the direction of two vectors, regardless of their magnitude, making it ideal for comparing text data.

Here's a simple example of how vector search might work in Python:

```python
import numpy as np

# Example movie vectors (genre scores for [comedy, action, romance])
movie1 = np.array([0.8, 0.6, 0.2])
movie2 = np.array([0.3, 0.9, 0.1])
movie3 = np.array([0.5, 0.4, 0.7])

# User preference vector
user_preference = np.array([0.7, 0.8, 0.3])

# Calculate similarity (dot product)
similarity1 = np.dot(movie1, user_preference)
similarity2 = np.dot(movie2, user_preference)
similarity3 = np.dot(movie3, user_preference)

print(f"Similarity scores: {similarity1:.2f}, {similarity2:.2f}, {similarity3:.2f}")
```

Now, you may wonder why similar embeddings are located closer together or have a smaller distance between them in the semantic space. This is because embedding models are trained to map semantically similar items (words, sentences, or documents) closer together in the vector space. The more similar two vectors are in terms of meaning, the smaller the angle between them (cosine similarity) or the closer they lie in the vector space (Euclidean distance).

In essence, vector search enables you to compare complex, multidimensional relationships in data, allowing systems to recommend content (like movies) that best aligns with a user's preferences. This approach is highly efficient for large datasets and delivers more relevant results by considering the contextual relationships between data points rather than simple keyword matching.

## Semantic Search

Unlike traditional keyword-based search, which relies on matching exact words, semantic search focuses on the meaning behind the words. It helps systems interpret a query contextually and deliver more relevant and accurate results.

### How It Works

At the core of semantic search is the concept of word embeddings—vector representations of words that capture their meanings based on their context in a large body of text. Two popular techniques for generating these embeddings are CBOW (Continuous Bag of Words) and Skip-gram, which are often used in training models like Word2Vec.

#### CBOW (Continuous Bag of Words)

In this method, the model tries to predict a word given its surrounding context words. For example, in the sentence, "The cat sat on the mat," if we remove the word "sat," the model will use the surrounding words ("The," "cat," "on," "the," "mat") to predict the missing word "sat." This method treats words as having a context-dependent meaning, generating embeddings based on the relationships between neighboring words.

#### Skip-gram

Skip-gram works in the opposite way. Here, the model takes a single word as input and tries to predict its surrounding context words. Using the previous example, the model would take "sat" as input and try to predict words like "The," "cat," "on," and "mat." This method is useful for learning word embeddings that capture a broader semantic context, especially when dealing with rare words.

In both methods, the words are represented as vectors, and these vectors are updated and learned as model parameters through the process of training on large datasets. Words with similar meanings end up having vectors that are closer together in the high-dimensional space. This concept of closeness, measured using techniques like cosine similarity or dot product, is essential for comparing words or phrases in semantic search.

Here's a simplified example of how you might use a pre-trained Word2Vec model for semantic search:

```python
from gensim.models import KeyedVectors

# Load pre-trained Word2Vec model
model = KeyedVectors.load_word2vec_format('path_to_pretrained_model', binary=True)

# Find similar words
similar_words = model.most_similar('computer', topn=5)

print("Words similar to 'computer':")
for word, score in similar_words:
    print(f"{word}: {score:.2f}")

# Compare word similarities
similarity = model.similarity('king', 'queen')
print(f"Similarity between 'king' and 'queen': {similarity:.2f}")
```

### Transformers and Contextual Embeddings

While CBOW and Skip-gram focus on individual word embeddings, modern semantic search leverages transformer-based models such as BERT or GPT for more sophisticated representations. Unlike static embeddings like Word2Vec, these models generate contextual embeddings. This means that the same word can have different embeddings depending on its context. For example, the word "bank" would have one vector when referring to a financial institution and another vector when referring to the side of a river.

In semantic search, these contextual embeddings are used to represent both the user query and the documents or knowledge base entries. When a user inputs a query like "How do I reset my password?" the model converts this query into an embedding that captures the query's semantic meaning. It then compares this query embedding with the embeddings of the documents to find the most relevant information based on their semantic similarity.

### Why Semantic Search?

Semantic search significantly enhances the accuracy and relevance of information retrieval systems by:

1. Understanding Context: It captures the meaning of words based on the surrounding context, allowing for more nuanced results.
2. Handling Synonyms: Semantic search can recognize synonyms and related concepts, so it returns results even if the user doesn't use the exact words present in the database.
3. Fighting Hallucinations: When integrated with language models, it grounds responses in real, meaningful content, reducing the chances of generating incorrect or irrelevant information.

### Conclusion

Semantic search empowers systems to move beyond simple keyword matching by leveraging deep learning models that understand the context and relationships between words. Whether it's chatbots answering user queries, search engines surfacing relevant documents, or systems making product recommendations, semantic search ensures that results are tailored to the true meaning behind the user's input. By incorporating techniques like CBOW, Skip-gram, and transformers, semantic search enables more intelligent and context-aware information retrieval, transforming how users interact with large datasets.

## Graph Search: Navigating Complex Relationships

When we talk about searching through data, we often think of text or document-based queries and solutions like vector search or semantic search. However, there are situations where the relationships between data points are as important—if not more important—than the individual pieces of data themselves. This is where graph search comes into play.

### What Is Graph Search?

At its core, a graph is a data structure composed of nodes (representing entities) and edges (representing relationships between entities). Graph search involves exploring these nodes and edges to find relevant information, particularly when the relationships between entities hold as much value as the entities themselves.

For example, imagine you're working with a social network like LinkedIn. Here, you're not just interested in isolated people (nodes); you're interested in how those people are connected (edges)—who works with whom, who is connected to whom, etc. This type of search is all about tracing paths and connections between nodes, uncovering relationships and patterns that might not be obvious through traditional search methods.

### How Graph Search Works

Graph search relies on traversal algorithms to explore the graph structure and find the relevant paths or nodes. Two primary methods of graph traversal are:

#### Breadth-First Search (BFS)

In BFS, you start from a given node and explore all its neighbours before moving on to their neighbours, layer by layer. It's a useful algorithm when you're looking for the shortest path between two nodes, such as finding the fewest degrees of connection between two people on a social network.

Here's a simple implementation of BFS in Python:

```python
from collections import deque

def bfs(graph, start, goal):
    queue = deque([[start]])
    visited = set([start])
    
    while queue:
        path = queue.popleft()
        node = path[-1]
        
        if node == goal:
            return path
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                new_path = list(path)
                new_path.append(neighbor)
                queue.append(new_path)
    
    return None

# Example usage
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

start = 'A'
goal = 'F'
path = bfs(graph, start, goal)
print(f"Shortest path from {start} to {goal}: {' -> '.join(path)}")
```

#### Depth-First Search (DFS)

In DFS, you dive deep into one path (following connections between nodes) before backtracking and exploring other paths. This can be useful when searching for specific relationships or when paths are long and complex.

Here's a simple implementation of DFS in Python:

```python
def dfs(graph, start, goal, path=None):
    if path is None:
        path = [start]
    
    if start == goal:
        return path
    
    for neighbor in graph[start]:
        if neighbor not in path:
            new_path = dfs(graph, neighbor, goal, path + [neighbor])
            if new_path:
                return new_path
    
    return None

# Example usage
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

start = 'A'
goal = 'F'
path = dfs(graph, start, goal)
print(f"A path from {start} to {goal}: {' -> '.join(path)}")
```

In both cases, the goal is to navigate the graph's structure to find meaningful connections, relationships, or patterns that would be difficult to uncover with other search methods.

### Why Use Graph Search?

Graph search is particularly effective when:

1. Data relationships are critical: Graph search shines in scenarios where the relationships between data points hold the key to valuable insights.
2. Pathfinding is necessary: Whether you're finding the shortest route in a transportation system or tracing how ideas flow in a network of research papers, graph search helps you navigate complex pathways.
3. Querying interconnected data: Many datasets—like social networks, recommendation systems, or knowledge graphs—are naturally structured as graphs. Graph search allows you to tap into the inherent structure of this data for better, more meaningful results.

### Graph Search vs. Vector/Semantic Search

While vector and semantic searches focus on capturing the content and meaning of documents or queries, graph search emphasizes relationships and connections between data points. It's not an either/or choice—each search technique has its strengths, depending on the use case.

- Vector Search: Great for finding items similar in content, such as movie recommendations based on genre or articles related to a specific topic.
- Semantic Search: Ideal for understanding the meaning behind a user's query and retrieving information that is contextually relevant, even if exact keywords don't match.
- Graph Search: Powerful when the connections between entities are crucial, such as in social networks, supply chains, or recommendation engines that rely on relationships rather than isolated content.
