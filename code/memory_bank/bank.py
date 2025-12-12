"""
Memory Bank core class.

Provides memory storage, retrieval, similarity search, and other functions.
Supports role-based filtering to ensure each agent can only access memories related to its role.
"""
import json
import os
import uuid
from typing import Callable, Optional
import numpy as np

from .types import MemoryEntry, MemoryUpdateEvent, MemoryOperation


class MemoryBank:
    """
    Memory Bank class.
    
    Responsible for storing all memory entries and providing similarity-based retrieval.
    Uses simple vector similarity search (via injected embedding function).
    Supports role-based filtering to ensure each agent only sees relevant memories.
    
    Attributes:
        entries: Dictionary of all memory entries {id: MemoryEntry}
        embeddings: Memory embedding vectors {id: np.ndarray}
        embed_fn: Embedding function that converts text to vectors
    """
    
    def __init__(self, embed_fn: Optional[Callable[[str], np.ndarray]] = None):
        """
        Initialize memory bank.
        
        Args:
            embed_fn: Optional embedding function. If None, uses SentenceTransformer.
        """
        self.entries: dict[str, MemoryEntry] = {}
        self.embeddings: dict[str, np.ndarray] = {}
        
        if embed_fn is None:
            # Default: use SentenceTransformer
            self.embed_fn = self._sentence_transformer_embedding
            self._init_sentence_transformer()
        else:
            self.embed_fn = embed_fn
    
    def _init_sentence_transformer(self):
        """Initialize SentenceTransformer model (called once on first use)"""
        try:
            from sentence_transformers import SentenceTransformer
            print("Loading SentenceTransformer model: all-MiniLM-L6-v2")
            self._st_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("SentenceTransformer model loaded successfully")
        except ImportError:
            print("Error: sentence-transformers not installed")
            print("Install with: pip install sentence-transformers")
            raise ImportError("sentence-transformers package is required")
    
    def _sentence_transformer_embedding(self, text: str) -> np.ndarray:
        """
        Use SentenceTransformer to generate embeddings (384 dimensions).
        """
        if not text or not text.strip():
            return np.zeros(384)
        
        # Generate normalized embedding
        embedding = self._st_model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return embedding
    
    def add(self, entry: MemoryEntry) -> MemoryUpdateEvent:
        """
        Add a new memory entry.
        
        Args:
            entry: Memory entry to add
        
        Returns:
            MemoryUpdateEvent recording this operation
        """
        if entry.id in self.entries:
            raise ValueError(f"Memory entry {entry.id} already exists. Use update() instead.")
        
        self.entries[entry.id] = entry
        # Compute and store embedding
        self.embeddings[entry.id] = self.embed_fn(entry.text)
        
        return MemoryUpdateEvent(
            entry_id=entry.id,
            operation=MemoryOperation.ADD,
            old_text=None,
            new_text=entry.text,
            reason="New memory added"
        )
    
    def update(self, entry_id: str, new_text: str, reason: str = "") -> MemoryUpdateEvent:
        """
        Update an existing memory entry.
        
        Args:
            entry_id: Memory ID to update
            new_text: New memory text
            reason: Update reason
        
        Returns:
            MemoryUpdateEvent recording this operation
        """
        if entry_id not in self.entries:
            raise ValueError(f"Memory entry {entry_id} not found.")
        
        old_entry = self.entries[entry_id]
        old_text = old_entry.text
        
        # Update entry
        old_entry.text = new_text
        old_entry.timestamp = datetime.now()
        # Update embedding
        self.embeddings[entry_id] = self.embed_fn(new_text)
        
        return MemoryUpdateEvent(
            entry_id=entry_id,
            operation=MemoryOperation.UPDATE,
            old_text=old_text,
            new_text=new_text,
            reason=reason or "Memory updated"
        )
    
    def delete(self, entry_id: str, reason: str = "") -> MemoryUpdateEvent:
        """
        Delete a memory entry.
        
        Args:
            entry_id: Memory ID to delete
            reason: Delete reason
        
        Returns:
            MemoryUpdateEvent recording this operation
        """
        if entry_id not in self.entries:
            raise ValueError(f"Memory entry {entry_id} not found.")
        
        old_entry = self.entries[entry_id]
        old_text = old_entry.text
        
        # Delete entry and embedding
        del self.entries[entry_id]
        if entry_id in self.embeddings:
            del self.embeddings[entry_id]
        
        return MemoryUpdateEvent(
            entry_id=entry_id,
            operation=MemoryOperation.DELETE,
            old_text=old_text,
            new_text=None,
            reason=reason or "Memory deleted"
        )
    
    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """
        Get memory entry by ID.
        
        Args:
            entry_id: Memory ID
        
        Returns:
            MemoryEntry or None (if not found)
        """
        return self.entries.get(entry_id)
    
    def all_entries(self, owner: Optional[str] = None) -> list[MemoryEntry]:
        """
        Get all memory entries (optionally filtered by owner).
        
        Args:
            owner: Optional owner filter (agent role name)
        
        Returns:
            List of memory entries
        """
        entries = list(self.entries.values())
        if owner is not None:
            entries = [e for e in entries if e.owner == owner]
        return entries
    
    def search(
        self,
        query: str,
        top_k: int = 20,
        owner: Optional[str] = None,
        min_similarity: float = 0.0
    ) -> list[MemoryEntry]:
        """
        Search memory entries based on similarity (supports role-based filtering).
        
        Args:
            query: Query text
            top_k: Return top k most similar results
            owner: Optional owner filter (agent role name, e.g., "Mathematician")
                   If provided, only returns memories for that role
            min_similarity: Minimum similarity threshold
        
        Returns:
            List of memory entries sorted by similarity (returns empty list [] if memory bank is empty)
        """
        # If memory bank is empty, return empty list directly (graceful handling)
        if not self.entries:
            return []
        
        # Compute query embedding
        query_embedding = self.embed_fn(query)
        
        # Compute similarity between all memories and query
        similarities = []
        for entry_id, entry in self.entries.items():
            # Filter by owner (role-based)
            if owner is not None and entry.owner != owner:
                continue
            
            if entry_id in self.embeddings:
                # Compute cosine similarity
                mem_embedding = self.embeddings[entry_id]
                similarity = np.dot(query_embedding, mem_embedding)
                
                # Filter by minimum similarity
                if similarity >= min_similarity:
                    similarities.append((similarity, entry))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in similarities[:top_k]]
    
    def save(self, filepath: str) -> None:
        """
        Save memory bank to JSON file (text only) and numpy file (embeddings).
        
        Creates two files:
        1. {filepath}: JSON with text (human-readable)
        2. {filepath}.npy: Numpy file with embeddings (fast loading)
        
        Args:
            filepath: Save path for JSON file
        """
        # Group memories by owner (agent role)
        data = {}
        embedding_data = {}
        
        for entry in self.entries.values():
            if entry.owner not in data:
                data[entry.owner] = {}
                embedding_data[entry.owner] = {}
            
            # Use a simple counter for each agent's memories
            memory_count = len(data[entry.owner]) + 1
            memory_id = str(memory_count)
            
            # Save text to JSON
            data[entry.owner][memory_id] = entry.text
            
            # Save embedding separately
            embedding_data[entry.owner][memory_id] = self.embeddings[entry.id]
        
        # Save JSON (text only, human-readable)
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # Save embeddings as numpy file
        embedding_filepath = filepath.replace('.json', '_embeddings.npz')
        np.savez_compressed(embedding_filepath, **{
            f"{owner}_{mem_id}": emb 
            for owner, mems in embedding_data.items() 
            for mem_id, emb in mems.items()
        })
    
    def load(self, filepath: str) -> None:
        """
        Load memory bank from JSON file.
        
        Loads text from JSON and embeddings from .npz file if available.
        If embeddings file doesn't exist, recomputes embeddings from text.
        
        Args:
            filepath: File path to JSON file
        """
        # Load text from JSON
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.entries = {}
        self.embeddings = {}
        
        # Try to load pre-computed embeddings
        embedding_filepath = filepath.replace('.json', '_embeddings.npz')
        precomputed_embeddings = {}
        
        if os.path.exists(embedding_filepath):
            try:
                loaded = np.load(embedding_filepath)
                precomputed_embeddings = {key: loaded[key] for key in loaded.files}
                print(f"Loaded precomputed embeddings from {embedding_filepath}")
            except Exception as e:
                print(f"Warning: Failed to load embeddings file: {e}")
        
        # Load memories
        for owner, memories in data.items():
            for memory_id, memory_text in memories.items():
                entry_id = f"mem_{owner}_{memory_id}"
                
                entry = MemoryEntry(
                    id=entry_id,
                    owner=owner,
                    text=memory_text
                )
                self.entries[entry_id] = entry
                
                # Use precomputed embedding if available, otherwise compute
                emb_key = f"{owner}_{memory_id}"
                if emb_key in precomputed_embeddings:
                    self.embeddings[entry_id] = precomputed_embeddings[emb_key]
                else:
                    # Recompute embedding from text
                    self.embeddings[entry_id] = self.embed_fn(memory_text)
    
    def create_entry(
        self,
        text: str,
        owner: str = "system"
    ) -> MemoryEntry:
        """
        Create a new memory entry (not automatically added to bank).
        
        This is a convenience method for generating a MemoryEntry with a unique ID.
        
        Args:
            text: Memory text (high-level experience summary)
            owner: Owner (agent role name, e.g., "Mathematician")
        
        Returns:
            New MemoryEntry (not yet added to bank)
        """
        entry_id = f"mem_{uuid.uuid4().hex[:8]}"
        return MemoryEntry(
            id=entry_id,
            owner=owner,
            text=text
        )

