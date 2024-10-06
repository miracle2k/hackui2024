import hashlib
from collections import defaultdict

class MinHashLSH:
    def __init__(self, num_hashes=100, bands=20):
        self.num_hashes = num_hashes
        self.bands = bands
        self.hash_functions = self._generate_hash_functions()
        self.document_sets = {}
        self.lsh_index = defaultdict(set)

    def _generate_hash_functions(self):
        return [lambda x, i=i: hashlib.sha256(f"{x}{i}".encode()).hexdigest() for i in range(self.num_hashes)]

    def _minhash_signature(self, document):
        words = set(document.lower().split())
        signature = [min(hash_func(word) for word in words) for hash_func in self.hash_functions]
        return signature

    def add_document(self, doc_id, document):
        signature = self._minhash_signature(document)
        self.document_sets[doc_id] = signature

        rows_per_band = self.num_hashes // self.bands
        for i in range(self.bands):
            band = tuple(signature[i * rows_per_band: (i + 1) * rows_per_band])
            self.lsh_index[band].add(doc_id)

    def find_similar_documents(self, query_doc, threshold=0.7):
        query_signature = self._minhash_signature(query_doc)
        candidate_docs = set()

        rows_per_band = self.num_hashes // self.bands
        for i in range(self.bands):
            band = tuple(query_signature[i * rows_per_band: (i + 1) * rows_per_band])
            candidate_docs.update(self.lsh_index.get(band, set()))

        similar_docs = []
        for doc_id in candidate_docs:
            similarity = self._jaccard_similarity(query_signature, self.document_sets[doc_id])
            if similarity >= threshold:
                similar_docs.append((doc_id, similarity))

        return sorted(similar_docs, key=lambda x: x[1], reverse=True)
    
    def has_similar_document(self, query_doc, threshold=0.7):
        query_signature = self._minhash_signature(query_doc)
        candidate_docs = set()
        rows_per_band = self.num_hashes // self.bands
        for i in range(self.bands):
            band = tuple(query_signature[i * rows_per_band: (i + 1) * rows_per_band])
            candidate_docs.update(self.lsh_index.get(band, set()))
        
        for doc_id in candidate_docs:
            similarity = self._jaccard_similarity(query_signature, self.document_sets[doc_id])
            if similarity >= threshold:
                return True
        return False

    def _jaccard_similarity(self, sig1, sig2):
        return sum(1 for a, b in zip(sig1, sig2) if a == b) / self.num_hashes
