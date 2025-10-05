# inventory.py
from typing import Optional, Any, Generic, TypeVar, Dict

T = TypeVar('T')

class Node(Generic[T]):
    def __init__(self, data: T):
        self.data: T = data
        self.next: Optional[Node[T]] = None
        self.prev: Optional[Node[T]] = None

class DoublyLinkedList(Generic[T]):
    def __init__(self):
        self.head: Optional[Node[T]] = None
        self.tail: Optional[Node[T]] = None
        self.size: int = 0
        self._id_map: Dict[T, Node[T]] = {} 

    def append(self, data: T) -> None:
        new_node = Node(data)
        if self.tail is None:
            self.head = self.tail = new_node
        else:
            self.tail.next = new_node
            new_node.prev = self.tail
            self.tail = new_node
        
        self._id_map[data] = new_node
        self.size += 1

    def remove(self, data: T) -> None:
        if data not in self._id_map:
            raise ValueError(f"Dato '{data}' no encontrado en la lista.")
        
        node_to_remove = self._id_map[data]
        
        if node_to_remove.prev is not None:
            node_to_remove.prev.next = node_to_remove.next
        else:
            self.head = node_to_remove.next
            
        if node_to_remove.next is not None:
            node_to_remove.next.prev = node_to_remove.prev
        else:
            self.tail = node_to_remove.prev

        del self._id_map[data]
        self.size -= 1
        
    def get_at(self, index: int) -> Optional[T]:
        if index < 0 or index >= self.size:
            return None
        
        current = self.head
        for _ in range(index):
            if current is None:
                return None
            current = current.next
        
        return current.data if current else None

    def __len__(self):
        return self.size
    
    def __iter__(self):
        current = self.head
        while current:
            yield current.data
            current = current.next